"""Deterministic shipped-answer checks. Stdlib-only; also runs standalone on workers.

This is a bounded static analysis, not a proof that arbitrary code hides no answers.
It never executes package code or treats intentionally shipped target CSV columns as
leaks. All files are inventoried and scanned; unsupported containers fail closed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

VERSION = "1.0.1"
LIMIT = 2 * 1024 * 1024
MAX_FINDINGS = 200
PLACEHOLDER = re.compile(r"@@[A-Za-z][A-Za-z0-9_]*@@")
FIXTURE = re.compile(r"fixture|known_event|answer_table|expected_answers", re.I)
ANSWER = re.compile(r"(?:^|_)(?:label|answer|verdict|escape|lift30|exceedance)(?:$|_)|expected_(?:outcome|target)", re.I)
SALT = re.compile(r"(?:^|_)(?:salt|pepper)(?:$|_)", re.I)
IDENTIFIER = re.compile(r"\b(?:\d{1,4}-[A-Za-z0-9-]{5,}|[A-Z]{2,8}\d{3,}|\d{4}-\d{2}-\d{2})\b")
OUTCOME = re.compile(r"\b(?:label|target|outcome|expected_verdict)\s*[=:]?\s*[`\"']?[01]\b|(?:->|→)\s*[01]\b", re.I)
DATES = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _has_answer(value):
    if isinstance(value, dict):
        return any(ANSWER.search(str(k)) or _has_answer(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        # Fixed rows with an ID and binary answer, not aggregate counts/exclusions.
        if (len(value) >= 2 and isinstance(value[-1], (int, bool)) and value[-1] in (0, 1)
                and any(IDENTIFIER.search(s) for s in _strings(value[:-1]))):
            return True
        return any(_has_answer(v) for v in value)
    return isinstance(value, str) and bool(OUTCOME.search(value))


def _fixture_ids(value):
    return any(IDENTIFIER.search(s) for s in _strings(value))


def scan_text(text, name, *, scored_ids=(), windows=()):
    findings = []
    seen = set()

    def add(rule, line, detail):
        key = (rule, line)
        if key not in seen and len(findings) < MAX_FINDINGS:
            seen.add(key)
            findings.append(dict(rule=rule, file=name, line=line, detail=detail))

    for m in PLACEHOLDER.finditer(text):
        add('unresolved_placeholder', text.count('\n', 0, m.start()) + 1,
            'Unresolved template token: ' + m.group())

    # Context windows span wrapped prose. No recovered answer values enter diagnostics.
    lines = text.splitlines()
    id_set = set(scored_ids)
    for i, line in enumerate(lines):
        context = ' '.join(lines[max(0, i - 2):i + 4])
        if OUTCOME.search(context):
            if any(token in id_set for token in re.findall(r"[\w:./-]+", line)):
                add('scored_row_answer', i + 1, 'Scored-row identifier appears beside an explicit outcome')
            for date in DATES.findall(line):
                if any(start <= date < end for start, end in windows):
                    add('scored_window_example', i + 1,
                        'Dated outcome example falls in an eval/holdout window; use train or abstract values')

    def inspect(value, path='', fixture_context=False):
        if isinstance(value, dict):
            for key, child in value.items():
                location = path + '.' + str(key)
                if SALT.search(str(key)) and isinstance(child, str) and child:
                    add('embedded_salt', 1, 'Literal pseudonym salt at ' + location)
                flagged = fixture_context or bool(FIXTURE.search(str(key)))
                if flagged and _has_answer(child):
                    add('fixture_answers', 1, 'Fixture answer data at ' + location)
                elif flagged and _fixture_ids(child):
                    add('raw_fixture_ids', 1, 'Raw fixture identifiers at ' + location)
                inspect(child, location, flagged)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child, path, fixture_context)

    if name.endswith('.json') or text.lstrip().startswith(('{', '[')):
        try:
            inspect(json.loads(text))
        except (ValueError, RecursionError):
            if name.endswith('.json'):
                add('unreadable_json', 1, 'Cannot inspect malformed JSON')
            else:
                # JSON Lines is common in generated evidence; a renamed stream
                # must not bypass structural fixture/salt inspection.
                for line_number,line in enumerate(text.splitlines(),1):
                    if line.lstrip().startswith(('{','[')):
                        try:
                            inspect(json.loads(line),f'line_{line_number}')
                        except (ValueError,RecursionError):
                            pass  # ordinary Markdown links and prose are still scanned above
    if name.endswith('.py') or re.search(r'(?im)^\s*\w*(?:fixture|salt|answer_table)\w*\s*=', text):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            add('unreadable_python', 1, 'Cannot inspect malformed Python')
            return findings
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, RecursionError):
                continue
            for var in names:
                if SALT.search(var) and isinstance(value, (str, bytes)) and value:
                    add('embedded_salt', node.lineno, 'Literal pseudonym salt: ' + var)
                if FIXTURE.search(var):
                    if _has_answer(value):
                        add('fixture_answers', node.lineno, 'Literal fixture answer table: ' + var)
                    elif _fixture_ids(value):
                        add('raw_fixture_ids', node.lineno, 'Literal fixture identifiers: ' + var)
                # Detect differently named tables too, without executing or importing them.
                if isinstance(value, dict) and _has_answer(value) and _fixture_ids(value):
                    add('answer_lookup', node.lineno, 'Literal identifiers mapped to outcomes: ' + var)
    return findings


def scored_context(dataset):
    """Read only identifiers and declared windows, never target values."""
    dataset = Path(dataset)
    quality = json.loads((dataset / 'quality.json').read_text())
    ids, windows, splits = set(), [], set()
    for split, relative in quality.get('splits', {}).items():
        path = dataset / relative
        if path.is_symlink() or not path.resolve().is_relative_to(dataset.resolve()):
            raise ValueError('split path escapes artifact')
        splits.add(path.resolve())
        if split not in ('eval', 'holdout'):
            continue
        window = quality.get('split_windows', {}).get(split, {})
        if window.get('start') and window.get('end'):
            windows.append((window['start'][:10], window['end'][:10]))
        if not path.is_file():
            continue  # packaging may use remote CSVs; context scope is reported
        with path.open(newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            indexes = [header.index(c) for c in quality.get('id_columns', [])
                       if c in header and c != quality.get('target')]
            for row in reader:
                for i in indexes:
                    if i < len(row) and len(row[i]) >= 6:
                        ids.add(row[i])
    return ids, windows, splits


def scan_tree(root, *, dataset=None):
    root = Path(root)
    findings, files = [], {}
    ids, windows, splits = set(), [], set()
    if dataset is not None:
        ids, windows, splits = scored_context(dataset)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('scan root must be a regular directory')
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(dict(rule='unsafe_path', file=name, line=0, detail='Symlink is not shippable'))
            continue
        if not path.is_file():
            continue
        files[name] = sha256(path)
        if path.resolve() in splits:
            continue  # intentionally labeled split; its identifiers were inspected above
        if path.stat().st_size > LIMIT:
            findings.append(dict(rule='uninspected_file', file=name, line=0,
                                 detail='Non-split file exceeds static scan limit; unpack or reduce it'))
            continue
        data = path.read_bytes()
        if data.startswith((b'PK\x03\x04', b'\x1f\x8b', b'%PDF')):
            findings.append(dict(rule='uninspected_container', file=name, line=0,
                                 detail='Unpack/extract document text for answer scanning'))
            continue
        # Scan unknown extensions too. A file rename must not hide a prose answer.
        try:
            source = data.decode('utf-8')
            if '\x00' in source:
                raise ValueError('binary content')
        except (UnicodeDecodeError, ValueError):
            findings.append(dict(rule='uninspected_binary', file=name, line=0,
                                 detail='Provide inspectable text for non-split binary content'))
            continue
        findings.extend(scan_text(source, name,
                                  scored_ids=ids, windows=windows))
    findings = sorted(findings, key=lambda f: (f['file'], f['line'], f['rule']))
    return dict(version=VERSION, ok=not findings, files=files,
                tree_digest=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                findings=findings[:MAX_FINDINGS], finding_count=len(findings),
                truncated=len(findings) > MAX_FINDINGS,
                scope={'identifier_count': len(ids), 'scored_windows': windows,
                       'split_files_excluded': len(splits), 'executes_code': False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    try:
        result = scan_tree(args.root, dataset=args.dataset)
    except (OSError, ValueError, KeyError) as exc:
        result = {'version': VERSION, 'ok': False, 'error': str(exc)}
    body = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.write_text(body)
    print(body, end='')
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
