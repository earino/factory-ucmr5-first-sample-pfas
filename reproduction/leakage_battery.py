"""Two-phase deterministic battery: freeze feature-only predictions, then score.

All output lives outside the shipped artifact. Prediction vectors are audit-only
and must never be packaged with data; only the compact aggregate receipt ships.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import zipfile

try:  # Same canonical files are copied beside the standalone worker gate.
    from .artifact_scrub import sha256, scan_tree
    from .recovery_recipes import Panel, exact_recipes, fixture_recipes, declared_recipes, number, NAN, Join
except ImportError:
    from artifact_scrub import sha256, scan_tree
    from recovery_recipes import Panel, exact_recipes, fixture_recipes, declared_recipes, number, NAN, Join

VERSION = '1.0.0'
MAX_COLUMNS = 128
TOKEN_LIMIT = 80
FLAG_LIMIT = 40


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def safe_file(root, relative):
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('artifact file must be a regular contained file: ' + str(relative))
    return path


def code_identity():
    names = ('leakage_battery.py', 'recovery_recipes.py', 'artifact_scrub.py')
    return {name: sha256(Path(__file__).with_name(name)) for name in names}


def input_identity(dataset):
    """Ignore only the generated aggregate receipt, avoiding a self-referential hash."""
    files = {}
    for path in sorted(Path(dataset).rglob('*')):
        if path.is_symlink():
            raise ValueError('artifact symlink refused')
        if path.is_file():
            name = path.relative_to(dataset).as_posix()
            if name == 'quality.json':
                quality = json.loads(path.read_text())
                quality.pop('adversarial_battery', None)
                files[name] = digest(quality)
            else:
                files[name] = sha256(path)
    return files


def load_features(dataset, quality):
    target = quality['target']; columns = {}; ranges = {}; offset = 0
    for split in ('train', 'eval', 'holdout'):
        path = safe_file(dataset, quality['splits'][split])
        with path.open(newline='') as stream:
            reader = csv.reader(stream); header = next(reader)
            if len(header) != len(set(header)) or header.count(target) != 1:
                raise ValueError('duplicate columns or missing target in ' + split)
            features = [c for c in header if c != target]
            if len(features) > MAX_COLUMNS:
                raise ValueError('battery column limit exceeded; incomplete inspection refuses')
            if columns and set(features) != set(columns):
                raise ValueError('feature schemas differ between splits')
            for c in features:
                columns.setdefault(c, [])
            positions = [(i,c) for i,c in enumerate(header) if c != target]
            start = offset
            for row in reader:
                if len(row) != len(header):
                    raise ValueError('ragged CSV row in ' + split)
                for i,c in positions:
                    columns[c].append(row[i])  # Target cell never enters the panel.
                offset += 1
            ranges[split] = (start, offset)
    return Panel(columns, ranges, target)


def read_labels(dataset, quality, split):
    with safe_file(dataset, quality['splits'][split]).open(newline='') as stream:
        reader = csv.DictReader(stream)
        return [int(row[quality['target']] == str(quality['positive_label'])) for row in reader]


def basic_probes(p):
    for split in ('eval', 'holdout'):
        rows = range(*p.ranges[split])
        yield 'row_order', 'statistical', split, rows, list(map(float, rows))
        for c in sorted(p.cols):
            values = [p.cols[c][i] for i in rows]
            nums = list(map(number, values))
            if any(math.isfinite(v) for v in nums):
                yield 'numeric:'+c, 'statistical', split, rows, nums
                yield 'absolute:'+c, 'statistical', split, rows, list(map(abs, nums))
            ranks = {value: i for i,value in enumerate(sorted(set(p.cols[c])))}
            yield 'lexical:'+c, 'statistical', split, rows, [float(ranks[v]) for v in values]
            yield 'length:'+c, 'statistical', split, rows, list(map(len, values))
            yield 'missing:'+c, 'statistical', split, rows, [float(v in ('','-1','NA','N/A','nan')) for v in values]
    # Tokens are selected by feature frequency before any eval/holdout label read.
    for c in sorted(p.cols):
        if max(map(len, p.cols[c]), default=0) <= 15:
            continue
        counts = Counter()
        for value in p.cols[c]:
            counts.update(tokens(value))
        selected = sorted((t for t,n in counts.items() if n >= 10),
                          key=lambda t: (-counts[t], t))[:TOKEN_LIMIT]
        for token in selected:
            for split in ('eval', 'holdout'):
                rows = range(*p.ranges[split])
                yield 'token:'+c+':'+token, 'statistical', split, rows, [
                    float(token in tokens(p.cols[c][i])) for i in rows]


def tokens(value):
    words = re.findall(r'\w+', value.lower())
    return set(words) | {' '.join(pair) for pair in zip(words, words[1:])}


def generic_joins(p, quality):
    """Bounded schema-independent exploratory joins, never promoted by lucky accuracy."""
    time = quality.get('time_column')
    if time not in p.cols:
        return
    names = [c for c in sorted(p.cols) if c != time and re.search(
        r'id|island|country|origin|asylum|station|site|authority|beds|rating|sector', c)]
    # Unique row IDs cannot provide successor donors. Choose repeated feature keys.
    names = [c for c in names if 1 < len(set(p.cols[c])) < max(2, p.n//2)][:8]
    keys = [(c,) for c in names] + list(itertools.combinations(names, 2))
    quasi = ['local_authority','care_home_beds','hsca_start_year','latest_overall_rating','provider_sector']
    if all(c in p.cols for c in quasi):
        keys.append(tuple(quasi))
    operands = [c for c in sorted(p.cols) if re.search(r'lag|prior|last|mean|rate|count|since', c)][:8]
    for scope, split, rows, donors in p.scopes():
        for key in keys:
            join = Join(p, donors, key, time)
            for offset in (1, 2, 7, 30, 365, 366):
                for column in operands:
                    yield f'{scope}:explore:{"+".join(key)}:{offset}:{column}', 'heuristic', split, rows, [
                        join.at(i, offset, column) for i in rows]
            # Quasi-identifier survival/missingness is a heuristic, not a label identity.
            for offset in (365, 366):
                values = []
                for i in rows:
                    d = join.days[i]
                    own = join.index.get((p.key(key,i),d))
                    future = join.index.get((p.key(key,i),d+offset)) if d is not None else None
                    values.append(0.0 if own is not None and future is not None else NAN)
                yield f'{scope}:quasi_successor:{"+".join(key)}:{offset}', 'heuristic', split, rows, values


def lookup_controls(p, train_labels):
    """Train-only single/pair tables and unanimity coverage; no fitting on scored labels."""
    train = range(*p.ranges['train'])
    columns = sorted(p.cols)
    for keys in itertools.chain(((c,) for c in columns), itertools.combinations(columns, 2)):
        table = defaultdict(lambda: [0,0])
        for i,y in zip(train, train_labels):
            stats = table[p.key(keys,i)]; stats[0] += 1; stats[1] += y
        for split in ('eval','holdout'):
            rows = range(*p.ranges[split]); values = []; unanimous = []
            for i in rows:
                n, yes = table.get(p.key(keys,i), (0,0))
                values.append(yes/n if n else NAN)
                unanimous.append(yes/n if n >= 2 and yes in (0,n) else NAN)
            yield 'lookup:'+'+'.join(keys), 'supervised', split, rows, values
            yield 'lookup_unanimous:'+'+'.join(keys), 'supervised', split, rows, unanimous


def freeze(dataset, output, quality=None):
    dataset, output = Path(dataset), Path(output)
    if output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError('audit predictions must be outside the shipped artifact')
    if output.exists():
        raise ValueError('prediction output must be a fresh owned directory')
    output.mkdir(parents=True)
    quality = quality or json.loads((dataset/'quality.json').read_text())
    identity = input_identity(dataset)
    p = load_features(dataset, quality)
    records = []; attempted = Counter(); skipped = Counter()

    def write_vectors(filename, probes):
        with zipfile.ZipFile(output/filename, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name, kind, split, rows, values in probes:
                attempted[kind] += 1
                if len(values) != len(rows):
                    raise ValueError('prediction alignment failed: '+name)
                values = [float(v) if math.isfinite(v) else NAN for v in values]
                finite = [v for v in values if math.isfinite(v)]
                if not finite or (kind in ('statistical','supervised','heuristic') and len(set(finite)) <= 1):
                    # Preserve constant unanimous controls and quasi joins: their coverage matters.
                    if not finite or not (name.startswith('lookup_unanimous:') or ':quasi_successor:' in name):
                        skipped[kind] += 1
                        continue
                vector = array('d', values)
                if sys.byteorder != 'little':
                    vector.byteswap()
                payload = vector.tobytes(); member = str(len(records))+'.bin'
                info = zipfile.ZipInfo(member, date_time=(1980,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload)
                records.append({'name':name,'kind':kind,'split':split,'archive':filename,
                                'member':member,'rows':len(rows),'sha256':hashlib.sha256(payload).hexdigest()})

    write_vectors('blind.zip', itertools.chain(basic_probes(p), generic_joins(p,quality),
                  exact_recipes(p), fixture_recipes(p,dataset), declared_recipes(p,quality)))
    blind_sha = sha256(output/'blind.zip')  # Seal blind vectors before reading even train targets.
    write_vectors('controls.zip', lookup_controls(p, read_labels(dataset,quality,'train')))
    if identity != input_identity(dataset):
        raise ValueError('artifact changed during prediction freeze')
    manifest = {'version':VERSION, 'code':code_identity(), 'inputs':identity,
                'input_digest':digest(identity), 'blind_sha256':blind_sha,
                'controls_sha256':sha256(output/'controls.zip'), 'vectors':records,
                'attempted':dict(attempted), 'constant_or_uncovered':dict(skipped),
                'scope':{'scored_labels_read':False,'train_labels':'supervised controls only',
                         'donors':['self','all'], 'max_columns':MAX_COLUMNS,
                         'tokens_per_column':TOKEN_LIMIT, 'exploratory_keys':8,'exploratory_operands':8}}
    (output/'freeze.json').write_bytes(canonical(manifest)+b'\n')
    return manifest


def auc(labels, scores):
    positive = sum(labels); negative = len(labels)-positive
    if not positive or not negative:
        return None
    pairs = sorted(zip(scores,labels)); total = 0; i = 0
    while i < len(pairs):
        end = i+1
        while end < len(pairs) and pairs[end][0] == pairs[i][0]:
            end += 1
        total += sum(y for _,y in pairs[i:end]) * ((i+1+end)/2)
        i = end
    return (total-positive*(positive+1)/2)/(positive*negative)


def metric(labels, values):
    covered = [(y,v) for y,v in zip(labels,values) if math.isfinite(v)]
    yy, vv = zip(*covered) if covered else ([],[])
    binary = bool(vv) and all(v in (0,1) for v in vv)
    return {'rows':len(labels), 'covered':len(vv), 'errors':sum(y!=v for y,v in covered) if binary else None,
            'auc_covered':auc(yy,vv), 'auc_full':auc(labels,[v if math.isfinite(v) else .5 for v in values])}


def score(dataset, output):
    dataset, output = Path(dataset), Path(output)
    frozen = json.loads((output/'freeze.json').read_text())
    if frozen['version'] != VERSION or frozen['code'] != code_identity():
        raise ValueError('battery implementation changed after freeze')
    if frozen['inputs'] != input_identity(dataset):
        raise ValueError('artifact changed after prediction freeze')
    for name, key in [('blind.zip','blind_sha256'),('controls.zip','controls_sha256')]:
        if sha256(output/name) != frozen[key]:
            raise ValueError('prediction archive changed after freeze')
    quality = json.loads((dataset/'quality.json').read_text())
    # First scored-label access occurs here, after every vector and input has been sealed.
    labels = {s:read_labels(dataset,quality,s) for s in ('eval','holdout')}
    exact, flags, failures, detail_files = [], [], [], {}
    detail_lines, detail_bytes = [], 0

    def flush_details():
        if detail_lines:
            name = f'results-{len(detail_files):04d}.jsonl'
            path = output/name
            path.write_bytes(b''.join(detail_lines))
            detail_files[name] = {'sha256':sha256(path),'bytes':path.stat().st_size}
            detail_lines.clear()

    try:
        for archive_name in ('blind.zip','controls.zip'):
            with zipfile.ZipFile(output/archive_name) as archive:
                for record in frozen['vectors']:
                    if record['archive'] != archive_name:
                        continue
                    payload = archive.read(record['member'])
                    if hashlib.sha256(payload).hexdigest() != record['sha256']:
                        raise ValueError('prediction vector changed')
                    values = array('d'); values.frombytes(payload)
                    if sys.byteorder != 'little': values.byteswap()
                    if len(values) != len(labels[record['split']]):
                        raise ValueError('scored labels do not align with frozen vector')
                    result = {k:record[k] for k in ('name','kind','split','sha256')}
                    result.update(metric(labels[record['split']],values))
                    line = canonical(result)+b'\n'
                    if detail_bytes+len(line) > 240*1024:
                        flush_details(); detail_bytes = 0
                    detail_lines.append(line); detail_bytes += len(line)
                    if record['kind'] == 'exact':
                        exact.append(result)
                        if result['covered'] >= 1 and result['errors'] == 0:
                            failures.append({'name':result['name'],'split':result['split'],
                                             'covered':result['covered'],'errors':0})
                    else:
                        a = result['auc_covered']
                        if a is not None and max(a,1-a) > .95:
                            flags.append(result)
    finally:
        flush_details()
    if frozen['inputs'] != input_identity(dataset):
        raise ValueError('artifact changed during scoring')
    flags.sort(key=lambda r: (-max(r['auc_covered'],1-r['auc_covered']),r['name'],r['split']))
    result = {'version':VERSION, 'ok':not failures, 'input_digest':frozen['input_digest'],
              'code':frozen['code'], 'freeze_sha256':sha256(output/'freeze.json'),
              'all_results_sha256':digest(detail_files), 'detail_files':detail_files, 'exact':exact,
              'failures':failures, 'flags':flags[:FLAG_LIMIT], 'flag_count':len(flags),
              'attempted':frozen['attempted'], 'constant_or_uncovered':frozen['constant_or_uncovered'],
              'scope':frozen['scope']}
    if len(canonical(result)) >= 256*1024:
        raise ValueError('compact battery report exceeds collection limit; inspection incomplete')
    (output/'battery.json').write_bytes(canonical(result)+b'\n')
    return result


def run(dataset, output):
    freeze(dataset,output)
    return score(dataset,output)


def record_quality(dataset, output):
    """Explicit construction-only step. Never call this on a frozen historical artifact."""
    dataset = Path(dataset)
    result = run(dataset, output)
    path = dataset/'quality.json'
    quality = json.loads(path.read_text())
    quality['adversarial_battery'] = result
    path.write_text(json.dumps(quality, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset, args.output)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['ok'] else 1)
