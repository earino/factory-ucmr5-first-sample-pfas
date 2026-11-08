"""Canonical construct -> qualify -> accept -> existing runner -> package worker."""
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

try:
    from .source_io import sha, capture_terms
except ImportError:
    from source_io import sha, capture_terms


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def persistence_predictions(training, validation, prior, direction=1):
    """Training-only range reference, with an explicit prior for absent history."""
    def number(value):
        if value is None or value == '':
            return None
        value = float(value)
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError('Persistence reference contains an infinite value')
        return value
    train = [number(v) for v in training]
    observed = [v for v in train if v is not None]
    values = [number(v) for v in validation]
    lo, hi = (min(observed), max(observed)) if observed else (None, None)
    usable = lo is not None and hi > lo
    predictions = []
    for value in values:
        if value is None or not usable:
            predictions.append(prior)
        else:
            p = min(1., max(0., (value - lo) / (hi - lo)))
            predictions.append(1 - p if direction == -1 else p)
    return predictions, {
        'training_observed_rows': len(observed), 'training_missing_rows': len(train) - len(observed),
        'validation_missing_rows': sum(v is None for v in values),
        'prior_fallback_rows': sum(v is None or not usable for v in values),
        'training_min': lo, 'training_max': hi, 'direction': direction,
        'fallback_probability': prior,
        'uninformative_training_range': not usable,
        'method': 'Training min/max ranking; missing history or no varying training range uses training label prior. Not fitted calibration.'}


def finalize(extract, descriptor, source):
    quality = dict(descriptor['quality'])
    quality['name'] = source['dataset_id']
    quality['positive_label'] = 1
    quality['splits'] = {'train': 'public/train.csv', 'eval': 'public/eval.csv', 'holdout': 'private/holdout.csv'}
    quality['expected_duplicate_ids'] = 0
    quality['positive_events'] = {}
    rows, rates, columns = {}, {}, None
    for split, relative in quality['splits'].items():
        with (extract / relative).open(newline='') as stream:
            reader = csv.DictReader(stream)
            required = {quality['target'], quality['event_key_column']}
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f'{split}: declared columns {missing!r} absent from CSV headers '
                                 f'{reader.fieldnames!r}; quality.target={quality["target"]!r}. '
                                 'Make the builder headers and descriptor agree; columns are not renamed.')
            if columns is not None and columns != reader.fieldnames:
                raise ValueError('Split schemas differ')
            columns = reader.fieldnames
            count = positives = 0
            events = set()
            for row in reader:
                label = row[quality['target']]
                if label not in ('0', '1'):
                    raise ValueError('Target must be binary 0/1')
                count += 1
                if label == '1':
                    positives += 1
                    # Match the existing gate's conservative event-day grouping,
                    # including its handling of empty keys. Do not claim a full-ID
                    # count where qualification will recompute date prefixes.
                    key = row[quality['event_key_column']]
                    if key:
                        events.add(key[:10])
            rows[split] = count
            rates[split] = positives / count if count else 0
            quality['positive_events'][split] = {'positives': positives, 'distinct_events': len(events)}
    write(extract / 'quality.json', quality)
    write(extract / 'meta.json', {'name': source['dataset_id'], 'description': source['task']['question'],
          'target': quality['target'], 'positive_label': 1, 'id_columns': quality['id_columns'],
          'columns': columns, 'rows': rows, 'positive_rate': rates, 'split': quality['split_windows'],
          'split_description': source['task']['split_policy']})
    return quality


def evaluate(extract, workspace, out, quality, references):
    # This is a trusted baseline-development run. It does not implement or claim
    # the deferred chronological agent-scoring adapter or a holdout benchmark.
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, brier_score_loss
    work = Path(tempfile.mkdtemp(prefix='factory-baseline-'))
    (work / 'data').mkdir()
    for name in ('train.py', 'validate.py', 'validate.sh'):
        shutil.copyfile(workspace / name, work / name)
    for split in ('train', 'eval'):
        shutil.copyfile(extract / 'public' / (split + '.csv'), work / 'data' / (split + '.csv'))
    meta = json.loads((extract / 'meta.json').read_text())
    write(work / 'task.json', {k: meta[k] for k in ('name', 'description', 'target', 'positive_label',
                                                  'id_columns', 'split', 'columns', 'rows')})
    previous = Path.cwd()
    try:
        os.chdir(work)
        with (out / 'baseline.log').open('w') as log, redirect_stdout(log):
            ns = runpy.run_path(str(work / 'train.py'), run_name='__train__')
        train, validation = pd.read_csv(work / 'data/train.csv'), pd.read_csv(work / 'data/eval.csv')
        target = quality['target']
        y = validation[target].to_numpy()
        predictions = np.asarray(ns['predict_proba'](validation.drop(columns=[target])), dtype=float).ravel()
        if len(predictions) != len(y) or not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any():
            raise ValueError('Existing runner prediction contract failed')
        scores = {'full_model': predictions, 'majority_prior': np.full(len(y), float(train[target].mean()))}
        entities = references.get('entity_columns', [])
        if any(c not in train or c == target for c in entities):
            raise ValueError('Invalid calendar/identity reference columns')
        months_train = pd.to_datetime(train[quality['time_column']], utc=True).dt.month
        months_eval = pd.to_datetime(validation[quality['time_column']], utc=True).dt.month
        keys_train = [tuple(str(row[c]) for c in entities) + (int(month),)
                      for (_, row), month in zip(train.iterrows(), months_train)]
        keys_eval = [tuple(str(row[c]) for c in entities) + (int(month),)
                     for (_, row), month in zip(validation.iterrows(), months_eval)]
        totals = {}
        for key, label in zip(keys_train, train[target]):
            total, count = totals.get(key, (0, 0)); totals[key] = (total + int(label), count + 1)
        prior = float(train[target].mean())
        scores['calendar_identity'] = np.array([totals[k][0] / totals[k][1] if k in totals else prior for k in keys_eval])
        persistence = references.get('persistence_column')
        persistence_details = None
        if persistence is not None:
            if persistence not in quality['features'] or persistence == target:
                raise ValueError('Persistence must use a declared available feature')
            predictions, persistence_details = persistence_predictions(
                train[persistence], validation[persistence], prior, references.get('persistence_direction', 1))
            scores['persistence'] = np.asarray(predictions)
        def metrics(mask):
            labels = y[mask]
            return {name: {'auc': float(roc_auc_score(labels, p[mask])) if len(set(labels)) == 2 else None,
                           'brier': float(brier_score_loss(labels, p[mask]))} for name, p in scores.items()}
        years = pd.to_datetime(validation[quality['time_column']], utc=True).dt.year.to_numpy()
        result = {'protocol': 'Existing unmodified runner baseline on public validation only; no holdout scoring',
                  'rows': len(y), 'metrics': metrics(np.ones(len(y), dtype=bool)),
                  'by_year': {str(year): {'rows': int((years == year).sum()),
                                        'positive_rate': float(y[years == year].mean()),
                                        'metrics': metrics(years == year)} for year in sorted(set(years))},
                  'majority_accuracy': float(np.mean(y == int(prior >= .5))),
                  'calendar_identity_unseen_rows': sum(k not in totals for k in keys_eval),
                  'persistence': 'Not applicable; see construction notes' if persistence is None else
                      persistence_details,
                  'limitations': ['No uncertainty interval; observed differences do not establish universal headroom'],
                  'runner_files': {name: sha(workspace / name) for name in ('train.py', 'validate.py', 'validate.sh')}}
        write(out / 'evaluation.json', result)
        return result
    finally:
        os.chdir(previous)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path('/output'))
    parser.add_argument('--source-cache', type=Path)
    args = parser.parse_args()
    workspace, out = args.workspace.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty output directory; existing work is never cleared')
    source = json.loads((workspace / 'source.json').read_text())
    descriptor = json.loads((workspace / 'construction.json').read_text())
    extract = out / 'extract'
    cache = args.source_cache or Path(tempfile.mkdtemp(prefix='factory-source-'))
    extract.mkdir()
    write(out / 'LICENSE-EVIDENCE.json', capture_terms(source['source'], cache))
    spec = importlib.util.spec_from_file_location('generated_builder', workspace / 'build.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    checks = builder.build(extract, cache)
    if not isinstance(checks, list) or not checks or any(c.get('passed') is not True for c in checks):
        raise ValueError('Construction source checks are missing or failed')
    write(out / 'source-checks.json', {'checks': checks})
    units = [check for check in checks if check.get('kind') == 'unit']
    if units:
        descriptor['quality']['unit_validation'] = {
            'checks': len(units), 'mismatches': 0, 'source': 'source-checks.json: unit checks',
            'evidence_sha256': sha(out / 'source-checks.json')}
    if not (cache / 'sources.json').is_file():
        raise ValueError('No source retrieval provenance recorded')
    quality = finalize(extract, descriptor, source)
    gate = workspace / 'qualify_dataset.py'
    subprocess.run([sys.executable, str(gate), str(extract), '--record-battery', '--json',
                    '--report', str(out / 'qualification.json')], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(gate), '--accept', str(out / 'qualification.json'), str(extract)], check=True)
    evaluate(extract, workspace, out, quality, descriptor['references'])
    subprocess.run([sys.executable, str(gate), '--accept', str(out / 'qualification.json'), str(extract)], check=True)
    # The canonical package step is added by the controller bundle. It runs after
    # exact-byte acceptance and never modifies extract.
    from dataset_package import package
    package(workspace, out, cache, source, descriptor)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        if Path('/output').is_dir():
            write(Path('/output/failure.json'), {'error': type(exc).__name__,
                  'diagnostic': traceback.format_exc()[-16000:]})
        raise
