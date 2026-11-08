"""Build one immutable manifest for both destinations from accepted worker bytes."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

try:
    from .source_io import sha
except ImportError:
    from source_io import sha

REPRODUCTION = ('build.py', 'construction.json', 'source.json', 'dataset_run.py',
                'dataset_package.py', 'source_io.py', 'qualify_dataset.py', 'artifact_scrub.py',
                'leakage_battery.py', 'recovery_recipes.py', 'train.py', 'validate.py', 'validate.sh',
                'requirements.txt', 'get_dataset.py', 'provenance.json', 'LICENSE-MIT.txt', 'LICENSE-CC0-1.0.txt')


def package(workspace, out, cache, source, descriptor):
    qualification = json.loads((out / 'qualification.json').read_text())
    if qualification.get('ok') is not True:
        raise ValueError('Cannot package a failed qualification')
    artifact = qualification['artifact']['artifact_version']['digest']
    reproduction = out / 'reproduction'
    reproduction.mkdir()
    for name in REPRODUCTION:
        shutil.copyfile(workspace / name, reproduction / name)
    # Preserve the exact upstream bytes for replay without relying on mutable URLs.
    with (out / 'source-snapshot.tar.gz').open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for path in sorted(cache.iterdir()):
                if path.is_symlink() or not path.is_file() or path.suffix == '.part':
                    raise ValueError('Source cache contains an unexpected entry')
                info = archive.gettarinfo(str(path), arcname=path.name)
                info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''
                with path.open('rb') as raw:
                    archive.addfile(info, raw)
    shutil.copyfile(cache / 'sources.json', out / 'SOURCES.json')
    licence = source['source']
    (out / 'LICENSE.md').write_text(
        '# Rights and attribution\n\nDerived dataset; not an official product of ' + licence['publisher'] + '.\n\n'
        'Upstream designation: ' + licence['license'] + '\n\n' + licence['terms_url'] + '\n\n'
        + licence['terms_excerpt'] + '\n\nRequired credit: ' + licence['attribution'] + '\n\n'
        + licence.get('citation', '') + '\n\n'
        'Upstream terms remain in force for the source snapshot and its derived data. '
        'CC0-1.0 dedicates only our additional compilation rights, subject to those terms. '
        'Our code/documentation uses MIT. Reproduction includes unchanged baseline runner '
        'copies from earino/harness_benchmark, distributed under MIT with permission confirmed '
        'by the copyright holder on 2026-09-19. Credits: E. Arino de la Rubia, Claude and Szilard.\n')
    evaluation = json.loads((out / 'evaluation.json').read_text())
    quality = json.loads((out / 'extract/quality.json').read_text())
    meta = json.loads((out / 'extract/meta.json').read_text())
    limitations = list(dict.fromkeys(source['limitations'] + quality['evaluation_contract']['limitations'] + evaluation['limitations']))
    (out / 'README.md').write_text(
        '---\nlicense: other\nlicense_name: upstream-terms-preserved\nlicense_link: LICENSE.md\n'
        'task_categories:\n- tabular-classification\nconfigs:\n- config_name: default\n  data_files:\n'
        '  - split: train\n    path: extract/public/train.csv\n'
        '  - split: validation\n    path: extract/public/eval.csv\n'
        '  - split: test\n    path: extract/private/holdout.csv\n---\n\n'
        '# ' + source['title'] + '\n\n' + source['task']['question'] + '\n\n'
        'Derived from ' + licence['publisher'] + '. This is not its official product.\n\n'
        'Prediction time: ' + source['task']['prediction_time'] + '\n\n'
        'Evaluation scope: `' + quality['evaluation_contract']['use'] + '`. '
        'Public validation is development data. The downloadable labeled holdout is public; '
        'it is not cryptographically sealed. No final holdout benchmark score is claimed.\n\n'
        '## Measured baseline\n\nThe existing runner was used unchanged on public validation. '
        'See `evaluation.json` for rows, controls, per-year results and limitations.\n\n'
        '```json\n' + json.dumps(evaluation['metrics'], indent=2) + '\n```\n\n'
        '## Limitations\n\n' + ''.join('- ' + item + '\n' for item in limitations) + '\n'
        '## Loading and reproduction\n\nVerify all downloaded files with `verify_dataset.py`. '
        'CSV splits live under `extract/`; `meta.json` declares the target and identifiers. '
        '`MANIFEST.json` binds data, source snapshot, code and measured evidence.\n\n'
        'To reconstruct without network source drift, extract `source-snapshot.tar.gz` into '
        'a new cache directory, install `reproduction/requirements.txt` in an isolated Python '
        'environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY '
        '--source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this '
        'manifest. This performs full validation and baseline evaluation and may need worker resources.\n\n'
        'See `LICENSE.md` for source terms and preserved attribution.\n')
    (out / 'verify_dataset.py').write_text(VERIFY_SCRIPT)
    shutil.copyfile(workspace / 'get_dataset.py', out / 'get_dataset.py')
    # Logs are diagnostics, not release assets. Everything else is explicitly named.
    selected = [p for p in out.rglob('*') if p.is_file() and p.suffix != '.log']
    assets = {}
    for path in sorted(selected):
        name = path.relative_to(out).as_posix()
        assets[name] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                        'asset_name': name.replace('/', '__')}
    version = 'v-' + hashlib.sha256(json.dumps(assets, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
    manifest = {'schema': 'factory-1', 'dataset': source['dataset_id'], 'version': version,
                'artifact_version': artifact, 'assets': assets,
                'source': licence, 'task': source['task'], 'rows': meta['rows'],
                'evaluation_contract': quality['evaluation_contract'], 'limitations': limitations,
                'qualification': {'gate_version': qualification['gate_version'], 'ok': True}}
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(entry['sha256'] + '  ' + name + '\n' for name, entry in assets.items()))


VERIFY_SCRIPT = '''"""Verify the complete downloaded release against its manifest; no credentials."""
import csv, hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text())
for name, entry in manifest['assets'].items():
    path = root / name
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise SystemExit('Missing or unsafe file: ' + name)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
        raise SystemExit('Digest mismatch: ' + name)
meta = json.loads((root / 'extract/meta.json').read_text())
for split, name in [('train','public/train.csv'),('eval','public/eval.csv'),('holdout','private/holdout.csv')]:
    with (root / 'extract' / name).open(newline='') as stream:
        reader = csv.reader(stream)
        assert next(reader) == meta['columns'], 'schema differs'
        assert sum(1 for row in reader) == meta['rows'][split], 'row count differs'
print(json.dumps({'ok':True,'dataset':manifest['dataset'],'artifact_version':manifest['artifact_version']}))
'''
