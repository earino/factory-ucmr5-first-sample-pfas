"""Verify the complete downloaded release against its manifest; no credentials."""
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
