"""Download and verify a public release from either destination, without credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


def download_once(url, destination, expected=None):
    request = urllib.request.Request(url, headers={'User-Agent': 'DatasetFactory-consumer/1.0'})
    temporary = destination.with_suffix(destination.suffix + '.part')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or temporary.is_symlink():
        raise ValueError('Download destination is a symlink')
    digest = hashlib.sha256(); size = 0
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as out:
        if urllib.parse.urlsplit(response.url).scheme != 'https':
            raise ValueError('Download redirected away from HTTPS')
        for block in iter(lambda: response.read(1024 * 1024), b''):
            size += len(block)
            if size > (expected['bytes'] if expected else 256 * 1024):
                raise ValueError('Download exceeds declared size')
            digest.update(block); out.write(block)
    if expected and (size != expected['bytes'] or digest.hexdigest() != expected['sha256']):
        raise ValueError('Downloaded artifact does not match the manifest')
    temporary.replace(destination)


def download(url, destination, expected=None):
    for attempt in range(3):
        try:
            return download_once(url, destination, expected)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def retrieve(repo, version, out, platform):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo) or not re.fullmatch(r'[A-Za-z0-9_.-]+', version):
        raise ValueError('Invalid repository or version')
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty download directory')
    base = ('https://github.com/' + repo + '/releases/download/' + version + '/' if platform == 'github'
            else 'https://huggingface.co/datasets/' + repo + '/resolve/' + version + '/')
    download(base + 'MANIFEST.json', out / 'MANIFEST.json')
    manifest = json.loads((out / 'MANIFEST.json').read_text())
    if manifest.get('version') != version:
        raise ValueError('Manifest version differs from requested version')
    for name, entry in manifest['assets'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe manifest path')
        remote = entry['asset_name'] if platform == 'github' else name
        download(base + urllib.parse.quote(remote, safe='/'), out / name, entry)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('github', 'huggingface'), required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = retrieve(args.repo, args.version, args.out, args.platform)
    print(json.dumps({'downloaded': len(manifest['assets']), 'dataset': manifest['dataset'],
                      'artifact_version': manifest['artifact_version']}))


if __name__ == '__main__':
    main()
