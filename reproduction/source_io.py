"""Reproducible public source downloads, with bounded transport retries."""
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_BYTES = 1024 * 1024 * 1024


class TermsEvidenceError(ValueError):
    pass


def excerpt_match(text, excerpt):
    # Inline HTML nodes can put whitespace before punctuation during extraction.
    # Explicit ellipses denote omissions, not invented words: every remaining
    # verbatim span must occur in order in the preserved publisher document.
    def normalize(value):
        value = ' '.join(html.unescape(value).split())
        return re.sub(r'\s+([,;:.!?])', r'\1', value)
    text, excerpt = normalize(text), normalize(excerpt)
    if excerpt and excerpt in text:
        return 'contiguous'
    spans = [span.strip() for span in re.split(r'\.{3}|…', excerpt) if span.strip()]
    if len(spans) < 2:
        return None
    offset = 0
    for span in spans:
        found = text.find(span, offset)
        if found < 0:
            return None
        offset = found + len(span)
    return 'ordered-verbatim-spans-with-explicit-omissions'


def capture_terms(source, cache):
    """Capture publisher evidence and establish that the quoted grant is present.

    Presence is mechanical; applicability/compatibility remains semantic review.
    """
    try:
        terms = fetch(source['terms_url'], cache, 'publisher-terms.snapshot')
        documentation = fetch(source['documentation_url'], cache, 'publisher-documentation.snapshot')
        raw = terms.read_bytes()
        if raw.startswith(b'%PDF-'):
            from pypdf import PdfReader
            text = '\n'.join(page.extract_text() or '' for page in PdfReader(terms).pages)
        else:
            class Text(HTMLParser):
                def __init__(self): super().__init__(); self.parts = []
                def handle_data(self, data): self.parts.append(data)
            parser = Text(); parser.feed(raw.decode('utf-8', errors='replace'))
            text = ' '.join(parser.parts)
        match = excerpt_match(text, source['terms_excerpt'])
        if not match:
            raise TermsEvidenceError('The claimed terms excerpt is absent from the captured publisher document')
        return {'ok': True, 'terms_url': source['terms_url'], 'terms_sha256': sha(terms),
                'documentation_url': source['documentation_url'], 'documentation_sha256': sha(documentation),
                'quoted_excerpt': source['terms_excerpt'], 'excerpt_found': True, 'excerpt_match': match,
                'scope': 'Captured publisher bytes and quote presence; semantic applicability reviewed separately'}
    except TermsEvidenceError:
        raise
    except Exception as exc:
        raise TermsEvidenceError('Publisher terms/documentation could not be captured: ' + type(exc).__name__) from exc


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, cache, name):
    cache = Path(cache)
    if (urllib.parse.urlsplit(url).scheme != 'https' or not name
            or Path(name).name != name or name in ('.', '..', 'sources.json')):
        raise ValueError('Use an HTTPS source and a plain cache filename')
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    lock = cache / 'sources.json'
    records = json.loads(lock.read_text()) if lock.exists() else {}
    if path.is_symlink():
        raise ValueError('Source cache symlink refused')
    if name in records:
        entry = records[name]
        if entry['url'] != url or not path.is_file() or sha(path) != entry['sha256']:
            raise ValueError('Cached source changed or filename reused')
        return path
    temporary = cache / (name + '.part')
    if temporary.is_symlink():
        raise ValueError('Source temporary path is a symlink')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'DatasetFactory/1.0 (research; reproducible public data)'})
            with urllib.request.urlopen(request, timeout=45) as response, temporary.open('wb') as out:
                if urllib.parse.urlsplit(response.url).scheme != 'https':
                    raise ValueError('Source redirect downgraded HTTPS')
                total = 0
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    total += len(block)
                    if total > MAX_BYTES:
                        raise ValueError('Source file exceeds 1 GiB bounded download')
                    out.write(block)
            if not total:
                raise ValueError('Source download is empty')
            temporary.replace(path)
            records[name] = {'url': url, 'bytes': total, 'sha256': sha(path),
                             'retrieved_at_unix': time.time(), 'transport_attempts': attempt + 1}
            pending = cache / 'sources.json.part'
            pending.write_text(json.dumps(records, indent=2) + '\n')
            pending.replace(lock)
            return path
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

