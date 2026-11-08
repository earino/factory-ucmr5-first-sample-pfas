import codecs
import csv
import io
import pathlib
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

from source_io import fetch

URL5 = 'https://www.epa.gov/system/files/other-files/2023-08/ucmr5-occurrence-data.zip'
URL3 = 'https://www.epa.gov/system/files/other-files/2024-04/ucmr3-occurrence-data.zip'
MRL5 = 0.004
TAB = chr(9)
TARGET_ANALYTES = {'PFOA', 'PFOS'}
PFOA_PATTERN = r'(?<![A-Z0-9])(?:PFOA|PERFLUOROOCTANOIC)(?![A-Z0-9])'
PFOS_PATTERN = r'(?<![A-Z0-9])(?:PFOS|PERFLUOROOCTANESULFONIC)(?![A-Z0-9])'
FEATURES = ['pws_size', 'facility_water_type', 'state', 'epa_region', 'prior_ucmr3_pfoa_pfos_detect', 'prior_ucmr3_max_pfoa_pfos', 'pfas_tested_before', 'pfas_treatment_before', 'potential_pfas_sources', 'potential_source_types', 'treatment_information', 'collection_year']
ADDTL_FEATURES = ['pfas_tested_before', 'pfas_treatment_before', 'potential_pfas_sources', 'potential_source_types', 'treatment_information']
WINDOWS = (('train', '2023-01-01', '2023-12-01'), ('eval', '2024-01-01', '2024-12-01'), ('holdout', '2025-01-01', '2026-07-01'))
OUT_COLS = ['pwsid', 'collection_date', 'sample_id'] + FEATURES + ['any_pfoa_pfos']
CANON = {
    'pwsid': ('PWSID',),
    'state': ('STATE', 'PWSSTATE'),
    'region': ('REGION', 'EPAREGION', 'PWSREGION', 'REGIONID'),
    'size': ('SIZE', 'PWSSIZE'),
    'facilityid': ('FACILITYID',),
    'facilitywatertype': ('FACILITYWATERTYPE', 'WATERTYPE', 'FACILITYWATERTYPES'),
    'samplepointid': ('SAMPLEPOINTID',),
    'samplepointtype': ('SAMPLEPOINTTYPE', 'SAMPLEPOINTTYPECODE', 'SAMPLINGPOINTTYPE'),
    'collectiondate': ('COLLECTIONDATE', 'SAMPLECOLLECTIONDATE'),
    'sampleid': ('SAMPLEID',),
    'sampleeventcode': ('SAMPLEEVENTCODE', 'SAMPLEEVENTID'),
    'contaminant': ('CONTAMINANT', 'CONTAMINANTNAME'),
    'sign': ('ANALYTICALRESULTSSIGN', 'ANALYTICALRESULTSIGN', 'RESULTSIGN'),
    'value': ('ANALYTICALRESULTVALUE', 'RESULTVALUE'),
    'mrl': ('MRL', 'MINIMUMREPORTINGLEVEL'),
    'units': ('UNITS', 'UNIT', 'UNITSOFMEASURE', 'RESULTUNITS', 'CONCENTRATIONUNITS'),
    'ade': ('ADDITIONALDATALELEMENT', 'ADDITIONALDATAELEMENT', 'DATAELEMENT'),
    'response': ('RESPONSE', 'RESPONSECODE', 'RESPONSEVALUE'),
    'ifothertext': ('IFOTHERTEXT', 'OTHER TEXT'),
}
ADE_TO_FEATURE = {
    'PFASOCCURRENCE': 'pfas_tested_before',
    'PFASTREATMENT': 'pfas_treatment_before',
    'POTENTIALPFASSOURCES': 'potential_pfas_sources',
    'POTENTIALPFASSOURCESDETAIL': 'potential_source_types',
    'TREATMENTINFORMATION': 'treatment_information',
}


def _norm(value):
    return ''.join(ch for ch in str(value).upper() if ch.isalnum())


def _base(name):
    return str(name).rsplit('/', 1)[-1].lower()


def _members(zf):
    return [info.filename for info in zf.infolist() if not info.is_dir()]


def _zip_member(zf, includes, excludes=()):
    best = None
    for info in zf.infolist():
        if info.is_dir():
            continue
        base = _base(info.filename)
        if not base.endswith(('.txt', '.tsv', '.csv')):
            continue
        if not any(token in base for token in includes):
            continue
        if any(token in base for token in excludes):
            continue
        if best is None or info.file_size > best[0]:
            best = (info.file_size, info.filename)
    return None if best is None else best[1]


def _open_text(zf, member):
    handle = zf.open(member)
    head = handle.read(3)
    if head == codecs.BOM_UTF8:
        return io.TextIOWrapper(handle, encoding='utf-8-sig', errors='replace', newline='')
    try:
        handle.seek(0)
    except Exception:
        handle = io.BytesIO(head + handle.read())
    return io.TextIOWrapper(handle, encoding='cp1252', errors='replace', newline='')


def _delimiter(line):
    candidates = [(line.count(TAB), TAB), (line.count(','), ','), (line.count(';'), ';'), (line.count('|'), '|')]
    best = TAB
    top = -1
    for count, char in candidates:
        if count > top:
            top = count
            best = char
    return best


def _head_info(zf, member):
    with _open_text(zf, member) as text:
        line = text.readline().rstrip()
    delim = _delimiter(line)
    raw = line.split(delim)
    stripped = [part.strip() for part in raw]
    if raw:
        raw[0] = raw[0].replace(chr(65279), '')
    if stripped:
        stripped[0] = stripped[0].replace(chr(65279), '')
    return raw, stripped, delim


def _resolve(header, aliases):
    lookup = {}
    for name in header:
        key = _norm(name)
        if key and key not in lookup:
            lookup[key] = name
    found = {}
    for concept, names in aliases.items():
        for name in names:
            key = _norm(name)
            if key in lookup:
                found[concept] = lookup[key]
                break
    return found


def _fuzzy(header, must, miss=()):
    hits = []
    for name in header:
        key = _norm(name)
        if all(token in key for token in must) and not any(token in key for token in miss):
            hits.append(name)
    return hits[0] if len(hits) == 1 else None


def _clean(series):
    return series.fillna('').astype(str).str.strip()


def _unit_norm(value):
    text = str(value).strip().lower().replace(chr(181), 'u').replace(chr(956), 'u')
    return ''.join(ch for ch in text if ch.isalnum() or ch == '/')


def _num(value):
    try:
        return float(str(value).replace(',', '').strip())
    except Exception:
        return None


def _parse_date(value):
    text = str(value).strip()
    for fmt in ('%m/%d/%Y', '%Y-%m-%d', '%m/%d/%y'):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass
    return None


def _yes_no(series):
    mapping = {'Y': 'YES', 'YES': 'YES', 'N': 'NO', 'NO': 'NO', 'DK': 'DK', 'DONTKNOW': 'DK', 'UNKNOWN': 'DK', 'UNK': 'DK', 'NOTSURE': 'DK'}
    def _map(value):
        text = str(value).upper().replace(chr(39), '').replace(chr(8217), '').replace(' ', '')
        parts = [part for part in text.split(';') if part]
        if not parts:
            return text
        return ';'.join(mapping.get(part, part) for part in parts)
    return _clean(series).map(_map)


def _check(checks, name, observed, expected, passed, kind=None):
    item = {'check': name, 'observed': str(observed)[:1500], 'expected': str(expected)[:800], 'passed': bool(passed)}
    if kind is not None:
        item['kind'] = kind
    checks.append(item)


def _fail(message):
    raise AssertionError(str(message)[:4000])


def _read_frame(zip_path, member, usecols, sep):
    with zipfile.ZipFile(zip_path) as zf:
        with _open_text(zf, member) as text:
            return pd.read_csv(text, sep=sep, dtype=str, keep_default_na=False, na_filter=False, usecols=usecols, quoting=csv.QUOTE_NONE, skipinitialspace=True)


def _read_chunks(zip_path, member, usecols, sep, chunksize=200000):
    with zipfile.ZipFile(zip_path) as zf:
        with _open_text(zf, member) as text:
            reader = pd.read_csv(text, sep=sep, dtype=str, keep_default_na=False, na_filter=False, usecols=usecols, quoting=csv.QUOTE_NONE, skipinitialspace=True, chunksize=chunksize, on_bad_lines='error')
            for chunk in reader:
                yield chunk


def _agg_text(series):
    values = sorted({str(value).strip() for value in series if str(value).strip() != ''})
    return ';'.join(values)


def _pivot_addtl(frame, keys):
    columns = keys + ADDTL_FEATURES
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    work = frame.loc[:, keys + ['feature', 'response']].copy()
    work['response'] = _clean(work['response'])
    work = work.loc[work['response'].ne(''), :]
    if work.empty:
        return pd.DataFrame(columns=columns)
    agg = work.groupby(keys + ['feature'], as_index=False, dropna=False).agg(value=('response', _agg_text))
    pivot = agg.pivot_table(index=keys, columns='feature', values='value', aggfunc='first').reset_index()
    for name in ADDTL_FEATURES:
        if name not in pivot.columns:
            pivot[name] = ''
    return pivot.loc[:, columns]


def _fill_from(left, right, on, names):
    if right is None or right.empty:
        return left
    use = [name for name in on if name in left.columns and name in right.columns]
    if not use:
        return left
    merged = left.merge(right, on=use, how='left', suffixes=('', '_r'))
    for name in names:
        alt = name + '_r'
        if name not in merged.columns:
            merged[name] = ''
        if alt in merged.columns:
            base = _clean(merged[name])
            replacement = _clean(merged[alt])
            merged[name] = np.where(base.ne(''), base, replacement)
            merged = merged.drop(columns=[alt])
    return merged


def _match_count(left, right, on):
    if right is None or right.empty:
        return 0
    use = [name for name in on if name in left.columns and name in right.columns]
    if not use:
        return 0
    keys = right.loc[:, use].drop_duplicates()
    return int(len(left.merge(keys, on=use, how='inner')))


def _verify_label_source(zip_path, member, delim, cols5, written, checks):
    required = ['pwsid', 'sampleid', 'samplepointtype', 'collectiondate', 'contaminant', 'sign', 'value']
    units_seen = set()
    mrl_seen = set()
    expected = {}
    positive_rows = 0
    bad_units = 0
    scanned = 0
    with zipfile.ZipFile(zip_path) as zf:
        with _open_text(zf, member) as text:
            reader = csv.reader(text, delimiter=delim, quoting=csv.QUOTE_NONE)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            header_index = {}
            for i, name in enumerate(header):
                key = str(name).replace(chr(65279), '').strip()
                if key and key not in header_index:
                    header_index[key] = i
            idx = {}
            missing_fields = []
            for concept in required:
                clean = cols5.get(concept)
                if clean is None or clean not in header_index:
                    missing_fields.append(concept)
                else:
                    idx[concept] = header_index[clean]
            for concept in ('units', 'mrl'):
                clean = cols5.get(concept)
                if clean is not None and clean in header_index:
                    idx[concept] = header_index[clean]
            if missing_fields:
                _check(checks, 'Frozen label rule recomputed from an independent source pass', 'missing fields ' + str(missing_fields) + '; selected member ' + str(member) + '; header ' + str(header)[:800], 'the occurrence text carries system identifier, sample identifier, sample point type, collection date, contaminant, result sign, and result value so the frozen label can be recomputed independently', False)
                return
            selected_keys = {(str(p).strip(), str(s).strip()) for p, s in zip(written['pwsid'], written['sample_id'])}
            def cell(row, concept):
                i = idx.get(concept)
                if i is None or i >= len(row):
                    return ''
                return row[i]
            top = max(idx.values())
            for row in reader:
                scanned += 1
                if len(row) <= top:
                    continue
                sp = cell(row, 'samplepointtype').strip().upper()
                if sp != 'EP':
                    continue
                cont = cell(row, 'contaminant').strip().upper().replace(' ', '')
                if cont not in TARGET_ANALYTES:
                    continue
                pws = cell(row, 'pwsid').strip()
                sid = cell(row, 'sampleid').strip()
                key = (pws, sid)
                if key not in selected_keys:
                    continue
                if _parse_date(cell(row, 'collectiondate')) is None:
                    continue
                sign = cell(row, 'sign').strip()
                if sign not in ('<', '='):
                    continue
                value = _num(cell(row, 'value'))
                units = _unit_norm(cell(row, 'units')) if 'units' in idx else ''
                if units:
                    units_seen.add(units)
                mrl = _num(cell(row, 'mrl')) if 'mrl' in idx else None
                if mrl is not None:
                    mrl_seen.add(mrl)
                expected.setdefault(key, 0)
                if sign == '=' and value is not None and value >= MRL5:
                    expected[key] = 1
                    positive_rows += 1
                    if units and units != 'ug/l':
                        bad_units += 1
    mismatches = []
    missing_keys = 0
    written_pos = 0
    for p, s, label in zip(written['pwsid'], written['sample_id'], written['any_pfoa_pfos']):
        key = (str(p).strip(), str(s).strip())
        exp = expected.get(key)
        actual = int(label)
        if exp is None:
            missing_keys += 1
        elif exp != actual:
            if len(mismatches) < 10:
                mismatches.append(key)
        written_pos += actual
    source_pos = sum(expected.values())
    unit_ok = bool(units_seen) and units_seen.issubset({'ug/l'}) and bad_units == 0
    mrl_ok = bool(mrl_seen) and all(abs(value - MRL5) < 1e-9 for value in mrl_seen)
    passed = missing_keys == 0 and not mismatches and unit_ok and mrl_ok and source_pos == written_pos
    observed = 'scanned rows ' + str(scanned) + '; selected raw keys ' + str(len(expected)) + '; source positive keys ' + str(source_pos) + '; written positive rows ' + str(written_pos) + '; positive source rows ' + str(positive_rows) + '; mismatches ' + str(mismatches) + '; missing keys ' + str(missing_keys) + '; units ' + str(sorted(units_seen)) + '; MRL values ' + str(sorted(mrl_seen))
    expected_text = 'the written labels equal an independent csv-module recomputation from the occurrence text using result sign equals and numeric value at or above 0.004 ug/L for PFOA or PFOS, with the reporting concentration unit and MRL verified from the same source rows'
    _check(checks, 'Frozen label rule recomputed from an independent source pass', observed, expected_text, passed)


def build(output: pathlib.Path, cache: pathlib.Path):
    output = pathlib.Path(output)
    cache = pathlib.Path(cache)
    checks = []
    signs = set()
    analytes = set()
    mrls = set()
    unit_values = set()
    cont_seen = set()
    date_seen = []
    u3signs = set()
    u3_mrls = set()
    u3_units = set()
    u3_cont_seen = set()
    counts = {'rows': 0, 'ep': 0, 'other': 0, 'bad_date': 0, 'missing_date': 0, 'blank_sign': 0, 'bad_sign': 0, 'eq_missing': 0, 'below': 0, 'positive': 0, 'u3_rows': 0, 'u3_unknown': 0, 'u3_eq_missing': 0, 'u3_non_detect_missing_mrl': 0}

    zip5 = pathlib.Path(fetch(URL5, cache, 'ucmr5-occurrence-data.zip'))
    zip3 = pathlib.Path(fetch(URL3, cache, 'ucmr3-occurrence-data.zip'))

    with zipfile.ZipFile(zip5) as zf5:
        members5 = _members(zf5)
        member5 = _zip_member(zf5, ('ucmr5', 'all'), ('addtl', 'additional', 'dataelem'))
        if member5 is None:
            member5 = _zip_member(zf5, ('ucmr5',), ('addtl', 'additional', 'dataelem'))
        _check(checks, 'UCMR 5 occurrence archive member located', 'selected member ' + str(member5) + '; archive members ' + str(members5), 'the published UCMR 5 occurrence text member is selected from the archive', member5 is not None)
        if member5 is None:
            _fail('UCMR 5 occurrence member not found; archive members ' + str(members5))
        raw5, hdr5, delim5 = _head_info(zf5, member5)
        cols5 = _resolve(hdr5, CANON)
        if 'units' not in cols5:
            cand = _fuzzy(hdr5, ('UNIT',), ())
            if cand is not None:
                cols5['units'] = cand
        if 'mrl' not in cols5:
            cand = _fuzzy(hdr5, ('MRL',), ())
            if cand is not None:
                cols5['mrl'] = cand
        required5 = ['pwsid', 'state', 'region', 'size', 'facilityid', 'facilitywatertype', 'samplepointid', 'samplepointtype', 'collectiondate', 'sampleid', 'contaminant', 'sign', 'value']
        missing5 = [name for name in required5 if name not in cols5]
        _check(checks, 'UCMR 5 occurrence fields resolved', 'missing ' + str(missing5) + '; delimiter ' + repr(delim5) + '; header ' + str(hdr5), 'every required occurrence field maps to a real header column of the occurrence text', not missing5)
        if missing5:
            _fail('UCMR 5 occurrence fields missing ' + str(missing5) + '; member ' + str(member5) + '; header ' + str(hdr5))
        raw_of = {}
        for raw_name, clean_name in zip(raw5, hdr5):
            raw_of.setdefault(clean_name, raw_name)
        read5 = []
        rename5 = {}
        for name in required5 + [item for item in ('mrl', 'units', 'sampleeventcode') if item in cols5]:
            if name in cols5:
                raw_name = raw_of.get(cols5[name])
                if raw_name is not None and raw_name not in read5:
                    read5.append(raw_name)
                    rename5[raw_name] = name

        member_add = _zip_member(zf5, ('addtl',), ())
        if member_add is None:
            member_add = _zip_member(zf5, ('additional',), ())
        _check(checks, 'UCMR 5 additional data element member located', 'selected member ' + str(member_add) + '; archive members ' + str(members5), 'the published UCMR 5 additional data element text member is selected from the archive', member_add is not None)
        if member_add is None:
            _fail('UCMR 5 additional data element member not found; archive members ' + str(members5))
        raw_add, hdr_add, delim_add = _head_info(zf5, member_add)
        cols_add = _resolve(hdr_add, CANON)
        required_add = ['pwsid', 'facilityid', 'samplepointid', 'ade', 'response']
        missing_add = [name for name in required_add if name not in cols_add]
        _check(checks, 'UCMR 5 additional data element exact key and response fields resolved', 'missing ' + str(missing_add) + '; delimiter ' + repr(delim_add) + '; header ' + str(hdr_add), 'the additional data element text carries the system, facility, and sample point keys plus the data element name and response fields needed for an exact sample-key join', not missing_add)
        if missing_add:
            _fail('UCMR 5 additional data element fields missing ' + str(missing_add) + '; member ' + str(member_add) + '; header ' + str(hdr_add))
        raw_of_add = {}
        for raw_name, clean_name in zip(raw_add, hdr_add):
            raw_of_add.setdefault(clean_name, raw_name)
        add_read = []
        rename_add = {}
        for name in ('pwsid', 'facilityid', 'samplepointid', 'sampleeventcode', 'ade', 'response', 'ifothertext'):
            if name in cols_add:
                raw_name = raw_of_add.get(cols_add[name])
                if raw_name is not None and raw_name not in add_read:
                    add_read.append(raw_name)
                    rename_add[raw_name] = name

    addtl = _read_frame(zip5, member_add, add_read, delim_add)
    addtl = addtl.rename(columns=rename_add)
    for key in ('pwsid', 'facilityid', 'samplepointid', 'sampleeventcode'):
        if key not in addtl.columns:
            addtl[key] = ''
        addtl[key] = _clean(addtl[key])
    addtl['ade_norm'] = _clean(addtl['ade']).map(_norm)
    addtl['feature'] = addtl['ade_norm'].map(ADE_TO_FEATURE)
    addtl['response'] = _clean(addtl['response'])
    ade_seen = sorted({value for value in addtl['ade_norm'] if value != ''})
    mapped_features = sorted({value for value in addtl['feature'].dropna() if value != ''})
    nonempty_add = addtl.loc[addtl['feature'].notna() & addtl['response'].ne(''), :]
    addtl_counts = {}
    if not nonempty_add.empty:
        for feature, group in nonempty_add.groupby('feature'):
            addtl_counts[feature] = int(len(group))
    missing_features = [name for name in ADDTL_FEATURES if addtl_counts.get(name, 0) == 0]
    _check(checks, 'UCMR 5 additional data element long-format concepts resolved', 'selected member ' + str(member_add) + '; data element values ' + str(ade_seen) + '; mapped features ' + str(mapped_features) + '; nonempty response counts ' + str(addtl_counts), 'each declared additional-data concept is a value of the AdditionalDataElement field with at least one nonempty Response', not missing_features)
    if missing_features:
        _fail('UCMR 5 additional data element concepts missing ' + str(missing_features) + '; member ' + str(member_add) + '; header ' + str(hdr_add) + '; data element values ' + str(ade_seen))
    sub_add = addtl.loc[addtl['feature'].notna() & addtl['response'].ne(''), :].copy()

    parts = []
    for chunk in _read_chunks(zip5, member5, read5, delim5):
        counts['rows'] += len(chunk)
        chunk = chunk.rename(columns=rename5)
        cont = _clean(chunk['contaminant']).str.upper().str.replace(' ', '', regex=False)
        cont_seen.update(sorted(set(cont)))
        mask = cont.str.contains(PFOA_PATTERN, regex=True) | cont.str.contains(PFOS_PATTERN, regex=True)
        if not mask.any():
            continue
        t = chunk.loc[mask].copy()
        analytes.update(sorted(set(cont[mask])))
        point_type = _clean(t['samplepointtype']).str.upper()
        entry = point_type.eq('EP')
        counts['ep'] += int(entry.sum())
        counts['other'] += int((~entry).sum())
        t = t.loc[entry].copy()
        if t.empty:
            continue
        raw = _clean(t['collectiondate'])
        counts['missing_date'] += int(raw.eq('').sum())
        if len(date_seen) < 5:
            date_seen.extend(sorted(set(raw))[:5])
        dates = pd.to_datetime(raw, format='%m/%d/%Y', errors='coerce')
        retry = dates.isna() & raw.ne('')
        if retry.any():
            dates.loc[retry] = pd.to_datetime(raw.loc[retry], errors='coerce')
        counts['bad_date'] += int((dates.isna() & raw.ne('')).sum())
        t = t.loc[dates.notna()].copy()
        t['collection_date'] = dates.loc[dates.notna()]
        sign = _clean(t['sign'])
        value = pd.to_numeric(_clean(t['value']).str.replace(',', '', regex=False), errors='coerce')
        signs.update(sorted(set(sign)))
        blank = sign.eq('')
        known = sign.isin(['<', '='])
        counts['blank_sign'] += int(blank.sum())
        counts['bad_sign'] += int((~known & ~blank).sum())
        counts['eq_missing'] += int((sign.eq('=') & value.isna()).sum())
        counts['below'] += int((sign.eq('=') & value.notna() & value.lt(MRL5)).sum())
        positive = (sign.eq('=') & value.ge(MRL5)).astype(int)
        counts['positive'] += int(positive.sum())
        t['positive'] = positive
        if 'mrl' in t.columns:
            mrls.update(sorted(set(_clean(t['mrl']))))
        if 'units' in t.columns:
            unit_values.update(sorted(set(_clean(t['units']))))
        t['pwsid'] = _clean(t['pwsid'])
        t['sampleid'] = _clean(t['sampleid'])
        keep = ['pwsid', 'sampleid', 'collection_date', 'positive', 'size', 'facilitywatertype', 'state', 'region', 'facilityid', 'samplepointid', 'samplepointtype']
        if 'sampleeventcode' in t.columns:
            t['sampleeventcode'] = _clean(t['sampleeventcode'])
            keep.append('sampleeventcode')
        parts.append(t.loc[known, keep].copy())

    if not parts:
        _fail('no entry-point PFOA or PFOS analytical result with a collection date was found; contaminant names seen ' + str(sorted(cont_seen)[:40]) + '; occurrence rows read ' + str(counts['rows']))

    rows = pd.concat(parts, ignore_index=True)
    rows = rows.sort_values(['pwsid', 'collection_date', 'sampleid'], kind='mergesort')
    agg_map = {
        'any_pfoa_pfos': ('positive', 'max'),
        'pws_size': ('size', 'first'),
        'facility_water_type': ('facilitywatertype', 'first'),
        'state': ('state', 'first'),
        'epa_region': ('region', 'first'),
        'facilityid': ('facilityid', 'first'),
        'samplepointid': ('samplepointid', 'first'),
        'samplepointtype': ('samplepointtype', 'first'),
    }
    if 'sampleeventcode' in rows.columns:
        agg_map['sampleeventcode'] = ('sampleeventcode', 'first')
    events = rows.groupby(['pwsid', 'sampleid', 'collection_date'], as_index=False).agg(**agg_map)
    events = events.sort_values(['pwsid', 'collection_date', 'sampleid'], kind='mergesort')
    selected = events.drop_duplicates('pwsid', keep='first').copy()
    selected = selected.rename(columns={'sampleid': 'sample_id'})

    with zipfile.ZipFile(zip3) as zf3:
        members3 = _members(zf3)
        member3 = _zip_member(zf3, ('ucmr3', 'all'))
        if member3 is None:
            member3 = _zip_member(zf3, ('ucmr3',))
        if member3 is None:
            _fail('UCMR 3 occurrence member not found; archive members ' + str(members3))
        raw3, hdr3, delim3 = _head_info(zf3, member3)
        cols3 = _resolve(hdr3, CANON)
        if 'mrl' not in cols3:
            cand = _fuzzy(hdr3, ('MRL',), ())
            if cand is not None:
                cols3['mrl'] = cand
        if 'units' not in cols3:
            cand = _fuzzy(hdr3, ('UNIT',), ())
            if cand is not None:
                cols3['units'] = cand
        required3 = ['pwsid', 'contaminant', 'sign', 'value', 'mrl', 'units']
        missing3 = [name for name in required3 if name not in cols3]
        _check(checks, 'UCMR 3 occurrence member and fields resolved', 'member ' + str(member3) + '; missing ' + str(missing3) + '; header ' + str(hdr3), 'UCMR 3 occurrence text carries the system identifier, contaminant, result sign, result value, MRL, and unit fields', member3 is not None and not missing3)
        if member3 is None or missing3:
            _fail('UCMR 3 occurrence data unavailable; member ' + str(member3) + '; archive members ' + str(members3) + '; missing fields ' + str(missing3) + '; header ' + str(hdr3))
        raw_of3 = {}
        for raw_name, clean_name in zip(raw3, hdr3):
            raw_of3.setdefault(clean_name, raw_name)
        read3 = []
        rename3 = {}
        for name in required3:
            raw_name = raw_of3.get(cols3[name])
            if raw_name is not None and raw_name not in read3:
                read3.append(raw_name)
                rename3[raw_name] = name

    parts3 = []
    for chunk in _read_chunks(zip3, member3, read3, delim3):
        chunk = chunk.rename(columns=rename3)
        cont = _clean(chunk['contaminant']).str.upper().str.replace(' ', '', regex=False)
        u3_cont_seen.update(sorted(set(cont)))
        mask = cont.str.contains(PFOA_PATTERN, regex=True) | cont.str.contains(PFOS_PATTERN, regex=True)
        if not mask.any():
            continue
        t = chunk.loc[mask].copy()
        t['pwsid'] = _clean(t['pwsid'])
        sign = _clean(t['sign'])
        value = pd.to_numeric(_clean(t['value']).str.replace(',', '', regex=False), errors='coerce')
        mrl = pd.to_numeric(_clean(t['mrl']).str.replace(',', '', regex=False), errors='coerce')
        u3signs.update(sorted(set(sign)))
        u3_mrls.update(sorted(set(_clean(t['mrl']))))
        u3_units.update(sorted(set(_clean(t['units']))))
        known3 = sign.isin(['<', '='])
        counts['u3_unknown'] += int((~known3 & sign.ne('')).sum())
        t = t.loc[known3].copy()
        sign = sign.loc[known3]
        value = value.loc[known3]
        mrl = mrl.loc[known3]
        detected = sign.eq('=')
        counts['u3_eq_missing'] += int((detected & value.isna()).sum())
        counts['u3_non_detect_missing_mrl'] += int((~detected & mrl.isna()).sum())
        t['positive'] = detected.astype(int)
        t['detected_value'] = value.where(detected, np.nan)
        t['censored_bound'] = mrl.where(~detected, np.nan)
        counts['u3_rows'] += len(t)
        parts3.append(t.loc[:, ['pwsid', 'positive', 'detected_value', 'censored_bound']])

    if not parts3:
        _fail('no UCMR 3 PFOA or PFOS rows were found; contaminant names seen ' + str(sorted(u3_cont_seen)[:40]))
    rows3 = pd.concat(parts3, ignore_index=True)
    prior = rows3.groupby('pwsid', as_index=False).agg(
        prior_ucmr3_pfoa_pfos_detect=('positive', 'max'),
        detected_max=('detected_value', 'max'),
        censored_max=('censored_bound', 'max'),
    )
    prior['prior_ucmr3_max_pfoa_pfos'] = prior['detected_max'].where(prior['detected_max'].notna(), prior['censored_max'])
    prior = prior.loc[:, ['pwsid', 'prior_ucmr3_pfoa_pfos_detect', 'prior_ucmr3_max_pfoa_pfos']]
    prior_pws = int(prior['pwsid'].nunique())

    selected = selected.merge(prior, on='pwsid', how='left')

    selected_has_event = 'sampleeventcode' in selected.columns and _clean(selected['sampleeventcode']).ne('').any()
    add_has_event = 'sampleeventcode' in sub_add.columns and _clean(sub_add['sampleeventcode']).ne('').any()
    if selected_has_event and add_has_event:
        join_keys = ['pwsid', 'facilityid', 'samplepointid', 'sampleeventcode']
        join_kind = 'exact system, facility, sample point, and sample event key'
    else:
        join_keys = ['pwsid', 'facilityid', 'samplepointid']
        join_kind = 'exact system, facility, and sample point key because the sample event code is unavailable'
    add_exact = _pivot_addtl(sub_add, join_keys)
    event_match = _match_count(selected, add_exact, join_keys)
    selected = _fill_from(selected, add_exact, join_keys, ADDTL_FEATURES)
    for name in ADDTL_FEATURES:
        if name not in selected.columns:
            selected[name] = ''
        selected[name] = _clean(selected[name])
    addtl_coverage = {name: int(selected[name].ne('').sum()) for name in ADDTL_FEATURES}
    addtl_missing_coverage = [name for name in ADDTL_FEATURES if addtl_coverage.get(name, 0) == 0]
    _check(checks, 'UCMR 5 additional data elements joined on the exact available sample key', 'join ' + join_kind + '; join keys ' + str(join_keys) + '; exact key matches ' + str(event_match) + '; selected rows ' + str(len(selected)) + '; no system-only fallback used', 'each additional-data response is attached to the selected first sample only through the same exact available sample key, never through a system-level fallback', event_match > 0 and not addtl_missing_coverage)
    _check(checks, 'UCMR 5 additional data element features have nonempty coverage after the exact join', 'coverage ' + str(addtl_coverage) + '; selected rows ' + str(len(selected)) + '; exact key matches ' + str(event_match) + '; unmatched selected rows ' + str(len(selected) - event_match), 'every declared additional-data feature has at least one nonempty value after the exact sample-key join, while unmatched rows remain missing rather than being filled from another event or system', not addtl_missing_coverage)
    if addtl_missing_coverage:
        _fail('UCMR 5 additional data element features without coverage ' + str(addtl_missing_coverage) + '; member ' + str(member_add) + '; header ' + str(hdr_add) + '; coverage ' + str(addtl_coverage) + '; join ' + join_kind)

    selected['pws_size'] = _clean(selected['pws_size']).str.upper()
    selected['facility_water_type'] = _clean(selected['facility_water_type']).str.upper()
    selected['state'] = _clean(selected['state']).str.upper()
    selected['epa_region'] = _clean(selected['epa_region'])
    selected['prior_ucmr3_pfoa_pfos_detect'] = pd.to_numeric(selected['prior_ucmr3_pfoa_pfos_detect'], errors='coerce').astype('Int64')
    selected['prior_ucmr3_max_pfoa_pfos'] = pd.to_numeric(selected['prior_ucmr3_max_pfoa_pfos'], errors='coerce')
    selected['pfas_tested_before'] = _yes_no(selected['pfas_tested_before'])
    selected['pfas_treatment_before'] = _clean(selected['pfas_treatment_before']).str.upper()
    selected['potential_pfas_sources'] = _yes_no(selected['potential_pfas_sources'])
    selected['potential_source_types'] = _clean(selected['potential_source_types']).str.upper()
    selected['treatment_information'] = _clean(selected['treatment_information']).str.upper()
    selected['collection_date'] = pd.to_datetime(selected['collection_date'])
    selected['collection_year'] = selected['collection_date'].dt.year.astype(int)

    assignment = pd.Series('', index=selected.index)
    for name, start, end in WINDOWS:
        mask = selected['collection_date'].ge(pd.Timestamp(start)) & selected['collection_date'].lt(pd.Timestamp(end))
        assignment = assignment.mask(mask, name)
    frames = {}
    for name, start, end in WINDOWS:
        frames[name] = selected.loc[assignment.eq(name)].copy()
    outside = int(assignment.eq('').sum())
    shipped = pd.concat(list(frames.values()), ignore_index=True)

    analyte_names = sorted(analytes)
    unexpected_analytes = sorted(set(analyte_names) - TARGET_ANALYTES)
    missing_target_analytes = sorted(TARGET_ANALYTES - set(analyte_names))
    _check(checks, 'PFOA and PFOS analytes recognized exactly', 'analyte names matched ' + str(analyte_names) + '; unexpected matched analytes ' + str(unexpected_analytes) + '; missing target analytes ' + str(missing_target_analytes) + '; contaminant names seen ' + str(sorted(cont_seen)[:40]), 'the contaminant field carries exactly the documented PFOA and PFOS acronyms among matched rows and no unrelated analyte is captured by the acronym rule', not unexpected_analytes and not missing_target_analytes)
    retained_types = sorted(value for value in set(_clean(selected['samplepointtype']).str.upper()) if value != '')
    _check(checks, 'UCMR 5 entry-point samples retained', 'retained sample point types ' + str(retained_types) + '; entry-point rows scanned ' + str(counts['ep']) + '; non-entry-point PFOA or PFOS rows excluded ' + str(counts['other']), 'the retained modeling rows have SamplePointType EP and non-entry-point sample rows are excluded from the population', retained_types == ['EP'] and counts['ep'] > 0)
    sign_codes = sorted(value for value in signs if value != '')
    _check(checks, 'UCMR 5 result signs documented', 'sign codes ' + str(sign_codes) + '; unexpected sign rows ' + str(counts['bad_sign']) + '; blank sign rows ' + str(counts['blank_sign']), 'only the less-than and equals result codes occur among included entry-point analyte rows', counts['bad_sign'] == 0 and len(sign_codes) > 0)
    _check(checks, 'UCMR 5 detected rows carry numeric results', 'equals sign rows without a numeric value ' + str(counts['eq_missing']), 'no detected row lacks its numeric analytical result', counts['eq_missing'] == 0)
    selected_missing_dates = int(selected['collection_date'].isna().sum())
    _check(checks, 'UCMR 5 collection dates parse and missing dates are counted', 'blank CollectionDate rows excluded and counted ' + str(counts['missing_date']) + '; unparseable nonmissing dates ' + str(counts['bad_date']) + '; selected rows with missing date ' + str(selected_missing_dates) + '; sample raw dates ' + str(date_seen[:5]), 'blank CollectionDate rows are excluded before first-sample selection and counted, no nonmissing CollectionDate row is unparseable, and every selected row carries a collection date', counts['bad_date'] == 0 and selected_missing_dates == 0)
    mrl_numeric = []
    for raw in mrls:
        try:
            mrl_numeric.append(float(str(raw).replace(',', '')))
        except Exception:
            pass
    unit_norms = {_unit_norm(value) for value in unit_values if str(value).strip() != ''}
    mrl_ok = bool(mrl_numeric) and all(abs(value - MRL5) < 1e-9 for value in mrl_numeric)
    unit_ok = bool(unit_norms) and unit_norms.issubset({'ug/l'})
    _check(checks, 'UCMR 5 PFOA and PFOS reporting threshold unit', 'MRL values ' + str(sorted(mrls)) + '; unit values ' + str(sorted(unit_values)) + '; normalized units ' + str(sorted(unit_norms)), 'reported MRL is 0.004 with concentration unit micrograms per liter, where the micro sign byte decodes to the micro prefix', mrl_ok and unit_ok, kind='unit')
    size_values = sorted(value for value in set(selected['pws_size']) if value != '')
    _check(checks, 'UCMR 5 size categories documented', 'size codes ' + str(size_values) + '; blank size rows ' + str(int(selected['pws_size'].eq('').sum())), 'documented categories S and L only among selected first-sample rows', len(size_values) > 0 and set(size_values).issubset({'S', 'L'}))
    water_values = sorted(value for value in set(selected['facility_water_type']) if value != '')
    _check(checks, 'UCMR 5 facility water types documented', 'water type codes ' + str(water_values) + '; blank water type rows ' + str(int(selected['facility_water_type'].eq('').sum())), 'documented types SW, GW, GU and MX only among selected first-sample rows', len(water_values) > 0 and set(water_values).issubset({'SW', 'GW', 'GU', 'MX'}))
    u3_sign_list = sorted(value for value in u3signs if value != '')
    _check(checks, 'UCMR 3 prior PFOA and PFOS results available', 'rows ' + str(counts['u3_rows']) + '; systems ' + str(prior_pws) + '; sign codes ' + str(u3_sign_list) + '; unexpected signs ' + str(counts['u3_unknown']) + '; detected rows without numeric value ' + str(counts['u3_eq_missing']) + '; non-detect rows without MRL ' + str(counts['u3_non_detect_missing_mrl']), 'prior PFOA or PFOS rows exist with documented result signs, numeric detected values, and a numeric MRL for every non-detect so the censoring bound is sourced', counts['u3_rows'] > 0 and prior_pws > 0 and counts['u3_unknown'] == 0 and counts['u3_eq_missing'] == 0 and counts['u3_non_detect_missing_mrl'] == 0)
    u3_mrl_numeric = []
    for raw in u3_mrls:
        try:
            u3_mrl_numeric.append(float(str(raw).replace(',', '')))
        except Exception:
            pass
    u3_unit_norms = {_unit_norm(value) for value in u3_units if str(value).strip() != ''}
    u3_unit_ok = bool(u3_mrl_numeric) and bool(u3_unit_norms) and all(value > 0 for value in u3_mrl_numeric) and u3_unit_norms.issubset({'ug/l'})
    _check(checks, 'UCMR 3 prior PFOA and PFOS reporting units', 'MRL values ' + str(sorted(u3_mrls)) + '; unit values ' + str(sorted(u3_units)) + '; normalized units ' + str(sorted(u3_unit_norms)), 'prior UCMR 3 PFOA and PFOS results carry a positive MRL and concentration unit micrograms per liter', u3_unit_ok, kind='unit')
    duplicate_ids = int(shipped['pwsid'].duplicated().sum())
    _check(checks, 'One row per public water system', 'duplicate system identifier rows ' + str(duplicate_ids), 'zero duplicate system identifier rows across the shipped splits', duplicate_ids == 0)
    target_counts = shipped['any_pfoa_pfos'].value_counts().to_dict()
    target_values = sorted(set(int(value) for value in shipped['any_pfoa_pfos']))
    _check(checks, 'Target is binary with both outcomes present', 'target counts ' + str(target_counts), 'only 0 and 1 occur and both classes are present', set(target_values).issubset({0, 1}) and 0 in target_values and 1 in target_values)
    missing_out = [name for name in OUT_COLS if name not in selected.columns]
    _check(checks, 'Declared output columns present', 'missing ' + str(missing_out) + '; shipped columns ' + str(list(selected.columns)), 'every declared identifier, time, feature, carry and target column exists before writing', not missing_out)
    if missing_out:
        _fail('declared output columns missing ' + str(missing_out) + '; shipped columns ' + str(list(selected.columns)))
    outside_dates = selected.loc[assignment.eq(''), 'collection_date']
    outside_span = 'none'
    if len(outside_dates) > 0:
        outside_span = outside_dates.min().strftime('%Y-%m-%d') + ' to ' + outside_dates.max().strftime('%Y-%m-%d')
    _check(checks, 'First-sample split window accounting', 'selected first samples ' + str(len(selected)) + '; shipped ' + str(len(shipped)) + '; excluded ' + str(outside) + '; excluded date span ' + outside_span + '; blank date rows counted before selection ' + str(counts['missing_date']), 'every selected first sample is either shipped in exactly one fixed half-open window or explicitly counted as excluded by the December embargo or the 2023 through 2026 outer bounds, and blank collection dates were excluded and counted before first-sample selection', len(selected) == len(shipped) + outside)
    split_summary = []
    split_ok = True
    for name, start, end in WINDOWS:
        frame = frames[name]
        if frame.empty:
            split_ok = False
            split_summary.append(name + ' empty')
            continue
        low = frame['collection_date'].min()
        high = frame['collection_date'].max()
        inside = bool(low >= pd.Timestamp(start) and high < pd.Timestamp(end))
        split_ok = split_ok and inside
        split_summary.append(name + ' rows ' + str(len(frame)) + ' min ' + low.strftime('%Y-%m-%d') + ' max ' + high.strftime('%Y-%m-%d'))
    _check(checks, 'Fixed chronological split windows populated', '; '.join(split_summary) + '; rows outside windows ' + str(outside), 'each split is nonempty and its dates fall inside the frozen half-open window', split_ok)

    public_dir = output / 'public'
    private_dir = output / 'private'
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    destinations = {'train': public_dir / 'train.csv', 'eval': public_dir / 'eval.csv', 'holdout': private_dir / 'holdout.csv'}
    for name, start, end in WINDOWS:
        frame = frames[name].copy()
        frame = frame.sort_values(['collection_date', 'pwsid', 'sample_id'], kind='mergesort')
        frame['collection_date'] = frame['collection_date'].dt.strftime('%Y-%m-%d')
        frame = frame.loc[:, OUT_COLS]
        frame['any_pfoa_pfos'] = frame['any_pfoa_pfos'].astype(int)
        frame.to_csv(destinations[name], index=False, na_rep='')

    written_headers = {}
    for name, dest in destinations.items():
        written_headers[name] = list(pd.read_csv(dest, nrows=0).columns)
    header_ok = all(value == OUT_COLS for value in written_headers.values())
    _check(checks, 'Written CSV headers equal declared columns', 'headers ' + str(written_headers), 'each written CSV header exactly equals ' + str(OUT_COLS), header_ok)
    if not header_ok:
        _fail('written CSV headers differ from declared columns; headers ' + str(written_headers) + '; expected ' + str(OUT_COLS))

    written = pd.concat([pd.read_csv(dest, dtype=str, keep_default_na=False) for dest in destinations.values()], ignore_index=True)
    written['any_pfoa_pfos'] = written['any_pfoa_pfos'].astype(int)
    _verify_label_source(zip5, member5, delim5, cols5, written, checks)

    failed = [item for item in checks if not item['passed']]
    if failed:
        _fail('; '.join(item['check'] + ' | observed ' + item['observed'] + ' | expected ' + item['expected'] for item in failed))
    return checks
