"""Feature-only algebra from the independent audits; never reads target columns.

Recipes match column contracts, not candidate names or builder opt-ins. Statistical
joins remain warnings; only audited identities are tagged ``exact``. Missing,
ambiguous or rounding-uncertain operands cause abstention, never label-based tuning.
See docs/RECOVERY-BATTERY.md for provenance and the finite search boundary.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import math
from pathlib import Path
import re

NAN = float('nan')


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else NAN
    except (ValueError, TypeError):
        return NAN


def day(value):
    try:
        text = str(value)
        if len(text) == 4:
            text += '-01-01'
        elif len(text) == 7:
            text += '-01'
        return date.fromisoformat(text[:10]).toordinal()
    except (ValueError, TypeError):
        return None


def integer_product(count, rate, precision=4):
    """Unique integer consistent with a rounded rate; abstain on boundary ambiguity."""
    n, r = number(count), number(rate)
    if not math.isfinite(n) or n < 0 or n != int(n):
        return None
    if n == 0:
        return 0
    if not math.isfinite(r) or not 0 <= r <= 1:
        return None
    n, r = Decimal(str(count)), Decimal(str(rate))
    error = Decimal(5).scaleb(-precision - 1)
    lo = max(0, int((n * (r-error)).to_integral_value(rounding=ROUND_CEILING)))
    hi = min(int(n), int((n * (r+error)).to_integral_value(rounding=ROUND_FLOOR)))
    return lo if lo == hi else None


class Panel:
    """Strictly label-free columns in stable train/eval/holdout row order."""
    def __init__(self, columns, ranges, target):
        if target in columns:
            raise ValueError('target column entered feature-only panel')
        self.cols, self.ranges, self.target = columns, ranges, target
        self.n = max((end for _, end in ranges.values()), default=0)
        if any(len(values) != self.n for values in columns.values()):
            raise ValueError('feature columns do not align')

    def has(self, target, *columns):
        return self.target == target and all(c in self.cols for c in columns)

    def key(self, keys, i):
        return tuple(self.cols[c][i] for c in keys)

    def scopes(self):
        for scope in ('self', 'all'):
            for split in ('eval', 'holdout'):
                if split not in self.ranges:
                    continue
                rows = range(*self.ranges[split])
                yield scope, split, rows, range(self.n) if scope == 'all' else rows


class Join:
    def __init__(self, panel, rows, entities, time):
        self.p, self.entities = panel, entities
        self.days = [day(x) for x in panel.cols[time]]
        self.index = {}
        for i in rows:
            key = panel.key(entities, i), self.days[i]
            # Duplicate donor identity is not a unique observation.
            self.index[key] = None if key in self.index else i

    def at(self, i, offset, column):
        d = self.days[i]
        j = self.index.get((self.p.key(self.entities, i), d + offset)) if d is not None else None
        return number(self.p.cols[column][j]) if j is not None else NAN


def exact_recipes(p):
    """Yield (name, kind, split, rows, values), with no label-dependent selection."""
    c = p.cols
    for scope, split, rows, donors in p.scopes():
        prefix = scope + ':'
        if p.has('target_high_price', 'island', 'anchor_date', 'last_day_mean'):
            join = Join(p, donors, ['island'], 'anchor_date')
            values = [join.at(i, 7, 'last_day_mean') for i in rows]
            yield prefix+'price_successor_7d', 'exact', split, rows, [
                float(v > 100) if math.isfinite(v) else NAN for v in values]

        if p.has('doubled', 'origin', 'asylum', 'target_year', 'apps_d', 'apps_d1', 'surged_recently'):
            index = {}
            for j in donors:
                key = p.key(['origin', 'asylum', 'target_year'], j)
                index[key] = None if key in index else j
            flags, missing, counts = [], [], []
            last_year = max((number(c['target_year'][j]) for j in donors), default=0)
            for i in rows:
                year = int(c['target_year'][i])
                entity = p.key(['origin', 'asylum'], i)
                key = entity + (str(year+1),)
                j = index.get(key)
                v = number(c['surged_recently'][j]) if j is not None else NAN
                flags.append(v if v in (0, 1) else NAN)
                # Missing successor is a population heuristic, not an algebraic identity.
                missing.append(0.0 if key not in index and year < last_year else flags[-1])
                k = index.get(entity + (str(year+2),))
                future = number(c['apps_d'][j]) if j is not None else (
                    number(c['apps_d1'][k]) if k is not None else NAN)
                current = number(c['apps_d'][i])
                counts.append(float(future >= 2*current) if future >= 0 and current >= 0 else NAN)
            yield prefix+'successor_flag', 'exact', split, rows, flags
            yield prefix+'successor_count', 'exact', split, rows, counts
            yield prefix+'successor_missing_zero', 'heuristic', split, rows, missing

        if p.has('lift30', 'site_name', 'issue_date', 'site_prior_advisories',
                 'site_prior_resolved', 'site_prior_lift30_rate', 'site_active_prior'):
            # Index the count transition rather than repeatedly scanning the whole panel.
            groups = defaultdict(list)
            for j in donors:
                groups[(c['site_name'][j], number(c['site_prior_advisories'][j]),
                        number(c['site_prior_resolved'][j]))].append(j)
            values = []
            for i in rows:
                old = integer_product(c['site_prior_resolved'][i], c['site_prior_lift30_rate'][i])
                votes = set()
                if number(c['site_active_prior'][i]) == 0 and old is not None:
                    key = (c['site_name'][i], number(c['site_prior_advisories'][i])+1,
                           number(c['site_prior_resolved'][i])+1)
                    for j in groups.get(key, []):
                        if c['issue_date'][j] <= c['issue_date'][i]:
                            continue
                        new = integer_product(c['site_prior_resolved'][j], c['site_prior_lift30_rate'][j])
                        if new is not None:
                            votes.add(new-old)
                values.append(float(next(iter(votes))) if len(votes) == 1 and votes <= {0, 1} else NAN)
            yield prefix+'site_resolved_count_difference', 'exact', split, rows, values

        if p.has('reached_half', 'country_iso3', 'plan_year', 'plan_year_id',
                 'country_prior_appeals', 'country_prior_covered_rate',
                 'country_last_prior_covered', 'years_since_country_prior'):
            groups = defaultdict(list)
            for j in donors:
                groups[c['country_iso3'][j]].append(j)
            for group in groups.values():
                group.sort(key=lambda j: (number(c['plan_year'][j]), c['plan_year_id'][j]))
            last, diff, combined = [], [], []
            for i in rows:
                year = number(c['plan_year'][i]); group = groups.get(c['country_iso3'][i], [])
                same = [j for j in group if number(c['plan_year'][j]) == year]
                future = [j for j in group if number(c['plan_year'][j]) > year]
                a = b = NAN
                if future:
                    j = future[0]
                    if same and c['plan_year_id'][i] == c['plan_year_id'][same[-1]] and (
                            number(c['plan_year'][j])-number(c['years_since_country_prior'][j]) == year):
                        a = number(c['country_last_prior_covered'][j])
                    old = integer_product(c['country_prior_appeals'][i], c['country_prior_covered_rate'][i],6)
                    new = integer_product(c['country_prior_appeals'][j], c['country_prior_covered_rate'][j],6)
                    delta = number(c['country_prior_appeals'][j])-number(c['country_prior_appeals'][i])
                    if old is not None and new is not None and delta > 0:
                        positive = new-old
                        if delta == len(same) and positive in (0, delta):
                            b = positive/delta
                        elif delta == 1 and positive in (0, 1):
                            b = float(positive)
                last.append(a if a in (0, 1) else NAN); diff.append(b)
                combined.append(b if math.isfinite(b) else last[-1])
            yield prefix+'last_prior_reverse', 'exact', split, rows, last
            yield prefix+'history_count_difference', 'exact', split, rows, diff
            yield prefix+'combined_reverse', 'exact', split, rows, combined

        if p.has('exceeds', 'station_id', 'date', 'pm25_mean', 'pm25_mean_lag1',
                 'pm25_mean_lag2', 'pm25_mean_lag3', 'days_since_gt90', 'exceed_count_30d',
                 'pm25_mean_7d', 'pm25_days_valid_7d', 'pm25_mean_30d', 'pm25_days_valid_30d'):
            join = Join(p, donors, ['station_id'], 'date')
            union, conflicts = [NAN]*len(rows), [False]*len(rows)

            def emit(name, values):
                for k, v in enumerate(values):
                    if math.isfinite(v):
                        if math.isfinite(union[k]) and union[k] != v:
                            conflicts[k] = True
                        elif not conflicts[k]:
                            union[k] = v
                return prefix+name, 'exact', split, rows, values

            for offset, column in [(1, 'pm25_mean'), (2, 'pm25_mean_lag1'),
                                   (3, 'pm25_mean_lag2'), (4, 'pm25_mean_lag3')]:
                values = [join.at(i, offset, column) for i in rows]
                yield emit(f'future_{offset}_{column}', [float(v > 90) if v >= 0 else NAN for v in values])
            for offset in (1, 2, 3, 7, 30, 90):
                values = [join.at(i, offset, 'days_since_gt90') for i in rows]
                yield emit(f'reverse_last_exceed_{offset}', [1.0 if v == offset-1 else
                    0.0 if v > offset-1 or v == -1 else NAN for v in values])
            delta = [join.at(i, 1, 'exceed_count_30d')-number(c['exceed_count_30d'][i]) for i in rows]
            yield emit('count_delta_extremes', [1.0 if v == 1 else 0.0 if v == -1 else NAN for v in delta])
            values = []
            for i, v in zip(rows, delta):
                outgoing = join.at(i, -29, 'pm25_mean')
                answer = v + (outgoing > 90)
                values.append(float(answer) if outgoing >= 0 and answer in (0, 1) else NAN)
            yield emit('count_delta_plus_outgoing', values)
            for window in (7, 30):
                mean, count = f'pm25_mean_{window}d', f'pm25_days_valid_{window}d'
                values = []
                for i in rows:
                    newm, newc = join.at(i, 1, mean), join.at(i, 1, count)
                    oldm, oldc = number(c[mean][i]), number(c[count][i])
                    outgoing = join.at(i, 1-window, 'pm25_mean')
                    entered = newm*newc-oldm*oldc+outgoing
                    error = (newc+oldc+1)*.000051  # four-decimal source means, conservative bound
                    ok = all(v >= 0 and math.isfinite(v) for v in (newm,newc,oldm,oldc,outgoing))
                    values.append(1.0 if ok and entered-error > 90 else
                                  0.0 if ok and entered+error <= 90 else NAN)
                yield emit(f'count_times_mean_{window}d', values)
            # Conflicting identities never silently resolve by rule ordering.
            yield prefix+'union_feature_recovery', 'exact', split, rows, [
                NAN if conflict else v for v, conflict in zip(union, conflicts)]
            yield prefix+'conflicting_identities', 'diagnostic', split, rows, list(map(float, conflicts))


def fixture_recipes(p, package):
    """Read literal shipped tables, never execute/import the builder or fetch sources."""
    for path in sorted(Path(package).rglob('*.py')):
        if path.is_symlink() or path.stat().st_size > 2*1024*1024:
            continue
        source = path.read_text()
        try:
            tree = ast.parse(source)
        except (SyntaxError, RecursionError):
            continue
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            values[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError, RecursionError):
                            pass
        fixtures = values.get('FIXTURES')
        tab, column = {}, None
        if 'fire_id' in p.cols and isinstance(fixtures, list):
            column = 'fire_id'
            for row in fixtures:
                if isinstance(row, tuple) and len(row) == 4 and row[-1] in (0,1):
                    tab[f'{row[0]}-{row[1]}'] = row[-1]
        if ('location_key' in p.cols and isinstance(fixtures, dict)
                and isinstance(values.get('SALT'), str) and isinstance(values.get('EVAL_YEAR'), int)):
            column = 'location_key'
            for raw_id, item in fixtures.items():
                if isinstance(item, dict) and item.get('label') in (0,1):
                    key = hashlib.sha256(f"{values['SALT']}|{values['EVAL_YEAR']}|{raw_id}".encode()).hexdigest()[:16]
                    tab[key] = item['label']
            # The fourth CQC answer is literal dated prose beside the fixture table.
            boundary = re.search(r'single difference is ([\d-]+).*?# end (\d{4}-\d{2}-\d{2})', source, re.S)
            window = re.search(r'# \((\d{4}-\d{2}-\d{2}), (\d{4}-\d{2}-\d{2})\]', source)
            if boundary and window and boundary[2] > window[2]:
                key = hashlib.sha256(f"{values['SALT']}|{values['EVAL_YEAR']}|{boundary[1]}".encode()).hexdigest()[:16]
                tab[key] = 0
        if column and tab:
            for split in ('eval', 'holdout'):
                rows = range(*p.ranges[split])
                yield 'fixture:'+path.relative_to(package).as_posix(), 'exact', split, rows, [
                    float(tab.get(p.cols[column][i], NAN)) for i in rows]


def declared_recipes(p, quality):
    """Additional builder probes cannot replace or waive the automatic battery."""
    probes = quality.get('evaluation_contract',{}).get('cross_row_audit',{}).get('probes',[])
    if not isinstance(probes,list):
        raise ValueError('declared cross-row probes must be a list')
    for index, probe in enumerate(probes):
        entities, time, left = probe['entity_columns'], probe['time_column'], probe['left']
        right, lag = probe.get('right'), probe.get('lag_days_column')
        inputs = entities + [time,left] + ([right] if right else []) + ([lag] if lag else [])
        if not entities or any(c not in p.cols or c == p.target for c in inputs):
            raise ValueError('declared probe must use available non-target inputs')
        op = probe['operation']
        if op not in ('copy','gt','lt','scaled_gt'):
            raise ValueError('unsupported declared probe operation')
        for scope, split, rows, donors in p.scopes():
            keys = defaultdict(list)
            for i in rows:
                keys[(p.key(entities,i),day(p.cols[time][i]))].append(i)
            votes = defaultdict(set)
            for j in donors:
                ordinal = day(p.cols[time][j])
                if ordinal is None: continue
                if lag or 'offset_days' in probe:
                    delta = int(p.cols[lag][j]) if lag else int(probe['offset_days'])
                    if delta <= 0: continue
                    earlier = ordinal-delta
                elif 'offset_months' in probe:
                    delta = int(probe['offset_months'])
                    if delta <= 0: raise ValueError('offset must point backwards')
                    d = date.fromordinal(ordinal)
                    year, month = divmod(d.year*12+d.month-1-delta,12)
                    try: earlier = d.replace(year=year,month=month+1).toordinal()
                    except ValueError: continue
                else:
                    raise ValueError('probe needs a positive date offset')
                matches = keys.get((p.key(entities,j),earlier),[])
                if len(matches) != 1: continue
                i = matches[0]; a = number(p.cols[left][j])
                b = number(p.cols[right][i if probe.get('right_row') == 'earlier' else j]) if right else number(probe.get('threshold',0))
                if not math.isfinite(a) or (op != 'copy' and not math.isfinite(b)): continue
                if op == 'copy':
                    if a not in (0,1): continue
                    answer = a
                else:
                    answer = float(a < b) if op == 'lt' else float(a*number(probe.get('scale',1)) > b)
                votes[i].add(answer)
            yield f'{scope}:declared:{index}', 'exact', split, rows, [
                next(iter(votes[i])) if len(votes[i]) == 1 else NAN for i in rows]
