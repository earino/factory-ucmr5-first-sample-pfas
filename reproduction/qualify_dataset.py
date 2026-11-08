#!/usr/bin/env python3
"""Dataset qualification gate.

Runs the five checks that stand between a constructed dataset and a scored claim:

  1. leakage        - no shipped column may determine the target
  2. prediction time- input availability supports the declared evaluation scope
  3. units / frames - values compared to an external threshold must declare unit and frame
  4. splits         - disjoint, correctly ordered, and carrying enough positives AND enough
                      negatives, judged from the actual rows rather than the declarations
  5. runner         - the extract must match the existing runner's input contract

Usage:

    python3 qualify_dataset.py <dataset-dir> [--json]
    python3 qualify_dataset.py --accept <report.json> <dataset-dir>
    python3 qualify_dataset.py --selftest

`<dataset-dir>` holds the runner layout (`public/train.csv`, `public/eval.csv`,
`private/holdout.csv`, `meta.json`) plus a `quality.json` descriptor. Exit status is 0 only
when every check passes.

`--accept` is the downstream gate: it re-verifies that a previously written report passed
**and** that its recorded artifact hashes still match the files on disk. Nothing may be
scored or accepted without one.

Design notes worth keeping:

* **A declared feature list proves nothing.** The runner hands every non-target column to
  prediction code, so the only reliable leak test is to look for a shipped column that
  separates the classes on its own.
* **Declarations are not evidence.** Split boundaries, positive counts and event counts are
  recorded in `quality.json` *and* recomputed from the rows. A dataset whose declarations are
  internally consistent but whose rows disagree must fail - that is exactly how an evaluation
  split ends up inside the training period, or with no negatives at all.
* **Unique-valued columns are not leaks by themselves.** An identifier or a free timestamp
  reaches a perfect AUC by memorising one row per level, which says nothing. Such columns are
  reported as high-cardinality and must be declared, not treated as leaks.
* **Uniform results are a red flag, not a finding.** Splits that are all-positive or
  all-negative are refused, because that is what a unit or reference-frame error looks like.
* **Full-dataset shape.** Splits are read column-wise, and the categorical scan builds its
  per-level statistics once instead of once per row, so a million-row extract costs linear
  memory and roughly linear time.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import leakage_battery as battery
    from artifact_scrub import scan_tree
except ModuleNotFoundError:
    if '__file__' in globals():
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from factory import leakage_battery as battery
    from factory.artifact_scrub import scan_tree

GATE_VERSION = "1.6.1"

MIN_GROUP_FOR_CATEGORICAL = 5
MIN_EVAL_POSITIVES = 30
MIN_DISTINCT_EVENTS = 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc_from_scores(labels: list[int], scores: list[float]) -> float | None:
    """Rank-based ROC AUC with ties averaged. Stdlib only."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must align and be non-empty")
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            if pairs[position][1] == 1:
                rank_sum += average_rank
        index = end + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def numeric_scores(values) -> list[float] | None:
    """Return the values as floats, or None when the column is not numeric."""
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def group_rates(values, labels) -> tuple[dict[str, float], int]:
    """Per-level positive rate, computed once for the whole column.

    Computing this inside the per-row loop is quadratic on a full dataset: each row would
    re-sum its entire level. One pass builds the level table, one pass scores the rows.
    """
    totals: dict[str, int] = {}
    positives: dict[str, int] = {}
    for value, label in zip(values, labels):
        totals[value] = totals.get(value, 0) + 1
        if label:
            positives[value] = positives.get(value, 0) + 1
    return ({value: positives.get(value, 0) / count for value, count in totals.items()},
            len(totals))


def column_leak_score(values, labels: list[int]) -> dict:
    """Score one column as a potential answer source.

    Numeric columns are used directly. Categorical columns are scored by their per-level
    positive rate, but only when every level carries at least MIN_GROUP_FOR_CATEGORICAL rows -
    otherwise the encoding memorises individual rows and a perfect score means nothing. Those
    columns are reported separately as high_cardinality.
    """
    scores = numeric_scores(values)
    if scores is not None:
        auc = auc_from_scores(labels, scores)
        return {"kind": "numeric", "auc": auc, "levels": None,
                "determines_label": auc in (0.0, 1.0)}

    rates, levels = group_rates(values, labels)
    if levels > max(1, len(labels) // MIN_GROUP_FOR_CATEGORICAL):
        return {"kind": "high_cardinality", "auc": None, "levels": levels,
                "determines_label": False}
    auc = auc_from_scores(labels, [rates[value] for value in values])
    return {"kind": "categorical", "auc": auc, "levels": levels,
            "determines_label": auc in (0.0, 1.0)}


def read_columns(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Read a CSV column-wise. A list-of-dicts costs about 1 KiB per row, which is hundreds
    of megabytes on a full extract; parallel column lists are a fraction of that."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = next(reader)
        except StopIteration:
            return [], {}
        columns: dict[str, list[str]] = {name: [] for name in fieldnames}
        for row in reader:
            for index, name in enumerate(fieldnames):
                columns[name].append(row[index] if index < len(row) else "")
    return fieldnames, columns


def header_of(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), [])


class Report:
    def __init__(self, dataset: Path):
        self.dataset = str(dataset)
        self.gate_version = GATE_VERSION
        self.results: list[dict] = []
        self.artifact: dict[str, dict] = {}

    def add(self, check: str, ok: bool, detail: str, evidence=None):
        self.results.append({"check": check, "ok": bool(ok), "detail": detail,
                             "evidence": evidence or {}})

    @property
    def failed_checks(self) -> list[str]:
        return [result["check"] for result in self.results if not result["ok"]]

    @property
    def ok(self) -> bool:
        return all(result["ok"] for result in self.results)

    def render(self) -> str:
        return "\n".join(
            f"[{'PASS' if result['ok'] else 'FAIL'}] {result['check']}: {result['detail']}"
            for result in self.results)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "dataset": self.dataset,
            "gate_version": self.gate_version,
            "artifact": self.artifact,
            "results": self.results,
            "failed_checks": [r["check"] for r in self.results if not r["ok"]],
        }


def check_leakage(dataset: Path, quality: dict, report: Report,
                  columns_by_split: dict) -> None:
    target = quality["target"]
    shipped: set[str] = set()
    for fieldnames, _ in columns_by_split.values():
        shipped |= set(fieldnames)
    declared = set(quality.get("features", [])) | set(quality.get("carry_columns", [])) \
        | set(quality.get("id_columns", [])) | set(quality.get("high_cardinality_columns", [])) \
        | {target}
    undeclared = sorted(shipped - declared)
    report.add("leakage.declared_columns",
               not undeclared,
               "every shipped column is declared" if not undeclared
               else f"undeclared shipped columns: {undeclared}",
               {"undeclared": undeclared})

    forbidden = set(quality.get("label_source_columns", [])) | set(quality.get("post_hoc_columns", []))
    leaked_columns = sorted(shipped & forbidden)
    report.add("leakage.answer_source_absent",
               not leaked_columns,
               "no answer source or post-hoc column is shipped" if not leaked_columns
               else f"shipped columns that determine or postdate the label: {leaked_columns}",
               {"shipped_forbidden": leaked_columns})

    offenders, high_card, missing, unscorable = [], [], [], []
    for name, (fieldnames, columns) in columns_by_split.items():
        if not columns or not columns.get(target):
            missing.append(name)
            continue
        labels = [int(value == str(quality["positive_label"])) for value in columns[target]]
        for column in fieldnames:
            if column == target:
                continue
            try:
                finding = column_leak_score(columns[column], labels)
            except (ValueError, TypeError, IndexError) as exc:
                # A malformed split (duplicate header, ragged row, everything blank) used to abort
                # the whole run with a bare ValueError. Name the column and the split instead, so
                # the fix is obvious and the report stays machine-readable.
                unscorable.append({"column": column, "split": name,
                                   "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if finding["determines_label"]:
                offenders.append({**finding, "column": column, "split": name})
            elif finding["kind"] == "high_cardinality":
                high_card.append({"column": column, "split": name,
                                  "levels": finding["levels"]})
    report.add("leakage.every_column_could_be_scored",
               not unscorable,
               "every column was scored for leakage" if not unscorable
               else "columns that could not be scored: "
                    + ", ".join(f"{item['column']} ({item['split']}): {item['reason']}"
                                for item in unscorable[:6]),
               {"unscorable": unscorable})

    report.add("leakage.no_single_column_determines_the_target",
               not offenders,
               "no shipped column separates the classes on its own" if not offenders
               else "columns reconstruct the label: "
                    + ", ".join(sorted({f"{o['column']}@{o['split']}" for o in offenders})),
               {"offenders": offenders,
                "high_cardinality": sorted({f"{h['column']}@{h['split']}" for h in high_card}),
                "missing_splits": missing})

    declared_free = set(quality.get("id_columns", [])) | set(quality.get("high_cardinality_columns", []))
    undeclared_ids = sorted({h["column"] for h in high_card if h["column"] not in declared_free})
    report.add("leakage.high_cardinality_declared",
               not undeclared_ids,
               "high-cardinality columns are declared as identifiers or free fields"
               if not undeclared_ids
               else f"high-cardinality columns are undeclared: {undeclared_ids}",
               {"undeclared": undeclared_ids, "declared": sorted(declared_free)})


def check_prediction_time(quality: dict, report: Report) -> None:
    task = quality.get("task", {})
    when = (task.get("prediction_time") or "").strip()
    report.add("timing.prediction_time_declared", bool(when),
               "prediction instant is declared" if when
               else "quality.json has no task.prediction_time")

    features = set(quality.get("features", []))
    post_hoc = set(quality.get("post_hoc_columns", []))
    overlap = sorted(features & post_hoc)
    report.add("timing.no_post_hoc_feature", not overlap,
               "no feature is documented as post-hoc" if not overlap
               else f"features documented as post-hoc: {overlap}", {"overlap": overlap})

    sources = set(quality.get("label_source_columns", []))
    source_features = sorted(features & sources)
    report.add("timing.label_source_is_not_a_feature", not source_features,
               "the label source is not a feature" if not source_features
               else f"label source used as a feature: {source_features}",
               {"source_features": source_features})

    documented = quality.get("features_documented_at")
    report.add("timing.every_feature_documented",
               documented == "prediction_time",
               f"features documented as available {documented!r}",
               {"features_documented_at": documented})

    # A declaration is not evidence. These checks exist because two datasets were packaged with a
    # prediction instant that preceded the publication of their own features: the string
    # "prediction_time" was true of the *period*, not of the *release*. Every dataset must now state
    # when its source actually publishes, with observed dates, and state that its row selection did
    # not use any later period.
    timing = quality.get("timing") if isinstance(quality.get("timing"), dict) else {}
    instant = timing.get("prediction_instant") if isinstance(timing.get("prediction_instant"), dict) else {}
    features = timing.get("features") if isinstance(timing.get("features"), dict) else {}
    selection = timing.get("row_selection") if isinstance(timing.get("row_selection"), dict) else {}
    observations = timing.get("observations") if isinstance(timing.get("observations"), list) else []

    report.add("timing.instant_declared",
               bool(instant.get("kind")) and bool(instant.get("citation")),
               f"prediction instant kind={instant.get('kind')!r} citation={bool(instant.get('citation'))}",
               {"prediction_instant": instant})

    def _date(value):
        try:
            return datetime.date.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    lags = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        end, published = _date(observation.get("period_end")), _date(observation.get("published"))
        if end and published and published >= end:
            lags.append((published - end).days)
    # A dataset drawn from an administrative system can honestly claim the record existed at the
    # event. A dataset drawn from a *statistical release* cannot: the publisher emits it later, and
    # the lag must be observed, not assumed. That distinction is what the nhs-ae audit turned on.
    source_kind = str(features.get("source_kind") or "")
    needs_observations = source_kind == "statistical_release"
    measured_ok = (len(observations) >= 1 and len(lags) == len(observations)) if needs_observations \
        else (not observations or len(lags) == len(observations))
    report.add("timing.publication_lag_measured",
               measured_ok,
               (f"{len(lags)} of {len(observations)} observed period_end -> published dates are usable"
                + (" (statistical release: at least one observed publication date is required)"
                   if needs_observations else "")),
               {"observations": observations[:4], "measured_lag_days": max(lags) if lags else None,
                "source_kind": source_kind})

    feature_lag = features.get("published_days_after_period_end")
    instant_lag = instant.get("days_after_period_end")
    contract = quality.get('evaluation_contract') if isinstance(quality.get('evaluation_contract'), dict) else {}
    vintage = contract.get('vintage') if isinstance(contract.get('vintage'), dict) else {}
    event_research = (source_kind == 'administrative_record' and instant.get('period') is None
                      and contract.get('use') == 'retrospective_research'
                      and vintage.get('kind') in ('current_snapshot', 'mixed')
                      and contract.get('training_label_timing') == 'unverified'
                      and bool(contract.get('limitations')))
    unverified_event = event_research and feature_lag is None and instant_lag is None and not observations
    numeric = all(isinstance(v, int) and v >= 0 for v in (feature_lag, instant_lag))
    report.add("timing.event_feature_availability_unverified" if unverified_event
               else "timing.features_available_at_instant",
               unverified_event or (numeric and feature_lag <= instant_lag and (not lags or feature_lag >= max(lags))),
               ("Administrative current-vintage retrospective research: event-time availability UNVERIFIED; no period lag invented"
                if unverified_event else
                f"features publish {feature_lag}d after the period, instant is {instant_lag}d after it"
                if numeric else "features or instant do not state a publication lag in days"),
               {"features_lag_days": feature_lag, "instant_lag_days": instant_lag,
                "measured_lag_days": max(lags) if lags else None})

    report.add("timing.instant_accounts_for_publication_lag",
               not (isinstance(feature_lag, int) and feature_lag > 0 and instant.get("kind") != "publication"),
               ("instant is a publication instant" if instant.get("kind") == "publication"
                else f"instant kind is {instant.get('kind')!r} while features publish "
                     f"{feature_lag}d after the period"),
               {"kind": instant.get("kind"), "features_lag_days": feature_lag})

    latest = str(selection.get("latest_period_used") or "")
    event_selection = (event_research and not latest and bool(selection.get('fields'))
                       and bool(selection.get('rule')))
    report.add("timing.event_selection_declared" if event_selection
               else "timing.selection_uses_only_available_data",
               event_selection or (bool(latest) and "+" not in latest),
               ("Event selection rule declared for retrospective review; prediction-time eligibility is not established"
                if event_selection else f"row selection declares its latest period as {latest!r}"
               + ("" if latest and "+" not in latest else " - a selection that reaches into a later "
                  "period (M+1 or beyond) makes the retained population depend on the future")),
               {"latest_period_used": latest, "fields": selection.get("fields"), 'rule': selection.get('rule')})


def check_evaluation_contract(dataset, quality, report, columns_by_split):
    """Check declared scientific scope; measured chronology requires a full-row private ledger.

    Retrospective artifacts may qualify with explicit assumptions. They cannot claim historical
    availability. Cross-row probes measure exposure, not whether an existing model used it.
    """
    contract = quality.get('evaluation_contract')
    report.add('contract.declared', isinstance(contract, dict),
               'evaluation_contract required: intended use, access, vintage and evidence')
    if not isinstance(contract, dict):
        return
    scope = contract.get('use')
    access = contract.get('access')
    report.add('contract.scope', scope in ('retrospective_research', 'historical_forecasting')
               and access in ('isolated_rows', 'chronological', 'common_cutoff_batch'),
               f'use={scope!r}; access={access!r}; eval is development, holdout is final')
    evidence = contract.get('feature_evidence', {})
    if not isinstance(evidence, dict):
        evidence = {}
    columns = set(next(iter(columns_by_split.values()))[0]) - {quality.get('target')} if columns_by_split else set()
    missing = [name for name in sorted(columns) if not isinstance(evidence.get(name), dict)
               or not evidence[name].get('source') or not evidence[name].get('availability')
               or evidence[name].get('status') not in ('verified', 'assumed', 'unverified')]
    report.add('contract.every_input_has_availability_evidence', not missing,
               f'Every solver-visible non-target column needs availability evidence; missing: {missing}')
    vintage = contract.get('vintage', {})
    selection = contract.get('selection', {})
    if not isinstance(vintage, dict): vintage = {}
    if not isinstance(selection, dict): selection = {}
    report.add('contract.vintage_and_selection',
               vintage.get('kind') in ('as_of', 'current_snapshot', 'mixed') and bool(vintage.get('evidence'))
               and selection.get('kind') in ('prediction_time', 'outcome_conditioned')
               and bool(selection.get('evidence')) and bool(contract.get('limitations')),
               'Vintage, selection and limitations must be explicit; declarations are not measurements')
    audit = contract.get('cross_row_audit', {})
    if not isinstance(audit, dict): audit = {}
    probes = audit.get('probes', [])
    report.add('contract.cross_row_review', isinstance(probes, list)
               and bool(audit.get('rationale')) and bool(audit.get('auxiliary_files_review')),
               'Document applicable cross-row probes and every solver-visible auxiliary file')
    if isinstance(probes, list):
        for index, probe in enumerate(probes):
            try:
                results = cross_row_probe(columns_by_split, quality, probe)
                exposed = any(v['covered'] for v in results.values())
                exact = any(v['covered'] >= 1 and v['mismatches'] == 0
                            for s,v in results.items() if s in ('eval','holdout'))
                report.add(f'contract.cross_row_probe_{index}',
                           not exact and (not exposed or access in ('isolated_rows', 'chronological')),
                           'Exact scored-row recovery blocks every scope; approximate exposure still requires restricted access', results)
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                report.add(f'contract.cross_row_probe_{index}', False, f'Invalid probe: {exc}')
    if scope == 'historical_forecasting':
        verified = not missing and all(evidence[c].get('status') == 'verified' for c in columns)
        report.add('contract.historical_evidence', verified and vintage.get('kind') == 'as_of'
                   and selection.get('kind') == 'prediction_time',
                   'Historical forecasting requires verified inputs, as-of vintage and prediction-time selection')
    ledger = contract.get('availability_ledger')
    if not ledger:
        report.add('contract.label_availability', scope == 'retrospective_research'
                   and contract.get('training_label_timing') == 'unverified',
                   'No full-row availability ledger: training-label timing UNVERIFIED; retrospective use only')
        return
    problems = []
    try:
        path = dataset / ledger
        if not path.resolve().is_relative_to(dataset.resolve()) or path.is_symlink():
            raise ValueError('availability ledger must be a regular file inside the artifact')
        report.artifact['availability_ledger'] = {'file': ledger, 'sha256': sha256_of(path),
                                                'bytes': path.stat().st_size}
        id_cols = quality['id_columns']
        if not id_cols:
            raise ValueError('availability ledger requires id_columns')
        expected = {}
        for split, (_, cols) in columns_by_split.items():
            expected[split] = {tuple(values) for values in zip(*(cols[c] for c in id_cols))}
            if len(expected[split]) != len(cols[quality['target']]):
                raise ValueError('availability ledger needs unique composite row identifiers')
        seen = {split: set() for split in expected}
        earliest, latest, last_prediction = {}, {}, {}
        def stamp(value):
            parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                raise ValueError('ledger timestamps must have explicit timezone')
            return parsed
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                split = row['split']; key = tuple(row[c] for c in id_cols)
                if split not in expected or key not in expected[split] or key in seen[split]:
                    raise ValueError(f'unknown or duplicate ledger row: {split}/{key}')
                seen[split].add(key)
                prediction = stamp(row['prediction_time'])
                available = stamp(row['features_available_at'])
                label = stamp(row['label_available_at'])
                if available > prediction:
                    problems.append(f'{split}/{key}: features arrive after prediction')
                earliest[split] = min(earliest.get(split, prediction), prediction)
                last_prediction[split] = max(last_prediction.get(split, prediction), prediction)
                latest[split] = max(latest.get(split, label), label)
        if access == 'common_cutoff_batch' and earliest != last_prediction:
            problems.append('common_cutoff_batch contains multiple prediction instants within a split')
        if seen != expected:
            problems.append('availability ledger does not cover every artifact row exactly once')
        for before, after in (('train', 'eval'), ('train', 'holdout'), ('eval', 'holdout')):
            if before not in latest or after not in earliest or latest[before] > earliest[after]:
                problems.append(f'{before} outcomes unavailable before first {after} prediction')
        report.add('contract.label_availability', not problems,
                   '; '.join(problems[:10]) or 'Full-row feature and development-label availability verified against supplied ledger',
                   {'problems': len(problems), 'rows_checked': {k: len(v) for k, v in seen.items()}})
    except (OSError, KeyError, ValueError, TypeError) as exc:
        report.add('contract.label_availability', False, f'Invalid availability ledger: {exc}')


def cross_row_probe(columns_by_split, quality, probe):
    """Full-artifact blind join, with singleton targets and unanimous donors. No model/AUC.

    The source row points to an earlier target by fixed day/month/year offset or a day-lag
    column. expression is copy, gt, lt, or scaled_gt; operands come only from donor features.
    Supports inspection histories, ABS, FluNet, EUTL and NOAA margin probes. Price ledger
    extensions can be expressed as multiple probes. Target labels are read only to compare.
    """
    entities = probe['entity_columns']; time_col = probe['time_column']
    if not entities or not isinstance(entities, list):
        raise ValueError('entity_columns must be nonempty')
    lhs, rhs = probe['left'], probe.get('right')
    target = quality['target']
    if lhs == target or rhs == target or target in entities or time_col == target:
        raise ValueError('probe inputs cannot read target labels')
    op = probe['operation']
    if op not in ('copy', 'gt', 'lt', 'scaled_gt'):
        raise ValueError('unsupported probe operation')
    result = {}
    for split, (_, cols) in columns_by_split.items():
        n = len(cols[target]); keys = []; donors = {}; counts = {}
        # Freeze donor operands across only the splits visible at this evaluation stage.
        ranks = {'train': 0, 'eval': 1, 'holdout': 2}
        donor_cols = {c: [] for c in cols}
        for donor_split, (_, values) in columns_by_split.items():
            if ranks.get(donor_split, 99) <= ranks.get(split, -1):
                for c in donor_cols:
                    donor_cols[c].extend(values[c])
        for i in range(n):
            day = datetime.date.fromisoformat(cols[time_col][i][:10] + ('-01' if len(cols[time_col][i]) == 7 else ''))
            key = tuple(cols[c][i] for c in entities) + (day.isoformat(),)
            keys.append(key); counts[key] = counts.get(key, 0) + 1
        target_cols = cols
        cols = donor_cols
        for i in range(len(cols[target])):
            day = datetime.date.fromisoformat(cols[time_col][i][:10] + ('-01' if len(cols[time_col][i]) == 7 else ''))
            key = tuple(cols[c][i] for c in entities) + (day.isoformat(),)
            if 'lag_days_column' in probe:
                if probe['lag_days_column'] == target:
                    raise ValueError('lag cannot read target')
                lag = int(cols[probe['lag_days_column']][i])
                if lag <= 0:
                    continue
                earlier = day - datetime.timedelta(days=lag)
            elif 'offset_months' in probe:
                months = int(probe['offset_months'])
                if months <= 0:
                    raise ValueError('offset must point backwards')
                y, m = divmod(day.year * 12 + day.month - 1 - months, 12)
                earlier = day.replace(year=y, month=m + 1)
            else:
                days = int(probe['offset_days'])
                if days <= 0:
                    raise ValueError('offset must point backwards')
                earlier = day - datetime.timedelta(days=days)
            left = float(cols[lhs][i])
            if op == 'copy' and left not in (0, 1):
                continue
            right = (None if probe.get('right_row') == 'earlier' else
                     float(cols[rhs][i]) if rhs else float(probe.get('threshold', 0)))
            donor_key = key[:-1] + (earlier.isoformat(),)
            donors.setdefault(donor_key, set()).add((left, right))
        covered = mismatches = ambiguous = 0
        for i, key in enumerate(keys):
            votes = donors.get(key, set())
            if counts[key] != 1 or len(votes) != 1:
                ambiguous += bool(votes)
                continue
            covered += 1
            left, right = next(iter(votes))
            if probe.get('right_row') == 'earlier':
                right = float(target_cols[rhs][i])
            prediction = int(left) if op == 'copy' else (int(left < right) if op == 'lt' else
                          int(left * float(probe.get('scale', 1)) > right))
            actual = int(target_cols[target][i] == str(quality['positive_label']))
            mismatches += prediction != actual
        result[split] = {'rows': n, 'covered': covered, 'mismatches': mismatches,
                         'ambiguous_or_conflicting': ambiguous}
    return result


def check_units_and_frames(quality: dict, report: Report) -> None:
    measured = quality.get("measurements", [])
    missing = [entry.get("column") for entry in measured
               if not entry.get("unit") or not entry.get("frame")]
    report.add("units.measured_columns_declare_unit_and_frame",
               bool(measured) and not missing,
               f"{len(measured)} measured column(s) declare unit and frame"
               if measured and not missing
               else f"columns missing unit or frame: {missing}" if measured
               else "no measurements declared",
               {"measurements": measured})

    if quality.get("external_threshold"):
        receipt = quality.get('unit_validation') or {}
        valid = (isinstance(receipt,dict) and type(receipt.get('checks')) is int
                 and receipt['checks'] > 0 and receipt.get('mismatches') == 0
                 and bool(receipt.get('source')) and isinstance(receipt.get('evidence_sha256'),str)
                 and len(receipt['evidence_sha256']) == 64
                 and all(c in '0123456789abcdef' for c in receipt['evidence_sha256']))
        report.add('units.validation_receipt', valid,
                   'External-threshold checks need a source-bound counts-only receipt; never ship fixture answers',
                   {'receipt':receipt})
    else:
        report.add('units.validation_receipt', True,
                   'No external threshold, so no unit-validation receipt is required')


def split_timestamp(value):
    """Compare ISO dates/timestamps as times, including an optional monthly grain."""
    value = str(value)
    if len(value) == 7:
        value += '-01'
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))


def check_splits(quality: dict, report: Report, columns_by_split: dict) -> None:
    splits = quality.get("splits", {})
    windows = quality.get("split_windows", {})
    time_column = quality.get("time_column")
    keys = ["train", "eval", "holdout"]

    report.add("splits.time_column_declared", bool(time_column),
               f"split overlap is checked against {time_column!r}" if time_column
               else "quality.json must declare time_column so split overlap can be checked "
                    "from the rows")

    declared_ok = all(key in windows and isinstance(windows[key], dict) for key in keys)
    report.add("splits.machine_readable_windows", declared_ok,
               "train/eval/holdout windows are declared with start/end"
               if declared_ok
               else "quality.json must declare split_windows with per-split start/end, so the "
                    "rows can be checked against them",
               {"split_windows": windows})

    present = [key for key in keys if key in splits]
    missing_files = [key for key in keys
                     if key not in splits or not (Path(report.dataset) / splits[key]).is_file()]
    report.add("splits.files_present", not missing_files,
               "all three split files exist" if not missing_files
               else f"missing split files: {missing_files}", {"missing": missing_files})
    if missing_files or not present:
        return

    target = quality["target"]
    positive = str(quality["positive_label"])
    counts: dict[str, dict] = {}
    for key in keys:
        fieldnames, columns = columns_by_split[key]
        labels = columns.get(target, [])
        pos = sum(1 for value in labels if value == positive)
        counts[key] = {"rows": len(labels), "positives": pos, "negatives": len(labels) - pos,
                       "positive_rate": round(pos / len(labels), 6) if labels else None}

    problems = [f"{key} is empty" for key in keys if counts[key]["rows"] == 0]
    report.add("splits.non_empty", not problems,
               "every split has rows" if not problems else "; ".join(problems), counts)

    # Both classes, judged from the rows. An eval or holdout split with no negatives scores
    # a meaningless AUC, and a declaration saying otherwise must not save it.
    single_class = [f"{key} has {counts[key]['positives']} positives and "
                    f"{counts[key]['negatives']} negatives"
                    for key in ("eval", "holdout")
                    if counts[key]["positives"] == 0 or counts[key]["negatives"] == 0]
    report.add("splits.eval_and_holdout_have_both_classes", not single_class,
               "eval and holdout each contain positives and negatives"
               if not single_class else "; ".join(single_class), counts)

    declared_events = quality.get("positive_events", {})
    missing_decl = [key for key in ("eval", "holdout")
                    if not (declared_events.get(key) or {}).get("positives")
                    or not (declared_events.get(key) or {}).get("distinct_events")]
    report.add("splits.positive_events_declared", not missing_decl,
               "eval and holdout declare positives and distinct events" if not missing_decl
               else f"missing clustered counts for {missing_decl}",
               {"declared": declared_events})

    mismatched = []
    for key in ("eval", "holdout"):
        entry = declared_events.get(key) or {}
        if "positives" in entry and entry["positives"] != counts[key]["positives"]:
            mismatched.append(f"{key}: declared {entry['positives']}, rows have "
                              f"{counts[key]['positives']}")
    report.add("splits.positive_events_match_rows", not mismatched,
               "declared positive counts match the rows" if not mismatched
               else "; ".join(mismatched), {"mismatched": mismatched})

    # The clustered event count is a claim like any other, so it is recomputed from the rows.
    # An "event" is one distinct date of the declared event key among the positive rows, which
    # is the conservative reading: positives sharing a day are not independent.
    event_key = quality.get("event_key_column")
    report.add("splits.event_key_column_declared", bool(event_key),
               f"clustered events are counted by {event_key!r} (date part)" if event_key
               else "quality.json must declare event_key_column so the clustered event count "
                    "can be recomputed from the rows")
    recomputed: dict[str, int | None] = {key: None for key in keys}
    event_mismatch = []
    if event_key:
        for key in keys:
            fieldnames, columns = columns_by_split[key]
            if event_key not in columns:
                event_mismatch.append(f"{key} has no {event_key} column")
                continue
            days = {value[:10] for value, label in zip(columns[event_key], columns[target])
                    if label == positive and value}
            recomputed[key] = len(days)
            declared = (declared_events.get(key) or {}).get("distinct_events")
            if declared is not None and declared != len(days):
                event_mismatch.append(f"{key}: declared {declared} distinct events, "
                                      f"rows have {len(days)}")
        report.add("splits.distinct_events_match_rows", not event_mismatch,
                   "declared clustered event counts match the rows" if not event_mismatch
                   else "; ".join(event_mismatch),
                   {"recomputed": recomputed, "declared": declared_events})

    thin = []
    for key in ("eval", "holdout"):
        entry = declared_events.get(key) or {}
        declared_pos = entry.get("positives", counts[key]["positives"])
        events = entry.get("distinct_events") if recomputed[key] is None else recomputed[key]
        if declared_pos < MIN_EVAL_POSITIVES or (events is not None
                                                 and events < MIN_DISTINCT_EVENTS):
            thin.append(f"{key}={declared_pos} positives / {events} distinct events "
                        f"(positives threshold {MIN_EVAL_POSITIVES}, "
                        f"event threshold {MIN_DISTINCT_EVENTS})")
    report.add("splits.enough_clustered_positives", not thin,
               "eval and holdout carry enough independent positive events to score"
               if not thin else "too few independent positives: " + "; ".join(thin),
               {"thin": thin, "recomputed_events": recomputed,
                "minimums": {"positives": MIN_EVAL_POSITIVES,
                             "distinct_events": MIN_DISTINCT_EVENTS}})

    rates = [counts[key]["positive_rate"] for key in keys
             if counts[key]["positive_rate"] is not None]
    uniform = bool(rates) and all(rate in (0.0, 1.0) for rate in rates)
    report.add("splits.rates_not_uniform", not uniform,
               "positive rates are between 0 and 1" if not uniform
               else "every split is all-positive or all-negative, which is a unit or frame "
                    "red flag", {"rates": rates})

    if not time_column:
        return

    ranges, parsed_ranges = {}, {}
    for key in keys:
        values = [v for v in columns_by_split[key][1].get(time_column, []) if v]
        try:
            minimum = maximum = None
            for value in values:
                item = (split_timestamp(value), value)
                minimum = item if minimum is None or item < minimum else minimum
                maximum = item if maximum is None or item > maximum else maximum
            ranges[key] = {"min": minimum[1], "max": maximum[1]} if minimum else None
            parsed_ranges[key] = (minimum[0], maximum[0]) if minimum else None
        except (ValueError, TypeError):
            ranges[key] = None
    report.add("splits.actual_ranges_computed",
               all(ranges[key] for key in keys),
               "row timestamps found for every split" if all(ranges[key] for key in keys)
               else f"missing or invalid/mixed-timezone {time_column} values in {[k for k in keys if not ranges[k]]}",
               {"ranges": ranges})
    if not all(ranges[key] for key in keys):
        return

    overlaps = []
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            try:
                overlap = (parsed_ranges[left][0] <= parsed_ranges[right][1]
                           and parsed_ranges[right][0] <= parsed_ranges[left][1])
            except TypeError:
                overlaps.append(f'{left}/{right} mix timezone-aware and naive timestamps')
                continue
            if overlap:
                overlaps.append(f"{left} [{ranges[left]['min']}..{ranges[left]['max']}] "
                                f"overlaps {right} [{ranges[right]['min']}..{ranges[right]['max']}]")
    report.add("splits.actual_ranges_disjoint", not overlaps,
               "the rows of the three splits occupy disjoint periods" if not overlaps
               else "; ".join(overlaps), {"ranges": ranges, "overlaps": overlaps})

    outside = []
    if declared_ok:
        for key in keys:
            start = windows[key].get("start")
            end = windows[key].get("end")
            try:
                if start and parsed_ranges[key][0] < split_timestamp(start):
                    outside.append(f"{key} starts {ranges[key]['min']}, before its declared start {start}")
                if end and parsed_ranges[key][1] >= split_timestamp(end):
                    outside.append(f"{key} reaches {ranges[key]['max']}, at or past its declared end {end}")
            except (ValueError, TypeError):
                outside.append(f'{key} has invalid or incompatible-timezone split boundaries')
    report.add("splits.rows_within_declared_windows", not outside,
               "every row falls inside its split's declared window" if not outside
               else "; ".join(outside), {"windows": windows, "ranges": ranges})

    id_columns = quality.get("id_columns", [])
    if id_columns:
        id_column = id_columns[0]
        target = quality.get("target")
        seen: dict[str, str] = {}
        first_label: dict[tuple, object] = {}    # (split, id) -> the label first seen for it
        shared = []                    # an identifier spanning two splits: leakage
        duplicate_rows: dict[str, int] = {}      # repeats inside one split: the source's grain
        duplicate_labels: dict[tuple, set] = {}
        for key in keys:
            values = columns_by_split[key][1].get(id_column, [])
            labels = columns_by_split[key][1].get(target, []) if target else []
            for index, value in enumerate(values):
                label = labels[index] if index < len(labels) else None
                first = seen.get(value)
                if first is None:
                    seen[value] = key
                    first_label[(key, value)] = label
                elif first != key:
                    shared.append(f"{value} in {first} and {key}")
                else:
                    duplicate_rows[key] = duplicate_rows.get(key, 0) + 1
                    # Both occurrences matter: comparing only against the repeat would call a
                    # contradictory pair consistent.
                    duplicate_labels.setdefault((key, value), set()).add(
                        first_label.get((key, value)))
                    duplicate_labels[(key, value)].add(label)
        # Two different properties, deliberately checked separately. A repeated identifier
        # *inside* one split is not split leakage - it is the source having more than one row
        # per identifier - and conflating them fails a usable dataset with a message that reads
        # "... in train and train". Only a cross-split identifier is a leak.
        report.add("splits.no_cross_split_entities", not shared,
                   f"no {id_column} appears in more than one split" if not shared
                   else f"{len(shared)} identifiers span two splits: {shared[:3]}",
                   {"shared_sample": shared[:10], "shared_count": len(shared)})

        repeated = sum(duplicate_rows.values())
        tolerance = quality.get("expected_duplicate_ids")
        if tolerance is None:
            report.add("splits.duplicate_entities_within_splits", not repeated,
                       "no identifier repeats inside a split" if not repeated
                       else f"{repeated} rows repeat an identifier within a split "
                            f"{duplicate_rows}; declare expected_duplicate_ids in "
                            f"quality.json to pin a known count",
                       {"duplicate_rows": duplicate_rows})
        else:
            report.add("splits.duplicate_entities_within_splits", repeated <= tolerance,
                       f"{repeated} repeated-identifier rows within splits, at most the "
                       f"declared {tolerance}" if repeated <= tolerance
                       else f"{repeated} repeated-identifier rows exceeds the declared "
                            f"expected_duplicate_ids={tolerance}",
                       {"duplicate_rows": duplicate_rows, "declared": tolerance})

        if duplicate_labels:
            contradictory = sorted(f"{value} in {key}"
                                   for (key, value), labels in duplicate_labels.items()
                                   if len(labels) > 1)
            # This check is only meaningful against a claim. If the descriptor says the id
            # column identifies a unique entity, one label per identifier follows and a
            # contradiction is label noise. If it makes no such claim, the identifier is just
            # reused - Austin's `incident_number` carries two *different* events hours apart,
            # with different sectors and different labels - and the right thing is to measure
            # the count and put it in the report rather than to refuse the dataset for a
            # property the source does not claim to have.
            claims_entity_key = quality.get("id_columns_are_entity_keys") is True
            if claims_entity_key:
                report.add("splits.repeated_identifiers_carry_one_label", not contradictory,
                           "each repeated identifier carries one label, as declared"
                           if not contradictory else
                           f"{len(contradictory)} repeated identifiers carry more than one "
                           f"label while id_columns_are_entity_keys is declared true: "
                           f"{contradictory[:3]}",
                           {"sample": contradictory[:10], "count": len(contradictory)})
            else:
                report.add("splits.repeated_identifiers_carry_one_label", True,
                           f"{len(contradictory)} of {len(duplicate_labels)} repeated "
                           f"identifiers carry more than one label; the descriptor does not "
                           f"claim id_columns_are_entity_keys, so this is recorded as a source "
                           f"property: {contradictory[:3]}",
                           {"sample": contradictory[:10], "count": len(contradictory),
                            "repeated_identifiers": len(duplicate_labels),
                            "claimed_entity_key": False})


def check_runner(dataset: Path, quality: dict, report: Report,
                 columns_by_split: dict) -> None:
    splits = quality.get("splits", {})
    layout_ok = splits.get("holdout", "").startswith("private/") and \
        splits.get("train", "").startswith("public/") and \
        splits.get("eval", "").startswith("public/")
    report.add("runner.layout", layout_ok,
               "holdout is private and train/eval are public" if layout_ok
               else f"unexpected layout: {splits}")

    meta_path = dataset / "meta.json"
    if not meta_path.is_file():
        report.add("runner.meta_json", False, "meta.json is missing")
        return
    meta = json.loads(meta_path.read_text())
    required = ["target", "positive_label", "id_columns", "columns", "rows", "positive_rate"]
    missing = [key for key in required if key not in meta]
    report.add("runner.meta_json_keys", not missing,
               "meta.json carries the keys the runner reads" if not missing
               else f"meta.json missing {missing}", {"keys": sorted(meta)})
    if missing:
        return

    target = meta["target"]
    problems = []
    if meta["positive_label"] != quality["positive_label"]:
        problems.append("meta positive_label disagrees with quality.json")
    if target != quality["target"]:
        problems.append("meta target disagrees with quality.json")
    if not set(meta["id_columns"]) <= set(meta["columns"]):
        problems.append("id_columns are not a subset of columns")
    for key in ("train", "eval", "holdout"):
        fieldnames, _ = columns_by_split[key]
        if target not in fieldnames:
            problems.append(f"{key} lacks the target column")
        if list(meta["columns"]) != fieldnames:
            problems.append(f"{key} header differs from meta columns")
    report.add("runner.meta_matches_files", not problems,
               "meta.json agrees with the split files" if not problems
               else "; ".join(problems), {"problems": problems})

    counts_ok, detail = True, "meta rows and rates match the files"
    for key in ("train", "eval", "holdout"):
        labels = columns_by_split[key][1].get(target, [])
        pos = sum(1 for value in labels if value == str(meta["positive_label"]))
        rate = pos / len(labels) if labels else 0.0
        if meta["rows"].get(key) != len(labels):
            counts_ok, detail = False, f"{key} row count mismatch"
        if meta["positive_rate"].get(key) is not None and \
                abs(meta["positive_rate"][key] - rate) > 1e-6:
            counts_ok, detail = False, f"{key} positive rate mismatch"
    report.add("runner.meta_counts_match_files", counts_ok, detail)

    id_cols = set(meta["id_columns"])
    id_feature_ok = not (id_cols & set(quality.get("features", [])))
    report.add("runner.id_columns_are_not_features", id_feature_ok,
               "declared identifiers are not also listed as features" if id_feature_ok
               else f"identifiers listed as features: {sorted(id_cols & set(quality.get('features', [])))}")


def qualify(dataset: Path, *, record_battery=False, battery_output=None) -> Report:
    report = Report(dataset)
    quality_path = dataset / "quality.json"
    if dataset.is_symlink() or quality_path.is_symlink() or not quality_path.is_file():
        report.add("descriptor", False, f"quality.json not found in {dataset}")
        return report
    try:
        quality = json.loads(quality_path.read_text())
        if not isinstance(quality,dict) or not isinstance(quality.get('splits'),dict):
            raise ValueError('descriptor and splits must be objects')
        for relative in quality['splits'].values():
            path = dataset/relative
            if path.is_symlink() or not path.resolve().is_relative_to(dataset.resolve()):
                raise ValueError('split path escapes artifact')
    except (OSError, ValueError, TypeError) as exc:
        report.add('descriptor',False,f'Invalid descriptor: {exc}')
        return report
    report.add("descriptor", True,
               f"quality.json loaded for {quality.get('name', dataset.name)!r}")

    # No scored target column is materialized before the blind prediction freeze.
    # The optional write is an explicit construction step, never historical audit.
    try:
        with tempfile.TemporaryDirectory(prefix='factory-battery-') as temporary:
            output = Path(battery_output) if battery_output else Path(temporary)/'audit'
            result = battery.run(dataset, output)
        if record_battery:
            quality['adversarial_battery'] = result
            quality_path.write_text(json.dumps(quality, indent=2, allow_nan=False)+'\n')
        report.add('battery.exact_recovery', result['ok'],
                   'Any zero-error exact reconstruction of a scored row blocks qualification', result)
        report.add('battery.recorded_in_quality', quality.get('adversarial_battery') == result,
                   'Construction must record the computed battery receipt in quality.json; use --record-battery')
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, csv.Error, RecursionError) as exc:
        report.add('battery.completed', False, f'Battery incomplete: {type(exc).__name__}: {exc}')
    try:
        scrub = scan_tree(dataset, dataset=dataset)
        report.add('artifacts.no_shipped_answers', scrub['ok'],
                   'Every generated artifact file is scanned for shipped answer scaffolds', scrub)
        # These exact auxiliary bytes, including newly added files, must survive acceptance.
        report.scanned_files = scrub['files']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.add('artifacts.no_shipped_answers', False, f'Artifact scan incomplete: {exc}')

    splits = quality.get("splits", {})
    columns_by_split: dict[str, tuple] = {}
    for key in ("train", "eval", "holdout"):
        relative = splits.get(key)
        path = dataset / relative if relative else None
        if path and path.is_file():
            columns_by_split[key] = read_columns(path)

    for key in ("train", "eval", "holdout"):
        relative = splits.get(key)
        path = dataset / relative if relative else None
        if path and path.is_file():
            report.artifact[key] = {"file": relative, "sha256": sha256_of(path),
                                    "bytes": path.stat().st_size}
    meta_path = dataset / "meta.json"
    if meta_path.is_file():
        report.artifact["meta.json"] = {"file": "meta.json", "sha256": sha256_of(meta_path),
                                        "bytes": meta_path.stat().st_size}
    report.artifact["quality.json"] = {"file": "quality.json",
                                       "sha256": sha256_of(quality_path),
                                       "bytes": quality_path.stat().st_size}
    report.artifact['scanned_files'] = getattr(report, 'scanned_files', {})
    check_evaluation_contract(dataset, quality, report, columns_by_split)
    report.artifact["artifact_version"] = {
        "digest": hashlib.sha256(
            "".join(sorted(f"{key}:{value['sha256']}"
                           for key, value in report.artifact.items()
                           if isinstance(value, dict) and "sha256" in value)).encode()
        ).hexdigest(),
        "components": sorted(key for key, value in report.artifact.items()
                             if isinstance(value, dict) and "sha256" in value),
    }

    check_leakage(dataset, quality, report, columns_by_split)
    check_prediction_time(quality, report)
    check_units_and_frames(quality, report)
    check_splits(quality, report, columns_by_split)
    check_runner(dataset, quality, report, columns_by_split)
    return report


def accept(report_path: Path, dataset: Path) -> int:
    """Downstream gate: a passing report whose artifact hashes still match the files."""
    if not report_path.is_file():
        print(f"ACCEPT FAILED: no qualification report at {report_path}")
        return 1
    report = json.loads(report_path.read_text())
    if report.get("gate_version") != GATE_VERSION:
        print(f"ACCEPT FAILED: report was produced by gate {report.get('gate_version')}, "
              f"this gate is {GATE_VERSION}")
        return 1
    if not report.get("ok"):
        print("ACCEPT FAILED: the qualification report did not pass. Failed checks: "
              + ", ".join(report.get("failed_checks", [])))
        return 1
    checks = report.get('results', [])
    required_checks = {'battery.exact_recovery', 'battery.recorded_in_quality', 'artifacts.no_shipped_answers'}
    if (not checks or any(r.get('ok') is not True for r in checks)
            or not required_checks <= {r.get('check') for r in checks}
            or report.get('failed_checks') != []):
        print('ACCEPT FAILED: missing, failed or inconsistent mandatory gate results')
        return 1
    recorded = report.get("artifact", {})
    problems = []
    required = ["train", "eval", "holdout", "meta.json", "quality.json"]
    descriptor = json.loads((dataset / "quality.json").read_text())
    receipt = descriptor.get('adversarial_battery')
    reported_receipt = next(r.get('evidence') for r in checks if r.get('check') == 'battery.exact_recovery')
    if (receipt != reported_receipt or not isinstance(receipt, dict) or not receipt.get('ok')
            or receipt.get('version') != battery.VERSION or receipt.get('code') != battery.code_identity()):
        problems.append('battery receipt is missing, stale or differs from computed qualification evidence')
    try:
        if not receipt or receipt.get('input_digest') != battery.digest(battery.input_identity(dataset)):
            problems.append('battery input digest differs from current full artifact')
        current = scan_tree(dataset, dataset=dataset)
        if not current['ok'] or current['files'] != recorded.get('scanned_files'):
            problems.append('generated artifact file inventory changed or contains answer scaffolds')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append('artifact reinspection incomplete: '+str(exc))
    if descriptor.get("evaluation_contract", {}).get("availability_ledger"):
        required.append("availability_ledger")
    for key in required:
        entry = recorded.get(key)
        if not entry:
            problems.append(f"{key} is missing from the report")
            continue
        path = dataset / entry["file"]
        if path.is_symlink() or not path.resolve().is_relative_to(dataset.resolve()):
            problems.append(f'{key} escapes artifact')
            continue
        if not path.is_file():
            problems.append(f"{entry['file']} is missing from {dataset}")
            continue
        digest = sha256_of(path)
        if digest != entry["sha256"]:
            problems.append(f"{entry['file']} has changed since qualification "
                            f"({digest[:12]} != {entry['sha256'][:12]})")
    if problems:
        print("ACCEPT FAILED: the artifact does not match the qualified version")
        for problem in problems:
            print(" -", problem)
        return 1
    print(f"ACCEPTED: {dataset} matches the qualified artifact "
          f"{recorded['artifact_version']['digest'][:16]}")
    return 0


SELFTEST_EXPECTATIONS = {
    "broken": {
        "expected_ok": False,
        "must_fail": {
            "leakage.answer_source_absent",
            "leakage.no_single_column_determines_the_target",
            "timing.no_post_hoc_feature",
            "units.validation_receipt",
            "runner.layout",
        },
    },
    "broken-overlap": {
        "expected_ok": False,
        "must_fail": {
            "splits.actual_ranges_disjoint",
            "splits.rows_within_declared_windows",
        },
    },
    "broken-single-class": {
        "expected_ok": False,
        "must_fail": {
            "splits.eval_and_holdout_have_both_classes",
        },
    },
    "broken-duplicates": {
        "expected_ok": False,
        "must_fail": {
            # The count is not declared, so a repeat is refused rather than tolerated.
            "splits.duplicate_entities_within_splits",
        },
    },
    "broken-duplicate-labels": {
        "expected_ok": False,
        "must_fail": {
            # The descriptor claims a unique entity key, so contradictory labels are noise.
            "splits.repeated_identifiers_carry_one_label",
        },
    },
    # The same rows without that claim, which is the real Austin case: measured and accepted.
    "documented-repeats": {"expected_ok": True, "must_fail": set()},
    "corrected": {"expected_ok": True, "must_fail": set()},
}


def selftest(fixtures_root: Path) -> int:
    failures = []
    for name, expectation in SELFTEST_EXPECTATIONS.items():
        dataset = fixtures_root / name
        if not dataset.is_dir():
            failures.append(f"{name}: fixture directory missing at {dataset}")
            continue
        with tempfile.TemporaryDirectory(prefix='qualification-selftest-') as tmp:
            copied = Path(tmp)/name
            shutil.copytree(dataset,copied)
            report = qualify(copied, record_battery=True)
        failed_checks = {result["check"] for result in report.results if not result["ok"]}
        print(f"--- fixture {name}: ok={report.ok} (expected {expectation['expected_ok']})")
        print(report.render())
        if report.ok != expectation["expected_ok"]:
            failures.append(f"{name}: checker said ok={report.ok}, expected "
                            f"{expectation['expected_ok']}")
        unfulfilled = expectation["must_fail"] - failed_checks
        if unfulfilled:
            failures.append(f"{name}: expected these checks to fail but they passed: "
                            f"{sorted(unfulfilled)}")
        print()
    if failures:
        print("SELFTEST FAILED")
        for failure in failures:
            print(" -", failure)
        return 1
    print("SELFTEST PASSED: every broken fixture is rejected with the expected checks and "
          "the corrected fixture is accepted.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Qualify a constructed dataset")
    parser.add_argument("dataset", nargs="?", help="dataset directory to check")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--report", help="write the JSON report to this path")
    parser.add_argument('--record-battery', action='store_true',
                        help='construction only: write computed aggregate results into quality.json')
    parser.add_argument('--battery-output', type=Path,
                        help='fresh directory outside the dataset for full audit diagnostics')
    parser.add_argument("--accept", metavar="REPORT",
                        help="verify a report and its artifact hashes against the dataset")
    parser.add_argument("--selftest", action="store_true",
                        help="check every shipped fixture pair")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest(Path(__file__).resolve().parent.parent / "fixtures")
    if not args.dataset:
        parser.error("a dataset directory is required unless --selftest is used")

    if args.accept:
        if args.record_battery or args.battery_output:
            parser.error('--accept cannot modify or recompute the artifact')
        return accept(Path(args.accept), Path(args.dataset))

    if args.report and Path(args.report).resolve().is_relative_to(Path(args.dataset).resolve()):
        parser.error('--report must be outside the artifact to avoid self-referential hashes')
    report = qualify(Path(args.dataset), record_battery=args.record_battery,
                     battery_output=args.battery_output)
    payload = report.as_dict()
    if args.report:
        Path(args.report).write_text(json.dumps(payload, indent=1) + "\n")
    if args.json:
        print(json.dumps(payload, indent=1))
    else:
        print(report.render())
        print()
        digest = (report.artifact.get("artifact_version") or {}).get("digest")
        print(f"artifact version: {digest[:16] if digest else 'not computed'}")
        print("QUALIFICATION PASSED" if report.ok else "QUALIFICATION FAILED")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
