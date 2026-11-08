---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

# Will a public water system's first UCMR 5 entry-point sample contain PFOA or PFOS at or above the reporting level?

For a U.S. public water system subject to UCMR 5, will its first entry-point sample contain PFOA or PFOS at or above the UCMR 5 minimum reporting level?

Derived from U.S. Environmental Protection Agency (EPA). This is not its official product.

Prediction time: Immediately before the first UCMR 5 entry-point sample is collected, recorded as CollectionDate; current-sample result fields are excluded. System attributes and prior UCMR 3 results are treated as pre-sample. Self-reported additional-data elements are attached only when the additional-data row matches the exact available system, facility, sample point, and sample event key, or the exact system, facility, and sample point key when no event code exists; unmatched rows remain missing and no system-level or cross-event fallback is used. The additional-data responses are undated self-reports, so their availability at the decision instant is assumed, not verified.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.7856421356421357,
    "brier": 0.06860848158758656
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.0766306076535297
  },
  "calendar_identity": {
    "auc": 0.5,
    "brier": 0.0766306076535297
  },
  "persistence": {
    "auc": 0.4525491493644361,
    "brier": 0.08101381885983604
  }
}
```

## Limitations

- UCMR 5 is occurrence monitoring at entry points, not compliance monitoring; a result at or above the MRL is not a violation or a determination that the system exceeds an MCL.
- The first sample is a single snapshot at one entry point; systems with multiple facilities or sample points can have heterogeneous PFOA/PFOS results, so the label is sample-level, not a system-wide contamination determination.
- The final UCMR 5 dataset was released in 2026; system size is as of February 1, 2021, prior UCMR 3 files were revised in 2024, and self-reported additional data elements may have been updated after sampling. This is current-vintage retrospective research, not proof of historical operational availability.
- Results below the UCMR 5 MRL are censored (null), so the label captures detection at the reporting threshold, not concentration magnitude or low-level exposure.
- First-sample collection dates are determined by monitoring schedules rather than random assignment; chronological splits can reflect rollout differences by system size, state, and laboratory capacity.
- UCMR 5 covered community water systems and non-transient non-community water systems, not transient non-community systems, and small-system participation was a nationally representative sample rather than a census.
- EPA's public-domain designation is a U.S. domestic basis; reusers outside the United States must assess local rights. Any CC0 release of the derived table applies only to that contribution and does not alter EPA's upstream designation.
- This is a predictive association task, not a causal analysis of source-water contamination or treatment effectiveness.
- The final UCMR 5 file includes samples collected through mid-2026, beyond the January 2023 through December 2025 monitoring-cycle text on the occurrence page; the frozen holdout window follows the observed data through 2026-06-30.
- UCMR 5 is occurrence monitoring at entry points, not compliance monitoring; a result at or above the minimum reporting level is not a violation or an MCL determination.
- The label is sample-level at one entry point; systems with multiple facilities or sample points can have heterogeneous PFOA or PFOS results.
- The final UCMR 5 dataset was released in 2026; system size reflects a 2021 vintage, UCMR 3 files were revised, and self-reported additional data elements may have been edited after sampling. This is current-vintage retrospective research, not proof of historical operational availability.
- Results below the UCMR 5 minimum reporting level are censored, so the label captures detection at the reporting threshold, not concentration magnitude.
- First-sample collection dates follow monitoring schedules rather than random assignment, so chronological splits can reflect rollout differences by system size, state, and laboratory capacity. First samples whose dates fall in the December embargo or outside the windows are excluded by the fixed first-sample selection and are counted in a source check, so a system whose earliest sample falls there is not represented even if a later in-window sample exists.
- Samples with a blank CollectionDate are excluded and counted before first-sample selection; they are not assigned to any split.
- UCMR 5 covered community water systems and non-transient non-community water systems, not transient non-community systems, and small-system participation was a representative sample rather than a census.
- Additional data element concepts are stored as rows in the AdditionalDataElement field rather than as header columns. The builder filters documented values and attaches Response values only through the exact available system, facility, sample point, and sample event key, or the exact system, facility, and sample point key when no event code exists. It never falls back to a system-only join. Unmatched rows remain missing. The additional-data responses are undated self-reports, so pre-sample availability is assumed rather than proven.
- For UCMR 3 systems with only non-detect PFOA or PFOS rows, the prior maximum uses the maximum reported MRL as a sourced censoring bound, not an observed concentration and not zero. The prior maximum is missing only when the system has no UCMR 3 PFOA or PFOS row.
- The occurrence text file carries a non-UTF-8 micro sign byte in the concentration unit text; the builder decodes publisher text with the Windows-1252 codec and checks UCMR 5 and UCMR 3 units.
- The label check uses an independent csv-module pass over the occurrence text and compares the written labels, source signs, source values, units, and MRLs; it does not rely on the in-memory pandas label expression.
- collection_year is constant within the train and eval windows and is a split tag, not a transferable feature.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
