# WEB-V0.3 — Scientific Explorer

## Scientific semantics

PM_FAO56_V1 is the primary calculated reference, not independently measured ET₀. EtPMon is a distinct historical calculated SIAR reference in legacy views. Errors measure concordance. R² is predictive 1−SSE/SST; bias is prediction−PM; MAE/RMSE/bias use mm/day. Coverage is availability, not accuracy. No synthetic score or clipping is applied.

The DEVELOPMENT-selected ML champion is read from the frozen selection. TEST-best is separately labeled descriptive. ML TEST (2021–2025) is `retrospective_recovery_evaluation`, including prior technical access, never a pristine holdout. AHC retains TRAIN adjustment (2004–2015), VALIDATION selection (2016–2020), TEST `retrospective_locked`. Annual and selected seasonal HG-Rs retain separate IDs. AHC_CLIMATOLOGY appears only in the coefficient view, without TEST performance. ANN is NOT STARTED.

[Canonical science](../../scientific_pipeline/README.md), [AHC visualization contract](../../scientific_pipeline/docs/visualization_contract.md), [ML rolling contract](../../scientific_pipeline/docs/ml_validation_contract_refinement.md), [recovery incident](../../scientific_pipeline/docs/ml-v1-test-recovery-access-incident.md).

The current narrative explanation of models, inputs and accepted results is [1.2.6 — ML clásico](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md), incorporated from main after PR #56.

## Verified data contract

`convert_explorer` reads an approved WEB-V0.2 package, official recovery CSVs and frozen `selection.json`. It imports no scientific implementation or serialized estimator. Recovery manifest SHA: `05834edaebf3f990729ebb8b15ecd7511bb96e006bfab69f11408a250f432aab`; selection SHA: `a41ee87956368d6c0e1fae2b39177ffe8c42cae7658477172c46725409f829c7`. Each source output hash and row count is checked. Protocol file SHA and canonical logical protocol SHA are distinct and both retained.

```bash
python -m web_app.backend.convert_explorer \
  --base /path/web-derived-v02-ahc-20261031 \
  --source /path/ml-v1-test-recovery-20261101T090828Z \
  --selection /path/selection.json \
  --output /path/NEW-web-derived-v03-explorer
```

The destination must be new. Base and official artifacts remain unchanged. Metrics are copied to Parquet without rounding. Recovery CSV predictions omit baseline columns; baseline values are joined by immutable station/date keys from the verified AHC derivative. Joined common-all and common-ML key hashes must exactly equal published scientific hashes before conversion succeeds. This verifies a representation of existing cohorts; it does not define a new cohort. Own cohorts follow the published finite prediction/reference rule.

Generated JSON metadata uses explicit UTF-8 / CRLF on every host, matching the finalized Windows preservation package byte for byte on Linux. Parquet source values and metric precision are unchanged. Verify the complete file inventory, not only the prediction Parquet hash, when matching deployment to preservation.

`explorer_models.json` preserves exact feature arrays, candidate IDs and frozen parameters. Negative counts count the producer's published negative-prediction flags. No model bundle is deployed. Audit records the source artifact/run, dataset, protocol, pipeline, manifest, selection and bundle hashes. Scaler diagnostics and local source paths are omitted. Startup validates every derived file hash. Web representations do not become scientific authorities.

## Read-only API

Existing V0.1/V0.2 endpoints remain available.

| Route | Content |
|---|---|
| GET /api/explorer/catalog | PM reference + 7 empirical/AHC + 35 ML identities, 12 station identities, full source audit and DEVELOPMENT selection |
| GET /api/explorer/metrics | Verbatim published metrics filtered by models, station, split, cohort, group and published year |
| GET /api/explorer/predictions | At most 3,000 preserved prediction rows, 1–8 models, deterministic visual sampling |

Metric parameters: `models` comma-separated, `station` optional, `split=TEST|TRAIN|VALIDATION|ALL`, `cohort=own|common_all|common_ml_35|common_7|common_selected`, `group=global|station|year|month|season|season_year`, `year` optional. Unsupported combinations return unavailable, never averaged RMSE/R². ALL enumerates split rows without a combined metric. ML TRAIN/VALIDATION metrics do not exist: DEVELOPMENT uses frozen rolling selection/folds in catalog/audit. AHC TRAIN/VALIDATION use their own published cohorts.

The recovery TEST metrics already include baseline/AHC comparators on common_all. `comparable=true` requires the same split, group key and cohort SHA for all returned rows. Different station/year groups have different populations. Source run/table, period and cohort hashes accompany each metric row.

Availability denominators count preserved source rows for global/station/year/month scopes; coverage is n/source_rows. Unsupported denominator scopes explicitly show unavailable coverage. This presentation ratio does not change producer metrics.

Prediction `from`/`to` filters affect graphs only. Official metrics use only published group/year filters. Responses give population size, returned sample size, sampling rule and source manifest. Residual subtraction and fixed-width histogram bins operate on visual samples and are labeled presentation, not newly published scientific metrics. Negative predictions and zero targets remain. Time/residual line views require a station; global scatter/distribution do not join unrelated stations with time lines.

## UX and audit

Summary shows DEVELOPMENT champion, selected AHC benchmark and ten ranking rows, expandable. Full catalog searches category, ID and variables. Comparison displays up to six models. HGB feature sets and FULL algorithms use selected cohort metrics, never hardcoded scores. Stations rank published RMSE and link to station analysis. Year, climatological month, climatological season and season/year groups appear where published; station-month/season are explicitly unavailable.

Each result exposes source, period and cohort audit; the provenance page exposes portable identities and frozen selection. Model cards preserve exact inputs and selection semantics. The glossary distinguishes computed references, metrics, availability and retrospective semantics. Legends use labels and patterns alongside color. Tables scroll within their containers; context remains explicit.

## Validation

Run synthetic API tests and frontend lint/test/build. Official acceptance is opt-in because full scientific payloads are not in Git:

```bash
EXPLORER_DATA_DIR=/path/verified-web-v03 \
EXPLORER_ML_SOURCE=/path/official-ml-recovery \
python -m pytest web_app/tests -q
```

Acceptance compares n, MAE, RMSE, bias, R² and cohort SHA directly to source CSV for global TEST, IB11, 2023, IB04/2023, champion, AHC SELECTED and a linear model. It checks champion candidate/parameters, all negative flags and an exact negative prediction without clipping. Existing APIs/integrity remain covered. Windows async checks need loopback access for Python's event-loop socket pair.

## Deployment, rollback and preservation

Keep `siar-et0-web` Compose project, LAN bind 192.168.1.43:8088, data read-only, root filesystems read-only, no-new-privileges and healthchecks. Sync/Forecast/AEMET remain isolated. Before replacement, tag running V0.2 backend/frontend image IDs with a unique rollback tag, save the prior WEB_DATA_DIR/Compose environment privately and record old Git SHA. Retain old images and derivative.

Build new images, generate a new derivative, switch WEB_DATA_DIR and recreate web services. Verify health, source integrity and UI acceptance. Rollback restores saved environment and uses saved previous images via Compose override; do not rebuild old code against newer dependencies. Never remove prior data for rollback.

Preserve the completed derivative through 02B explicit finalization, signed READY, transfer, ingestion and remote verification. Parquet outputs are web representations. Receipt and REMOTE_VERIFIED are required before declaring preservation PASS.

## Known limitations

- Nine primary stations have ML TEST; all 12 preserved identities remain selectable, including IB07 without TEST, IB09 Quality FAIL and short IB101.
- Recovery has no station-month/season or ML training metrics. These combinations remain unavailable.
- DEVELOPMENT fold statistics describe selection; sorting TEST results does not select models.
- Histograms/scatter are visual samples; metrics always come from full official populations.
- Units/provenance remain partial; wind at 2 m follows SIAR convention. No station error plot demonstrates causal explanation.
