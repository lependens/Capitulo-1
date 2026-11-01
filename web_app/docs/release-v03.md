# WEB-V0.3 Scientific Explorer — operational closeout

Verified on 2026-11-01. **WEB-V0.3-SCIENTIFIC-EXPLORER = PASS.**

## Deployment and review

- LAN: <http://192.168.1.43:8088/>; version 0.3.0.
- Implementation PRs: [57](https://github.com/lependens/Capitulo-1/pull/57), [58](https://github.com/lependens/Capitulo-1/pull/58), [59](https://github.com/lependens/Capitulo-1/pull/59).
- Deployed application merge SHA: `c44d1187760e19ed67269c0dbc05d2c5b0255ad1`.
- Final implementation CI: [41305975804](https://github.com/lependens/Capitulo-1/actions/runs/41305975804), backend/frontend PASS.
- Full local acceptance: 49 backend tests, four frontend tests, lint, build, applicable links and diff check PASS.
- Production API: 21 successful requests; maximum observed request 0.125 s on this LAN check, not a load benchmark. Twelve metric rows (global, IB11, 2023 and IB04/2023; champion, AHC SELECTED and linear model) matched source CSV exactly for n, MAE, RMSE, bias, R² and cohort SHA. Negative prediction retained exactly; 1,007 published ML negative flags, champion zero.
- Both deployed containers healthy, root filesystems read-only. Backend data mount `RW=false`; only frontend bound to `192.168.1.43:8088`. No Sync, Forecast or AEMET changes.
- Desktop and responsive mobile (390×844 CSS px) PASS; tablet layout checked. Station ranking, IB11 n=1,822 / RMSE 0.081764, filtering, model cards, wind-input search, comparison, residuals, provenance and legacy AHC coefficients/temporal/seasonal views checked in Chrome.

## Published scope and semantics

43 model identities: PM calculated reference, three original deterministic baselines, four AHC variants and 35 ML configurations (seven algorithms × five feature sets). Twelve station identities remain selectable; nine primary stations have ML TEST.

Splits: ALL / TRAIN / VALIDATION / TEST. AHC TRAIN is fitting and VALIDATION selection; TEST is retrospective_locked. ML selection is frozen DEVELOPMENT 2004–2020 rolling-origin; its TEST 2021–2025 is retrospective_recovery_evaluation. No ML TRAIN/VALIDATION metrics are invented.

ML/recovery cohorts: common_all / common_ml_35 / own. AHC additionally exposes published common_7 / common_selected. Compare only matching split, group and cohort SHA. PM is a computed reference, not independently measured ET₀. TEST sorting never changes the DEVELOPMENT champion.

Summary, model/input catalog, station analysis, comparison, temporal/scatter/residual/distribution views, feature-set/algorithm comparison, information cards, scientific glossary and audit/provenance are available. The [contract and limitations](explorer.md) remain authoritative for API behavior.

## Derived bytes and 02B preservation

| Identity | Verified value |
|---|---|
| Artifact ID | `web-derived-v03-explorer-20261101-v1` |
| Files / bytes | 19 / 27,205,051 |
| Derived prediction Parquet SHA256 | `922e3df2a716a4d8457e08183bc3c55a67801c9c653beb9458d327dc72f9d203` |
| Package SHA256 | `9975441ad2f625e4a5016b01b99f7909558d5607b3e85637b2f78b50632f009a` |
| info.json SHA256 | `9893055e97d75684a311f7270ee57d90dd9b7e934d01efb0679e0c90c4da3b94` |
| Batch | `science-20261101T120323Z-e58f9d30` |
| Signed inventory SHA256 | `ec52d0090efc7bd3adab7bdbf930044a94d5c5ced5280afbfe8337f15c133cca` |
| Ingestion receipt | `/srv/siar-project/ingestion-receipts/science-20261101T120323Z-e58f9d30_3c9a0bc6.json` |
| Ingestion status | `published-verified` |
| Preservation snapshot | `backup_20261101T123421Z_8022d3fd` |
| Snapshot manifest SHA256 | `cc0e912100c5eb3c502fe23bc79f75a80bfb8ef84356aa3ba6780d329847c320` |
| Remote verification receipt | `/srv/siar-project/preservation/verified/backup_20261101T123421Z_8022d3fd.json` |
| Remote destination | `forecast-offsite-crypt:SIAR-project/snapshots/backup_20261101T123421Z_8022d3fd` |
| Final preservation state | **REMOTE_VERIFIED**, logged 2026-11-01 12:35:00 UTC |

Package SHA256 hashes the UTF-8 concatenation of sorted relative filename + NUL + file SHA256 + newline. Windows and Linux derivatives matched all 19 file hashes. The signed READY flow verified ingestion; its ACK identifies all 19 files. Those same 19 path/hash pairs match the snapshot manifest; COMPLETE and the verified receipt match its SHA256. The snapshot also includes an ingestion inventory entry. This is the new release's preservation evidence, independent of previous releases.

The deployment reads `/home/josep/siar-et0-web-data/releases/web-v0.3-explorer-20261101/data-verified`. WEB recalculates science = NO; scientific artifacts, Dataset and raw modified = NO. Conversion only represents already published tables and joins verified station/date identities.

## Rollback and limits

Previous V0.2 environment and running images are retained in `/home/josep/siar-et0-web-rollback/web-v02-20261101-b`. Old deployment SHA: `3a0acc8be244a4c9aebad52b6ff83ac7fe8ae823`. Backend/frontend rollback tags end in `rollback-web-v02-20261101-b`. The backend fallback captures the running container root filesystem when its original image is unavailable; mounted data is excluded. Saved image identities were verified before switching.

```bash
python3 ~/siar-et0-web/web_app/ops/release.py rollback \
  --repo ~/siar-et0-web \
  --record ~/siar-et0-web-rollback/web-v02-20261101-b
```

This restores the saved environment and images without rebuilding; previous data is retained. Rollback readiness was checked; an actual rollback was unnecessary after successful deployment.

Known limits: station-month/season and ML training metrics are unavailable; charts may sample predictions, while accuracy metrics always use official full populations. AHC_CLIMATOLOGY is operational coefficients only, without TEST performance. IB07 has no TEST, IB09 fails Quality and IB101 is short/descriptive. Wind height follows the SIAR convention; input/station comparisons establish no causality. Next science step: authorize a prospective validation protocol and fresh data independently of this retrospective TEST, preserving the frozen champion.
