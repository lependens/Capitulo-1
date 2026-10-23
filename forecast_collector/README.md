# Raw Forecast Capturer v0.1

Batch-only Python component for preserving SIAR forecast responses as immutable, append-only raw captures. It does not assign forecast horizons, normalize ET₀, deduplicate captures, use DuckDB, or depend on SIAR Sync.

## State and deployment boundary

v0.1 entered `main` through PR #8. PR #12 added the validated shared-group storage permissions. PR #13 adapts the client to the demonstrated SIAR E2E contract. The implementation includes JSON Schema validation, the secret guard, safe ZIP handling, SHA-256 checksums, `complete` / `partial` / `failed` states, append-only captures, and atomic publication. `source.source_issue_at` is `null`.

The SIAR E2E contract has been manually demonstrated for IB04 — Son Ferriol, Mallorca, Aguacate: `/inicio` → AJAX `/validarForm` → normal form POST `/calculo` → same-session GET `/exportCSV`. R34 demonstrated `C1 + G1 + G2` as a sufficient payload for the tested IB04/Aguacate `/calculo` scenario. The current example configuration conservatively retains the additional G3 fields (`sEstacion`, `chkComarca`, `comarcaSelect`); their presence is not evidence that they are required. `fechaInicial` and `fechaFinal` are intentionally absent. SIAR returned a ZIP containing one CSV with seven forecast rows and columns `Fecha`, `Kc`, `ET0`, `ETc`, `Pe`, and `ETc - Pe`; the HTML result and CSV agreed on dates and forecast values.

### Permanent runtime and backup

The permanent runtime is validated as `siar-forecast`, with functional release `e850827e24cff11ae46f51c9bc3e8a2f36b0d2da` and Python 3.14.4. `/opt/siar-forecast` and `/etc/siar-forecast` are operational; `data_root = /srv/siar-forecast/raw` is provisioned and validated.

`SMOKE-FC-001 = PASS` demonstrated a complete IB04/Aguacate run and capture, verified TLS, hashes/schemas, shared-group permissions and safe omission of sensitive HTML. `BACKUP-FC-001 = PASS` demonstrated Google Drive through rclone crypt and an independent restore with hashes/schemas validated. Backup uses `rclone copy`, not `sync`; raw remains outside Git. Recovery after total server loss still requires separately recoverable crypt material and is not implied by the backup test alone.

As of the latest operational report supplied by direction (documented 2026-10-23), timers were paused after the storage incident. Reactivation is independent of SIAR Sync, which remains HOLD. One demonstrated real automatic cycle establishes SIAR-FORECAST-CAPTURE = RUNNING; at least two automatic cycles establish SIAR-FORECAST-PILOT = RUNNING. These gates are pending evidence, not inferred from timer configuration. Do not reconstruct emissions missed during the pause. Every relevant new raw capture requires external backup. Automated tests remain synthetic and do not contact SIAR. See the [consolidated project status](../docs/PROJECT_STATUS.md) for current gates.

FC-STATIONS-001 passed; the multi-smoke attempted nine ACTIVE stations, with eight complete and IB08 failed. Operational targets are IB01, IB02, IB03, IB04, IB05, IB06, IB10 and IB11. IB08 remains officially ACTIVE / TEMPORARILY_UNAVAILABLE_EXTERNAL, monitored separately without blocking those eight; the exact internal SIAR/MAPA cause is not established.

The example station and form values are configuration examples only. The SIAR E2E form contract was demonstrated for IB04; verify station configuration before a future capture. No secrets belong in the TOML file.

## Install and one-shot execution

Python 3.11+; runtime dependencies are `requests` and `beautifulsoup4`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
cp config/config.example.toml config/config.toml
siar-forecast-raw capture --config config/config.toml
```

The command performs one run and exits. Scheduling is deliberately external and not included. The approved experimental cadence is about 08:00, 14:00 and 20:00 Europe/Madrid, subject to the external scheduler's reactivation and cycle validation.

## Storage layout

The configured `data_root` is outside Git. Runs are partitioned using `started_at_utc`:

```text
<data_root>/runs/YYYY/MM/DD/<run_id>/
  run.json
  SHA256SUMS
  <station_code>/<capture_id>/
    metadata.json
    result.html       # optional: present only when the secret guard accepts it
    forecast.zip      # original ZIP bytes
    forecast.csv      # single CSV member, copied byte-for-byte in memory
    SHA256SUMS
```

Incomplete station outcomes are also published with metadata. If HTML contains reusable session/auth values, it remains in RAM only, is not persisted, and the flow continues to `/exportCSV`. Metadata records `html_persistence.status = omitted_sensitive` plus sanitized finding categories only. New captures always emit `html_persistence`; historical `siar-raw-capture-v1` metadata created before this policy may legitimately omit that field. A capture may still be `complete` when the validated original ZIP and its byte-for-byte CSV member are preserved even though sensitive HTML was deliberately omitted. If export then fails, normal partial/failed rules apply; HTML omission alone never upgrades the outcome. A station failure does not prevent attempts for other configured stations.

Every capture is first assembled in a sibling temporary directory and exposed by same-filesystem rename. A whole run is staged under `<data_root>/tmp/` and published by rename. Existing run/capture IDs raise an error; nothing is overwritten or deduplicated.

Cookies, session identifiers, CSRF values, request bodies, and full HTTP headers are never written to metadata. They remain in the in-memory requests session. Only method, URL, timings, status, content type/length, selected Content-Disposition, and safe error summaries are recorded.

## Raw recovery and integrity

No database or collector package is needed to read a capture. Read `run.json` and `metadata.json` as UTF-8 JSON; the raw files are ordinary bytes. Verify the run-level and capture-level SHA-256 records with:

```python
from pathlib import Path
from siar_forecast_raw.hashing import verify_checksums

failures = verify_checksums(Path("<run-or-capture-directory>"))
```

An empty list means every listed file matches. `source_issue_at` remains JSON `null` until SIAR exposes an actual emission timestamp.

## ZIP handling and secret guard

ZIP members are read directly from memory, without `extractall()` or filesystem path use. Exactly one bounded `.csv` member is accepted; absolute paths, drive paths, traversal segments, multiple files, and non-CSV content fail closed.

The HTML guard checks values in sensitive form fields and credential assignments, cookie headers, bearer/JWT forms, opaque token contexts, and exact session values observed in memory. Empty CSRF markers and ordinary page IDs/static asset hashes are allowed. When the guard finds reusable sensitive material, only sanitized finding categories are recorded; the HTML is kept in RAM, is neither redacted nor persisted, and export may continue. Safe HTML is persisted byte-for-byte.

## Synthetic examples

`examples/run.json` and `examples/metadata.json` contain fabricated values and no SIAR response material. Test fixtures are synthetic; no live or historical response is stored in the repository.

## Verification

```sh
python -m pytest
```
