# Raw Forecast Capturer v0.1

Batch-only Python component for preserving SIAR forecast responses as immutable, append-only raw captures. It does not assign forecast horizons, normalize ET₀, deduplicate captures, use DuckDB, or depend on SIAR Sync.

## State and deployment boundary

v0.1 was merged into `main` through PR #8. Synthetic validation is complete; the pre-merge technical gate passed **29 tests**. The implementation includes JSON Schema validation, the secret guard, safe ZIP handling, SHA-256 checksums, `complete` / `partial` / `failed` states, append-only captures, and atomic publication. `source.source_issue_at` is `null`.

The example form parameters are not yet the proven minimal live contract. The current SIAR form contract must be revalidated before the IB04 live test; additional serialized province/community fields may prove required.

It has not been live-tested against SIAR and is not deployed. The next gate is a runbook plus a manual live test with IB04. Permanent storage under `/srv` and off-site backup remain pending coordination with 02. Do not set up a scheduler or deploy before those gates are approved.

The example station and form values are configuration examples only. Review and verify the SIAR form contract and station IDs before any live capture. No secrets belong in the TOML file.

## Install and one-shot execution

Python 3.11+; runtime dependencies are `requests` and `beautifulsoup4`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
cp config/config.example.toml config/config.toml
siar-forecast-raw capture --config config/config.toml
```

The command performs one run and exits. Scheduling is deliberately external and not included. The approved experimental cadence is about 08:00, 14:00 and 20:00 Europe/Madrid, configurable by whichever scheduler may later be approved.

## Storage layout

The configured `data_root` is outside Git. Runs are partitioned using `started_at_utc`:

```text
<data_root>/runs/YYYY/MM/DD/<run_id>/
  run.json
  SHA256SUMS
  <station_code>/<capture_id>/
    metadata.json
    result.html       # only after the secret guard accepts it
    forecast.zip      # original ZIP bytes
    forecast.csv      # single CSV member, copied byte-for-byte in memory
    SHA256SUMS
```

Incomplete station outcomes are also published with metadata. If HTML contains reusable session/auth values, it is not persisted. If a safe HTML response exists but ZIP export fails, that is a `partial` capture. If no useful artifact exists, the station capture is `failed`. A station failure does not prevent attempts for other configured stations.

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

The HTML guard checks values in sensitive form fields and credential assignments, cookie headers, bearer/JWT forms, opaque token contexts, and exact session values observed in memory. Empty CSRF markers and ordinary page IDs/static asset hashes are allowed. A finding is reported in error metadata; the HTML is not silently redacted or persisted.

## Synthetic examples

`examples/run.json` and `examples/metadata.json` contain fabricated values and no SIAR response material. Test fixtures are synthetic; no live or historical response is stored in the repository.

## Verification

```sh
python -m pytest
```
