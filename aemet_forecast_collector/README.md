# AEMET Raw Municipal Forecast Collector v0.1

Standalone, one-shot Python collector for preserving AEMET's municipal daily forecast responses. It keeps the wrapper and the response fetched from its `datos` URL as separate raw byte artifacts. It does not calculate ET₀, normalize weather variables, assign D+n labels, schedule captures, or depend on the SIAR collector.

## Product and demonstrated contract

The collector uses the authenticated AEMET OpenData endpoint:

```text
GET https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/{municipality_id}
Header: api_key: supplied from AEMET_API_KEY at runtime
```

The response wrapper is saved as `wrapper.raw` and parsed only after its original bytes have been retained. Its `datos` URL is fetched without the API key; that response is saved separately as `payload.raw`. The data URL is treated as temporary and is never constructed or reused across captures.

The live A1 check for municipality `07040` returned a JSON wrapper and a seven-day JSON payload. The payload arrived with `Content-Type: text/plain;charset=ISO-8859-15`; the parser follows the declared charset and recognizes JSON independently of MIME type. Variables observed in the payload were `temperatura`, `humedadRelativa`, `viento`, `rachaMax`, `probPrecipitacion`, `estadoCielo`, `sensTermica`, `cotaNieveProv`, and `uvMax`. Presence can vary by forecast day. Empty strings, JSON nulls, numeric zeros, and absent fields remain distinct; raw values are not converted or normalized.

The observed payload municipality `id` is `7040` (JSON integer), while the canonical configured identifier is `"07040"`. PR #19 canonicalizes the payload identifier only for comparison with the configured five-digit string. It does not change `payload.raw`, reserialize JSON, or rewrite the original value. Canonical comparison is not normalization of the raw artifact.

The `elaborado` value is stored literally as `source.issue_time_raw`; `issue_timezone` remains JSON `null`. Each `dia.fecha` and each `periodo` value is stored literally. There are no generated `D+n`, `forecast_horizon`, or UTC issue-time fields.

The municipality forecast represents the municipal capital. `spatial_match = "municipal_capital"` does not mean a station-exact forecast. This is relevant to the initial IB04 → Son Ferriol → Palma mapping.

### Demonstrated and not demonstrated

**Demonstrated by A1:** wrapper → `datos` → payload flow; runtime `api_key` header; separate byte-preservable bodies; ISO-8859-15 decoding; literal `elaborado`, dates and periods; seven forecast days; the variable field names above; municipal-capital spatial meaning.

**Not demonstrated:** recovery of historical forecast emissions, the exact update cadence for each municipality, timezone semantics for `elaborado` or `fecha`, D+n semantics, station-exact forecasts, or a complete FAO-56 Penman-Monteith input set. No ET₀ calculation is performed.

### Implementation and operational boundary

PR #18 merged the Track A Collector, PR #19 merged municipality-id comparison, and PR #20 merged POSIX ancestor permissions. `AEMET-A1 = PASS`, `AEMET-A2 = PASS`, and `AEMET-A3 = COMPLETE`. Final E2E confirmation in the target runtime is still pending after an operational timeout. That timeout is not evidence of a Collector defect and does not revoke those completed milestones; no final E2E PASS or periodic deployment is claimed. The next gates are final E2E confirmation, prospective deployment and backup. Track B, investigation of the exact AEMET feed used by SIAR, remains OPEN.

See the [consolidated project status](../docs/PROJECT_STATUS.md), [milestones](../docs/MILESTONES.md), [AEMET OpenData API documentation](https://opendata.aemet.es/dist/) and AEMET's [municipal forecast interpretation](https://www.aemet.es/es/eltiempo/prediccion/municipios/horas/abertura-id10002/ayuda).

## Configuration and one-shot execution

Python 3.11 or newer. Runtime code uses only the Python standard library.

```sh
cd aemet_forecast_collector
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
cp config/config.example.toml config/config.toml
export AEMET_API_KEY='provided-out-of-band-at-runtime'
aemet-forecast-raw capture --config config/config.toml
```

`AEMET_API_KEY` is read from the process environment only. In an interactive Linux shell, read it without echoing it or placing it in shell history:

```sh
read -rsp "AEMET API key: " AEMET_API_KEY
echo
export AEMET_API_KEY
```

It is never written to configuration, metadata, raw files, or error messages. The program does not print request headers. HTTP 401/403 responses are summarized safely; renew or replace an expired key out of band.

Optionally set `AEMET_API_KEY_EXPIRES_AT` in the runtime environment as a timezone-aware ISO-8601 timestamp (for example, `2027-06-01T00:00:00Z`). If absent, expiry is unknown and capture is allowed. A malformed value fails closed; an expired value prevents requests and reports that key renewal is required. The expiry value is not stored, and renewal is manual.

The example config contains the initial target:

```text
station_code = IB04
station_name = Son Ferriol
municipality_id = "07040"
municipality_name = Palma
spatial_match = municipal_capital
```

Each command executes one batch and exits. It makes one authenticated wrapper GET per target and, when the wrapper is valid, one unauthenticated GET to the returned `datos` URL. No retry loop or scheduler is included.

## Raw storage and append-only behavior

`data_root` is configured outside the repository. Each execution receives new run and capture IDs, even when the issue timestamp and payload checksum repeat. Changed payloads for the same `issue_time_raw` are also retained.

```text
<data_root>/runs/YYYY/MM/DD/<run_id>/
  run.json
  SHA256SUMS
  <station_code>/<capture_id>/
    metadata.json
    wrapper.raw       # when a safe successful wrapper entity was received
    payload.raw       # when a safe payload response entity was received
    SHA256SUMS
```

The collector buffers the response body as bytes, checks it for the runtime credential, and writes those bytes without JSON reserialization or UTF-8 conversion. It calculates independent SHA-256 digests for the wrapper and payload. Parsing uses the declared HTTP charset and happens after the raw bytes have been captured. A `text/plain` MIME type does not prevent JSON parsing.

The complete run is assembled in a temporary sibling directory, manifests are written and verified, and the run is published with one same-filesystem rename. Existing run paths are never overwritten. On POSIX systems PR #20 enforces the managed ancestor chain as well as the published tree:

| Managed path/type | Mode | SGID | Other permissions |
|---|---|---|---|
| `data_root`, `data_root/runs`, `YYYY`, `MM`, `DD` | `2770` | YES | `0` |
| All directories in the published run/capture tree | `2770` | YES | `0` |
| Files in the published tree | `0660` | Not applicable | `0` |

Thus the complete `data_root/runs/YYYY/MM/DD` chain and the published directories share the same group policy, rather than only the final run directory. The parent of `data_root` must already exist as a normal directory; the package does not provision permanent infrastructure or change directories outside the managed chain. This is the POSIX policy, not a claim about Windows permissions or completion of the final runtime E2E test.

Capture states are:

- `complete`: safe wrapper and payload raw files exist, both parse, and the expected forecast structure and metadata validate.
- `partial`: at least one safe raw response was preserved, but a later request or parse step failed.
- `failed`: no useful raw response was preserved, including missing credentials and wrapper authentication failures.

The JSON Schemas are in `schemas/`. Tests use synthetic responses only and never contact AEMET.

## Verification

```sh
python -m pytest
```
