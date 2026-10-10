# Secret scan evidence — v0.1

Verification performed locally on synthetic material only; no live SIAR response was requested or stored.

- `tests/test_secret_scan.py` checks reusable hidden CSRF values and exact in-memory session values are blocked.
- False-positive controls cover empty CSRF markers, ordinary forecast wording/IDs, and a long static asset hash; all are allowed. A reusable unquoted access token in a URL is blocked.
- `tests/test_zip_safe.py` checks traversal paths, absolute/drive paths, multiple entries, and non-CSV entries are rejected without filesystem extraction.
- `examples/run.json` and `examples/metadata.json` use fabricated identifiers and an `example.invalid` URL. The metadata explicitly keeps `source_issue_at` null.
- All test HTML is inline synthetic markup. No HTTP response fixture or real credential is in the repository.

Run evidence: `19 passed` with the local pytest environment. This verifies the guard's synthetic cases; it is not a substitute for a pre-deployment scan of an actual SIAR HTML response. If a real response contains a reusable secret, the implementation refuses to persist the HTML and records the station outcome as failed.
