# Secret scan evidence — v0.1

Verification performed locally on synthetic material only; no live SIAR response was requested or stored.

- `tests/test_secret_scan.py` checks reusable hidden CSRF values and exact in-memory session values are blocked.
- False-positive controls explicitly cover bare names such as `_csrf`, `Cookie`, `XSRF-TOKEN`, `JSESSIONID` and `Authorization`, plus empty CSRF markers, ordinary forecast wording/IDs, and a long static asset hash; these are allowed when no reusable value is present.
- A reusable unquoted access token in a URL is blocked.
- `tests/test_zip_safe.py` checks traversal paths, absolute/drive paths, multiple entries, and non-CSV entries are rejected without filesystem extraction.
- `tests/test_storage.py` checks append-only behavior, preservation of byte-identical captures under distinct IDs, SHA-256 manifest coverage, and that a failed atomic rename does not publish a capture.
- `tests/test_siar_client.py` checks the observed SIAR validation contract is URL-encoded form data with the `X-XSRF-TOKEN` header, not JSON.
- `examples/run.json` and `examples/metadata.json` use fabricated identifiers and an `example.invalid` URL. The metadata explicitly keeps `source_issue_at` null.
- Synthetic example artifact hashes are valid SHA-256 values for fabricated HTML/ZIP/CSV byte sequences; they do not represent SIAR data.
- All test HTML is inline synthetic markup. No HTTP response fixture or real credential is in the repository.

Independent review run evidence: `26 passed` with the synthetic pytest suite after the acceptance-criteria fixes. This is not a substitute for the pre-deployment scan of an actual SIAR HTML response. If a real response contains a reusable session/authentication value, the implementation refuses to persist that HTML and records the station outcome as failed.
