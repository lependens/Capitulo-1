# Secret scan evidence — v0.1

Automated verification uses synthetic material only; the suite does not contact SIAR. The E2E SIAR contract was separately demonstrated in a manual session, but no live response, ZIP, CSV, cookie, or token is stored in the repository.

- `tests/test_secret_scan.py` checks reusable hidden CSRF values and exact in-memory session values are blocked.
- False-positive controls explicitly cover bare names such as `_csrf`, `Cookie`, `XSRF-TOKEN`, `JSESSIONID` and `Authorization`, plus empty CSRF markers, ordinary forecast wording/IDs, and a long static asset hash; these are allowed when no reusable value is present.
- A reusable unquoted access token in a URL is blocked.
- `tests/test_zip_safe.py` checks traversal paths, absolute/drive paths, multiple entries, and non-CSV entries are rejected without filesystem extraction.
- `tests/test_storage.py` checks append-only behavior, preservation of byte-identical captures under distinct IDs, SHA-256 manifest coverage, and that a failed atomic rename does not publish a capture.
- `tests/test_siar_client.py` checks the observed SIAR flow: distinct hidden/meta CSRF values, AJAX validation headers, a normal calculation POST without AJAX headers, and same-session export GET with no body or query parameters.
- `examples/run.json` and `examples/metadata.json` use fabricated identifiers and an `example.invalid` URL. The metadata explicitly keeps `source_issue_at` null.
- Synthetic example artifact hashes are valid SHA-256 values for fabricated HTML/ZIP/CSV byte sequences; they do not represent SIAR data.
- All test HTML is inline synthetic markup. No HTTP response fixture or real credential is in the repository.

The current synthetic suite checks the verified IB04 form payload, absence of G4 dates, expected CSV encoding/delimiter/headers/seven rows, ZIP limits and CRC, the secret guard, append-only behavior, and storage safety. This is not a substitute for the pre-deployment scan of an actual SIAR HTML response. If a real response contains a reusable session/authentication value, the implementation refuses to persist that HTML and records the station outcome as failed.

- `tests/test_schema_validation.py` validates both synthetic examples and a generated run/capture against the Draft 2020-12 schemas using `jsonschema` with format checking.
