from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

VALID_STATES = {"RECOVERED", "PENDING_RETRY", "CONFIRMED_NO_DATA"}
UNOBSERVED_STATES = {"PENDING_RETRY", "CONFIRMED_NO_DATA"}


def _key(row: Mapping[str, str], normalize_date: Callable[[object], str]) -> tuple[str, str]:
    return (row["station_code"], normalize_date(row["observation_date"]))


def merge_gap_ledger(
    base_missing: Iterable[Mapping[str, str]],
    work: Iterable[Mapping[str, str]],
    columns: list[str],
    normalize_date: Callable[[object], str],
    audit_id: str,
    repo_sha: str,
) -> list[dict[str, str]]:
    """Validate audited absences and emit every persisted recovery-ledger row.

    The audit output is used only to verify which work-ledger dates remain
    unobserved. Historical fields, including first_detected_at_utc, come from
    the persisted work ledger rather than being re-stamped by a later audit.
    """
    work_rows = [dict(item) for item in work]
    work_by_key: dict[tuple[str, str], dict[str, str]] = {}
    ordered_keys: list[tuple[str, str]] = []
    for item in work_rows:
        key = _key(item, normalize_date)
        if key in work_by_key:
            raise ValueError(f"duplicate recovery ledger key: {key}")
        if item.get("gap_status") not in VALID_STATES:
            raise ValueError(f"invalid recovery ledger state for {key}: {item.get('gap_status')}")
        work_by_key[key] = item
        ordered_keys.append(key)

    base_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for item in base_missing:
        key = _key(item, normalize_date)
        if key in base_by_key:
            raise ValueError(f"duplicate audited missing key: {key}")
        base_by_key[key] = dict(item)

    expected_unobserved = {
        key for key, item in work_by_key.items()
        if item["gap_status"] in UNOBSERVED_STATES
    }
    if set(base_by_key) != expected_unobserved:
        raise ValueError("audited missing dates do not match unresolved recovery dates")

    merged: list[dict[str, str]] = []
    for key in ordered_keys:
        source = work_by_key[key]
        row = {column: source.get(column, "") for column in columns}
        row["audit_id"] = audit_id
        row["repo_git_sha"] = repo_sha
        merged.append(row)

    if len(merged) != len(work_rows):
        raise ValueError("merged ledger does not contain every recovery row exactly once")
    return merged
