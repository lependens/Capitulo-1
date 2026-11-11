"""IB09 scientific V2 exception; Quality V1 is immutable."""
from datetime import date
from .quality import eligibility

IB09_RAW_SHA256 = "2cbd9bb289404264fc78dad21322562d9d22b64ce4bbc87e51512ca763ea7a8a"
IB09_START = date(2005, 10, 24)
IB09_END = date(2018, 10, 10)


def ib09_v2_eligibility(row, station, method, *, verified_raw_sha256):
    """Return V2 method reasons; caller must verify the raw bytes independently.

    This opt-in path is limited to the audited snapshot and does not change
    either the row or the station-level Quality V1 result.
    """
    if verified_raw_sha256 != IB09_RAW_SHA256 or row.get("source_data_sha") != IB09_RAW_SHA256:
        return ["unverified_ib09_snapshot"]
    if row.get("station_code") != "IB09" or row.get("quality_status") != "FAIL":
        return ["ib09_v2_exception_not_applicable"]
    try:
        day = date.fromisoformat(row.get("date", ""))
    except (TypeError, ValueError):
        return ["invalid_date"]
    if not IB09_START <= day <= IB09_END:
        return ["OUTSIDE_OFFICIAL_PERIOD"]
    # V1 FAIL remains on the input. Only this V2 calculation is evaluated with
    # its source gate satisfied; all method-specific checks remain in force.
    return eligibility({**row, "quality_status": "WARN"}, station, method)
