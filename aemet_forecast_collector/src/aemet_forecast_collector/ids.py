from datetime import datetime, timezone
import secrets


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("UTC timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def new_run_id(started_at: datetime) -> str:
    return f"run_{_stamp(started_at)}_{secrets.token_hex(6)}"


def new_capture_id(station_code: str, started_at: datetime) -> str:
    return f"cap_{_stamp(started_at)}_{station_code.upper()}_{secrets.token_hex(6)}"
