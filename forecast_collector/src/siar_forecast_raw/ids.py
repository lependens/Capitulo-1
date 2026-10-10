from datetime import datetime, timezone
import secrets


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"


def new_run_id(started_at: datetime | None = None) -> str:
    return f"run_{_stamp(started_at or utc_now())}_{secrets.token_hex(3)}"


def new_capture_id(station_code: str, started_at: datetime | None = None) -> str:
    safe_code = "".join(ch for ch in station_code.upper() if ch.isalnum() or ch in "-_")
    if not safe_code:
        raise ValueError("station_code must contain an alphanumeric character")
    return f"cap_{_stamp(started_at or utc_now())}_{safe_code}_{secrets.token_hex(3)}"
