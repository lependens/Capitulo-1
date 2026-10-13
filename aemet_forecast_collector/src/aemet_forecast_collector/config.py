from dataclasses import dataclass
import os
from pathlib import Path
import re
import tomllib
from typing import Any


@dataclass(frozen=True)
class Target:
    station_code: str
    station_name: str
    municipality_id: str
    municipality_name: str
    spatial_match: str


@dataclass(frozen=True)
class Config:
    data_root: Path
    request_timeout_seconds: float
    targets: tuple[Target, ...]


def _nonempty_string(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration field {key} must be a non-empty string")
    return value.strip()


def _contains_credential_setting(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if "api_key" in normalized or "token" in normalized or normalized in {"authorization", "password", "secret"}:
                return True
            if _contains_credential_setting(item):
                return True
    elif isinstance(value, list):
        return any(_contains_credential_setting(item) for item in value)
    return False


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path or os.environ.get("AEMET_FORECAST_CONFIG", "config/config.toml"))
    if not config_path.is_file():
        raise FileNotFoundError(f"AEMET config file not found: {config_path}")
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    if _contains_credential_setting(raw):
        raise ValueError("Credentials must be supplied through runtime environment only")

    targets: list[Target] = []
    for item in raw.get("targets", []):
        if not isinstance(item, dict):
            raise ValueError("Each target must be a TOML table")
        station_code = _nonempty_string(item, "station_code")
        municipality_id = _nonempty_string(item, "municipality_id")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", station_code):
            raise ValueError("station_code contains unsupported path characters")
        if not re.fullmatch(r"\d{5}", municipality_id):
            raise ValueError("municipality_id must contain five digits")
        targets.append(Target(
            station_code=station_code,
            station_name=_nonempty_string(item, "station_name"),
            municipality_id=municipality_id,
            municipality_name=_nonempty_string(item, "municipality_name"),
            spatial_match=_nonempty_string(item, "spatial_match"),
        ))

    if not targets:
        raise ValueError("At least one target must be configured")
    codes = [target.station_code for target in targets]
    if len(codes) != len(set(codes)):
        raise ValueError("station_code values must be unique")

    timeout = float(raw.get("request_timeout_seconds", 30))
    if not 0 < timeout <= 300:
        raise ValueError("request_timeout_seconds must be greater than 0 and at most 300")
    return Config(Path(raw["data_root"]), timeout, tuple(targets))
