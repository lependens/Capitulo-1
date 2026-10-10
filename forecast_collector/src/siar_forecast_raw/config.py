from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Station:
    code: str
    label: str
    siar_station_form_id: str
    form_params: dict[str, str]


@dataclass(frozen=True)
class Config:
    data_root: Path
    base_url: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_attempts: int
    retry_backoff_seconds: tuple[float, ...]
    stations: tuple[Station, ...]


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path or os.environ.get("SIAR_FORECAST_CONFIG", "config/config.toml"))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}; copy config/config.example.toml first")
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    stations = tuple(Station(
        code=str(item["code"]), label=str(item["label"]),
        siar_station_form_id=str(item["siar_station_form_id"]),
        form_params={str(k): str(v) for k, v in item.get("form_params", {}).items()},
    ) for item in raw.get("stations", []))
    if not stations:
        raise ValueError("At least one station must be configured")
    codes = [station.code for station in stations]
    if len(codes) != len(set(codes)):
        raise ValueError("Station codes must be unique")
    return Config(
        data_root=Path(raw["data_root"]), base_url=raw.get("base_url", "https://servicio.mapa.gob.es/siarweb").rstrip("/"),
        connect_timeout_seconds=float(raw.get("connect_timeout_seconds", 10)),
        read_timeout_seconds=float(raw.get("read_timeout_seconds", 30)),
        max_attempts=max(1, int(raw.get("max_attempts", 2))),
        retry_backoff_seconds=tuple(float(x) for x in raw.get("retry_backoff_seconds", [2, 5])),
        stations=stations,
    )
