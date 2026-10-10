from collections import Counter
import json
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config
from .errors import CaptureError, SensitiveHtmlError
from .ids import iso_utc, new_capture_id, new_run_id, utc_now
from .metadata import artifact_record
from .siar_client import SiarClient
from .storage import begin_run, publish_capture, publish_run

_REQUIRED_ARTIFACTS = {"result.html", "forecast.zip", "forecast.csv"}


def _status(results: list[dict[str, Any]]) -> str:
    counts = Counter(item["status"] for item in results)
    if len(results) > 0 and counts["complete"] == len(results):
        return "complete"
    if counts["complete"] or counts["partial"]:
        return "partial"
    return "failed"


def capture_run(config: Config, client_factory=SiarClient, now=utc_now) -> Path:
    started = now()
    run_id = new_run_id(started)
    started_iso = iso_utc(started)
    temp, final = begin_run(config.data_root, started_iso, run_id)
    outcomes: list[dict[str, Any]] = []
    for station in config.stations:
        station_started = now()
        capture_id = new_capture_id(station.code, station_started)
        http: dict[str, Any] = {}
        artifacts: dict[str, bytes] = {}
        details: dict[str, Any] = {}
        error: str | None = None
        try:
            details, artifacts = client_factory(config).capture_station(station)
            http = details.get("http", {})
            missing = sorted(name for name in _REQUIRED_ARTIFACTS if not artifacts.get(name))
            if missing:
                raise CaptureError(
                    "Capture client returned incomplete artifact set: " + ", ".join(missing),
                    artifacts=artifacts,
                    http=http,
                )
            status = "complete"
        except SensitiveHtmlError as exc:
            http, artifacts, error, status = exc.http, exc.artifacts, str(exc), "failed"
            # Secret-bearing HTML is never persisted, even in a partial capture.
        except CaptureError as exc:
            http, artifacts, error = exc.http, exc.artifacts, str(exc)
            status = "partial" if artifacts else "failed"
        except Exception as exc:  # keep station failures independent
            error, status = f"{type(exc).__name__}: {exc}", "failed"

        finished = now()
        artifact_keys = {"result.html": "html", "forecast.zip": "zip", "forecast.csv": "csv"}
        records = {artifact_keys[name]: artifact_record(name, data) for name, data in artifacts.items()}
        if "csv" in records and details.get("csv_source_filename"):
            records["csv"]["source_filename"] = details["csv_source_filename"]
        metadata = {
            "schema_version": "siar-raw-capture-v1", "collector_version": __version__,
            "run_id": run_id, "capture_id": capture_id,
            "station": details.get("station", {"code": station.code, "label": station.label, "siar_station_form_id": station.siar_station_form_id}),
            "capture": {"started_at_utc": iso_utc(station_started), "finished_at_utc": iso_utc(finished), "status": status},
            "source": details.get("source", {"system": "SIAR", "product": "necesidadesHidricas", "source_issue_at": None}),
            "request": details.get("request", {}), "http": http, "artifacts": records,
            "errors": [error] if error else [],
        }
        publish_capture(temp, station.code, capture_id, metadata, artifacts)
        outcomes.append({"station_code": station.code, "capture_id": capture_id, "status": status, "error": error})

    finished = now()
    counts = Counter(item["status"] for item in outcomes)
    run_metadata = {
        "schema_version": "raw-capture-run-v1", "collector_version": __version__, "run_id": run_id,
        "started_at_utc": started_iso, "finished_at_utc": iso_utc(finished),
        "target_scope": "configured", "stations_resolved": [
            {"code": s.code, "label": s.label, "siar_station_form_id": s.siar_station_form_id}
            for s in config.stations
        ],
        "stations_attempted": len(outcomes), "stations_complete": counts["complete"],
        "stations_partial": counts["partial"], "stations_failed": counts["failed"],
        "status": _status(outcomes), "captures": outcomes,
    }
    return publish_run(temp, final, run_metadata)


def load_run(path: str | Path) -> dict[str, Any]:
    """Read a previously published raw run without the collector or a DB."""
    return json.loads((Path(path) / "run.json").read_text(encoding="utf-8"))
