import os
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Callable

from . import __version__
from .client import AemetClient
from .config import Config
from .ids import iso_utc, new_capture_id, new_run_id, utc_now
from .metadata import validate_capture_metadata, validate_run_metadata
from .secrets import contains_secret, redact_metadata
from .storage import create_run_stage, discard_stage, publish_run, write_capture
from .transport import HttpResponse, http_get


def capture_run(
    config: Config,
    api_key: str | None = None,
    fetcher: Callable = http_get,
    now: Callable[[], datetime] = utc_now,
) -> Path:
    if api_key is None:
        api_key = os.environ.get("AEMET_API_KEY")
    started = now()
    started_iso = iso_utc(started)
    run_id = new_run_id(started)
    stage, final = create_run_stage(config.data_root, started_iso, run_id)
    results: list[dict] = []
    try:
        for target in config.targets:
            station_started = now()
            capture_id = new_capture_id(target.station_code, station_started)
            outcome = AemetClient(api_key, config.request_timeout_seconds, fetcher, now).capture(target)
            finished = now()
            status = "complete" if outcome.complete else ("partial" if outcome.artifacts else "failed")
            artifact_records = {
                name.removesuffix(".raw"): {
                    "filename": name,
                    "byte_length": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
                for name, body in outcome.artifacts.items()
            }
            metadata = {
                "schema_version": "aemet-raw-capture-v1",
                "collector_version": __version__,
                "run_id": run_id,
                "capture_id": capture_id,
                "station": {"code": target.station_code, "name": target.station_name},
                "municipality": {
                    "id": target.municipality_id,
                    "name": target.municipality_name,
                    "spatial_match": target.spatial_match,
                },
                "capture": {
                    "started_at_utc": iso_utc(station_started),
                    "finished_at_utc": iso_utc(finished),
                    "status": status,
                },
                "source": {
                    "provider": "AEMET",
                    "product": "prediccion_municipio_diaria",
                    "issue_time_raw": outcome.issue_time_raw,
                    "issue_timezone": None,
                },
                "valid_times_raw": outcome.valid_times_raw,
                "periods_raw": outcome.periods_raw,
                "payload_structure": outcome.payload_structure,
                "http": {
                    "wrapper": outcome.http.get("wrapper"),
                    "payload": outcome.http.get("payload"),
                },
                "artifacts": artifact_records,
                "errors": outcome.errors,
            }
            metadata = redact_metadata(metadata, api_key)
            validate_capture_metadata(metadata)
            metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
            if api_key and contains_secret(metadata_bytes, api_key):
                raise ValueError("credential safety check rejected capture metadata")
            write_capture(stage, target.station_code, capture_id, metadata, outcome.artifacts)
            results.append({"station_code": target.station_code, "capture_id": capture_id, "status": status})

        finished = now()
        complete_count = sum(item["status"] == "complete" for item in results)
        if results and complete_count == len(results):
            run_status = "complete"
        elif any(item["status"] in {"complete", "partial"} for item in results):
            run_status = "partial"
        else:
            run_status = "failed"
        run_metadata = {
            "schema_version": "aemet-raw-run-v1",
            "collector_version": __version__,
            "run_id": run_id,
            "started_at_utc": started_iso,
            "finished_at_utc": iso_utc(finished),
            "status": run_status,
            "captures": results,
        }
        validate_run_metadata(run_metadata)
        return publish_run(stage, final, run_metadata)
    finally:
        discard_stage(stage)
