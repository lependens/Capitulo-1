from datetime import datetime
import re
from typing import Any


_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def validate_capture_metadata(value: dict[str, Any]) -> None:
    required = {
        "schema_version", "collector_version", "run_id", "capture_id", "station",
        "municipality", "capture", "source", "valid_times_raw", "periods_raw",
        "payload_structure", "http", "artifacts", "errors",
    }
    if set(value) != required:
        raise ValueError("capture metadata fields do not match aemet-raw-capture-v1")
    if value["schema_version"] != "aemet-raw-capture-v1":
        raise ValueError("unsupported capture metadata schema")
    for key in ("run_id", "capture_id"):
        if not isinstance(value[key], str) or not value[key].startswith("run_" if key == "run_id" else "cap_"):
            raise ValueError(f"invalid {key}")
    station = value["station"]
    municipality = value["municipality"]
    if not isinstance(station, dict) or set(station) != {"code", "name"}:
        raise ValueError("invalid station metadata")
    if not isinstance(municipality, dict) or set(municipality) != {"id", "name", "spatial_match"}:
        raise ValueError("invalid municipality metadata")
    capture = value["capture"]
    if not isinstance(capture, dict) or capture.get("status") not in {"complete", "partial", "failed"}:
        raise ValueError("invalid capture status")
    for key in ("started_at_utc", "finished_at_utc"):
        if not isinstance(capture.get(key), str):
            raise ValueError("capture UTC timestamps are required")
        datetime.fromisoformat(capture[key].replace("Z", "+00:00"))
    source = value["source"]
    if not isinstance(source, dict) or source.get("provider") != "AEMET":
        raise ValueError("invalid source metadata")
    if source.get("product") != "prediccion_municipio_diaria" or source.get("issue_timezone") is not None:
        raise ValueError("invalid source product or issue timezone")
    if source.get("issue_time_raw") is not None and not isinstance(source["issue_time_raw"], str):
        raise ValueError("issue_time_raw must remain a string or null")
    if not isinstance(value["valid_times_raw"], list) or not all(isinstance(x, str) for x in value["valid_times_raw"]):
        raise ValueError("valid_times_raw must be literal strings")
    if not isinstance(value["periods_raw"], list) or not all(isinstance(x, str) for x in value["periods_raw"]):
        raise ValueError("periods_raw must be literal strings")
    if value["payload_structure"] is not None and not isinstance(value["payload_structure"], dict):
        raise ValueError("payload_structure must be an object or null")
    if not isinstance(value["http"], dict) or not isinstance(value["artifacts"], dict):
        raise ValueError("http and artifacts must be objects")
    if set(value["http"]) != {"wrapper", "payload"} or set(value["artifacts"]) - {"wrapper", "payload"}:
        raise ValueError("http or artifact keys do not match the capture schema")
    for record in value["http"].values():
        if record is None:
            continue
        expected_http = {
            "method", "url", "url_redacted", "status_code", "content_type",
            "content_encoding", "charset", "fetched_at_utc", "byte_length", "sha256",
        }
        if not isinstance(record, dict) or set(record) != expected_http or record["method"] != "GET":
            raise ValueError("invalid HTTP metadata record")
        if not isinstance(record["url"], str) or not isinstance(record["content_encoding"], str):
            raise ValueError("invalid HTTP URL or encoding metadata")
        if record["status_code"] is not None and not isinstance(record["status_code"], int):
            raise ValueError("invalid HTTP status metadata")
        if not isinstance(record["byte_length"], int) or record["byte_length"] < 0:
            raise ValueError("invalid HTTP body length")
        if not isinstance(record["sha256"], str) or not _SHA256.fullmatch(record["sha256"]):
            raise ValueError("invalid HTTP body checksum")
        datetime.fromisoformat(record["fetched_at_utc"].replace("Z", "+00:00"))
    if value["payload_structure"] is not None:
        structure = value["payload_structure"]
        expected_structure = {
            "format", "top_level_type", "top_level_count", "root_fields", "day_count",
            "field_names_by_day", "variables_observed", "json_null_count", "empty_string_count",
        }
        if not isinstance(structure, dict) or set(structure) != expected_structure or structure["format"] != "JSON":
            raise ValueError("invalid payload structure metadata")
        if structure["day_count"] != len(value["valid_times_raw"]):
            raise ValueError("payload day count does not match literal valid times")
    if not isinstance(value["errors"], list) or not all(isinstance(x, str) for x in value["errors"]):
        raise ValueError("errors must contain safe strings")
    for artifact_name, record in value["artifacts"].items():
        if not isinstance(record, dict) or not isinstance(record.get("byte_length"), int):
            raise ValueError("invalid artifact record")
        expected_filename = "wrapper.raw" if artifact_name == "wrapper" else "payload.raw"
        if set(record) != {"filename", "byte_length", "sha256"} or record["filename"] != expected_filename:
            raise ValueError("invalid artifact filename metadata")
        if not isinstance(record.get("sha256"), str) or not _SHA256.fullmatch(record["sha256"]):
            raise ValueError("invalid artifact checksum")


def validate_run_metadata(value: dict[str, Any]) -> None:
    required = {"schema_version", "collector_version", "run_id", "started_at_utc", "finished_at_utc", "status", "captures"}
    if set(value) != required or value.get("schema_version") != "aemet-raw-run-v1":
        raise ValueError("run metadata fields do not match aemet-raw-run-v1")
    if value.get("status") not in {"complete", "partial", "failed"}:
        raise ValueError("invalid run status")
    if not isinstance(value.get("captures"), list):
        raise ValueError("run captures must be a list")
    for key in ("started_at_utc", "finished_at_utc"):
        if not isinstance(value.get(key), str):
            raise ValueError("run UTC timestamps are required")
        datetime.fromisoformat(value[key].replace("Z", "+00:00"))
    for capture in value["captures"]:
        if not isinstance(capture, dict) or set(capture) != {"station_code", "capture_id", "status"}:
            raise ValueError("invalid run capture entry")
        if capture["status"] not in {"complete", "partial", "failed"}:
            raise ValueError("invalid run capture status")
