from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from aemet_forecast_collector.capture import capture_run
from aemet_forecast_collector.client import ENDPOINT
from aemet_forecast_collector.config import Config, Target, load_config
from aemet_forecast_collector.metadata import validate_capture_metadata
from aemet_forecast_collector.parsing import declared_charset
from aemet_forecast_collector.storage import verify_checksums
from aemet_forecast_collector.transport import HttpResponse


KEY = "synthetic-aemet-runtime-key-0123456789"
TARGET = Target("IB04", "Son Ferriol", "07040", "Palma", "municipal_capital")
NOW = datetime(2026, 10, 13, 5, 35, 21, tzinfo=timezone.utc)


def _payload(*, include_humidity=True, issue="2026-10-12T21:09:07", first_temp=27):
    day = {
        "fecha": "2026-10-12T00:00:00",
        "temperatura": {"minima": 14, "maxima": first_temp, "dato": [{"value": 23, "hora": 18}]},
        "viento": [{"direccion": "SO", "velocidad": 10, "periodo": "00-24"}],
        "rachaMax": [{"value": "", "periodo": "00-24"}],
        "probPrecipitacion": [{"value": 25, "periodo": "00-24"}],
        "estadoCielo": [{"value": "43", "descripcion": "Cielo húmedo", "periodo": "00-24"}],
        "cotaNieveProv": [{"value": "", "periodo": "00-24"}],
        "sensTermica": {"minima": 12, "maxima": 26},
        "uvMax": 4,
    }
    if include_humidity:
        day["humedadRelativa"] = {"minima": 55, "maxima": 95, "dato": []}
    return [{
        "elaborado": issue,
        "id": "07040",
        "nombre": "Palma",
        "origen": "Agencia Estatal de Meteorología",
        "provincia": "Illes Balears (Mallorca)",
        "version": 1,
        "prediccion": {"dia": [day]},
    }]


def _json(value, encoding="ISO-8859-15"):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(encoding)


def _wrapper(data_url="https://opendata.aemet.es/opendata/sh/1be9d7ec", estado=200):
    return _json({"descripcion": "exito", "estado": estado, "datos": data_url, "metadatos": "https://opendata.aemet.es/opendata/sh/meta"})


class SequenceFetcher:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, dict(headers), timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _config(root, target=TARGET):
    return Config(Path(root), 5, (target,))


def _successful_fetcher(payload=None, wrapper_body=None):
    payload_body = _json(payload if payload is not None else _payload())
    wrapper_body = wrapper_body if wrapper_body is not None else _wrapper()
    return SequenceFetcher(
        HttpResponse(200, {"Content-Type": "application/json;charset=ISO-8859-15", "Content-Encoding": "identity"}, wrapper_body),
        HttpResponse(200, {"Content-Type": "text/plain;charset=ISO-8859-15", "Content-Encoding": "identity"}, payload_body),
    ), wrapper_body, payload_body


def _metadata(run_path):
    return next(run_path.glob("IB04/cap_*/metadata.json")), json.loads(next(run_path.glob("IB04/cap_*/metadata.json")).read_text(encoding="utf-8"))


def _validate(instance, schema_name):
    schema_path = Path(__file__).parents[1] / "schemas" / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)


def test_wrapper_then_payload_are_saved_byte_for_byte_with_charset_and_hashes(tmp_path):
    fetcher, wrapper_bytes, payload_bytes = _successful_fetcher()
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    metadata_path, metadata = _metadata(run_path)
    capture_dir = metadata_path.parent

    assert fetcher.calls[0][0] == ENDPOINT.format(municipality_id="07040")
    assert fetcher.calls[0][1]["api_key"] == KEY
    assert "api_key" not in fetcher.calls[1][1]
    assert fetcher.calls[1][0] == "https://opendata.aemet.es/opendata/sh/1be9d7ec"
    assert (capture_dir / "wrapper.raw").read_bytes() == wrapper_bytes
    assert (capture_dir / "payload.raw").read_bytes() == payload_bytes
    assert metadata["capture"]["status"] == "complete"
    assert metadata["station"] == {"code": "IB04", "name": "Son Ferriol"}
    assert metadata["municipality"] == {"id": "07040", "name": "Palma", "spatial_match": "municipal_capital"}
    assert metadata["source"]["issue_time_raw"] == "2026-10-12T21:09:07"
    assert metadata["source"]["issue_timezone"] is None
    assert metadata["valid_times_raw"] == ["2026-10-12T00:00:00"]
    assert metadata["periods_raw"] == ["00-24", "00-24", "00-24", "00-24", "00-24"]
    assert metadata["http"]["payload"]["content_type"] == "text/plain;charset=ISO-8859-15"
    assert metadata["http"]["payload"]["charset"] == "iso-8859-15"
    assert metadata["artifacts"]["wrapper"]["byte_length"] == len(wrapper_bytes)
    assert metadata["artifacts"]["payload"]["byte_length"] == len(payload_bytes)
    assert metadata["artifacts"]["wrapper"]["sha256"] == metadata["http"]["wrapper"]["sha256"]
    assert metadata["artifacts"]["payload"]["sha256"] == metadata["http"]["payload"]["sha256"]
    assert verify_checksums(run_path) == []
    assert verify_checksums(capture_dir) == []
    _validate(metadata, "capture.schema.json")
    _validate(json.loads((run_path / "run.json").read_text()), "run.schema.json")
    assert not any(KEY.encode() in path.read_bytes() for path in run_path.rglob("*.*") if path.is_file())
    assert "forecast_horizon" not in metadata and "issue_time_utc" not in metadata


def test_payload_text_plain_json_and_structural_null_empty_absent_distinction(tmp_path):
    payload = _payload(include_humidity=False)
    payload[0]["prediccion"]["dia"][0]["datoAdicional"] = {"missing_value": None, "blank_value": ""}
    fetcher, _, _ = _successful_fetcher(payload=payload)
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    structure = metadata["payload_structure"]
    assert structure["format"] == "JSON"
    assert structure["json_null_count"] == 1
    assert structure["empty_string_count"] >= 1
    assert "humedadRelativa" not in structure["variables_observed"]
    assert "humedadRelativa" not in structure["field_names_by_day"][0]
    raw_payload = json.loads(next(run_path.glob("IB04/cap_*/payload.raw")).read_bytes().decode("ISO-8859-15"))
    day = raw_payload[0]["prediccion"]["dia"][0]
    assert day["datoAdicional"]["missing_value"] is None
    assert day["datoAdicional"]["blank_value"] == ""


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_failed_and_never_persist_response_body_or_key(tmp_path, status):
    fetcher = SequenceFetcher(HttpResponse(status, {"Content-Type": "application/json"}, b'{"error":"unauthorized"}'))
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "failed"
    assert metadata["http"]["wrapper"]["status_code"] == status
    assert not list(run_path.rglob("wrapper.raw"))
    assert KEY not in json.dumps(metadata)


def test_missing_api_key_does_not_make_a_request_or_persist_a_credential(tmp_path, monkeypatch):
    monkeypatch.delenv("AEMET_API_KEY", raising=False)
    fetcher = SequenceFetcher()
    run_path = capture_run(_config(tmp_path / "raw"), fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert fetcher.calls == []
    assert metadata["capture"]["status"] == "failed"
    assert metadata["errors"] == ["AEMET_API_KEY missing from runtime environment"]
    assert metadata["http"] == {"wrapper": None, "payload": None}


def test_expired_api_key_makes_zero_http_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("AEMET_API_KEY_EXPIRES_AT", "2026-10-13T05:35:20Z")
    fetcher = SequenceFetcher()
    capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    assert fetcher.calls == []


def test_expired_api_key_returns_only_safe_renewal_error(tmp_path, monkeypatch):
    monkeypatch.setenv("AEMET_API_KEY_EXPIRES_AT", "2026-10-13T05:35:20Z")
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=SequenceFetcher(), now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "failed"
    assert metadata["errors"] == ["AEMET API key renewal required"]
    assert KEY not in json.dumps(metadata)


def test_future_api_key_expiry_allows_capture_without_persisting_key_or_expiry(tmp_path, monkeypatch):
    expiry = "2026-10-13T05:35:22Z"
    monkeypatch.setenv("AEMET_API_KEY_EXPIRES_AT", expiry)
    fetcher, _, _ = _successful_fetcher()
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "complete"
    assert len(fetcher.calls) == 2
    assert expiry not in json.dumps(metadata)
    assert KEY not in json.dumps(metadata)
    assert not any(KEY.encode() in path.read_bytes() for path in run_path.rglob("*.*") if path.is_file())
    assert all(KEY not in error for error in metadata["errors"])


def test_invalid_api_key_expiry_fails_closed_with_safe_configuration_error(tmp_path, monkeypatch):
    invalid_expiry = "not-a-valid-timestamp"
    monkeypatch.setenv("AEMET_API_KEY_EXPIRES_AT", invalid_expiry)
    fetcher = SequenceFetcher()
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert fetcher.calls == []
    assert metadata["capture"]["status"] == "failed"
    assert metadata["errors"] == ["AEMET_API_KEY_EXPIRES_AT must be a timezone-aware ISO-8601 timestamp"]
    serialized = json.dumps(metadata)
    assert KEY not in serialized
    assert invalid_expiry not in serialized


@pytest.mark.parametrize("wrapper_body", [b"not json", _json({"descripcion": "exito", "estado": 200})])
def test_malformed_or_missing_datos_wrapper_is_partial_and_preserved(tmp_path, wrapper_body):
    fetcher = SequenceFetcher(HttpResponse(200, {"Content-Type": "application/json"}, wrapper_body))
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "partial"
    assert next(run_path.rglob("wrapper.raw")).read_bytes() == wrapper_body
    assert len(fetcher.calls) == 1
    assert fetcher.calls[0][0] == ENDPOINT.format(municipality_id="07040")
    assert "api_key" in fetcher.calls[0][1]


def test_malformed_payload_is_preserved_as_partial(tmp_path):
    wrapper_bytes = _wrapper()
    bad_payload = b"{bad json"
    fetcher = SequenceFetcher(
        HttpResponse(200, {"Content-Type": "application/json"}, wrapper_bytes),
        HttpResponse(200, {"Content-Type": "text/plain;charset=ISO-8859-15"}, bad_payload),
    )
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "partial"
    assert next(run_path.rglob("payload.raw")).read_bytes() == bad_payload
    assert metadata["payload_structure"] is None


def test_wrapper_internal_estado_must_be_200_before_payload_fetch(tmp_path):
    wrapper_body = _wrapper(estado=500)
    fetcher = SequenceFetcher(HttpResponse(200, {"Content-Type": "application/json"}, wrapper_body))
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "partial"
    assert metadata["errors"] == ["wrapper estado is not 200"]
    assert next(run_path.rglob("wrapper.raw")).read_bytes() == wrapper_body
    assert len(fetcher.calls) == 1
    assert metadata["http"]["payload"] is None


def test_payload_municipality_id_must_match_configured_target(tmp_path):
    payload = _payload()
    payload[0]["id"] = "07041"
    fetcher, _, payload_bytes = _successful_fetcher(payload=payload)
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "partial"
    assert metadata["payload_structure"] is None
    assert next(run_path.rglob("payload.raw")).read_bytes() == payload_bytes


def test_payload_with_multiple_municipality_roots_cannot_be_complete(tmp_path):
    payload = _payload() + _payload()
    fetcher, _, payload_bytes = _successful_fetcher(payload=payload)
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "partial"
    assert metadata["payload_structure"] is None
    assert next(run_path.rglob("payload.raw")).read_bytes() == payload_bytes


@pytest.mark.parametrize("invalid_payload", [
    _payload(issue=""),
    [{**_payload()[0], "prediccion": {"dia": [{**_payload()[0]["prediccion"]["dia"][0], "fecha": ""}]}}],
])
def test_payload_with_empty_literal_times_is_partial(tmp_path, invalid_payload):
    fetcher, _, payload_bytes = _successful_fetcher(payload=invalid_payload)
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "partial"
    assert metadata["payload_structure"] is None
    assert next(run_path.rglob("payload.raw")).read_bytes() == payload_bytes


def test_malformed_content_type_charset_is_nonfatal():
    assert declared_charset("application/json; charset=\"unterminated") is None


def test_payload_http_failure_keeps_wrapper_and_safe_payload_bytes(tmp_path):
    wrapper_bytes = _wrapper()
    error_body = b"temporary service error"
    fetcher = SequenceFetcher(
        HttpResponse(200, {"Content-Type": "application/json"}, wrapper_bytes),
        HttpResponse(503, {"Content-Type": "text/plain"}, error_body),
    )
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "partial"
    assert next(run_path.rglob("wrapper.raw")).read_bytes() == wrapper_bytes
    assert next(run_path.rglob("payload.raw")).read_bytes() == error_body
    assert metadata["http"]["payload"]["status_code"] == 503


def test_transport_failure_is_sanitized_and_does_not_expose_exception_text(tmp_path):
    fetcher = SequenceFetcher(TimeoutError(f"timeout including {KEY}"))
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "failed"
    assert metadata["http"]["wrapper"]["status_code"] is None
    assert KEY not in json.dumps(metadata)
    assert all(KEY.encode() not in p.read_bytes() for p in run_path.rglob("*.*") if p.is_file())


def test_credential_echo_in_payload_is_withheld_from_raw_and_metadata(tmp_path):
    wrapper_bytes = _wrapper()
    fetcher = SequenceFetcher(
        HttpResponse(200, {"Content-Type": "application/json"}, wrapper_bytes),
        HttpResponse(200, {"Content-Type": "text/plain"}, KEY.encode()),
    )
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "partial"
    assert not list(run_path.rglob("payload.raw"))
    assert KEY not in json.dumps(metadata)
    assert all(KEY.encode() not in p.read_bytes() for p in run_path.rglob("*.*") if p.is_file())


def test_secret_query_in_datos_url_is_not_written_to_raw_or_metadata(tmp_path):
    secret_url = "https://opendata.aemet.es/opendata/sh/1be9d7ec?token=runtime-secret"
    wrapper_bytes = _wrapper(secret_url)
    fetcher = SequenceFetcher(HttpResponse(200, {"Content-Type": "application/json"}, wrapper_bytes))
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    assert metadata["capture"]["status"] == "failed"
    assert not list(run_path.rglob("wrapper.raw"))
    serialized = json.dumps(metadata)
    assert "runtime-secret" not in serialized
    assert "?token=" not in serialized


@pytest.mark.parametrize("payload_delta", [0, 1])
def test_each_run_is_append_only_for_same_or_changed_issue_payload(tmp_path, payload_delta):
    first_fetcher, _, first_payload = _successful_fetcher()
    second_fetcher, _, second_payload = _successful_fetcher(payload=_payload(first_temp=27 + payload_delta))
    first = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=first_fetcher, now=lambda: NOW)
    second = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=second_fetcher, now=lambda: NOW)
    assert first != second
    assert next(first.rglob("payload.raw")).read_bytes() == first_payload
    assert next(second.rglob("payload.raw")).read_bytes() == second_payload
    assert first.exists() and second.exists()
    assert json.loads(next(first.rglob("metadata.json")).read_text())["source"]["issue_time_raw"] == "2026-10-12T21:09:07"
    assert json.loads(next(second.rglob("metadata.json")).read_text())["source"]["issue_time_raw"] == "2026-10-12T21:09:07"
    if payload_delta:
        assert first_payload != second_payload
    else:
        assert first_payload == second_payload


def test_atomic_publication_failure_leaves_no_visible_run(tmp_path, monkeypatch):
    import aemet_forecast_collector.capture as capture_module
    import aemet_forecast_collector.storage as storage

    monkeypatch.setattr(capture_module, "new_run_id", lambda _started: "run_atomic")
    fetcher, _, _ = _successful_fetcher()
    real_rename = storage.os.rename

    def fail_final_rename(source, destination):
        if Path(destination).name == "run_atomic":
            raise OSError("synthetic atomic publish failure")
        return real_rename(source, destination)

    monkeypatch.setattr(storage.os, "rename", fail_final_rename)
    with pytest.raises(OSError):
        capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    final_parent = tmp_path / "raw/runs/2026/10/13"
    assert not (final_parent / "run_atomic").exists()
    assert list(final_parent.iterdir()) == []


def test_config_rejects_inline_api_key_and_loads_station_fixture(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'data_root = "../raw"\nrequest_timeout_seconds = 15\n'
        '[[targets]]\nstation_code = "IB04"\nstation_name = "Son Ferriol"\n'
        'municipality_id = "07040"\nmunicipality_name = "Palma"\nspatial_match = "municipal_capital"\n',
        encoding="utf-8",
    )
    loaded = load_config(config_path)
    assert loaded.targets == (TARGET,)
    config_path.write_text(config_path.read_text() + f'api_key = "{KEY}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="runtime environment only"):
        load_config(config_path)


@pytest.mark.parametrize("spatial_match", ["station_exact", "arbitrary"])
def test_invalid_spatial_match_is_rejected_before_http(tmp_path, spatial_match):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'data_root = "../raw"\nrequest_timeout_seconds = 15\n'
        '[[targets]]\nstation_code = "IB04"\nstation_name = "Son Ferriol"\n'
        f'municipality_id = "07040"\nmunicipality_name = "Palma"\nspatial_match = "{spatial_match}"\n',
        encoding="utf-8",
    )
    fetcher = SequenceFetcher()
    with pytest.raises(ValueError, match="spatial_match must be municipal_capital"):
        config = load_config(config_path)
        capture_run(config, api_key=KEY, fetcher=fetcher)
    assert fetcher.calls == []


def test_station_exact_is_rejected_by_runtime_metadata_and_schema(tmp_path):
    fetcher, _, _ = _successful_fetcher()
    run_path = capture_run(_config(tmp_path / "raw"), api_key=KEY, fetcher=fetcher, now=lambda: NOW)
    _, metadata = _metadata(run_path)
    metadata["municipality"]["spatial_match"] = "station_exact"

    with pytest.raises(ValueError, match="municipality spatial_match must be municipal_capital"):
        validate_capture_metadata(metadata)
    with pytest.raises(ValidationError):
        _validate(metadata, "capture.schema.json")
