import io
import tomllib
import zipfile
from pathlib import Path

import pytest
import requests

from siar_forecast_raw.config import Config, Station
from siar_forecast_raw.errors import CaptureError
from siar_forecast_raw.siar_client import SiarClient


ROOT = Path(__file__).parents[1]
EXPECTED_FORM_PARAMS = {
    "esCalculo": "0",
    "tipoCalculo": "2",
    "idCultivo": "6",
    "idComarca": "702",
    "comarcaEstacion": "Mallorca",
    "idCAEstacion": "-1",
    "provinciaEstHidden": "7",
    "chkProvincia": "1",
    "radioProvCom": "on",
    "idProvinciaEstacion": "7",
    "chkComRegante": "0",
    "idComunidadRegantes": "-1",
    "sEstacion": "IB04 - Son Ferriol",
    "chkComarca": "0",
    "comarcaSelect": "-1",
}
HIDDEN_CSRF = "synthetic-body-csrf-token-123"
META_CSRF = "synthetic-meta-csrf-token-456"


def _zip_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "Pronostico_IB04_Son_Ferriol.csv",
            b"\xef\xbb\xbfFecha;Kc;ET0;ETc;Pe;ETc - Pe\n11/10/2026;0,80;2,30;1,84;0,00;1,84\n",
        )
    return stream.getvalue()


def _response(url, body, content_type="text/html;charset=UTF-8"):
    response = requests.Response()
    response.status_code = 200
    response.url = url
    response._content = body
    response._content_consumed = True
    if content_type:
        response.headers["Content-Type"] = content_type
    return response


def test_example_config_matches_observed_ib04_payload_without_g4():
    with (ROOT / "config" / "config.example.toml").open("rb") as stream:
        config = tomllib.load(stream)

    station = config["stations"][0]
    assert station["code"] == "IB04"
    assert station["siar_station_form_id"] == "4_7"
    assert station["form_params"] == EXPECTED_FORM_PARAMS
    assert "fechaInicial" not in station["form_params"]
    assert "fechaFinal" not in station["form_params"]


def test_validation_calculation_and_export_match_observed_transport_contract(tmp_path):
    config = Config(
        tmp_path,
        "https://example.invalid/siarweb",
        1,
        1,
        1,
        (),
        (Station("IB04", "Son Ferriol", "4_7", EXPECTED_FORM_PARAMS),),
    )
    session = requests.Session()
    calls = []

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/necesidadesHidricas/inicio"):
            return _response(
                url,
                (
                    f'<html><meta name="_csrf" content="{META_CSRF}">'
                    f'<input type="hidden" name="_csrf" value="{HIDDEN_CSRF}"></html>'
                ).encode(),
            )
        if url.endswith("/necesidadesHidricasRest/validarForm"):
            return _response(url, b"", None)
        if url.endswith("/necesidadesHidricas/calculo"):
            return _response(url, b"<html><table id='tablaDatos'></table></html>")
        if url.endswith("/necesidadesHidricas/exportCSV"):
            response = _response(url, _zip_bytes(), "application/zip;charset=UTF-8")
            response.headers["Content-Disposition"] = 'attachment; filename="PronosticoNecesidadesNetas.zip"'
            return response
        raise AssertionError(url)

    session.request = fake_request
    client = SiarClient(config, session=session)
    details, artifacts = client.capture_station(config.stations[0])

    assert [call[0] for call in calls] == ["GET", "POST", "POST", "GET"]
    validation = next(call for call in calls if call[1].endswith("/necesidadesHidricasRest/validarForm"))
    validation_headers = validation[2]["headers"]
    assert validation[2]["data"]["_csrf"] == HIDDEN_CSRF
    assert validation_headers["X-XSRF-TOKEN"] == META_CSRF
    assert validation_headers["X-Requested-With"] == "XMLHttpRequest"
    assert validation_headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert validation_headers["Origin"] == "https://example.invalid"
    assert validation_headers["Referer"].endswith("/necesidadesHidricas/inicio")
    assert validation[2]["allow_redirects"] is False
    assert client._validation_accepted(b"")
    assert client._validation_accepted(b"{}")
    assert not client._validation_accepted(b'{"error":"invalid"}')

    calculation = next(call for call in calls if call[1].endswith("/necesidadesHidricas/calculo"))
    calc_headers = calculation[2]["headers"]
    assert calc_headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert calc_headers["Origin"] == "https://example.invalid"
    assert calc_headers["Referer"].endswith("/necesidadesHidricas/inicio")
    assert "X-Requested-With" not in calc_headers
    assert "X-XSRF-TOKEN" not in calc_headers
    assert calculation[2]["data"] == {**EXPECTED_FORM_PARAMS, "idEstacion": "4_7", "_csrf": HIDDEN_CSRF}

    export = next(call for call in calls if call[1].endswith("/necesidadesHidricas/exportCSV"))
    assert export[0] == "GET"
    assert export[2]["headers"] == {
        "Referer": "https://example.invalid/siarweb/necesidadesHidricas/calculo",
        "Accept": "*/*",
    }
    assert "data" not in export[2]
    assert "params" not in export[2]
    assert export[2]["stream"] is True
    assert client.session is session

    assert details["request"] == {**EXPECTED_FORM_PARAMS, "idEstacion": "4_7"}
    assert set(artifacts) == {"result.html", "forecast.zip", "forecast.csv"}
    assert artifacts["forecast.zip"] == _zip_bytes()
    assert artifacts["forecast.csv"].startswith(b"\xef\xbb\xbfFecha;Kc;ET0;ETc;Pe;ETc - Pe")


def test_nonempty_validation_error_stops_before_calculation(tmp_path):
    config = Config(
        tmp_path,
        "https://example.invalid/siarweb",
        1,
        1,
        1,
        (),
        (Station("IB04", "Son Ferriol", "4_7", EXPECTED_FORM_PARAMS),),
    )
    session = requests.Session()
    calls = []

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append((method, url))
        if url.endswith("/necesidadesHidricas/inicio"):
            return _response(
                url,
                (
                    f'<meta name="_csrf" content="{META_CSRF}">'
                    f'<input name="_csrf" value="{HIDDEN_CSRF}">'
                ).encode(),
            )
        if url.endswith("/necesidadesHidricasRest/validarForm"):
            return _response(url, b'{"error":"invalid"}', "application/json")
        raise AssertionError("Calculation must not run after validation rejects")

    session.request = fake_request
    client = SiarClient(config, session=session)
    with pytest.raises(CaptureError, match="validation rejected"):
        client.capture_station(config.stations[0])
    assert [url.rsplit("/", 1)[-1] for _, url in calls] == ["inicio", "validarForm"]


def test_export_stream_failure_preserves_safe_calculation_html(tmp_path):
    config = Config(
        tmp_path,
        "https://example.invalid/siarweb",
        1,
        1,
        1,
        (),
        (Station("IB04", "Son Ferriol", "4_7", EXPECTED_FORM_PARAMS),),
    )
    session = requests.Session()

    def fake_request(method, url, timeout=None, **kwargs):
        if url.endswith("/necesidadesHidricas/inicio"):
            return _response(
                url,
                (
                    f'<meta name="_csrf" content="{META_CSRF}">'
                    f'<input name="_csrf" value="{HIDDEN_CSRF}">'
                ).encode(),
            )
        if url.endswith("/necesidadesHidricasRest/validarForm"):
            return _response(url, b"", None)
        if url.endswith("/necesidadesHidricas/calculo"):
            return _response(url, b"<html><table id='tablaDatos'></table></html>")
        if url.endswith("/necesidadesHidricas/exportCSV"):
            response = _response(url, b"", "application/zip")

            def failed_stream(chunk_size):
                raise requests.ConnectionError("synthetic stream interruption")

            response.iter_content = failed_stream
            return response
        raise AssertionError(url)

    session.request = fake_request
    client = SiarClient(config, session=session)
    with pytest.raises(CaptureError, match="export download failed") as caught:
        client.capture_station(config.stations[0])
    assert caught.value.artifacts == {"result.html": b"<html><table id='tablaDatos'></table></html>"}
    assert "export_get" in caught.value.http
