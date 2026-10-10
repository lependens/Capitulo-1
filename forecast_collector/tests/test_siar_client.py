import io
import zipfile

import requests

from siar_forecast_raw.config import Config, Station
from siar_forecast_raw.siar_client import SiarClient


def _zip_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("synthetic.csv", b"Fecha;ET0\n10/10/2026;2,30\n")
    return stream.getvalue()


def _response(url, body, content_type="text/html;charset=UTF-8"):
    response = requests.Response()
    response.status_code = 200
    response.url = url
    response._content = body
    response.headers["Content-Type"] = content_type
    return response


def test_validation_matches_observed_urlencoded_contract(tmp_path):
    config = Config(
        tmp_path,
        "https://example.invalid/siarweb",
        1,
        1,
        1,
        (),
        (Station("IB04", "Synthetic", "4_7", {"esCalculo": "0", "tipoCalculo": "2"}),),
    )
    session = requests.Session()
    calls = []

    def fake_request(method, url, timeout=None, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/necesidadesHidricas/inicio"):
            return _response(url, b'<html><input name="_csrf" value="synthetic-csrf-token-123"></html>')
        if url.endswith("/necesidadesHidricasRest/validarForm"):
            return _response(url, b"{}", "application/json")
        if url.endswith("/necesidadesHidricas/calculo"):
            return _response(url, b"<html><table><tr><td>forecast</td></tr></table></html>")
        if url.endswith("/necesidadesHidricas/exportCSV"):
            response = _response(url, _zip_bytes(), "application/zip")
            response.headers["Content-Disposition"] = 'attachment; filename="PronosticoNecesidadesNetas.zip"'
            return response
        raise AssertionError(url)

    session.request = fake_request
    client = SiarClient(config, session=session)
    details, artifacts = client.capture_station(config.stations[0])

    validation = next(call for call in calls if call[1].endswith("/necesidadesHidricasRest/validarForm"))
    assert "data" in validation[2]
    assert "json" not in validation[2]
    assert validation[2]["headers"]["X-XSRF-TOKEN"] == "synthetic-csrf-token-123"
    assert validation[2]["data"]["_csrf"] == "synthetic-csrf-token-123"
    calculation = next(call for call in calls if call[1].endswith("/necesidadesHidricas/calculo"))
    assert calculation[2]["data"]["idEstacion"] == "4_7"
    assert details["source_issue_at"] is None
    assert set(artifacts) == {"result.html", "forecast.zip", "forecast.csv"}
