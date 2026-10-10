"""HTTP transport for the SIAR public forecast flow; session state stays in RAM."""
from datetime import datetime, timezone
import re
import time
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .config import Config, Station
from .errors import CaptureError
from .ids import iso_utc
from .secret_scan import assert_html_safe
from .zip_safe import extract_single_csv


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _http_record(method: str, url: str, started: datetime, response: requests.Response | None, error: str | None = None) -> dict[str, Any]:
    ended = _utc()
    record: dict[str, Any] = {"method": method, "url": url, "started_at_utc": iso_utc(started), "finished_at_utc": iso_utc(ended), "duration_ms": round((ended-started).total_seconds()*1000)}
    if response is not None:
        record.update(status_code=response.status_code, content_type=response.headers.get("Content-Type"), content_length=len(response.content))
        if response.headers.get("Content-Disposition"):
            record["content_disposition"] = response.headers["Content-Disposition"]
    if error:
        record["error"] = error
    return record


class SiarClient:
    def __init__(self, config: Config, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        self.timeout = (config.connect_timeout_seconds, config.read_timeout_seconds)
        self.known_values: set[str] = set()
        self.base = config.base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs: Any) -> tuple[requests.Response | None, dict[str, Any]]:
        url = urljoin(self.base + "/", path.lstrip("/"))
        last_error = None
        for attempt in range(self.config.max_attempts):
            started = _utc()
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                self.known_values.update(value for value in self.session.cookies.get_dict().values() if value)
                record = _http_record(method, url, started, response)
                response.raise_for_status()
                return response, record
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self.config.max_attempts:
                    delay_index = min(attempt, max(0, len(self.config.retry_backoff_seconds)-1))
                    if self.config.retry_backoff_seconds:
                        time.sleep(self.config.retry_backoff_seconds[delay_index])
                else:
                    return None, _http_record(method, url, started, getattr(exc, "response", None), last_error)
        return None, _http_record(method, url, _utc(), None, last_error)

    def _csrf(self, html: bytes) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for selector in (("input", "_csrf"), ("meta", "_csrf")):
            tag = soup.find(selector[0], attrs={"name": selector[1]})
            if tag and tag.get("value") or tag and tag.get("content"):
                value = str(tag.get("value") or tag.get("content"))
                self.known_values.add(value)
                return value
        match = re.search(rb"(?i)[\"']_csrf[\"']\s*:\s*[\"']([^\"']+)", html)
        if match:
            value = match.group(1).decode("utf-8", errors="replace")
            self.known_values.add(value)
            return value
        raise CaptureError("Could not find CSRF value on SIAR start page")

    def capture_station(self, station: Station) -> tuple[dict[str, Any], dict[str, bytes]]:
        http: dict[str, Any] = {}
        artifacts: dict[str, bytes] = {}
        start, http["initial_get"] = self._request("GET", "/necesidadesHidricas/inicio")
        if start is None:
            raise CaptureError("SIAR start page request failed", http=http)
        csrf = self._csrf(start.content)
        form = {**station.form_params, "idEstacion": station.siar_station_form_id, "_csrf": csrf}
        validation, http["validation_post"] = self._request("POST", "/necesidadesHidricasRest/validarForm", json=form)
        if validation is None:
            raise CaptureError("SIAR form validation failed", http=http)
        calculation, http["calculation_post"] = self._request("POST", "/necesidadesHidricas/calculo", data=form)
        if calculation is None:
            raise CaptureError("SIAR calculation request failed", http=http)
        try:
            assert_html_safe(calculation.content, self.known_values)
        except CaptureError as exc:
            exc.http = http
            raise
        artifacts["result.html"] = calculation.content
        export, http["export_get"] = self._request("GET", "/necesidadesHidricas/exportCSV")
        if export is None:
            raise CaptureError("SIAR CSV export request failed", artifacts=artifacts, http=http)
        try:
            filename, csv_bytes = extract_single_csv(export.content)
        except CaptureError as exc:
            raise CaptureError(str(exc), artifacts=artifacts, http=http) from exc
        artifacts["forecast.zip"] = export.content
        artifacts["forecast.csv"] = csv_bytes
        return ({"station": {"code": station.code, "label": station.label, "siar_station_form_id": station.siar_station_form_id}, "http": http, "source_issue_at": None, "source": {"system": "SIAR", "product": "necesidadesHidricas"}, "csv_source_filename": filename}, artifacts)
