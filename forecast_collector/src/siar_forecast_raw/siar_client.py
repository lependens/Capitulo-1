"""HTTP transport for the SIAR public forecast flow; session state stays in RAM."""
from datetime import datetime, timezone
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from .config import Config, Station
from .errors import CaptureError
from .ids import iso_utc
from .secret_scan import assert_html_safe
from .zip_safe import MAX_ARCHIVE_BYTES, extract_single_csv


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _http_record(method: str, url: str, started: datetime, response: requests.Response | None, error: str | None = None) -> dict[str, Any]:
    ended = _utc()
    record: dict[str, Any] = {"method": method, "url": url, "started_at_utc": iso_utc(started), "finished_at_utc": iso_utc(ended), "duration_ms": round((ended-started).total_seconds()*1000)}
    if response is not None:
        content = response._content
        if content is not False:
            content_length = len(content)
        else:
            try:
                content_length = int(response.headers["Content-Length"])
            except (KeyError, TypeError, ValueError):
                content_length = None
        record.update(status_code=response.status_code, content_type=response.headers.get("Content-Type"), content_length=content_length)
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
        parsed_base = urlsplit(self.base)
        self.origin = f"{parsed_base.scheme}://{parsed_base.netloc}"

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

    def _csrf_values(self, html: bytes) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        hidden = soup.find("input", attrs={"name": "_csrf"})
        meta = soup.find("meta", attrs={"name": "_csrf"})
        body_value = str(hidden.get("value") or "") if hidden else ""
        header_value = str(meta.get("content") or "") if meta else ""
        if not body_value or not header_value:
            raise CaptureError("Could not find both SIAR CSRF values on the start page")
        self.known_values.update((body_value, header_value))
        return body_value, header_value

    @staticmethod
    def _validation_accepted(body: bytes) -> bool:
        if not body.strip():
            return True
        try:
            import json
            response = json.loads(body)
        except (UnicodeDecodeError, ValueError):
            return False
        if response is None:
            return False
        if isinstance(response, (dict, list, str)):
            return len(response) == 0
        # Object.keys on non-null JSON number/boolean values has no keys.
        return True

    @staticmethod
    def _read_limited(response: requests.Response, max_bytes: int = MAX_ARCHIVE_BYTES) -> bytes:
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            if len(content) + len(chunk) > max_bytes:
                raise CaptureError("SIAR export exceeded the configured ZIP size limit")
            content.extend(chunk)
        return bytes(content)

    def capture_station(self, station: Station) -> tuple[dict[str, Any], dict[str, bytes]]:
        http: dict[str, Any] = {}
        artifacts: dict[str, bytes] = {}
        start, http["initial_get"] = self._request("GET", "/necesidadesHidricas/inicio")
        if start is None:
            raise CaptureError("SIAR start page request failed", http=http)
        try:
            csrf_body, csrf_header = self._csrf_values(start.content)
        except CaptureError as exc:
            exc.http = http
            raise
        form = {**station.form_params, "idEstacion": station.siar_station_form_id, "_csrf": csrf_body}
        browser_headers = {"Origin": self.origin, "Referer": start.url}
        validation, http["validation_post"] = self._request(
            "POST",
            "/necesidadesHidricasRest/validarForm",
            data=form,
            headers={
                **browser_headers,
                "X-Requested-With": "XMLHttpRequest",
                "X-XSRF-TOKEN": csrf_header,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "*/*",
            },
            allow_redirects=False,
        )
        if validation is None:
            raise CaptureError("SIAR form validation failed", http=http)
        if not self._validation_accepted(validation.content):
            raise CaptureError("SIAR form validation rejected the submitted data", http=http)
        calculation, http["calculation_post"] = self._request(
            "POST",
            "/necesidadesHidricas/calculo",
            data=form,
            headers={
                **browser_headers,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        if calculation is None:
            raise CaptureError("SIAR calculation request failed", http=http)
        try:
            assert_html_safe(calculation.content, self.known_values)
        except CaptureError as exc:
            exc.http = http
            raise
        artifacts["result.html"] = calculation.content
        export, http["export_get"] = self._request(
            "GET",
            "/necesidadesHidricas/exportCSV",
            headers={"Referer": calculation.url, "Accept": "*/*"},
            stream=True,
        )
        if export is None:
            raise CaptureError("SIAR CSV export request failed", artifacts=artifacts, http=http)
        try:
            zip_bytes = self._read_limited(export)
            http["export_get"]["content_length"] = len(zip_bytes)
            filename, csv_bytes = extract_single_csv(zip_bytes, max_archive_bytes=MAX_ARCHIVE_BYTES)
        except requests.RequestException as exc:
            raise CaptureError(
                f"SIAR export download failed: {type(exc).__name__}",
                artifacts=artifacts,
                http=http,
            ) from exc
        except CaptureError as exc:
            exc.artifacts = artifacts
            exc.http = http
            raise CaptureError(str(exc), artifacts=artifacts, http=http) from exc
        finally:
            export.close()
        artifacts["forecast.zip"] = zip_bytes
        artifacts["forecast.csv"] = csv_bytes
        return ({"station": {"code": station.code, "label": station.label, "siar_station_form_id": station.siar_station_form_id}, "http": http, "source": {"system": "SIAR", "product": "necesidadesHidricas", "source_issue_at": None}, "request": {**station.form_params, "idEstacion": station.siar_station_form_id}, "csv_source_filename": filename}, artifacts)
