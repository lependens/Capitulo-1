from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from html import unescape
import json
from typing import Any, Callable, Mapping

from .config import Target
from .ids import iso_utc, utc_now
from .parsing import decode_entity, header_value, parse_json_bytes, summarize_payload
from .secret_safety import contains_secret, is_safe_aemet_data_url, safe_url
from .transport import HttpResponse, http_get


ENDPOINT = "https://opendata.aemet.es/opendata/api/prediccion/especifica/municipio/diaria/{municipality_id}"


@dataclass
class CaptureOutcome:
    http: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, bytes] = field(default_factory=dict)
    issue_time_raw: str | None = None
    valid_times_raw: list[str] = field(default_factory=list)
    periods_raw: list[str] = field(default_factory=list)
    payload_structure: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    complete: bool = False


class AemetClient:
    def __init__(
        self,
        api_key: str | None,
        timeout_seconds: float = 30,
        fetcher: Callable[[str, Mapping[str, str], float], HttpResponse] = http_get,
        clock: Callable[[], datetime] = utc_now,
        api_key_expires_at: str | None = None,
    ):
        self._api_key = api_key
        self._api_key_expires_at = api_key_expires_at
        self._timeout_seconds = timeout_seconds
        self._fetcher = fetcher
        self._clock = clock

    def _http_record(self, url: str, response: HttpResponse) -> dict[str, Any]:
        content_type = header_value(dict(response.headers), "Content-Type")
        encoding = header_value(dict(response.headers), "Content-Encoding") or "identity"
        charset = None
        if content_type:
            from .parsing import declared_charset
            charset = declared_charset(content_type)
        safe, redacted = safe_url(url, self._api_key)
        return {
            "method": "GET",
            "url": safe,
            "url_redacted": redacted,
            "status_code": response.status_code,
            "content_type": content_type,
            "content_encoding": encoding,
            "charset": charset,
            "fetched_at_utc": iso_utc(self._clock()),
            "byte_length": len(response.body),
            "sha256": hashlib.sha256(response.body).hexdigest(),
        }

    def _request(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        try:
            return self._fetcher(url, headers, self._timeout_seconds)
        except Exception as exc:
            # Do not retain exception text or chained transport data.
            kind = "timeout" if type(exc).__name__ in {"TimeoutError", "Timeout", "ReadTimeout"} else "network"
            return HttpResponse(None, {}, b"", kind)

    def _body_has_secret(self, response: HttpResponse) -> bool:
        if contains_secret(response.body, self._api_key or ""):
            return True
        try:
            decoded = decode_entity(response.body, dict(response.headers))
        except Exception:
            # Fail closed when an encoded response cannot be inspected safely.
            return True
        if contains_secret(decoded, self._api_key or ""):
            return True
        content_type = header_value(dict(response.headers), "Content-Type")
        from .parsing import declared_charset
        try:
            text = decoded.decode(declared_charset(content_type) or "utf-8")
        except Exception:
            return False
        if contains_secret(unescape(text).encode("utf-8"), self._api_key or ""):
            return True
        try:
            parsed = parse_json_bytes(response.body, dict(response.headers))
        except Exception:
            return False
        return contains_secret(json.dumps(parsed, ensure_ascii=False).encode("utf-8"), self._api_key or "")

    def capture(self, target: Target) -> CaptureOutcome:
        outcome = CaptureOutcome()
        if not self._api_key:
            outcome.errors.append("AEMET_API_KEY missing from runtime environment")
            return outcome
        if self._api_key_expires_at is not None:
            try:
                raw_expiry = self._api_key_expires_at.strip()
                expiry = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
                if expiry.tzinfo is None or expiry.utcoffset() is None:
                    raise ValueError("timezone required")
            except (AttributeError, OverflowError, TypeError, ValueError):
                outcome.errors.append("AEMET_API_KEY_EXPIRES_AT must be a timezone-aware ISO-8601 timestamp")
                return outcome
            now = self._clock()
            if now.tzinfo is None or now.utcoffset() is None:
                outcome.errors.append("AEMET_API_KEY_EXPIRES_AT cannot be checked against a naive runtime clock")
                return outcome
            if expiry <= now:
                outcome.errors.append("AEMET API key renewal required")
                return outcome

        wrapper_url = ENDPOINT.format(municipality_id=target.municipality_id)
        wrapper_response = self._request(wrapper_url, {"Accept": "application/json", "api_key": self._api_key})
        outcome.http["wrapper"] = self._http_record(wrapper_url, wrapper_response)
        if wrapper_response.status_code != 200:
            status = wrapper_response.status_code
            if status in (401, 403):
                outcome.errors.append(f"wrapper request unauthorized (HTTP {status})")
            elif status is None:
                outcome.errors.append(f"wrapper request transport failure ({wrapper_response.transport_error or 'network'})")
            else:
                outcome.errors.append(f"wrapper request returned HTTP {status}")
            return outcome
        if self._body_has_secret(wrapper_response):
            outcome.errors.append("wrapper response withheld by credential safety check")
            return outcome

        try:
            wrapper = parse_json_bytes(wrapper_response.body, dict(wrapper_response.headers))
        except Exception:
            # A successful but malformed response is useful as exact raw evidence.
            outcome.artifacts["wrapper.raw"] = wrapper_response.body
            outcome.errors.append("wrapper body is not parseable JSON")
            return outcome

        if not isinstance(wrapper, dict):
            outcome.artifacts["wrapper.raw"] = wrapper_response.body
            outcome.errors.append("wrapper JSON root is not an object")
            return outcome
        if wrapper.get("estado") != 200:
            outcome.artifacts["wrapper.raw"] = wrapper_response.body
            outcome.errors.append("wrapper estado is not 200")
            return outcome
        datos_url = wrapper.get("datos")
        if not isinstance(datos_url, str) or not datos_url:
            outcome.artifacts["wrapper.raw"] = wrapper_response.body
            outcome.errors.append("wrapper does not contain a datos URL")
            return outcome
        if not is_safe_aemet_data_url(datos_url) or contains_secret(datos_url.encode("utf-8"), self._api_key):
            outcome.errors.append("datos URL rejected by safe URL policy")
            return outcome

        # Save the original wrapper entity only after it has passed secret and URL checks.
        outcome.artifacts["wrapper.raw"] = wrapper_response.body
        payload_response = self._request(datos_url, {"Accept": "application/json, text/plain;q=0.9, */*;q=0.8"})
        outcome.http["payload"] = self._http_record(datos_url, payload_response)
        if payload_response.status_code is not None and self._body_has_secret(payload_response):
            outcome.errors.append("payload response withheld by credential safety check")
            return outcome
        if payload_response.status_code is not None:
            outcome.artifacts["payload.raw"] = payload_response.body
        if payload_response.status_code != 200:
            status = payload_response.status_code
            if status is None:
                outcome.errors.append(f"payload request transport failure ({payload_response.transport_error or 'network'})")
            else:
                outcome.errors.append(f"payload request returned HTTP {status}")
            return outcome

        try:
            # Parse JSON regardless of MIME type; raw bytes are already retained.
            payload = parse_json_bytes(payload_response.body, dict(payload_response.headers))
            summary = summarize_payload(payload, target.municipality_id)
        except Exception:
            outcome.errors.append("payload JSON or forecast structure is invalid")
            return outcome

        outcome.issue_time_raw = summary["issue_time_raw"]
        outcome.valid_times_raw = summary["valid_times_raw"]
        outcome.periods_raw = summary["periods_raw"]
        outcome.payload_structure = summary["payload_structure"]
        outcome.complete = True
        return outcome
