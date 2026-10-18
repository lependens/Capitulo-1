from dataclasses import dataclass
from http.client import HTTPMessage
import socket
import ssl
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class HttpResponse:
    status_code: int | None
    headers: Mapping[str, str]
    body: bytes
    transport_error: str | None = None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _header_map(headers: HTTPMessage | Mapping[str, str] | None) -> dict[str, str]:
    if headers is None:
        return {}
    return {str(key): str(value) for key, value in headers.items()}


def _transport_error(exc: BaseException) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, ssl.SSLError):
        return "tls"
    if isinstance(exc, URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout"
        if isinstance(reason, ssl.SSLError):
            return "tls"
    return "network"


def http_get(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    """Perform one GET without following redirects or persisting request headers."""
    request = Request(url, headers=dict(headers), method="GET")
    opener = build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return HttpResponse(response.status, _header_map(response.headers), response.read())
    except HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        return HttpResponse(exc.code, _header_map(exc.headers), body)
    except (URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as exc:
        # Deliberately discard the exception text: it can contain URLs or headers.
        return HttpResponse(None, {}, b"", _transport_error(exc))
