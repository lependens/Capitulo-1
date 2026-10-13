import base64
import re
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit


def secret_variants(secret: str) -> set[str]:
    if not secret:
        return set()
    encoded = secret.encode("utf-8")
    b64 = base64.b64encode(encoded).decode("ascii")
    b64_url = base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")
    values = {
        secret,
        quote(secret, safe=""),
        quote_plus(secret, safe=""),
        b64,
        b64.rstrip("="),
        b64_url,
    }
    return {item for item in values if item}


def contains_secret(data: bytes, secret: str) -> bool:
    lowered = data.lower()
    return any(
        value.encode("utf-8") in data or value.encode("utf-8").lower() in lowered
        for value in secret_variants(secret)
    )


def redact_string(value: str, secret: str | None) -> str:
    if not secret:
        return value
    for variant in sorted(secret_variants(secret), key=len, reverse=True):
        value = value.replace(variant, "[REDACTED]")
    return value


def redact_metadata(value, secret: str | None):
    if isinstance(value, str):
        return redact_string(value, secret)
    if isinstance(value, list):
        return [redact_metadata(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: redact_metadata(item, secret) for key, item in value.items()}
    return value


def safe_url(url: str, secret: str | None = None) -> tuple[str, bool]:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if not parsed.scheme or not host:
            return "[invalid-url]", True
        netloc = host
        if parsed.port and parsed.port not in (80, 443):
            netloc = f"{netloc}:{parsed.port}"
        path = redact_string(parsed.path, secret)
        query_redacted = bool(parsed.query)
        fragment_redacted = bool(parsed.fragment)
        safe_query = "[REDACTED]" if query_redacted else ""
        safe_fragment = "[REDACTED]" if fragment_redacted else ""
        result = urlunsplit((parsed.scheme, netloc, path, safe_query, safe_fragment))
        return redact_string(result, secret), query_redacted or fragment_redacted or parsed.username is not None
    except ValueError:
        return "[invalid-url]", True


def is_safe_aemet_data_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.hostname == "opendata.aemet.es"
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
            and parsed.path.startswith("/opendata/sh/")
            and bool(re.fullmatch(r"[A-Fa-f0-9]{8,64}", parsed.path.removeprefix("/opendata/sh/")))
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False
