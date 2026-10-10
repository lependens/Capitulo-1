"""Conservative detection of reusable session/authentication values in HTML."""
import re
from urllib.parse import unquote

from bs4 import BeautifulSoup

_ASSIGNMENT = re.compile(
    r"(?i)(?:_csrf(?:_header)?|csrf(?:token)?|xsrf(?:token)?|jsessionid|session(?:id|token)?|"
    r"access[_-]?token|auth(?:orization)?|api[_-]?key|password)\s*[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9._~+/=-]{8,})"
)
_COOKIE = re.compile(r"(?i)(?:set-cookie|cookie)\s*:\s*([^\r\n<]+)")
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_OPAQUE = re.compile(r"\b[A-Fa-f0-9]{32,}\b")


def _looks_reusable(value: str) -> bool:
    value = value.strip()
    return len(value) >= 8 and not re.fullmatch(r"(?i)(?:null|undefined|false|true|placeholder|test|example|none|0+)", value)


def find_reusable_session_values(html: bytes | str, known_values: set[str] | None = None) -> list[str]:
    text = html.decode("utf-8", errors="replace") if isinstance(html, bytes) else html
    decoded = unquote(text)
    soup = BeautifulSoup(decoded, "html.parser")
    found: set[str] = set()
    # Inspect values, not names alone: empty/non-reusable CSRF fields are safe.
    for tag in soup.find_all(["input", "meta"]):
        value = tag.get("value") if tag.name == "input" else tag.get("content")
        name = " ".join(str(tag.get(key, "")) for key in ("name", "id", "content"))
        if value and re.search(r"(?i)csrf|xsrf|session|token|auth|cookie", name) and _looks_reusable(str(value)):
            found.add("csrf/session field")
    for pattern, label in ((_ASSIGNMENT, "credential/session assignment"), (_COOKIE, "cookie header"), (_BEARER, "bearer token"), (_JWT, "JWT")):
        for match in pattern.finditer(decoded):
            value = match.group(1) if pattern in (_ASSIGNMENT, _COOKIE) else match.group(0)
            if _looks_reusable(value):
                found.add(label)
    if known_values:
        for value in known_values:
            if value and len(value) >= 8 and value in decoded:
                found.add("known live session value")
    # Long opaque values are blocked only in credential-bearing contexts, not
    # ordinary IDs, hashes, static assets, or generic page content.
    for match in re.finditer(r"(?i)(?:token|csrf|xsrf|session|auth)[^<>]{0,40}([A-Fa-f0-9]{32,})", decoded):
        if _looks_reusable(match.group(1)):
            found.add("opaque credential value")
    return sorted(found)


def assert_html_safe(html: bytes, known_values: set[str] | None = None) -> None:
    findings = find_reusable_session_values(html, known_values)
    if findings:
        from .errors import SensitiveHtmlError
        raise SensitiveHtmlError("HTML secret guard blocked persistence: " + ", ".join(findings))
