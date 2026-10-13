from email.message import Message
import codecs
import gzip
import json
import zlib
from typing import Any


VARIABLE_FIELDS = (
    "temperatura",
    "humedadRelativa",
    "viento",
    "rachaMax",
    "probPrecipitacion",
    "estadoCielo",
    "sensTermica",
    "cotaNieveProv",
    "uvMax",
)


class PayloadContractError(ValueError):
    pass


def header_value(headers: dict[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.casefold() == name.casefold():
            return value
    return None


def declared_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    try:
        message = Message()
        message["content-type"] = content_type
        charset = message.get_content_charset()
        if charset is not None:
            codecs.lookup(charset)
        return charset
    except (TypeError, ValueError):
        return None
    except LookupError:
        return None


def _entity_bytes(body: bytes, content_encoding: str | None) -> bytes:
    encoding = (content_encoding or "identity").strip().casefold()
    if encoding in ("", "identity"):
        return body
    if encoding in ("gzip", "x-gzip"):
        return gzip.decompress(body)
    if encoding == "deflate":
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -zlib.MAX_WBITS)
    raise ValueError("unsupported content encoding")


def decode_entity(body: bytes, headers: dict[str, str]) -> bytes:
    return _entity_bytes(body, header_value(headers, "Content-Encoding"))


def parse_json_bytes(body: bytes, headers: dict[str, str]) -> Any:
    entity = decode_entity(body, headers)
    content_type = header_value(headers, "Content-Type")
    charset = declared_charset(content_type) or "utf-8"
    text = entity.decode(charset)
    return json.loads(text.lstrip("\ufeff"))


def _counts(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        child = [_counts(item) for item in value.values()]
    elif isinstance(value, list):
        child = [_counts(item) for item in value]
    else:
        return int(value is None), int(value == "")
    return sum(item[0] for item in child), sum(item[1] for item in child)


def _collect_periods(value: Any, output: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "periodo" and isinstance(item, str):
                output.append(item)
            else:
                _collect_periods(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_periods(item, output)


def summarize_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, list) or not payload or not all(isinstance(root, dict) for root in payload):
        raise PayloadContractError("payload root must be a non-empty list of objects")

    roots = payload
    root = roots[0]
    issue_time_raw = root.get("elaborado")
    if not isinstance(issue_time_raw, str) or not issue_time_raw.strip():
        raise PayloadContractError("payload elaborado field is missing, empty or not a string")

    dates: list[str] = []
    field_names_by_day: list[list[str]] = []
    periods: list[str] = []
    days: list[dict[str, Any]] = []
    for item in roots:
        prediction = item.get("prediccion")
        day_list = prediction.get("dia") if isinstance(prediction, dict) else None
        if not isinstance(day_list, list):
            raise PayloadContractError("payload prediccion.dia field is missing or not a list")
        for day in day_list:
            if not isinstance(day, dict) or not isinstance(day.get("fecha"), str) or not day["fecha"].strip():
                raise PayloadContractError("forecast day is missing a literal fecha string")
            days.append(day)
            dates.append(day["fecha"])
            field_names_by_day.append(list(day.keys()))
            _collect_periods(day, periods)
    if not days:
        raise PayloadContractError("payload contains no forecast days")

    observed = [name for name in VARIABLE_FIELDS if any(name in day for day in days)]
    json_null_count, empty_string_count = _counts(payload)
    return {
        "issue_time_raw": issue_time_raw,
        "valid_times_raw": dates,
        "periods_raw": periods,
        "payload_structure": {
            "format": "JSON",
            "top_level_type": "list",
            "top_level_count": len(roots),
            "root_fields": list(root.keys()),
            "day_count": len(days),
            "field_names_by_day": field_names_by_day,
            "variables_observed": observed,
            "json_null_count": json_null_count,
            "empty_string_count": empty_string_count,
        },
    }
