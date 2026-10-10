from typing import Any


def artifact_record(filename: str, data: bytes, source_filename: str | None = None) -> dict[str, Any]:
    from .hashing import sha256_bytes
    result = {"filename": filename, "size_bytes": len(data), "sha256": sha256_bytes(data)}
    if source_filename:
        result["source_filename"] = source_filename
    return result
