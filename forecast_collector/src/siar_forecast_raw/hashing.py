import hashlib
import os
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_text_synced(path: Path, text: str) -> None:
    with path.open("w", encoding="ascii", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def write_checksums(directory: Path, files: list[str]) -> None:
    lines = [f"{sha256_bytes((directory / name).read_bytes())}  {name}" for name in sorted(files)]
    _write_text_synced(directory / "SHA256SUMS", "\n".join(lines) + "\n")


def verify_checksums(directory: Path) -> list[str]:
    failures = []
    for line in (directory / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            failures.append(name)
    return failures
