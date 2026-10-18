import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, content: dict[str, Any]) -> None:
    data = (json.dumps(content, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _write_bytes(path, data)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checksums(directory: Path, files: list[str]) -> None:
    lines = [f"{sha256_file(directory / name)}  {name}" for name in sorted(files)]
    _write_bytes(directory / "SHA256SUMS", ("\n".join(lines) + "\n").encode("ascii"))


def verify_checksums(directory: Path) -> list[str]:
    failures: list[str] = []
    for line in (directory / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError:
            failures.append(line)
            continue
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            failures.append(name)
    return failures


def _set_shared_permissions(path: Path) -> None:
    if os.name == "nt":
        return
    for current, directories, files in os.walk(path, topdown=False):
        current_path = Path(current)
        for filename in files:
            os.chmod(current_path / filename, 0o660)
        for dirname in directories:
            os.chmod(current_path / dirname, 0o2770)
    os.chmod(path, 0o2770)


def create_run_stage(data_root: Path, started_at_utc: str, run_id: str) -> tuple[Path, Path]:
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id contains unsupported path characters")
    date_part = started_at_utc[:10].split("-")
    if len(date_part) != 3 or not all(part.isdigit() for part in date_part):
        raise ValueError("started_at_utc must begin with YYYY-MM-DD")
    final_parent = data_root / "runs" / date_part[0] / date_part[1] / date_part[2]
    final_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=final_parent))
    return stage, final_parent / run_id


def write_capture(stage: Path, station_code: str, capture_id: str, metadata: dict[str, Any], artifacts: dict[str, bytes]) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", station_code) or not _SAFE_ID.fullmatch(capture_id):
        raise ValueError("capture path contains unsupported characters")
    capture_dir = stage / station_code / capture_id
    capture_dir.mkdir(parents=True, exist_ok=False)
    for name, body in artifacts.items():
        if name not in {"wrapper.raw", "payload.raw"}:
            raise ValueError("unsupported raw artifact name")
        _write_bytes(capture_dir / name, body)
    _write_json(capture_dir / "metadata.json", metadata)
    write_checksums(capture_dir, [*artifacts.keys(), "metadata.json"])
    return capture_dir


def publish_run(stage: Path, final: Path, run_metadata: dict[str, Any]) -> Path:
    _write_json(stage / "run.json", run_metadata)
    files = [path.relative_to(stage).as_posix() for path in sorted(stage.rglob("*")) if path.is_file()]
    write_checksums(stage, files)
    for capture_manifest in stage.glob("*/*/SHA256SUMS"):
        if verify_checksums(capture_manifest.parent):
            raise OSError("capture checksum verification failed")
    if verify_checksums(stage):
        raise OSError("run checksum verification failed")
    if final.exists():
        raise FileExistsError("append-only run target already exists")
    _set_shared_permissions(stage)
    os.rename(stage, final)
    return final


def discard_stage(stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
