import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .hashing import write_checksums


def _write_bytes_synced(path: Path, content: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _write_bytes_synced(path, payload)


def _fsync_dir(path: Path) -> None:
    # Directory fsync is required for durable rename semantics on the Linux
    # production target. Windows does not support opening directories this way.
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_capture(run_temp: Path, station_code: str, capture_id: str, metadata: dict[str, Any], artifacts: dict[str, bytes]) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", station_code):
        raise ValueError("station_code must be a safe directory name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", capture_id):
        raise ValueError("capture_id must be a safe directory name")
    station_dir = run_temp / station_code
    station_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{capture_id}.", dir=station_dir))
    try:
        for name, content in artifacts.items():
            _write_bytes_synced(staging / name, content)
        _write_json(staging / "metadata.json", metadata)
        write_checksums(staging, [*artifacts, "metadata.json"])
        _fsync_dir(staging)
        target = station_dir / capture_id
        if target.exists():
            raise FileExistsError(f"Append-only target already exists: {target}")
        os.replace(staging, target)
        _fsync_dir(station_dir)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def begin_run(data_root: Path, started_at_utc: str, run_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id must be a safe directory name")
    date_part = started_at_utc[:10].split("-")
    if len(date_part) != 3 or not all(part.isdigit() for part in date_part):
        raise ValueError("started_at_utc must begin with YYYY-MM-DD")
    final_parent = data_root / "runs" / date_part[0] / date_part[1] / date_part[2]
    final_parent.mkdir(parents=True, exist_ok=True)
    temp_root = data_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    run_temp = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=temp_root))
    return run_temp, final_parent / run_id


def publish_run(run_temp: Path, final_path: Path, run_metadata: dict[str, Any]) -> Path:
    _write_json(run_temp / "run.json", run_metadata)
    root_manifest = run_temp / "SHA256SUMS"
    files = []
    for path in sorted(run_temp.rglob("*")):
        if path.is_file() and path != root_manifest:
            files.append(path.relative_to(run_temp).as_posix())
    write_checksums(run_temp, files)
    _fsync_dir(run_temp)
    if final_path.exists():
        raise FileExistsError(f"Append-only run target already exists: {final_path}")
    os.replace(run_temp, final_path)
    _fsync_dir(final_path.parent)
    return final_path
