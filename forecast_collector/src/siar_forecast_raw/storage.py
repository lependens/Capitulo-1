import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .hashing import write_checksums


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def publish_capture(run_temp: Path, station_code: str, capture_id: str, metadata: dict[str, Any], artifacts: dict[str, bytes]) -> Path:
    station_dir = run_temp / station_code
    station_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{capture_id}.", dir=station_dir))
    try:
        for name, content in artifacts.items():
            (staging / name).write_bytes(content)
        _write_json(staging / "metadata.json", metadata)
        write_checksums(staging, [*artifacts, "metadata.json"])
        target = station_dir / capture_id
        if target.exists():
            raise FileExistsError(f"Append-only target already exists: {target}")
        os.replace(staging, target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def begin_run(data_root: Path, started_at_utc: str, run_id: str) -> tuple[Path, Path]:
    date_part = started_at_utc[:10].split("-")
    final_parent = data_root / "runs" / date_part[0] / date_part[1] / date_part[2]
    final_parent.mkdir(parents=True, exist_ok=True)
    temp_root = data_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    run_temp = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=temp_root))
    return run_temp, final_parent / run_id


def publish_run(run_temp: Path, final_path: Path, run_metadata: dict[str, Any]) -> Path:
    _write_json(run_temp / "run.json", run_metadata)
    files = []
    for path in sorted(run_temp.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(path.relative_to(run_temp).as_posix())
    write_checksums(run_temp, files)
    if final_path.exists():
        raise FileExistsError(f"Append-only run target already exists: {final_path}")
    os.replace(run_temp, final_path)
    return final_path
