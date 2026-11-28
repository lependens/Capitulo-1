#!/opt/siar-forecast/venvs/e850827e24cff11ae46f51c9bc3e8a2f36b0d2da/bin/python
"""Idempotent, non-destructive SIAR Forecast Raw backup reconciler."""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

RAW_ROOT = Path("/srv/siar-forecast/raw/runs")
RCLONE = "/opt/siar-forecast/tools/rclone/1.75.2/rclone"
RCLONE_CONFIG = os.environ.get("SIAR_FORECAST_RCLONE_CONFIG", "")
REMOTE_ROOT = os.environ.get("SIAR_FORECAST_REMOTE_ROOT", "")
RUN_ID_RE = re.compile(r"^run_[A-Za-z0-9._-]+$")
DATE_PART_RE = re.compile(r"^[0-9]{2,4}$")
EXPECTED_CSV_HEADER = ["Fecha", "Kc", "ET0", "ETc", "Pe", "ETc - Pe"]
SECRET_KEYS = {
    "cookie", "set-cookie", "csrf", "csrf_token", "xsrf_token",
    "authorization", "authorization_header", "bearer", "access_token",
    "refresh_token", "client_secret", "password", "password2",
    "session_cookie", "session_token", "id_token",
}

class ReconcileError(Exception):
    pass

def emit(message):
    print(message, flush=True)

def run_checked(argv, label, timeout=600):
    try:
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise ReconcileError(label + "_timeout") from None
    except OSError as exc:
        raise ReconcileError(label + "_unavailable_" + type(exc).__name__) from None
    if result.returncode != 0:
        raise ReconcileError(label + "_failed_rc_" + str(result.returncode))
    return result.stdout

def rclone(args, label, timeout=600):
    return run_checked([RCLONE, "--config", RCLONE_CONFIG, "--contimeout", "20s",
                        "--timeout", "60s", "--retries", "2", "--low-level-retries", "2"]
                       + args, label, timeout)

def reject_secret_fields(value, context):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9_-]", "", str(key).casefold())
            if normalized in SECRET_KEYS:
                raise ReconcileError(context + "_sensitive_field_present")
            reject_secret_fields(child, context)
    elif isinstance(value, list):
        for child in value:
            reject_secret_fields(child, context)

def safe_relative(name):
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or ".." in rel.parts or "." in rel.parts:
        raise ReconcileError("unsafe_manifest_path")
    return rel

def tree_files(root):
    if root.is_symlink() or not root.is_dir():
        raise ReconcileError("run_directory_missing_or_symlink")
    result = {}
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for dirname in list(dirs):
            if (current_path / dirname).is_symlink():
                raise ReconcileError("symlink_in_run_tree")
        for filename in files:
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise ReconcileError("nonregular_file_in_run_tree")
            rel = path.relative_to(root).as_posix()
            result[rel] = (path, path.stat().st_size)
    return result

def verify_manifest(root, files, manifest_name):
    manifest_path = root / manifest_name
    if manifest_name not in files or manifest_path.is_symlink():
        raise ReconcileError("manifest_missing")
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ReconcileError("manifest_unreadable") from None
    expected = {}
    for line in lines:
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not match:
            raise ReconcileError("manifest_format_invalid")
        digest, name = match.groups()
        rel = safe_relative(name)
        key = rel.as_posix()
        if key == manifest_name or key in expected:
            raise ReconcileError("manifest_duplicate_or_self_reference")
        expected[key] = digest.lower()
    actual = {name for name in files if name != manifest_name}
    if set(expected) != actual:
        raise ReconcileError("manifest_file_set_mismatch")
    for name, digest in expected.items():
        path = root.joinpath(*PurePosixPath(name).parts)
        h = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
        except OSError:
            raise ReconcileError("manifest_file_unreadable") from None
        if h.hexdigest() != digest:
            raise ReconcileError("manifest_hash_mismatch")

def validate_csv_and_zip(capture_dir):
    def validate_csv_bytes(raw_csv):
        try:
            text = raw_csv.decode("utf-8-sig")
            rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=";"))
        except (UnicodeError, csv.Error):
            raise ReconcileError("csv_unreadable") from None
        if not rows or rows[0] != EXPECTED_CSV_HEADER:
            raise ReconcileError("csv_schema_invalid")
        for row in rows[1:]:
            if len(row) != len(EXPECTED_CSV_HEADER):
                raise ReconcileError("csv_row_width_invalid")
            date_value = row[0].strip()
            parsed = False
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
                try:
                    datetime.strptime(date_value, fmt)
                    parsed = True
                    break
                except ValueError:
                    pass
            if not parsed:
                raise ReconcileError("csv_date_invalid")
        return raw_csv

    csv_path = capture_dir / "forecast.csv"
    zip_path = capture_dir / "forecast.zip"
    raw_csv = None
    if csv_path.is_file():
        try:
            raw_csv = validate_csv_bytes(csv_path.read_bytes())
        except OSError:
            raise ReconcileError("csv_unreadable") from None
    if zip_path.is_file():
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                members = archive.infolist()
                if len(members) != 1:
                    raise ReconcileError("zip_member_count_invalid")
                member = members[0]
                member_path = PurePosixPath(member.filename)
                if (member.is_dir() or member_path.is_absolute() or len(member_path.parts) != 1
                        or ".." in member_path.parts or member_path.suffix.casefold() != ".csv"):
                    raise ReconcileError("zip_member_name_invalid")
                if archive.testzip() is not None:
                    raise ReconcileError("zip_crc_failed")
                zipped_csv = validate_csv_bytes(archive.read(member))
        except (OSError, zipfile.BadZipFile, RuntimeError):
            raise ReconcileError("zip_unreadable_or_invalid") from None
        if raw_csv is not None and zipped_csv != raw_csv:
            raise ReconcileError("zip_csv_mismatch")

def validate_run(run_dir):
    run_id = run_dir.name
    if not RUN_ID_RE.fullmatch(run_id):
        raise ReconcileError("invalid_run_id")
    files = tree_files(run_dir)
    verify_manifest(run_dir, files, "SHA256SUMS")
    run_json_path = run_dir / "run.json"
    if "run.json" not in files:
        raise ReconcileError("run_json_missing")
    try:
        run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ReconcileError("run_json_invalid") from None
    reject_secret_fields(run_data, "run_json")
    if run_data.get("run_id") != run_id or run_data.get("status") not in ("complete", "partial", "failed"):
        raise ReconcileError("run_status_or_id_invalid")
    captures = run_data.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ReconcileError("captures_missing")
    if run_data.get("status") == "complete" and any(not isinstance(c, dict) or c.get("status") != "complete" for c in captures):
        raise ReconcileError("complete_run_capture_status_invalid")
    for capture in captures:
        if not isinstance(capture, dict) or capture.get("status") not in ("complete", "partial", "failed"):
            raise ReconcileError("capture_status_invalid")
        capture_status = capture.get("status")
        station = capture.get("station_code")
        capture_id = capture.get("capture_id")
        if not isinstance(station, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", station):
            raise ReconcileError("station_code_invalid")
        if not isinstance(capture_id, str) or not re.fullmatch(r"cap_[A-Za-z0-9._-]+", capture_id):
            raise ReconcileError("capture_id_invalid")
        capture_dir = run_dir / station / capture_id
        capture_files = tree_files(capture_dir)
        verify_manifest(capture_dir, capture_files, "SHA256SUMS")
        for required in ("metadata.json",):
            if required not in capture_files:
                raise ReconcileError("capture_required_file_missing")
        try:
            metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ReconcileError("capture_metadata_invalid") from None
        reject_secret_fields(metadata, "capture_metadata")
        if metadata.get("run_id") != run_id or metadata.get("capture_id") != capture_id:
            raise ReconcileError("capture_metadata_identity_invalid")
        metadata_capture = metadata.get("capture")
        if not isinstance(metadata_capture, dict) or metadata_capture.get("status") != capture_status:
            raise ReconcileError("capture_metadata_status_invalid")
        artifacts = metadata.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ReconcileError("capture_artifacts_invalid")
        has_csv = "forecast.csv" in capture_files
        has_zip = "forecast.zip" in capture_files
        if ("csv" in artifacts) != has_csv or ("zip" in artifacts) != has_zip:
            raise ReconcileError("capture_artifact_declaration_mismatch")
        if capture_status == "complete" and not (has_csv and has_zip):
            raise ReconcileError("complete_capture_artifact_missing")
        html_state = metadata.get("html_persistence")
        if not isinstance(html_state, dict) or html_state.get("status") not in ("omitted_sensitive", "not_available"):
            raise ReconcileError("html_persistence_policy_invalid")
        if (capture_dir / "result.html").exists():
            raise ReconcileError("sensitive_result_html_present")
        if has_csv or has_zip:
            validate_csv_and_zip(capture_dir)
    if "result.html" in files or any(name.casefold().endswith("/result.html") for name in files):
        raise ReconcileError("sensitive_result_html_present")
    return files, sum(size for _, size in files.values())

def iter_runs():
    if RAW_ROOT.is_symlink() or not RAW_ROOT.is_dir():
        raise ReconcileError("raw_runs_root_unavailable")
    found = []
    for year in sorted(RAW_ROOT.iterdir()):
        if year.is_symlink() or not year.is_dir() or not re.fullmatch(r"[0-9]{4}", year.name):
            continue
        for month in sorted(year.iterdir()):
            if month.is_symlink() or not month.is_dir() or not re.fullmatch(r"[0-9]{2}", month.name):
                continue
            for day in sorted(month.iterdir()):
                if day.is_symlink() or not day.is_dir() or not re.fullmatch(r"[0-9]{2}", day.name):
                    continue
                for run_dir in sorted(day.iterdir()):
                    if run_dir.is_dir() and not run_dir.is_symlink() and run_dir.name.startswith("run_"):
                        found.append(run_dir)
    return found

def remote_file_map(remote_dir):
    raw = rclone(["lsjson", "--recursive", "--files-only", remote_dir], "remote_listing")
    try:
        entries = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ReconcileError("remote_listing_invalid") from None
    if not isinstance(entries, list):
        raise ReconcileError("remote_listing_invalid")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("IsDir"):
            continue
        name = entry.get("Path")
        size = entry.get("Size")
        if not isinstance(name, str) or not isinstance(size, int):
            raise ReconcileError("remote_listing_entry_invalid")
        name = PurePosixPath(name).as_posix().lstrip("/")
        if not name or ".." in PurePosixPath(name).parts or name in result:
            raise ReconcileError("remote_listing_path_invalid")
        result[name] = size
    return result

def compare_remote_payload(run_dir, run_id, local_files, remote_dir, has_complete):
    expected = {name: size for name, (_, size) in local_files.items()}
    actual = remote_file_map(remote_dir)
    marker_size = actual.pop("COMPLETE", None)
    if has_complete and marker_size is None:
        raise ReconcileError("complete_marker_disappeared")
    if not has_complete and marker_size is not None:
        raise ReconcileError("unexpected_complete_marker")
    if actual != expected:
        raise ReconcileError("remote_plaintext_inventory_mismatch")
    run_checked([RCLONE, "--config", RCLONE_CONFIG, "--contimeout", "20s", "--timeout", "60s",
                 "--retries", "2", "--low-level-retries", "2", "check", str(run_dir), remote_dir,
                 "--download", "--one-way"], "remote_plaintext_check", 900)
    return len(expected), sum(expected.values())

def parse_marker(raw, run_id, expected_files, expected_bytes):
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        raise ReconcileError("complete_marker_not_utf8") from None
    values = {}
    lines = text.splitlines()
    if not lines or lines[0] != "BACKUP-FC-001":
        raise ReconcileError("complete_marker_invalid")
    for line in lines[1:]:
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if values.get("run_id") != run_id:
        raise ReconcileError("complete_marker_identity_invalid")
    modern = values.get("status") == "PASS" and values.get("COMPLETE") == "PASS"
    legacy = (values.get("backup_status") == "PASS" and
              values.get("copy_status") == "PASS" and
              values.get("restore_status") == "PASS")
    if not (modern or legacy):
        raise ReconcileError("complete_marker_status_invalid")
    if modern:
        files_value = values.get("payload_files")
        bytes_value = values.get("payload_logical_bytes")
    else:
        files_value = values.get("files")
        bytes_value = values.get("logical_bytes")
    if files_value != str(expected_files) or bytes_value != str(expected_bytes):
        raise ReconcileError("complete_marker_size_mismatch")

def current_version():
    output = run_checked([RCLONE, "version"], "rclone_version")
    first = output.decode("utf-8", "replace").splitlines()[0]
    match = re.search(r"v([0-9][A-Za-z0-9.+-]*)", first)
    return match.group(1) if match else "unknown"

def process_run(run_dir, remote_dirs, version):
    run_id = run_dir.name
    local_files, logical_bytes = validate_run(run_dir)
    run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    capture_records = run_data.get("captures", [])
    capture_statuses = ",".join(sorted({str(c.get("status", "unknown")) for c in capture_records if isinstance(c, dict)}))
    total_captures = len(capture_records)
    csv_count = sum(Path(name).name == "forecast.csv" for name in local_files)
    zip_count = sum(Path(name).name == "forecast.zip" for name in local_files)
    csv_status = "PASS" if csv_count else "NOT_PRESENT"
    zip_status = "PASS" if zip_count else "NOT_PRESENT"
    html_states = set()
    for capture in capture_records:
        if isinstance(capture, dict):
            metadata_path = run_dir / capture["station_code"] / capture["capture_id"] / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            html_state = metadata.get("html_persistence")
            if isinstance(html_state, dict):
                html_states.add(str(html_state.get("status", "unknown")))
    html_summary = ",".join(sorted(html_states)) if html_states else "unknown"
    source_issue = json.dumps(run_data.get("source_issue_at"), separators=(",", ":"))
    relative = run_dir.relative_to(RAW_ROOT).as_posix()
    remote_dir = REMOTE_ROOT + "/" + relative
    has_dir = relative in remote_dirs
    if has_dir:
        remote_files = remote_file_map(remote_dir)
        has_complete = "COMPLETE" in remote_files
        if has_complete:
            count, size = compare_remote_payload(run_dir, run_id, local_files, remote_dir, True)
            marker = rclone(["cat", remote_dir + "/COMPLETE"], "complete_marker_read")
            parse_marker(marker, run_id, count, size)
            emit("NOOP COMPLETE verified run_id=" + run_id + " files=" + str(count) + " bytes=" + str(size))
            return
    else:
        has_complete = False
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    emit("BACKUP begin run_id=" + run_id + " files=" + str(len(local_files)) + " bytes=" + str(logical_bytes))
    rclone(["copy", "--ignore-existing", str(run_dir), remote_dir], "copy", 900)
    count, size = compare_remote_payload(run_dir, run_id, local_files, remote_dir, False)
    marker_text = "\n".join([
        "BACKUP-FC-001",
        "run_id=" + run_id,
        "backup_started_utc=" + started,
        "backup_completed_utc=" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status=PASS",
        "provider_remote=forecast-offsite-provider",
        "crypt_remote=" + REMOTE_ROOT.split(":",1)[0],
        "rclone_version=" + version,
        "payload_files=" + str(count),
        "payload_logical_bytes=" + str(size),
        "copy_exit=0",
        "remote_plaintext_check=PASS; " + str(count) + " files; " + str(size) + " bytes; 0 differences",
        "run_status=" + str(run_data.get("status")),
        "capture_status=" + capture_statuses,
        "zip_crc=" + zip_status,
        "zip_present=" + str(zip_count) + "/" + str(total_captures),
        "csv=" + csv_status,
        "csv_present=" + str(csv_count) + "/" + str(total_captures),
        "html_persistence=" + html_summary,
        "result.html=absent",
        "source_issue_at=" + source_issue,
        "secrets_persisted=0",
        "COMPLETE=PASS",
        "",
    ])
    temp_base = Path(os.environ.get("TMPDIR", "/run/siar-forecast-backup"))
    if temp_base.is_symlink() or not temp_base.is_dir():
        raise ReconcileError("private_runtime_directory_unavailable")
    with tempfile.TemporaryDirectory(prefix="complete-", dir=str(temp_base)) as temporary:
        payload = Path(temporary)
        (payload / "COMPLETE").write_text(marker_text, encoding="utf-8")
        os.chmod(payload / "COMPLETE", 0o600)
        rclone(["copy", str(payload), remote_dir], "complete_publish", 300)
    published = rclone(["cat", remote_dir + "/COMPLETE"], "complete_readback")
    if published != marker_text.encode("utf-8"):
        raise ReconcileError("complete_readback_mismatch")
    emit("PASS COMPLETE published run_id=" + run_id + " files=" + str(count) + " bytes=" + str(size))

def main():
    parser = argparse.ArgumentParser(description="Reconcile complete Forecast Raw runs to encrypted off-site storage")
    parser.add_argument("--only-run-id", help="limit operation to one exact published run ID")
    args = parser.parse_args()
    if not RCLONE_CONFIG or not REMOTE_ROOT:
        emit("ERROR backup_configuration_missing")
        return 2
    if args.only_run_id and not RUN_ID_RE.fullmatch(args.only_run_id):
        emit("ERROR invalid_run_id")
        return 2
    try:
        runs = iter_runs()
        if args.only_run_id:
            runs = [path for path in runs if path.name == args.only_run_id]
            if len(runs) != 1:
                raise ReconcileError("requested_run_not_unique_or_missing")
        if not runs:
            emit("PASS no_published_runs")
            return 0
        dirs_raw = rclone(["lsf", "--recursive", "--dirs-only", REMOTE_ROOT], "remote_directory_listing")
        remote_dirs = {line.strip().strip("/") for line in dirs_raw.decode("utf-8").splitlines() if line.strip()}
        version = current_version()
        for run_dir in runs:
            process_run(run_dir, remote_dirs, version)
        return 0
    except ReconcileError as exc:
        emit("ERROR " + str(exc))
        return 1
    except Exception as exc:
        emit("ERROR unexpected_" + type(exc).__name__)
        return 1

if __name__ == "__main__":
    sys.exit(main())
