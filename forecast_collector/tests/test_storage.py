import json
import os
import stat
from pathlib import Path
import pytest

from siar_forecast_raw.hashing import verify_checksums
from siar_forecast_raw.storage import begin_run, publish_capture, publish_run


def test_atomic_publication_and_checksums(tmp_path):
    started = "2026-10-10T08:00:00.000Z"
    temp, final = begin_run(tmp_path, started, "run_test")
    cap = {"capture": {"status": "failed"}}
    publish_capture(temp, "IB04", "cap_test", cap, {})
    published = publish_run(temp, final, {"status": "failed"})
    assert published.parts[-4:] == ("2026", "10", "10", "run_test")
    assert json.loads((published / "IB04/cap_test/metadata.json").read_text())["capture"]["status"] == "failed"
    assert verify_checksums(published) == []
    root_manifest = (published / "SHA256SUMS").read_text()
    assert "IB04/cap_test/SHA256SUMS" in root_manifest
    (published / "run.json").write_text("changed")
    assert "run.json" in verify_checksums(published)


def test_append_only_rejects_existing_run(tmp_path):
    temp, final = begin_run(tmp_path, "2026-10-10T08:00:00Z", "run_same")
    publish_run(temp, final, {"status": "failed"})
    temp2, _ = begin_run(tmp_path, "2026-10-10T08:00:00Z", "run_same")
    try:
        publish_run(temp2, final, {"status": "failed"})
        assert False, "expected append-only collision"
    except FileExistsError:
        pass


def test_identical_capture_payloads_are_both_preserved(tmp_path):
    temp, final = begin_run(tmp_path, "2026-10-10T08:00:00Z", "run_identical")
    metadata = {"capture": {"status": "complete"}}
    artifacts = {"result.html": b"<html>same</html>", "forecast.zip": b"same-zip", "forecast.csv": b"same-csv"}
    first = publish_capture(temp, "IB04", "cap_first", metadata, artifacts)
    second = publish_capture(temp, "IB04", "cap_second", metadata, artifacts)
    assert first != second
    assert (first / "result.html").read_bytes() == (second / "result.html").read_bytes()
    published = publish_run(temp, final, {"status": "complete"})
    assert (published / "IB04/cap_first").is_dir()
    assert (published / "IB04/cap_second").is_dir()


def test_capture_is_not_published_if_atomic_rename_fails(tmp_path, monkeypatch):
    import siar_forecast_raw.storage as storage

    run_temp = tmp_path / "run"
    run_temp.mkdir()
    real_replace = os.replace

    def fail_replace(src, dst):
        if "cap_atomic" in str(dst):
            raise OSError("synthetic rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError):
        publish_capture(run_temp, "IB04", "cap_atomic", {"capture": {"status": "failed"}}, {})
    assert not (run_temp / "IB04/cap_atomic").exists()
    assert not list((run_temp / "IB04").glob(".cap_atomic.*"))


def test_rejects_unsafe_station_directory_name(tmp_path):
    with pytest.raises(ValueError):
        publish_capture(tmp_path, "../escape", "cap_test", {}, {})


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission and SGID semantics are required")
def test_staging_modes_normalize_before_atomic_publication(tmp_path, monkeypatch):
    import siar_forecast_raw.storage as storage

    old_umask = os.umask(0o007)
    try:
        data_root = tmp_path / "raw"
        data_root.mkdir(mode=0o2770)
        os.chmod(data_root, 0o2770)

        staging_modes = {"capture": [], "run": []}
        original_set_permissions = storage._set_final_permissions

        def inspect_then_normalize(path):
            path = Path(path)
            mode = stat.S_IMODE(path.stat().st_mode)
            key = "capture" if path.parent.name == "IB04" else "run"
            staging_modes[key].append(mode)
            original_set_permissions(path)

        replaces = []
        original_replace = os.replace

        def record_replace(source, destination):
            replaces.append((Path(source), Path(destination)))
            return original_replace(source, destination)

        monkeypatch.setattr(storage, "_set_final_permissions", inspect_then_normalize)
        monkeypatch.setattr(storage.os, "replace", record_replace)

        run_temp, run_target = begin_run(data_root, "2026-10-11T08:00:00.000Z", "run_permission_test")
        capture = publish_capture(
            run_temp,
            "IB04",
            "cap_permission_test",
            {"capture": {"status": "complete"}},
            {"result.html": b"synthetic html", "forecast.zip": b"synthetic zip"},
        )
        expected_capture_target = run_temp / "IB04" / "cap_permission_test"
        published_run = publish_run(run_temp, run_target, {"status": "complete"})

        assert staging_modes == {"capture": [0o2700], "run": [0o2700]}
        assert stat.S_IMODE((published_run / "IB04/cap_permission_test").stat().st_mode) == 0o2770
        assert stat.S_IMODE(published_run.stat().st_mode) == 0o2770

        for directory in [published_run, published_run / "IB04", published_run / "IB04/cap_permission_test"]:
            mode = stat.S_IMODE(directory.stat().st_mode)
            assert mode & stat.S_ISGID
            assert mode & 0o007 == 0
            assert mode == 0o2770
        for directory in [published_run, published_run / "IB04/cap_permission_test"]:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    mode = stat.S_IMODE(file_path.stat().st_mode)
                    assert mode == 0o660
                    assert mode & 0o007 == 0

        assert verify_checksums(published_run / "IB04/cap_permission_test") == []
        assert verify_checksums(published_run) == []
        assert len(replaces) == 2
        assert all(source.name.startswith(".") for source, _ in replaces)
        assert any(destination == published_run for _, destination in replaces)
        assert any(destination == expected_capture_target for _, destination in replaces)
    finally:
        os.umask(old_umask)
