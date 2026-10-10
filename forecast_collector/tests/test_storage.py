import json
import os
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
