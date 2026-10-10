import json
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


def test_rejects_unsafe_station_directory_name(tmp_path):
    with pytest.raises(ValueError):
        publish_capture(tmp_path, "../escape", "cap_test", {}, {})
