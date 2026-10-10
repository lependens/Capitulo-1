import json

from siar_forecast_raw.capture import capture_run
from siar_forecast_raw.config import Config, Station
from siar_forecast_raw.errors import CaptureError


def config(path):
    return Config(path, "https://example.invalid", 1, 1, 1, (), (Station("IB04", "Synthetic", "4_7", {}), Station("IB05", "Synthetic 2", "5_7", {})))


class FakeClient:
    calls = 0
    def __init__(self, _config): pass
    def capture_station(self, station):
        type(self).calls += 1
        if station.code == "IB05":
            raise CaptureError("synthetic export timeout", {"result.html": b"<html>safe forecast</html>"}, {"calculation_post": {"status_code": 200}})
        return ({"station": {"code": station.code, "label": station.label, "siar_station_form_id": station.siar_station_form_id}, "http": {}, "csv_source_filename": "synthetic.csv"}, {"result.html": b"<html>safe forecast</html>", "forecast.zip": b"synthetic", "forecast.csv": b"Fecha;ET0\n"})


def test_complete_and_partial_are_both_published(tmp_path):
    FakeClient.calls = 0
    final = capture_run(config(tmp_path), client_factory=FakeClient)
    run = json.loads((final / "run.json").read_text())
    assert run["status"] == "partial"
    assert run["collector_version"] == "0.1.0"
    assert [station["code"] for station in run["stations_resolved"]] == ["IB04", "IB05"]
    assert (final / "IB04").is_dir() and (final / "IB05").is_dir()
    statuses = [json.loads(p.read_text())["capture"]["status"] for p in final.glob("*/cap_*/metadata.json")]
    assert sorted(statuses) == ["complete", "partial"]
    assert all(json.loads(p.read_text())["source_issue_at"] is None for p in final.glob("*/cap_*/metadata.json"))


class FailedClient:
    def __init__(self, _config): pass
    def capture_station(self, station): raise CaptureError("synthetic timeout")


def test_failed_status_is_published(tmp_path):
    cfg = Config(tmp_path, "https://example.invalid", 1, 1, 1, (), (Station("IB01", "Synthetic", "1", {}),))
    final = capture_run(cfg, client_factory=FailedClient)
    assert json.loads((final / "run.json").read_text())["status"] == "failed"


class IncompleteClient:
    def __init__(self, _config): pass
    def capture_station(self, station):
        return ({"http": {}}, {"result.html": b"<html>safe forecast</html>"})


def test_incomplete_return_cannot_be_marked_complete(tmp_path):
    cfg = Config(tmp_path, "https://example.invalid", 1, 1, 1, (), (Station("IB01", "Synthetic", "1", {}),))
    final = capture_run(cfg, client_factory=IncompleteClient)
    metadata_path = next(final.glob("IB01/cap_*/metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["capture"]["status"] == "partial"
    assert "incomplete artifact set" in metadata["errors"][0]
