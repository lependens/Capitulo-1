from datetime import datetime, timezone

from siar_forecast_raw.ids import new_capture_id, new_run_id


def test_ids_are_timestamped_and_distinct():
    moment = datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)
    assert new_run_id(moment).startswith("run_20261010T060000.000Z_")
    first = new_capture_id("IB04", moment)
    second = new_capture_id("IB04", moment)
    assert first.startswith("cap_20261010T060000.000Z_IB04_")
    assert first != second
