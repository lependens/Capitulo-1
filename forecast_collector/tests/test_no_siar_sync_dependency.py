from pathlib import Path


def test_package_has_no_siar_sync_dependency():
    source_root = Path(__file__).parents[1] / "src" / "siar_forecast_raw"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
    assert "siar_worker" not in source
    assert "import siar_sync" not in source
    assert "from siar_sync" not in source
