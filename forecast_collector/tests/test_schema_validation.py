import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from siar_forecast_raw.capture import capture_run
from siar_forecast_raw.config import Config, Station


ROOT = Path(__file__).parents[1]
SCHEMAS = ROOT / "schemas"
EXAMPLES = ROOT / "examples"


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def _validate(instance: dict, schema_name: str) -> None:
    Draft202012Validator(
        _schema(schema_name),
        format_checker=FormatChecker(),
    ).validate(instance)


class SchemaClient:
    def __init__(self, _config):
        pass

    def capture_station(self, station):
        return (
            {
                "station": {
                    "code": station.code,
                    "label": station.label,
                    "siar_station_form_id": station.siar_station_form_id,
                },
                "source": {
                    "system": "SIAR",
                    "product": "necesidadesHidricas",
                    "source_issue_at": None,
                },
                "request": {
                    "esCalculo": "0",
                    "tipoCalculo": "2",
                    "idEstacion": station.siar_station_form_id,
                },
                "http": {},
                "csv_source_filename": "synthetic.csv",
            },
            {
                "result.html": b"<html>safe forecast</html>",
                "forecast.zip": b"synthetic-zip",
                "forecast.csv": b"Fecha;ET0\n",
            },
        )


def test_example_run_matches_schema():
    _validate(
        json.loads((EXAMPLES / "run.json").read_text(encoding="utf-8")),
        "run.schema.json",
    )


def test_example_metadata_matches_schema():
    _validate(
        json.loads((EXAMPLES / "metadata.json").read_text(encoding="utf-8")),
        "capture.schema.json",
    )


def test_generated_run_and_capture_match_schemas(tmp_path):
    config = Config(
        tmp_path,
        "https://example.invalid",
        1,
        1,
        1,
        (),
        (Station("IB04", "Synthetic", "4_7", {}),),
    )
    final = capture_run(config, client_factory=SchemaClient)
    run = json.loads((final / "run.json").read_text(encoding="utf-8"))
    metadata_path = next(final.glob("IB04/cap_*/metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    _validate(run, "run.schema.json")
    _validate(metadata, "capture.schema.json")

    assert metadata["source"] == {
        "system": "SIAR",
        "product": "necesidadesHidricas",
        "source_issue_at": None,
    }
    assert "source_issue_at" not in metadata
