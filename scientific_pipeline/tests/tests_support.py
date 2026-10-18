"""Controlled synthetic fixtures; never reads canonical CSVs."""
from copy import deepcopy
import os
from tempfile import TemporaryDirectory
from siar_science.datasets import COLUMNS, UNITS


def temporary_directory():
    return TemporaryDirectory(dir=os.environ.get("SIAR_SCIENCE_TEST_TMP"))


def dataset_fixture():
    values = dict.fromkeys(COLUMNS)
    values.update(Fecha="2001-07-06", Estacion="SYNTH", TempMedia=99, TempMin=12.3,
                  TempMax=21.5, HumedadMin=63, HumedadMax=84, VelViento=2.078,
                  Radiacion=22.07, EtPMon=0)
    row = dict(station_code="SYNTH", date="2001-07-06", values=values,
               observation_status="OBSERVED", quality_status="PASS", quality_flags=[],
               provenance={"source_file": "controlled-synthetic.csv", "source_row": 2}, source_data_sha="b"*64)
    second = deepcopy(row)
    second.update(date="2001-07-07")
    second["values"].update(Fecha="2001-07-07", Radiacion=None, EtPMon=2)
    third = deepcopy(row)
    third.update(date="2001-07-08")
    third["values"].update(Fecha="2001-07-08", EtPMon=None, TempMedia=None)
    return dict(schema_version="observational-v1", dataset_id="controlled-synthetic-only",
                source_repo_sha="a"*40, source_data_hashes={"controlled-synthetic.csv": "b"*64},
                quality_post={"id": "SYNTHETIC-POST", "sha256": "c"*64}, units=dict(UNITS),
                stations={"SYNTH": {"latitude_deg": 50.8, "altitude_m": 100}}, rows=[row, second, third],
                missing_days=[dict(station_code="SYNTH", date="2001-07-09", observation_status="PENDING_RETRY")])
