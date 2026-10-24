"""Lossless observation/prediction CSV export; explicit caller-owned I/O."""
import csv
import json
from .datasets import COLUMNS
from .metrics import reference_value
from .quality import finite

PREDICTIONS = {
    "pm": ("ET0_FAO56", "ET0_PM_FAO56_V1"),
    "hargreaves": ("ET0_Hargreaves", "ET0_HS_RA_V1"),
    "valiantzas": ("ET0_Valiantzas", "ET0_VALIANTZAS_V1"),
}


def write_combined_csv(path, rows, identity):
    """UTF-8 BOM, comma, decimal point, ISO date, null as empty, no rounding.

    Identity is repeated in every row for self-contained downstream exports;
    full producer/configuration/source hashes remain in the run manifest.
    The output must not exist. All source observations are retained.
    """
    headers = list(COLUMNS) + ["date", "station_code"] + [p[1] for p in PREDICTIONS.values()]
    labels = {"pm": "PM", "hargreaves": "HS", "valiantzas": "VAL"}
    headers += [prefix+label for label in labels.values() for prefix in ("eligible_", "evaluated_", "eligibility_reasons_")]
    headers += ["observation_status", "quality_status", "quality_flags", "scientific_flags", "provenance"] + list(identity)
    encode = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = dict(row["values"], date=row["date"], station_code=row["station_code"], **identity)
            for method, (field, column) in PREDICTIONS.items():
                record[column] = row.get(field)
                label = labels[method]
                record["eligible_"+label] = str(row["eligibility"][method]["eligible"]).lower()
                record["evaluated_"+label] = str(finite(row.get(field)) and reference_value(row) is not None).lower()
                record["eligibility_reasons_"+label] = encode(row["eligibility"][method]["reasons"])
            for key in ("observation_status", "quality_status"):
                record[key] = row[key]
            for key in ("quality_flags", "scientific_flags", "provenance"):
                record[key] = encode(row.get(key, [] if key.endswith("flags") else {}))
            writer.writerow(record)
