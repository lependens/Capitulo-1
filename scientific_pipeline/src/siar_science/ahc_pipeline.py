"""Explicit I/O for four chronological AHC stages, hash-bound and append-only."""
import csv
from copy import deepcopy
import json
from pathlib import Path
import uuid
from collections import Counter

from .ahc import train_analysis, validate_selection, apply_validated, closeout_analysis
from .calibration import EMPIRICAL, validate_protocol, partition
from .datasets import read_dataset
from .experimental_metrics import ANNUAL_MODELS, experiment_scores, reference
from .pipeline import _identity, _merge_predictions, _utc, safe_new_output, PARAMETERS
from .serialization import digest, json_bytes


def _read(path):
    content = Path(path).read_bytes()
    return json.loads(content), digest(content)


def _write(path, value):
    with path.open("xb") as stream:
        stream.write(json_bytes(value))
    return digest(path.read_bytes())


def _csv(path, rows, fieldnames):
    def encode(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return str(value).lower() if isinstance(value, bool) else value
    count = 0
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: encode(row.get(k)) for k in fieldnames})
            count += 1
    return dict(path=path.name, sha256=digest(path.read_bytes()), rows=count,
                encoding="utf-8-sig", delimiter=",", decimal=".", null="empty")


def _context(dataset_path, baseline_manifest_path, protocol_path):
    protocol, protocol_file_sha = _read(protocol_path)
    validate_protocol(protocol)
    content = Path(dataset_path).read_bytes()
    dataset_sha = digest(content)
    dataset = read_dataset(dataset_path)
    if digest(Path(dataset_path).read_bytes()) != dataset_sha:
        raise ValueError("Dataset changed during load")
    if dataset_sha != protocol["dataset_sha256"] or dataset["dataset_id"] != protocol["dataset_id"]:
        raise ValueError("Protocol dataset identity mismatch")
    baseline, baseline_sha = _read(baseline_manifest_path)
    if baseline_sha != protocol["baseline_manifest_sha256"] or baseline.get("status") != "SUCCESS":
        raise ValueError("Protocol baseline manifest mismatch or unsuccessful baseline")
    if baseline.get("parameters") != PARAMETERS or set(baseline.get("methods", [])) != set(PARAMETERS):
        raise ValueError("Require the four frozen versioned baselines")
    for key in ("dataset_id", "dataset_sha256", "audit_source_repo_sha", "source_repo_sha"):
        expected = dataset_sha if key == "dataset_sha256" else dataset[key]
        if baseline.get(key) != expected:
            raise ValueError("Baseline repository/dataset identity mismatch")
    if protocol["require_clean_upstream"] and (baseline["pipeline"].get("dirty") is not False or baseline["pipeline"].get("package_matches_checkout") is not True):
        raise ValueError("Require a clean verified baseline producer")
    prediction = Path(baseline_manifest_path).resolve().parent / baseline["outputs"]["predictions.json"]["path"]
    if digest(prediction.read_bytes()) != baseline["outputs"]["predictions.json"]["sha256"]:
        raise ValueError("Baseline prediction hash mismatch")
    imported, methods, producers = _merge_predictions(dataset, [prediction], dataset_sha)
    if len(methods) != 4:
        raise ValueError("Require all four baseline methods")
    rows = imported["rows"]
    for row in rows:
        row["EtPMon"] = row["values"]["EtPMon"]
    identity = {k: dataset[k] for k in ("dataset_id", "audit_source_repo_sha", "source_repo_sha", "quality_post", "source_data_hashes", "units")}
    identity.update(dataset_sha256=dataset_sha, catalog=dataset.get("catalog"),
                    baseline_manifest_sha256=baseline_sha, baseline_manifest_path=str(Path(baseline_manifest_path).resolve()),
                    baseline_prediction_inputs=producers, protocol_sha256=digest(json_bytes(protocol)),
                    protocol_file_sha256=protocol_file_sha, protocol=protocol,
                    split_assignment_sha256=digest(json_bytes(sorted([r["station_code"], r["date"], partition(r["date"], protocol)] for r in rows))))
    return dataset, rows, protocol, identity


def _parent(path, expected_stage, identity, code):
    manifest, manifest_sha = _read(path)
    if manifest.get("status") != "SUCCESS" or manifest.get("stage") != expected_stage:
        raise ValueError("Parent stage must be successful " + expected_stage)
    for key in ("dataset_id", "dataset_sha256", "protocol_sha256", "baseline_manifest_sha256", "source_repo_sha", "audit_source_repo_sha", "split_assignment_sha256"):
        if manifest.get(key) != identity[key]:
            raise ValueError("Parent experiment identity mismatch: " + key)
    if manifest.get("pipeline", {}).get("git_sha") != code["git_sha"] or manifest["pipeline"].get("source_files_sha256") != code["source_files_sha256"]:
        raise ValueError("Parent pipeline identity differs; use a new experiment for changed code")
    root = Path(path).resolve().parent
    for name, output in manifest["outputs"].items():
        if digest((root/output["path"]).read_bytes()) != output["sha256"]:
            raise ValueError("Parent output hash mismatch: " + name)
    return manifest, manifest_sha, root


def _coefficients_csv(path, coefficients):
    names = ("coefficient_id", "station_code", "base_method_id", "kind", "reference_id", "granularity", "group_key",
             "coefficient", "status", "deployable", "fit_split", "calibration_start", "calibration_end", "fit_key_sha256", "support", "estimate")
    return _csv(path, coefficients, names)


def _model_producer(producer, method):
    matches = [entry["producer"] for entry in producer.get("upstream", []) if method in entry["producer"].get("parameters", {})]
    if len(matches) > 1:
        raise ValueError("Ambiguous upstream model producer")
    return _model_producer(matches[0], method) if matches else producer["pipeline"]["git_sha"]


def _catalog_metadata(dataset):
    records, audit = {}, {}
    catalog = dataset.get("catalog")
    if catalog and catalog.get("path"):
        path = Path(catalog["path"])
        if digest(path.read_bytes()) != catalog["sha256"]:
            raise ValueError("Catalog metadata hash mismatch")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = {r["Codigo"]: r for r in csv.DictReader(stream)}
    report = dataset["quality_post"].get("reports", {}).get("siar_station_audit.csv")
    if report and report.get("path"):
        path = Path(report["path"])
        if digest(path.read_bytes()) != report["sha256"]:
            raise ValueError("Station audit metadata hash mismatch")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            audit = {r["station_code"]: r for r in csv.DictReader(stream)}
    return records, audit


def _views(output, rows, scores, dataset, manifest, coefficients, selection):
    identity = {k: manifest[k] for k in ("dataset_id", "dataset_sha256", "audit_source_repo_sha", "source_repo_sha", "protocol_sha256", "run_id")}
    models = sorted({m for r in rows for m in r["predictions"]})
    daily_names = list(identity) + ["date", "station_code", "split", "EtPMon"] + models + ["eligibility", "evaluated_PM", "evaluated_SIAR", "coefficient_ids", "observation_status", "quality_status", "quality_flags", "scientific_flags", "source_data_sha", "provenance"]
    def daily():
        for row in rows:
            record = dict(identity, **{k: row[k] for k in daily_names if k in row}, **row["predictions"])
            record["evaluated_PM"] = {m: row["predictions"][m] is not None and reference(row, "PM_FAO56_V1") is not None for m in models}
            record["evaluated_SIAR"] = {m: row["predictions"][m] is not None and reference(row, "EtPMon") is not None for m in models}
            yield record
    manifest["outputs"]["daily_predictions.csv"] = _csv(output/"daily_predictions.csv", daily(), daily_names)
    manifest["outputs"]["ahc_coefficients.csv"] = _coefficients_csv(output/"ahc_coefficients.csv", coefficients)
    coefficient_artifact_sha = manifest["outputs"]["ahc_coefficients.csv"]["sha256"]
    instances = {m: "model-"+digest(json_bytes(dict(method=m, protocol=manifest["protocol_sha256"],
                     coefficients=coefficient_artifact_sha if m not in PARAMETERS else None))) for m in models}
    for score in scores:
        score["model_instance_id"] = instances[score["method_id"]]
    score_names = list(identity) + list(scores[0]) if scores else list(identity) + ["split", "reference_id", "group", "key", "method_id", "n", "mae", "rmse", "bias", "r2"]
    tables = {"metrics.csv": None, "station_metrics.csv": {"station"},
              "monthly_metrics.csv": {"year_month", "station_year_month"},
              "seasonal_metrics.csv": {"season_year", "station_season_year"},
              "yearly_metrics.csv": {"year", "station_year"},
              "climatological_month_metrics.csv": {"month"}, "climatological_season_metrics.csv": {"season"}}
    for filename, groups in tables.items():
        selected = (dict(identity, **s) for s in scores if groups is None or s["group"] in groups)
        manifest["outputs"][filename] = _csv(output/filename, selected, score_names)
    metadata = []
    requirements = {
        "PM_FAO56_V1": (["TempMin", "TempMax", "HumedadMin", "HumedadMax", "Radiacion", "VelViento"], ["date", "latitude_deg", "altitude_m/derived_pressure", "wind_2m_convention"]),
        "HS_RA_V1": (["TempMin", "TempMax"], ["date", "latitude_deg"]),
        "HG_RS_V1": (["TempMin", "TempMax", "Radiacion"], []),
        "VALIANTZAS_V1": (["TempMin", "TempMax", "HumedadMedia"], ["date", "latitude_deg"]),
    }
    for model in models:
        base = model if model in PARAMETERS else next(m for m, (_, adjusted) in EMPIRICAL.items() if model.startswith(adjusted))
        original_producer = _model_producer(manifest["baseline_prediction_inputs"][0]["producer"], base)
        granularity = next((s["selected_granularity"] for s in selection if s["base_method_id"] == base), "annual") if model.endswith("_SELECTED") else "annual" if model not in PARAMETERS else None
        metadata.append(dict(method_id=model, model_instance_id=instances[model], base_method_id=base,
            version="AHC_MULTIPLIER_OLS0_V1" if model not in PARAMETERS else base,
            reference_id="PM_FAO56_V1", input_parameters=PARAMETERS[base],
            units="mm/day", input_requirements=dict(weather=requirements[base][0], context=requirements[base][1],
                station_coefficient_required=model not in PARAMETERS), granularity=granularity,
            coefficient_artifact_sha256=coefficient_artifact_sha if model not in PARAMETERS else None,
            producer_pipeline_sha=manifest["pipeline"]["git_sha"] if model not in PARAMETERS else original_producer,
            base_producer_pipeline_sha=original_producer, test_status=manifest["protocol"]["test_status"],
            adjustment_kind="AHC_VALIDATED" if model not in PARAMETERS else None,
            calibration_split="TRAIN" if model not in PARAMETERS else None,
            selection=next((s for s in selection if s["base_method_id"] == base), None) if model.endswith("_SELECTED") else None,
            interpretation="emulation of calculated reference, not measured ET", **identity))
    manifest["outputs"]["model_metadata.csv"] = _csv(output/"model_metadata.csv", metadata, list(metadata[0]))
    station_rows = []
    catalog, audit = _catalog_metadata(dataset)
    for code, station in sorted(dataset["stations"].items()):
        series = [r for r in rows if r["station_code"] == code]
        source_metadata = catalog.get(code, {})
        official = audit.get(code, {})
        station_rows.append(dict(station_code=code, latitude_deg=station["latitude_deg"], altitude_m=station.get("altitude_m"),
            station_name=source_metadata.get("Estacion"), catalog_metadata=source_metadata,
            official_start=official.get("expected_start_date") or (source_metadata.get("Fecha_Instalacion") or "")[:10] or None,
            official_end=official.get("expected_end_date") or (source_metadata.get("Fecha_Baja") or "")[:10] or None,
            source_row_count=len(series), first_source_date=min((r["date"] for r in series), default=None),
            last_source_date=max((r["date"] for r in series), default=None),
            source_quality_statuses=sorted({r["quality_status"] for r in series}),
            available_splits=sorted({r["split"] for r in series}),
            station_metadata=station, catalog_sha256=dataset.get("catalog", {}).get("sha256"), **identity))
    manifest["outputs"]["station_metadata.csv"] = _csv(output/"station_metadata.csv", station_rows, list(station_rows[0]))
    coverage, complexity = [], []
    for model in models:
        base = next(m["base_method_id"] for m in metadata if m["method_id"] == model)
        alias = {"PM_FAO56_V1": "pm", "HS_RA_V1": "hargreaves", "HG_RS_V1": "hargreaves_rs", "VALIANTZAS_V1": "valiantzas"}[base] if model in PARAMETERS else model
        complexity.append(dict(method_id=model, model_instance_id=instances[model], weather_inputs=requirements[base][0],
            weather_input_count=len(requirements[base][0]), context_inputs=requirements[base][1],
            prior_station_calibration_required=model not in PARAMETERS,
            full_PM_inputs_required_at_prediction=base == "PM_FAO56_V1", **identity))
        for split in ("TRAIN", "VALIDATION", "TEST"):
            for station in (None, *sorted(dataset["stations"])):
                group = [r for r in rows if r["split"] == split and (station is None or r["station_code"] == station)]
                reasons = Counter(reason for r in group for reason in r["eligibility"].get(alias, {}).get("reasons", []))
                coverage.append(dict(method_id=model, model_instance_id=instances[model], station_code=station, split=split,
                    source_rows=len(group), eligible_n=sum(r["eligibility"].get(alias, {}).get("eligible", False) for r in group),
                    predicted_n=sum(r["predictions"][model] is not None for r in group),
                    evaluated_PM_n=sum(r["predictions"][model] is not None and reference(r, "PM_FAO56_V1") is not None for r in group),
                    evaluated_SIAR_n=sum(r["predictions"][model] is not None and reference(r, "EtPMon") is not None for r in group),
                    overlapping_reason_counts=dict(reasons), **identity))
    manifest["outputs"]["coverage_metrics.csv"] = _csv(output/"coverage_metrics.csv", coverage, list(coverage[0]))
    manifest["outputs"]["input_complexity.csv"] = _csv(output/"input_complexity.csv", complexity, list(complexity[0]))
    manifest["view_contract"] = dict(schema_version="science-view-v1", science_recalculation_allowed=False,
        formats=["CSV UTF-8 BOM"], raw_weather_columns_duplicated=False,
        arbitrary_filtered_metrics="not available unless a matching scientific cohort is published")
    schema = {}
    json_columns = {"eligibility", "evaluated_PM", "evaluated_SIAR", "coefficient_ids", "quality_flags", "scientific_flags", "provenance",
                    "input_parameters", "input_requirements", "selection", "source_quality_statuses", "available_splits", "station_metadata", "catalog_metadata", "support", "estimate", "overlapping_reason_counts", "weather_inputs", "context_inputs"}
    numeric = {"mae", "rmse", "bias", "r2", "coefficient", "latitude_deg", "altitude_m", "EtPMon"} | set(models)
    integer = {"n", "source_row_count", "source_rows", "eligible_n", "predicted_n", "evaluated_PM_n", "evaluated_SIAR_n", "weather_input_count"}
    boolean = {"deployable", "prior_station_calibration_required", "full_PM_inputs_required_at_prediction"}
    for filename, entry in manifest["outputs"].items():
        if filename.endswith(".csv"):
            with (output/filename).open(encoding="utf-8-sig", newline="") as stream:
                headers = next(csv.reader(stream))
            schema[filename] = dict(rows=entry["rows"], columns=[dict(name=name, type="json" if name in json_columns else
                "float64_nullable" if name in numeric else "int64" if name in integer else "boolean" if name in boolean else "string_nullable") for name in headers])
    manifest["outputs"]["view_schemas.json"] = dict(path="view_schemas.json", sha256=_write(output/"view_schemas.json", schema))
    view = {k: manifest[k] for k in ("dataset_id", "dataset_sha256", "audit_source_repo_sha", "source_repo_sha", "quality_post", "catalog",
            "source_data_hashes", "units", "protocol_sha256", "split_assignment_sha256", "pipeline", "run_id", "parent", "baseline_prediction_inputs")}
    view.update(schema_version="science-view-v1", generated_at_utc=_utc(), status="SUCCESS", outputs=deepcopy(manifest["outputs"]),
                science_recalculation_allowed=False, test_status=manifest["protocol"]["test_status"])
    manifest["outputs"]["view-manifest.json"] = dict(path="view-manifest.json", sha256=_write(output/"view-manifest.json", view))


def execute_ahc(dataset_path, baseline_manifest_path, protocol_path, output_path, stage, *, parent_manifest=None, pipeline_sha=None):
    """Four explicit append-only stages: train→validate→test→closeout.

    STOP manifests preserve scientific diagnostics. I/O/software exceptions
    leave FAILED manifests and must be repaired, never relabeled as success.
    """
    if stage not in ("train", "validate", "test", "closeout"):
        raise ValueError("Unknown AHC stage")
    if (stage == "train") != (parent_manifest is None):
        raise ValueError("Only TRAIN has no parent; later stages require --parent-manifest")
    code = _identity(pipeline_sha)
    output = safe_new_output(output_path, directory=True)
    manifest = dict(schema_version="ahc-run-v1", stage=stage, status="FAILED", run_id="ahc-"+uuid.uuid4().hex,
                    started_at_utc=_utc(), pipeline=code, outputs={})
    try:
        dataset, rows, protocol, identity = _context(dataset_path, baseline_manifest_path, protocol_path)
        manifest.update(identity)
        manifest["configuration_sha256"] = identity["protocol_sha256"]
        manifest["input_rows"] = len(rows)
        if stage == "train":
            payload = train_analysis(rows, dataset["stations"], protocol)
            name = "training.json"
            assignments = (dict(station_code=r["station_code"], date=r["date"], split=partition(r["date"], protocol),
                                dataset_id=identity["dataset_id"], dataset_sha256=identity["dataset_sha256"]) for r in rows)
            manifest["outputs"]["split_assignments.csv"] = _csv(output/"split_assignments.csv", assignments,
                ("station_code", "date", "split", "dataset_id", "dataset_sha256"))
        else:
            expected = {"validate": "train", "test": "validate", "closeout": "test"}[stage]
            previous, previous_sha, root = _parent(parent_manifest, expected, identity, code)
            manifest["parent"] = dict(path=str(Path(parent_manifest).resolve()), sha256=previous_sha, stage=expected)
            if stage == "validate":
                training, _ = _read(root/"training.json")
                payload = validate_selection(rows, training, protocol)
                payload["training"] = dict(path=str(root/"training.json"), sha256=previous["outputs"]["training.json"]["sha256"], coefficients=training["coefficients"])
                name = "selection.json"
            elif stage == "test":
                selection, _ = _read(root/"selection.json")
                coefficients = selection["training"]["coefficients"]
                if digest(json_bytes(coefficients)) != selection["frozen_training_coefficients_sha256"]:
                    raise ValueError("Frozen TRAIN coefficients changed")
                training_path = Path(selection["training"]["path"])
                if digest(training_path.read_bytes()) != selection["training"]["sha256"]:
                    raise ValueError("Frozen training artifact hash mismatch")
                predicted = apply_validated(rows, coefficients, protocol, selection["selections"])
                scores = list(experiment_scores(predicted))
                payload = dict(status="PASS", reference="PM_FAO56_V1", test_status=protocol["test_status"],
                    coefficients_refitted=False, selection=selection["selections"],
                    frozen_training_coefficients_sha256=selection["frozen_training_coefficients_sha256"],
                    primary_model_ids=list(ANNUAL_MODELS),
                    global_metrics=[s for s in scores if s["group"] == "global"],
                    split_counts={split: sum(r["split"] == split for r in predicted) for split in ("TRAIN", "VALIDATION", "TEST", "OUTSIDE_EXPERIMENT")})
                _views(output, predicted, scores, dataset, manifest, coefficients, selection["selections"])
                name = "test_results.json"
            else:
                payload = closeout_analysis(rows, dataset["stations"], protocol, test_completed=True)
                name = "closeout.json"
                if payload["status"] == "PASS":
                    manifest["outputs"]["ahc_climatology_coefficients.csv"] = _coefficients_csv(output/"ahc_climatology_coefficients.csv", payload["climatology"])
        manifest["outputs"][name] = dict(path=name, sha256=_write(output/name, payload))
        if stage == "train":
            manifest["outputs"]["ahc_validated_coefficients.csv"] = _coefficients_csv(output/"ahc_validated_coefficients.csv", payload["coefficients"])
        manifest["status"] = "STOP" if payload["status"] == "STOP" else "SUCCESS"
        if payload["status"] == "STOP":
            manifest["stop_reasons"] = payload["stop_reasons"]
        if digest(Path(dataset_path).read_bytes()) != identity["dataset_sha256"]:
            raise ValueError("Dataset changed during experiment")
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
    finally:
        manifest["finished_at_utc"] = _utc()
        _write(output/"manifest.json", manifest)
    return manifest
