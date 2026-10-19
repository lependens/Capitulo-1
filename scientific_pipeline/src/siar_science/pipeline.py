"""Pure orchestration plus explicit, isolated run I/O. Never acquires SIAR data."""
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

from . import __version__
from .atmosphere import actual_vapour_pressure
from .datasets import read_dataset, validate_dataset
from .hargreaves import HS_RA_V1, HS_RA_V1_PARAMETERS, hargreaves_samani
from .metrics import evaluate, reference_value
from .penman_monteith import PM_FAO56_V1, PM_FAO56_V1_PARAMETERS, fao56_daily, net_longwave_radiation
from .quality import eligibility, scientific_flags, finite
from .solar import extraterrestrial_radiation, clear_sky_radiation

METHODS = {"pm": (PM_FAO56_V1, "ET0_FAO56"), "hargreaves": (HS_RA_V1, "ET0_Hargreaves")}
ASSUMPTIONS = {
    "units_interpretation": "RESOLVED_FOR_V1",
    "units_provenance": "UNITS_PROVENANCE_PARTIAL",
    "wind_height_assumption": "2 m / SIAR convention",
    "station_period_provenance": "not exhaustively verified",
    "wind_input": "VelViento used directly as u2; no /3.6",
    "radiation_input": "Radiacion MJ/m2/day directly; no *0.0864",
    "reference": "EtPMon supplied/calculated by SIAR; not independent ground truth",
}
POLICIES = {
    "version": "SCIENTIFIC_QC_V1",
    "row_policy": "preserve all observations and source values; no imputation or sigma filter",
    "temperature": "(TempMax+TempMin)/2; SIAR TempMedia preserved",
    "reference_policy": "finite nonnegative EtPMon; zero included; PASS/WARN and in official period",
    "negative_predictions": "retain signed output and flag; no clipping",
    "metrics": "n,MAE,RMSE,bias=prediction-reference,R2=1-SSE/SST; own and paired cohorts",
    "seasons": "northern DJF/MAM/JJA/SON; December belongs to next season-year",
    "split": "uncalibrated v1; future TRAIN<VALIDATION<TEST, global calendar cuts after POST",
}


def calculate(dataset, methods=("pm", "hargreaves")):
    """Pure calculation: input preserved, model-specific eligibility, no I/O."""
    validate_dataset(dataset)
    if len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Unknown or duplicate model")
    result = deepcopy(dataset)
    for row in result["rows"]:
        station = result["stations"][row["station_code"]]
        values = row["values"]
        row["scientific_flags"] = scientific_flags(row, station)
        row["eligibility"] = {}
        row["model_diagnostics"] = {}
        for method, (_, field) in METHODS.items():
            row[field] = None  # never carry predictions forward from another run
            reasons = eligibility(row, station, method)
            if method in methods and any(reason.startswith("STOP:") for reason in reasons):
                raise ValueError("STOP: incompatible wind-height evidence for " + row["station_code"] + ":" + row["date"])
            row["eligibility"][method] = dict(eligible=not reasons, reasons=reasons,
                                                status="NOT_REQUESTED" if method not in methods else "INELIGIBLE" if reasons else "READY")
            if method not in methods or reasons:
                continue
            tmin, tmax = values["TempMin"], values["TempMax"]
            tmean = (tmin + tmax) / 2
            day = date.fromisoformat(row["date"]).timetuple().tm_yday
            try:
                if method == "pm":
                    value = fao56_daily(tmean_c=tmean, tmin_c=tmin, tmax_c=tmax,
                                        rhmin_pct=values["HumedadMin"], rhmax_pct=values["HumedadMax"],
                                        wind_2m_m_s=values["VelViento"], solar_radiation_mj_m2_day=values["Radiacion"],
                                        latitude_deg=station["latitude_deg"], elevation_m=station["altitude_m"], day_of_year=day)
                    ra = extraterrestrial_radiation(station["latitude_deg"], day)
                    rso = clear_sky_radiation(ra, station["altitude_m"])
                    ea = actual_vapour_pressure(tmin, tmax, values["HumedadMin"], values["HumedadMax"])
                    rnl = net_longwave_radiation(tmin, tmax, ea, values["Radiacion"], rso)
                    row["model_diagnostics"][PM_FAO56_V1] = dict(tmean_calc_c=tmean, ra_mj_m2_day=ra,
                        rso_mj_m2_day=rso, rnl_mj_m2_day=rnl, rn_mj_m2_day=.77*values["Radiacion"]-rnl,
                        rs_rso=values["Radiacion"]/rso)
                    if rnl < 0:
                        row["scientific_flags"].append("negative_net_longwave_radiation")
                else:
                    ra = extraterrestrial_radiation(station["latitude_deg"], day)
                    value = hargreaves_samani(tmean_c=tmean, tmin_c=tmin, tmax_c=tmax,
                                             ra_mj_m2_day=ra)
                    row["model_diagnostics"][HS_RA_V1] = dict(tmean_calc_c=tmean, ra_mj_m2_day=ra)
                if not finite(value):
                    raise ValueError("Nonfinite result")
                row[field] = value
                row["eligibility"][method]["status"] = "CALCULATED"
                if value < 0:
                    row["scientific_flags"].append("negative_prediction:" + METHODS[method][0])
            except (ValueError, OverflowError) as exc:
                row[field] = None
                row["eligibility"][method]["status"] = "FAILED"
                row["eligibility"][method]["reasons"].append("calculation_failed:" + str(exc))
    return result


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _identity(provided_sha=None):
    package = Path(__file__).resolve().parent
    identity = {"package_version": __version__, "python_version": sys.version,
                "source_files_sha256": {p.name: digest(p.read_bytes()) for p in sorted(package.glob("*.py"))}}
    sha, dirty, root = None, None, None
    for candidate in (package, Path.cwd()):
        try:
            sha = subprocess.check_output(["git", "-C", str(candidate), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
            root = Path(subprocess.check_output(["git", "-C", str(candidate), "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True).strip())
            dirty = bool(subprocess.check_output(["git", "-C", str(candidate), "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    if root is not None:
        checkout_package = root/"scientific_pipeline/src/siar_science"
        matching = all((checkout_package/name).is_file() and digest((checkout_package/name).read_bytes()) == sha256
                       for name, sha256 in identity["source_files_sha256"].items())
        identity["package_matches_checkout"] = matching
        dirty = dirty or not matching
    if provided_sha is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", provided_sha):
            raise ValueError("--pipeline-sha must be a 40-character Git SHA")
        if sha is not None and sha != provided_sha:
            raise ValueError("Provided pipeline SHA differs from the available checkout")
    if provided_sha is None and sha is None:
        raise ValueError("No checkout identity available; provide --pipeline-sha for an installed release")
    identity.update(git_sha=provided_sha or sha, dirty=dirty,
                    git_sha_source="provided" if provided_sha else "checkout")
    return identity


def safe_new_output(path, *, directory):
    """Explicit output protection; never overwrite or target acquisition/reference."""
    path = Path(path).resolve()
    protected = {"sources", "raw", "datos_siar_baleares", "legacy", "data_quality", "siar-sync",
                 "forecast_collector", "aemet_forecast_collector", ".git"}
    if any(part.casefold() in protected for part in path.parts):
        raise ValueError("Output cannot target acquisition or reference directories")
    if path.exists():
        raise ValueError("Output already exists; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    if directory:
        path.mkdir()
    return path


def _write(path, value):
    content = json_bytes(value)
    with path.open("xb") as stream:
        stream.write(content)
    return digest(content)


def _utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _cohort_hashes(rows):
    result = {}
    for method, (identifier, field) in METHODS.items():
        keys = [[r["station_code"], r["date"]] for r in rows if finite(r.get(field))]
        result[identifier] = dict(n=len(keys), sha256=digest(json_bytes(sorted(keys))))
        evaluation_keys = [[r["station_code"], r["date"]] for r in rows
                           if finite(r.get(field)) and reference_value(r) is not None]
        result[identifier + ":evaluation"] = dict(n=len(evaluation_keys), sha256=digest(json_bytes(sorted(evaluation_keys))))
    paired = [[r["station_code"], r["date"]] for r in rows
              if all(finite(r.get(field)) for _, field in METHODS.values())]
    result["paired_predictions"] = dict(n=len(paired), sha256=digest(json_bytes(sorted(paired))))
    evaluated = [[r["station_code"], r["date"]] for r in rows
                 if reference_value(r) is not None and all(finite(r.get(field)) for _, field in METHODS.values())]
    result["paired_evaluation"] = dict(n=len(evaluated), sha256=digest(json_bytes(sorted(evaluated))))
    return result


def _merge_predictions(dataset, paths, dataset_sha):
    result = calculate(dataset, methods=())
    keyed = {(r["station_code"], r["date"]): r for r in result["rows"]}
    identifiers = {identifier: (method, field) for method, (identifier, field) in METHODS.items()}
    loaded, used = [], set()
    for path in paths:
        content = Path(path).read_bytes()
        artifact = json.loads(content)
        if (not isinstance(artifact, dict) or artifact.get("schema_version") != "predictions-v1" or artifact.get("dataset_id") != dataset["dataset_id"]
                or artifact.get("dataset_sha256") != dataset_sha):
            raise ValueError("Predictions belong to a different dataset or schema")
        if any(artifact.get(key) != dataset[key] for key in ("audit_source_repo_sha", "source_repo_sha")):
            raise ValueError("Prediction repository provenance mismatch")
        declared = artifact.get("methods")
        if not isinstance(declared, list) or not declared or any(m not in identifiers or m in used for m in declared) or len(set(declared)) != len(declared):
            raise ValueError("Unknown, duplicate or overlapping prediction methods")
        producer = artifact.get("producer")
        if not isinstance(producer, dict) or producer.get("status") != "SUCCESS":
            raise ValueError("Predictions require successful producer provenance")
        identity = producer.get("pipeline")
        if not isinstance(identity, dict) or not isinstance(identity.get("git_sha"), str) or not re.fullmatch(r"[0-9a-f]{40}", identity["git_sha"]):
            raise ValueError("Missing producer code identity")
        source_hashes = identity.get("source_files_sha256")
        if not isinstance(source_hashes, dict) or not source_hashes or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in source_hashes.values()):
            raise ValueError("Missing producer source hashes")
        expected = {PM_FAO56_V1: PM_FAO56_V1_PARAMETERS, HS_RA_V1: HS_RA_V1_PARAMETERS}
        if producer.get("parameters") != {m: expected[m] for m in declared}:
            raise ValueError("Prediction parameters differ from versioned method contract")
        predictions = artifact.get("rows")
        if not isinstance(predictions, list) or any(not isinstance(row, dict) for row in predictions):
            raise ValueError("Missing prediction rows")
        keys = [(r.get("station_code"), r.get("date")) for r in predictions]
        if len(set(keys)) != len(keys) or set(keys) != set(keyed):
            raise ValueError("Predictions do not preserve dataset keys")
        for row, key in zip(predictions, keys):
            target = keyed[key]
            for identifier in declared:
                method, field = identifiers[identifier]
                value = row.get(field)
                if value is not None and not finite(value):
                    raise ValueError("Invalid prediction value")
                eligible = target["eligibility"][method]["eligible"]
                if not eligible and value is not None:
                    raise ValueError("Prediction on a scientifically ineligible row")
                if eligible and value is None:
                    raise ValueError("Missing prediction on eligible row; incomplete run cannot be evaluated")
                target[field] = value
                target["eligibility"][method]["status"] = "CALCULATED" if value is not None else "INELIGIBLE"
                if value is not None and value < 0:
                    target["scientific_flags"].append("negative_prediction:" + identifier)
                if isinstance(row.get("model_diagnostics"), dict) and identifier in row["model_diagnostics"]:
                    diagnostics = row["model_diagnostics"][identifier]
                    if not isinstance(diagnostics, dict) or any(not finite(value) for value in diagnostics.values()):
                        raise ValueError("Invalid model diagnostics")
                    target["model_diagnostics"][identifier] = diagnostics
                    if target["model_diagnostics"][identifier].get("rnl_mj_m2_day", 0) < 0:
                        target["scientific_flags"].append("negative_net_longwave_radiation")
        used.update(declared)
        loaded.append(dict(path=str(Path(path).resolve()), sha256=digest(content), producer=producer))
    return result, [method for method, (identifier, _) in METHODS.items() if identifier in used], loaded


def execute_run(dataset_path, output_path, command="run-baselines", pipeline_sha=None, prediction_paths=()):
    """Explicit I/O boundary; write a new run and hash every successful output.

    Fatal validation/calculation failures leave a FAILED manifest and diagnostics.
    An existing output is rejected before any write. Dataset v1 is never built here.
    """
    if command not in ("validate", "calculate-pm", "calculate-hargreaves", "evaluate", "run-baselines"):
        raise ValueError("Unknown operation")
    code = _identity(pipeline_sha)
    output = safe_new_output(output_path, directory=True)
    manifest = dict(schema_version="science-run-v1", run_id="run-" + uuid.uuid4().hex,
                    started_at_utc=_utc(), command=command, status="FAILED", pipeline=code,
                    assumptions=deepcopy(ASSUMPTIONS), policies=deepcopy(POLICIES),
                    parameters={PM_FAO56_V1: deepcopy(PM_FAO56_V1_PARAMETERS), HS_RA_V1: deepcopy(HS_RA_V1_PARAMETERS)}, outputs={})
    manifest["configuration_sha256"] = digest(json_bytes({key: manifest[key] for key in ("assumptions", "policies", "parameters")}))
    try:
        content = Path(dataset_path).read_bytes()
        dataset_sha = digest(content)
        manifest.update(dataset_sha256=dataset_sha, dataset_path=str(Path(dataset_path).resolve()))
        dataset = read_dataset(dataset_path)
        # Detect input changes between hashing and parsing instead of silently
        # linking predictions to different bytes.
        if digest(Path(dataset_path).read_bytes()) != dataset_sha:
            raise ValueError("Dataset changed during read")
        manifest.update(dataset_id=dataset["dataset_id"], dataset_sha256=dataset_sha,
                        audit_source_repo_sha=dataset["audit_source_repo_sha"],
                        source_repo_sha=dataset["source_repo_sha"], source_data_hashes=dataset["source_data_hashes"],
                        quality_post=dataset["quality_post"], dataset_path=str(Path(dataset_path).resolve()))
        if "catalog" in dataset:
            manifest["catalog"] = dataset["catalog"]
        methods = {"validate": (), "calculate-pm": ("pm",), "calculate-hargreaves": ("hargreaves",),
                   "run-baselines": ("pm", "hargreaves"), "evaluate": ()}[command]
        if command == "evaluate":
            if not prediction_paths:
                raise ValueError("evaluate requires --predictions from a completed calculation")
            manifest["prediction_input_paths"] = [str(Path(path).resolve()) for path in prediction_paths]
            manifest["prediction_inputs"] = [dict(path=str(Path(path).resolve()), sha256=digest(Path(path).read_bytes())) for path in prediction_paths]
            result, methods, inputs = _merge_predictions(dataset, prediction_paths, dataset_sha)
            manifest["prediction_inputs"] = inputs
        else:
            result = calculate(dataset, methods)
        rows = result["rows"]
        manifest["methods"] = [METHODS[m][0] for m in methods]
        manifest["cohorts"] = _cohort_hashes(rows)
        manifest["counts"] = {}
        failures = 0
        for method in methods:
            states = [r["eligibility"][method] for r in rows]
            count = dict(input=len(rows), eligible=sum(s["eligible"] for s in states),
                         calculated=sum(s["status"] == "CALCULATED" for s in states), failed=sum(s["status"] == "FAILED" for s in states))
            failures += count["failed"]
            manifest["counts"][METHODS[method][0]] = count
        prediction_artifact = dict(schema_version="predictions-v1", dataset_id=dataset["dataset_id"],
                                   dataset_sha256=dataset_sha, audit_source_repo_sha=dataset["audit_source_repo_sha"],
                                   source_repo_sha=dataset["source_repo_sha"], methods=manifest["methods"], rows=rows)
        prediction_artifact["producer"] = dict(run_id=manifest["run_id"], operation=command,
            status="FAILED" if failures else "SUCCESS", pipeline=code,
            parameters={METHODS[m][0]: manifest["parameters"][METHODS[m][0]] for m in methods},
            assumptions=manifest["assumptions"], policies=manifest["policies"],
            upstream=manifest.get("prediction_inputs", []))
        scores = evaluate(rows, prediction_fields=tuple(METHODS[m][1] for m in methods)) if command in ("run-baselines", "evaluate") else []
        for score in scores:
            score.update(audit_source_repo_sha=dataset["audit_source_repo_sha"], source_repo_sha=dataset["source_repo_sha"])
        for method in methods:
            manifest["counts"][METHODS[method][0]]["evaluated"] = manifest["cohorts"][METHODS[method][0] + ":evaluation"]["n"]
        for name, artifact in (("predictions.json", prediction_artifact), ("metrics.json", scores)):
            manifest["outputs"][name] = dict(path=name, sha256=_write(output/name, artifact))
        manifest["status"] = "FAILED" if failures else "SUCCESS"
        if failures:
            manifest["error"] = "One or more model calculations failed; outputs are partial diagnostics"
    except (ValueError, OSError, TypeError, KeyError, OverflowError) as exc:
        manifest["error"] = str(exc)
    finally:
        manifest["finished_at_utc"] = _utc()
        _write(output/"manifest.json", manifest)
    return manifest
