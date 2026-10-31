"""Append-only, hash-bound TRAIN → VALIDATION → TEST runner for ML v1."""
import csv
from copy import deepcopy
import json
from pathlib import Path
import platform
import uuid

from .datasets import read_dataset
from .ml_classical import (ALGORITHMS, BASELINES, feature_names, fit_candidates, importance,
                           predict, prepare_rows, score_predictions, validate_candidates, validate_ml_protocol)
from .ml_rolling import evaluate_rolling_candidates, linux_smoke_candidates
from .pipeline import _identity, _utc, safe_new_output
from .serialization import digest, json_bytes


def _read(path):
    content = Path(path).read_bytes()
    return json.loads(content), digest(content)


def _write(path, value):
    with Path(path).open("xb") as stream: stream.write(json_bytes(value))
    return digest(Path(path).read_bytes())


def _csv(path, rows, fields):
    def encode(value):
        if isinstance(value, (dict,list,tuple)): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
        return str(value).lower() if isinstance(value,bool) else value
    count=0
    with Path(path).open("x",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for row in rows: writer.writerow({name:encode(row.get(name)) for name in fields});count+=1
    return dict(path=Path(path).name,sha256=digest(Path(path).read_bytes()),rows=count,encoding="utf-8-sig",delimiter=",",decimal=".",null="empty")


def _load_official(dataset, dataset_sha, ahc_manifest_path, expected_sha, *, period=None):
    manifest, manifest_sha = _read(ahc_manifest_path)
    if manifest_sha != expected_sha or manifest.get("status") != "SUCCESS" or manifest.get("stage") != "test":
        raise ValueError("Require the frozen successful AHC TEST manifest")
    for key in ("dataset_id","audit_source_repo_sha","source_repo_sha"):
        if manifest.get(key) != dataset[key]: raise ValueError("AHC/dataset identity mismatch: "+key)
    if manifest.get("dataset_sha256") != dataset_sha: raise ValueError("AHC/dataset byte identity mismatch")
    root=Path(ahc_manifest_path).resolve().parent
    for name, entry in manifest["outputs"].items():
        if digest((root/entry["path"]).read_bytes()) != entry["sha256"]: raise ValueError("AHC output hash mismatch: "+name)
    entry=manifest["outputs"]["daily_predictions.csv"]
    official={}
    with (root/entry["path"]).open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            # The rolling campaign asks for one temporal partition at a time.
            # Skip non-partition rows before reading target or baseline values.
            if period is not None and not period[0] <= row["date"] <= period[1]:
                continue
            key=(row["station_code"],row["date"])
            if key in official: raise ValueError("Duplicate AHC daily key")
            official[key]={name:(float(row[name]) if row.get(name) not in (None,"") else None) for name in ("EtPMon",*BASELINES)}
    return official, manifest, manifest_sha


def _context(dataset_path, ahc_manifest_path, protocol_path):
    protocol, protocol_file_sha=_read(protocol_path);validate_ml_protocol(protocol)
    dataset_bytes=Path(dataset_path).read_bytes();dataset_sha=digest(dataset_bytes);dataset=read_dataset(dataset_path)
    if digest(Path(dataset_path).read_bytes()) != dataset_sha: raise ValueError("Dataset changed during read")
    if dataset_sha != protocol["dataset_sha256"] or dataset["dataset_id"] != protocol["dataset_id"]: raise ValueError("Protocol dataset mismatch")
    official, ahc, ahc_sha=_load_official(dataset,dataset_sha,ahc_manifest_path,protocol["ahc_test_manifest_sha256"])
    rows=prepare_rows(dataset,official,protocol)
    split_sha=digest(json_bytes(sorted([r["station_code"],r["date"],r["split"]] for r in rows)))
    identity={key:dataset[key] for key in ("dataset_id","audit_source_repo_sha","source_repo_sha","quality_post","source_data_hashes","units")}
    identity.update(dataset_sha256=dataset_sha,catalog=dataset.get("catalog"),protocol=protocol,
        protocol_sha256=digest(json_bytes(protocol)),protocol_file_sha256=protocol_file_sha,
        ahc_test_manifest_sha256=ahc_sha,ahc_test_manifest_path=str(Path(ahc_manifest_path).resolve()),
        ahc_pipeline=ahc["pipeline"],split_assignment_sha256=split_sha)
    return dataset,rows,protocol,identity


def _parent(path, stage, identity, code):
    manifest, manifest_sha=_read(path)
    if manifest.get("status") != "SUCCESS" or manifest.get("stage") != stage: raise ValueError("Require successful parent "+stage)
    for key in ("dataset_id","dataset_sha256","audit_source_repo_sha","source_repo_sha","protocol_sha256","ahc_test_manifest_sha256","split_assignment_sha256"):
        if manifest.get(key) != identity[key]: raise ValueError("Parent identity mismatch: "+key)
    if manifest["pipeline"]["git_sha"] != code["git_sha"] or manifest["pipeline"]["source_files_sha256"] != code["source_files_sha256"]:
        raise ValueError("Parent code identity mismatch")
    root=Path(path).resolve().parent
    for name,entry in manifest["outputs"].items():
        if digest((root/entry["path"]).read_bytes()) != entry["sha256"]: raise ValueError("Parent output hash mismatch: "+name)
    return manifest,manifest_sha,root


def _dump_models(path, payload):
    import joblib
    joblib.dump(payload,path,compress=3,protocol=5)
    return dict(path=Path(path).name,sha256=digest(Path(path).read_bytes()),bytes=Path(path).stat().st_size,format="joblib-pickle-trusted-internal-only")


def _load_models(path, expected):
    if digest(Path(path).read_bytes()) != expected: raise ValueError("Model bundle hash mismatch")
    import joblib
    return joblib.load(path)


def _runtime():
    import joblib,numpy,scipy,sklearn
    return dict(python=platform.python_version(),scikit_learn=sklearn.__version__,numpy=numpy.__version__,scipy=scipy.__version__,joblib=joblib.__version__)


def _peak_memory_mb():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes. The canonical smoke is Linux.
        return value / (1024 if platform.system() != "Darwin" else 1024 * 1024)
    except (ImportError, AttributeError):
        return None


def _prediction_rows(rows, selected, models):
    maps={r["model_id"]:predict(models[r["candidate_id"]],rows,r["feature_set"]) for r in selected}
    output=[]
    for row in rows:
        key=(row["station_code"],row["date"]);values=dict(row["baselines"])
        values.update({identifier:mapping.get(key) for identifier,mapping in maps.items()})
        flags=list(row["scientific_flags"])
        flags.extend("negative_prediction:"+name for name,value in values.items() if value is not None and value < 0)
        output.append({**{k:row[k] for k in ("station_code","date","split","target","EtPMon","quality_status","observation_status","quality_flags","provenance","source_data_sha")},
            "scientific_flags":flags,"predictions":values,"eligibility_reasons":row["eligibility_reasons"]})
    return output


def _model_diagnostics(model, *, fitted_on="TRAIN"):
    estimator=model.named_steps["model"] if hasattr(model,"named_steps") else model
    result={}
    if hasattr(model,"named_steps"):
        scaler=model.named_steps["scale"]
        result["standard_scaler"]={"mean":[float(v) for v in scaler.mean_],"scale":[float(v) for v in scaler.scale_],"fitted_on":fitted_on}
    for name in ("n_features_in_","n_iter_","max_depth_","n_leaves_"):
        value=getattr(estimator,name,None)
        if value is not None:
            result[name]=int(value) if not hasattr(value,"tolist") else value.tolist()
    if hasattr(estimator,"get_depth"): result["fitted_depth"]=int(estimator.get_depth())
    if hasattr(estimator,"get_n_leaves"): result["fitted_leaves"]=int(estimator.get_n_leaves())
    if hasattr(estimator,"estimators_"): result["fitted_estimators"]=len(estimator.estimators_)
    return result


def _write_test_views(output, predicted, scores, selected, models_by_candidate, importance_rows, manifest):
    ml_ids=[r["model_id"] for r in selected];models=list(BASELINES)+ml_ids
    common=("dataset_id","dataset_sha256","audit_source_repo_sha","source_repo_sha","protocol_sha256","run_id")
    identity={k:manifest[k] for k in common}
    daily_fields=list(common)+["station_code","date","split","EtPMon"]+models+["eligibility_reasons","quality_status","observation_status","quality_flags","scientific_flags","provenance","source_data_sha"]
    def daily():
        for row in predicted:
            record=dict(identity,station_code=row["station_code"],date=row["date"],split=row["split"],EtPMon=row["EtPMon"])
            record.update(row["predictions"])
            record.update(PM_FAO56_V1=row["target"],eligibility_reasons=row["eligibility_reasons"],quality_status=row["quality_status"],
                observation_status=row["observation_status"],quality_flags=row["quality_flags"],scientific_flags=row["scientific_flags"],
                provenance=row["provenance"],source_data_sha=row["source_data_sha"])
            yield record
    manifest["outputs"]["daily_predictions.csv"]=_csv(output/"daily_predictions.csv",daily(),daily_fields)
    score_fields=list(common)+list(scores[0])
    tables={"metrics.csv":None,"station_metrics.csv":{"station"},"yearly_metrics.csv":{"year","station_year"},
            "monthly_metrics.csv":{"month"},"seasonal_metrics.csv":{"season","season_year"}}
    for filename,groups in tables.items():
        manifest["outputs"][filename]=_csv(output/filename,(dict(identity,**s) for s in scores if groups is None or s["group"] in groups),score_fields)
    matrix=[dict(identity,**s) for s in scores if s["split"]=="TEST" and s["reference_id"]=="PM_FAO56_V1" and s["group"]=="global" and s["cohort_id"]=="common_all"]
    manifest["outputs"]["test_matrix.csv"]=_csv(output/"test_matrix.csv",matrix,list(matrix[0]))
    importance_fields=list(common)+list(importance_rows[0]) if importance_rows else list(common)
    manifest["outputs"]["feature_importance.csv"]=_csv(output/"feature_importance.csv",(dict(identity,**r) for r in importance_rows),importance_fields)
    metadata=[]
    for record in selected:
        metadata.append(dict(identity,model_id=record["model_id"],candidate_id=record["candidate_id"],algorithm=record["algorithm"],
            feature_set=record["feature_set"],features=record["features"],parameters=record["parameters"],fixed_parameters=record["fixed_parameters"],
            train_n=record["train_n"],validation_metrics=record["metrics"],selected_on="VALIDATION",refit_train_plus_validation=False,
            target="PM_FAO56_V1",target_meaning="calculated reference; not measured ET",test_status=manifest["protocol"]["test_status"],
            model_bundle_sha256=manifest["selected_model_bundle_sha256"],
            fitted_diagnostics=_model_diagnostics(models_by_candidate[record["candidate_id"]])))
    manifest["outputs"]["model_metadata.csv"]=_csv(output/"model_metadata.csv",metadata,list(metadata[0]))
    coverage=[]
    for family in manifest["protocol"]["feature_sets"]:
        for split in ("TRAIN","VALIDATION","TEST"):
            group=[r for r in predicted if r["split"]==split]
            eligible=sum(not r["eligibility_reasons"][family] for r in group)
            reasons={}
            for row in group:
                for reason in row["eligibility_reasons"][family]: reasons[reason]=reasons.get(reason,0)+1
            coverage.append(dict(identity,feature_set=family,split=split,source_rows=len(group),eligible_n=eligible,exclusion_reasons=reasons))
    manifest["outputs"]["coverage.csv"]=_csv(output/"coverage.csv",coverage,list(coverage[0]))
    schema={}
    for name,entry in manifest["outputs"].items():
        if name.endswith(".csv"):
            with (output/entry["path"]).open(encoding="utf-8-sig",newline="") as stream: headers=next(csv.reader(stream))
            schema[name]=dict(rows=entry["rows"],columns=headers)
    manifest["outputs"]["view_schemas.json"]=dict(path="view_schemas.json",sha256=_write(output/"view_schemas.json",schema))
    view={key:manifest[key] for key in ("dataset_id","dataset_sha256","audit_source_repo_sha","source_repo_sha","quality_post","catalog","source_data_hashes","units","protocol_sha256","split_assignment_sha256","pipeline","runtime","run_id","parent","ahc_test_manifest_sha256")}
    view.update(schema_version="ml-science-view-v1",status="SUCCESS",test_status=manifest["protocol"]["test_status"],
                science_recalculation_allowed=False,outputs=deepcopy(manifest["outputs"]))
    manifest["outputs"]["view-manifest.json"]=dict(path="view-manifest.json",sha256=_write(output/"view-manifest.json",view))


def execute_ml(dataset_path, ahc_manifest_path, protocol_path, output_path, stage, *, parent_manifest=None, pipeline_sha=None):
    if stage not in ("train","validate","test"): raise ValueError("Unknown ML stage")
    if (stage=="train") != (parent_manifest is None): raise ValueError("Only TRAIN has no parent")
    code=_identity(pipeline_sha);output=safe_new_output(output_path,directory=True)
    manifest=dict(schema_version="ml-classical-run-v1",stage=stage,status="FAILED",run_id="ml-"+uuid.uuid4().hex,
                  started_at_utc=_utc(),pipeline=code,runtime=_runtime(),outputs={})
    try:
        dataset,rows,protocol,identity=_context(dataset_path,ahc_manifest_path,protocol_path);manifest.update(identity);manifest["input_rows"]=len(rows)
        if protocol.get("schema_version") == "ml-classical-protocol-v2":
            raise ValueError("Legacy single-split ml-train/ml-validate/ml-test is superseded; use ml-linux-smoke, then the rolling-origin campaign")
        if manifest["runtime"]["scikit_learn"] != protocol["runtime"]["scikit_learn"].lstrip("="):
            raise ValueError("scikit-learn version differs from frozen protocol")
        if stage=="train":
            models,candidates=fit_candidates(rows,protocol)
            bundle=dict(schema_version="ml-model-bundle-v1",stage="TRAIN",protocol_sha256=identity["protocol_sha256"],models=models,candidates=candidates)
            manifest["outputs"]["candidate_models.joblib"]=_dump_models(output/"candidate_models.joblib",bundle)
            payload=dict(status="PASS",fit_split="TRAIN",candidate_count=len(candidates),candidates=candidates,
                test_inspected=False,validation_inspected=False,model_bundle_sha256=manifest["outputs"]["candidate_models.joblib"]["sha256"])
            name="training.json"
            assignments=(dict(station_code=r["station_code"],date=r["date"],split=r["split"],dataset_id=identity["dataset_id"],dataset_sha256=identity["dataset_sha256"]) for r in rows)
            manifest["outputs"]["split_assignments.csv"]=_csv(output/"split_assignments.csv",assignments,("station_code","date","split","dataset_id","dataset_sha256"))
        else:
            expected="train" if stage=="validate" else "validate";previous,previous_sha,root=_parent(parent_manifest,expected,identity,code)
            manifest["parent"]=dict(path=str(Path(parent_manifest).resolve()),sha256=previous_sha,stage=expected)
            if stage=="validate":
                training,_=_read(root/"training.json");entry=previous["outputs"]["candidate_models.joblib"]
                bundle=_load_models(root/entry["path"],entry["sha256"])
                selection=validate_candidates(rows,bundle["models"],bundle["candidates"],protocol)
                for record in selection["selected"]:
                    record["model_id"]="ML_"+record["algorithm"]+"__"+record["feature_set"]
                champion=next(r for r in selection["selected"] if r["candidate_id"]==selection["campaign_champion"]["candidate_id"])
                selection["campaign_champion_model_id"]=champion["model_id"]
                selection["permutation_importance"]=importance(rows,bundle["models"],selection,protocol)
                selected_models={r["candidate_id"]:bundle["models"][r["candidate_id"]] for r in selection["selected"]}
                selected_bundle=dict(schema_version="ml-selected-model-bundle-v1",stage="VALIDATION",protocol_sha256=identity["protocol_sha256"],models=selected_models,selected=selection["selected"])
                manifest["outputs"]["selected_models.joblib"]=_dump_models(output/"selected_models.joblib",selected_bundle)
                selection["selected_model_bundle_sha256"]=manifest["outputs"]["selected_models.joblib"]["sha256"]
                payload=selection;name="selection.json"
            else:
                selection,_=_read(root/"selection.json");entry=previous["outputs"]["selected_models.joblib"]
                bundle=_load_models(root/entry["path"],entry["sha256"])
                if entry["sha256"] != selection["selected_model_bundle_sha256"]: raise ValueError("Frozen selected model bundle changed")
                predicted=_prediction_rows(rows,bundle["selected"],bundle["models"])
                ml_ids=[r["model_id"] for r in bundle["selected"]];scores=list(score_predictions(predicted,ml_ids))
                manifest["selected_model_bundle_sha256"]=entry["sha256"]
                _write_test_views(output,predicted,scores,bundle["selected"],bundle["models"],selection["permutation_importance"],manifest)
                payload=dict(status="PASS",test_status=protocol["test_status"],selection_split="VALIDATION",models_refitted=False,
                    selected_model_bundle_sha256=entry["sha256"],campaign_champion_model_id=selection["campaign_champion_model_id"],
                    selected=[{k:r[k] for k in ("model_id","candidate_id","algorithm","feature_set","parameters","metrics")} for r in bundle["selected"]],
                    split_counts={split:sum(r["split"]==split for r in predicted) for split in ("TRAIN","VALIDATION","TEST","OUTSIDE_EXPERIMENT")},
                    global_metrics=[s for s in scores if s["group"]=="global"])
                name="test_results.json"
        manifest["outputs"][name]=dict(path=name,sha256=_write(output/name,payload))
        manifest["status"]="SUCCESS"
        if digest(Path(dataset_path).read_bytes()) != identity["dataset_sha256"]: raise ValueError("Dataset changed during run")
    except (OSError,ValueError,TypeError,KeyError,OverflowError) as exc:
        manifest["status"]="FAILED";manifest["error"]=str(exc)
    finally:
        manifest["finished_at_utc"]=_utc();_write(output/"manifest.json",manifest)
    return manifest


def execute_ml_smoke(dataset_path, ahc_manifest_path, protocol_path, output_path, *, pipeline_sha=None):
    """Run the non-selecting LINEAR/RIDGE Linux smoke through all dev folds."""
    code = _identity(pipeline_sha)
    output = safe_new_output(output_path, directory=True)
    manifest = dict(schema_version="ml-linux-smoke-run-v1", stage="linux-smoke", status="FAILED",
                    run_id="ml-smoke-"+uuid.uuid4().hex, started_at_utc=_utc(), pipeline=code,
                    runtime=_runtime(), outputs={})
    try:
        dataset, rows, protocol, identity = _context(dataset_path, ahc_manifest_path, protocol_path)
        manifest.update(identity)
        manifest["input_rows"] = len(rows)
        if platform.system() != "Linux":
            raise ValueError("Canonical ML smoke must execute natively on Linux")
        if manifest["runtime"]["scikit_learn"] != protocol["runtime"]["scikit_learn"].lstrip("="):
            raise ValueError("scikit-learn version differs from frozen protocol")
        outcome = evaluate_rolling_candidates(rows, protocol, linux_smoke_candidates(protocol))
        outcome.update(status="PASS", purpose="infrastructure smoke only; not scientific model selection",
                       scientific_selection_allowed=False, test_inspected=False,
                       peak_memory_mb=_peak_memory_mb(), platform=platform.platform())
        fields = ("candidate_id", "algorithm", "feature_set", "parameters", "fold_id", "train_period",
                  "validation_period", "train_n", "validation_n", "train_key_sha256",
                  "validation_key_sha256", "metrics", "fit_warnings", "test_inspected")
        manifest["outputs"]["rolling_fold_metrics.csv"] = _csv(
            output/"rolling_fold_metrics.csv", outcome["fold_records"], fields)
        summary = {key: value for key, value in outcome.items() if key != "fold_records"}
        manifest["outputs"]["smoke_results.json"] = dict(
            path="smoke_results.json", sha256=_write(output/"smoke_results.json", summary))
        manifest["status"] = "SUCCESS"
        manifest["scientific_selection_allowed"] = False
        manifest["test_inspected"] = False
        if digest(Path(dataset_path).read_bytes()) != identity["dataset_sha256"]:
            raise ValueError("Dataset changed during smoke")
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
    finally:
        manifest["finished_at_utc"] = _utc()
        _write(output/"manifest.json", manifest)
    return manifest
