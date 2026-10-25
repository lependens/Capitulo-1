"""Synthetic ML-v1 tests: feature contract, anti-leakage and staged artifacts."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import unittest

from siar_science.ahc_pipeline import execute_ahc
from siar_science.cli import main
from siar_science.ml_classical import (ALGORITHMS, candidate_id, feature_names, fit_candidates,
                                       prepare_rows, validate_candidates, validate_ml_protocol)
from siar_science.pipeline import calculate, execute_run
from siar_science.serialization import digest, json_bytes
from tests_support import dataset_fixture, temporary_directory


def ahc_protocol(root, dataset, dataset_bytes):
    p=json.loads((Path(__file__).parents[1]/"experiments/ahc-v1.json").read_bytes())
    p.update(dataset_id=dataset["dataset_id"],dataset_sha256=digest(dataset_bytes),require_clean_upstream=False,
             baseline_manifest_sha256=digest((root/"baseline/manifest.json").read_bytes()))
    p["splits"]={"TRAIN":["2001-01-01","2001-12-31"],"VALIDATION":["2002-01-01","2002-12-31"],"TEST":["2003-01-01","2003-12-31"]}
    p["training_stability_blocks"]=[[2001,2001]]
    p["post_test_descriptive_blocks"]=[[2001,2001],[2002,2002],[2003,2003]]
    p["support"]={k:1 for k in p["support"]}
    p["stability_gate"]["minimum_robust_blocks"]=2
    p["selection"]["station_minimum_validation_n"]=1
    return p


def ml_protocol(dataset, dataset_bytes, ahc_manifest):
    p=json.loads((Path(__file__).parents[1]/"experiments/ml-v1-classical.json").read_bytes())
    p.update(dataset_id=dataset["dataset_id"],dataset_sha256=digest(dataset_bytes),
             ahc_test_manifest_sha256=digest(Path(ahc_manifest).read_bytes()),primary_stations=["SYNTH"],excluded_stations={})
    p["splits"]={"TRAIN":["2001-01-01","2001-12-31"],"VALIDATION":["2002-01-01","2002-12-31"],"TEST":["2003-01-01","2003-12-31"]}
    p["candidate_grids"]={
        "LINEAR":[{}],"RIDGE":[{"alpha":1.0}],"ELASTIC_NET":[{"alpha":0.1,"l1_ratio":0.5}],
        "TREE":[{"max_depth":3,"min_samples_leaf":1,"min_samples_split":2}],
        "RANDOM_FOREST":[{"n_estimators":5,"max_depth":3,"min_samples_leaf":1,"min_samples_split":2,"max_features":1.0}],
        "EXTRA_TREES":[{"n_estimators":5,"max_depth":3,"min_samples_leaf":1,"min_samples_split":2,"max_features":1.0}],
        "HIST_GRADIENT_BOOSTING":[{"max_iter":10,"learning_rate":0.1,"max_leaf_nodes":5,"l2_regularization":0.0}]}
    return p


def synthetic_dataset():
    dataset=dataset_fixture();rows=[]
    for year in (2001,2002,2003):
        for month in (1,4,7,10):
            row=deepcopy(dataset["rows"][0]);day=f"{year}-{month:02}-15";i=len(rows)
            row.update(date=day)
            row["values"].update(Fecha=day,TempMin=5+month/4,TempMax=15+month/3,
                HumedadMedia=60+month/5,HumedadMin=40+month/6,HumedadMax=80+month/8,
                Radiacion=8+month,VelViento=1+month/20)
            row["provenance"]["source_row"]=i+2;rows.append(row)
    dataset.update(rows=rows,missing_days=[])
    pm=calculate(dataset,("pm",))
    for source,calculated in zip(dataset["rows"],pm["rows"]): source["values"]["EtPMon"]=calculated["ET0_FAO56"]
    return dataset


class MLClassicalPureTests(unittest.TestCase):
    def test_protocol_candidate_identity_and_feature_contract(self):
        p=json.loads((Path(__file__).parents[1]/"experiments/ml-v1-classical.json").read_bytes())
        validate_ml_protocol(p)
        self.assertEqual(sum(len(v) for v in p["candidate_grids"].values()),37)
        self.assertEqual(len(p["feature_sets"])*len(ALGORITHMS),35)
        self.assertNotIn("EtPMon",feature_names(p,"FEATURE_SET_FULL"))
        self.assertEqual(candidate_id("RIDGE","FEATURE_SET_T",{"alpha":1}),candidate_id("RIDGE","FEATURE_SET_T",{"alpha":1}))
        q=deepcopy(p);q["selection"]["test_tuning_allowed"]=True
        with self.assertRaises(ValueError): validate_ml_protocol(q)

    def test_test_mutation_cannot_change_validation_selection(self):
        dataset=synthetic_dataset();raw=json_bytes(dataset)
        official={}
        computed=calculate(dataset,("pm","hargreaves","hargreaves_rs","valiantzas"))
        for row in computed["rows"]:
            official[row["station_code"],row["date"]]={"PM_FAO56_V1":row["ET0_FAO56"],"EtPMon":row["values"]["EtPMon"],
                "HS_RA_V1":row["ET0_Hargreaves"],"HG_RS_V1":row["ET0_Hargreaves_Rs"],"VALIANTZAS_V1":row["ET0_Valiantzas"],
                "HS_RA_AHC_VALIDATED":row["ET0_Hargreaves"],"HG_RS_AHC_VALIDATED":row["ET0_Hargreaves_Rs"],
                "HG_RS_AHC_VALIDATED_SELECTED":row["ET0_Hargreaves_Rs"],"VALIANTZAS_AHC_VALIDATED":row["ET0_Valiantzas"]}
        p=ml_protocol(dataset,raw,Path(__file__))
        p["ahc_test_manifest_sha256"]="0"*64
        rows=prepare_rows(dataset,official,p);models,candidates=fit_candidates(rows,p)
        before=validate_candidates(rows,models,candidates,p)
        changed=deepcopy(rows)
        for row in changed:
            if row["split"]=="TEST": row["target"]=999999
        self.assertEqual(before,validate_candidates(changed,models,candidates,p))
        self.assertTrue(all(r["features"]["FEATURE_SET_FULL"] is not None for r in rows))


class MLClassicalPipelineTests(unittest.TestCase):
    def test_three_stages_preserve_hashes_and_never_refit(self):
        with temporary_directory() as directory,redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            root=Path(directory);dataset=synthetic_dataset();dataset_bytes=json_bytes(dataset)
            source=root/"dataset.json";source.write_bytes(dataset_bytes)
            baseline=execute_run(source,root/"baseline","run-four-baselines")
            self.assertEqual(baseline["status"],"SUCCESS")
            ap=ahc_protocol(root,dataset,dataset_bytes);apath=root/"ahc.json";apath.write_bytes(json_bytes(ap))
            previous=None
            for stage in ("train","validate","test"):
                target=root/("ahc-"+stage);kwargs={} if previous is None else {"parent_manifest":previous}
                result=execute_ahc(source,root/"baseline/manifest.json",apath,target,stage,**kwargs)
                self.assertEqual(result["status"],"SUCCESS",result.get("error"));previous=target/"manifest.json"
            mp=ml_protocol(dataset,dataset_bytes,previous);mpath=root/"ml.json";mpath.write_bytes(json_bytes(mp))
            common=["--dataset",str(source),"--ahc-manifest",str(previous),"--protocol",str(mpath)]
            code=main(["ml-train",*common,"--output",str(root/"train")])
            self.assertEqual(code,0,json.loads((root/"train/manifest.json").read_bytes()).get("error"));parent=root/"train/manifest.json"
            self.assertEqual(main(["ml-validate",*common,"--parent-manifest",str(parent),"--output",str(root/"validation")]),0);parent=root/"validation/manifest.json"
            code=main(["ml-test",*common,"--parent-manifest",str(parent),"--output",str(root/"test")])
            self.assertEqual(code,0,json.loads((root/"test/manifest.json").read_bytes()).get("error"))
            selection=json.loads((root/"validation/selection.json").read_bytes())
            test=json.loads((root/"test/test_results.json").read_bytes())
            self.assertEqual(len(selection["selected"]),35);self.assertFalse(selection["test_inspected"])
            self.assertFalse(test["models_refitted"])
            self.assertEqual(selection["selected_model_bundle_sha256"],test["selected_model_bundle_sha256"])
            with (root/"test/daily_predictions.csv").open(encoding="utf-8-sig",newline="") as stream:
                daily=list(csv.DictReader(stream))
            self.assertEqual(len(daily),12)
            self.assertTrue(all("ML_LINEAR__FEATURE_SET_T" in r for r in daily))
            self.assertTrue((root/"test/test_matrix.csv").is_file())
            self.assertEqual(json.loads((root/"test/view-manifest.json").read_bytes())["schema_version"],"ml-science-view-v1")
            self.assertEqual(source.read_bytes(),dataset_bytes)
            manifest=json.loads((root/"test/manifest.json").read_bytes())
            for entry in manifest["outputs"].values():
                self.assertEqual(digest((root/"test"/entry["path"]).read_bytes()),entry["sha256"])
            (root/"validation/selected_models.joblib").write_bytes(b"tampered")
            failed=main(["ml-test",*common,"--parent-manifest",str(parent),"--output",str(root/"tampered")])
            self.assertEqual(failed,1)


if __name__=="__main__": unittest.main()
