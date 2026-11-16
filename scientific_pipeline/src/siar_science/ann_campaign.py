"""Append-only ANN v1 DEVELOPMENT campaign; TEST is a separate entry point."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from functools import cmp_to_key
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

from .ann_modern import (_epochs, _matrix, _model, final_epochs, fold_candidate,
                         summarize_folds, validate_ann_contract)
from .ml_campaign import _context_for_split


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_checksums(out, names):
    inventory = ''.join(f'{_sha(out/name)}  {name}{chr(10)}' for name in sorted(names))
    (out/'SHA256SUMS').write_text(inventory, encoding='ascii')

def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                                     allow_nan=False)+'\n', encoding='utf-8')


def _now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')


def _parameter_count(n_features, widths):
    sizes = [n_features, *widths, 1]
    return sum((left + 1) * right for left, right in zip(sizes, sizes[1:], strict=False))

def _compare_candidates(left, right, protocol):
    a, b = left['aggregate'], right['aggregate']
    delta = a['weighted_mean_rmse'] - b['weighted_mean_rmse']
    if abs(delta) > 1e-12:
        return -1 if delta < 0 else 1
    def secondary(candidate):
        family = candidate['family']
        features = len(protocol['shared_context']) + len(protocol['feature_sets'][family])
        return (candidate['aggregate']['weighted_sd_rmse'], features,
                _parameter_count(features, candidate['widths']),
                family, tuple(candidate['widths']))
    return (secondary(left) > secondary(right)) - (secondary(left) < secondary(right))

def _fit_development(rows, family, widths, ann, folds):
    train=[r for r in rows if r['split']=='DEVELOPMENT' and r['features'][family] is not None]
    _,x,y=_matrix(train,family)
    scaler=StandardScaler().fit(x)
    epoch_map=final_epochs(folds,ann['seeds'])
    models=[]
    for seed in ann['seeds']:
        model=_epochs(_model(widths,seed,ann),scaler.transform(x),y,epoch_map[str(seed)])
        models.append(model)
    return {'scaler':scaler,'models':models,'epoch_map':epoch_map,'n':len(train)}


def develop(dataset_path, ahc_manifest_path, ml_protocol_path, ann_protocol_path, output_dir):
    start=time.monotonic()
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    manifest={'stage':'ANN_V1_DEVELOPMENT','status':'RUNNING','started_at_utc':_now(),
              'outputs':{}}
    _write_json(out/'manifest.json',manifest)
    try:
        ann_bytes=Path(ann_protocol_path).read_bytes()
        ann=json.loads(ann_bytes)
        ml_bytes=Path(ml_protocol_path).read_bytes()
        ml=json.loads(ml_bytes)
        ann_sha=validate_ann_contract(ann,ml,ann_bytes=ann_bytes,ml_bytes=ml_bytes)
        if _sha(dataset_path)!=ann['dataset_sha256']:
            raise ValueError('Dataset SHA mismatch')
        context=_context_for_split(dataset_path,ahc_manifest_path,ml_protocol_path,'DEVELOPMENT')
        rows, identity = context[1], context[3]
        if any(r['split']=='TEST' for r in rows):
            raise ValueError('TEST rows entered DEVELOPMENT')
        if identity['dataset_sha256']!=ann['dataset_sha256']:
            raise ValueError('Context Dataset mismatch')
        candidates=[]; fold_rows=[]; seed_rows=[]
        for family in ann['feature_sets']:
            for widths in ann['architectures']:
                folds=[]
                for fold in ann['folds']:
                    result=fold_candidate(rows,fold,family,widths,ann)
                    folds.append(result)
                    fold_rows.append({'family':family,'widths':json.dumps(widths),
                                      'fold':fold['id'],'n':result['n'],'rmse':result['rmse'],
                                      'mae':result['mae'],'bias':result['bias']})
                    for seed,rmse,epoch in zip(ann['seeds'],result['seed_rmse'],result['best_epochs'],strict=True):
                        seed_rows.append({'family':family,'widths':json.dumps(widths),
                                          'fold':fold['id'],'seed':seed,'rmse':rmse,'best_epoch':epoch})
                candidates.append({'family':family,'widths':widths,'aggregate':summarize_folds(folds),
                                   'folds':folds})
        by_family={family:min((c for c in candidates if c['family']==family),
                              key=cmp_to_key(lambda left,right:_compare_candidates(left,right,ml))) for family in ann['feature_sets']}
        champion=min(by_family.values(),key=cmp_to_key(lambda left,right:_compare_candidates(left,right,ml)))
        frozen={}
        for family,selection in by_family.items():
            frozen[family]=_fit_development(rows,family,selection['widths'],ann,selection['folds'])
        selection={'status':'FROZEN','selection_split':'DEVELOPMENT',
                   'test_inspected':False,'champion':{'family':champion['family'],
                   'widths':champion['widths'],'aggregate':champion['aggregate']},
                   'selected':{k:{'widths':v['widths'],'aggregate':v['aggregate'],
                                  'epoch_map':frozen[k]['epoch_map']} for k,v in by_family.items()}}
        _write_json(out/'candidate_summaries.json',candidates)
        _write_json(out/'selection.json',selection)
        for name,records in [('fold_metrics.csv',fold_rows),('seed_metrics.csv',seed_rows)]:
            with (out/name).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        joblib.dump(frozen,out/'refitted_models.joblib',compress=3)
        _write_checksums(out, ('candidate_summaries.json','selection.json','fold_metrics.csv','seed_metrics.csv','refitted_models.joblib'))
        manifest.update(status='PRE_TEST_FROZEN',finished_at_utc=_now(),
                        ann_protocol_sha256=ann_sha,ml_protocol_sha256=_sha(ml_protocol_path),
                        dataset_sha256=identity['dataset_sha256'],
                        ahc_manifest_sha256=_sha(ahc_manifest_path),
                        base_repo_sha=ann['base_repo_sha'],pipeline_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dataset_id=identity['dataset_id'],audit_source_repo_sha=identity['audit_source_repo_sha'],source_repo_sha=identity['source_repo_sha'],quality_post_id=identity['quality_post']['id'],source_data_hashes=identity['source_data_hashes'],catalog_sha256=identity['catalog']['sha256'],candidate_count=len(candidates),
                        selected_cell_count=len(by_family),test_access_count=0,
                        champion=selection['champion'],runtime_seconds=time.monotonic()-start,
                        environment={'python':platform.python_version(),
                                     'scikit_learn':sklearn.__version__,'numpy':np.__version__},
                        outputs={name:_sha(out/name) for name in ('candidate_summaries.json',
                            'selection.json','fold_metrics.csv','seed_metrics.csv','refitted_models.joblib','SHA256SUMS')})
        _write_json(out/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        # A failure cannot be mistaken for a frozen experiment.
        for child in out.iterdir():
            if child.name!='manifest.json' and child.is_file():
                child.unlink()
        manifest.update(status='FAILED',finished_at_utc=_now(),error=str(exc),outputs={})
        _write_json(out/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['develop'])
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--ahc-manifest',required=True)
    parser.add_argument('--ml-protocol',required=True)
    parser.add_argument('--ann-protocol',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    develop(args.dataset,args.ahc_manifest,args.ml_protocol,args.ann_protocol,args.output)


if __name__=='__main__':
    main()
