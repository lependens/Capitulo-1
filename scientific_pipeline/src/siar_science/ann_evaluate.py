"""Evaluate frozen ANN v1 models once against retrospective TEST."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .ann_campaign import _sha, _write_json, _write_checksums
from .ann_modern import validate_ann_contract
from .metrics import metrics
from .ml_campaign import _context_for_split


def _journal(path, state, **extra):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),
                                 'state':state,**extra},sort_keys=True)+'\n')


def _groups(row):
    d=date.fromisoformat(row['date'])
    season='DJF' if d.month in (12,1,2) else 'MAM' if d.month<6 else 'JJA' if d.month<9 else 'SON'
    return [('global','all'),('station',row['station_code']),('year',str(d.year)),
            ('month',f'{d.month:02}'),('season',season),
            ('station_year',f'{row["station_code"]}:{d.year}')]


def evaluate(dev_dir, dataset_path, ahc_path, ml_path, ann_path, hgb_dir, output_dir):
    dev=Path(dev_dir);out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    manifest={'stage':'ANN_V1_TEST','status':'RUNNING','outputs':{},'test_access_count':0}
    _write_json(out/'manifest.json',manifest)
    journal=out/'journal.jsonl'
    try:
        ann_bytes=Path(ann_path).read_bytes();ml_bytes=Path(ml_path).read_bytes()
        ann=json.loads(ann_bytes);ml=json.loads(ml_bytes)
        ann_sha=validate_ann_contract(ann,ml,ann_bytes=ann_bytes,ml_bytes=ml_bytes)
        if _sha(dataset_path)!=ann['dataset_sha256']:
            raise ValueError('Dataset SHA mismatch')
        frozen=json.loads((dev/'manifest.json').read_text())
        if (frozen['status']!='PRE_TEST_FROZEN' or frozen['test_access_count']!=0
            or frozen['ann_protocol_sha256']!=ann_sha
            or frozen['dataset_sha256']!=ann['dataset_sha256']):
            raise ValueError('Development freeze identity mismatch')
        for name,expected in frozen['outputs'].items():
            if _sha(dev/name)!=expected:
                raise ValueError('Frozen output mismatch: '+name)
        selection=json.loads((dev/'selection.json').read_text())
        if selection['status']!='FROZEN' or selection['test_inspected'] is not False:
            raise ValueError('Selection was not frozen before TEST')
        bundle=joblib.load(dev/'refitted_models.joblib')
        if set(bundle)!=set(selection['selected']):
            raise ValueError('Frozen bundle mismatch')
        for family in bundle:
            if (len(bundle[family]['models'])!=len(ann['seeds'])
                or bundle[family]['epoch_map']!=selection['selected'][family]['epoch_map']):
                raise ValueError('Frozen seed model mismatch')
        _journal(journal,'PRE_TEST_VERIFIED',ann_protocol_sha256=ann_sha)
        _journal(journal,'TEST_ACCESS_INTENT_RECORDED')
        _journal(journal,'TEST_OPENED')
        manifest['test_access_count']=1
        _write_json(out/'manifest.json',manifest)
        context=_context_for_split(dataset_path,ahc_path,ml_path,'TEST')
        rows=context[1]
        if any(r['split']!='TEST' for r in rows):
            raise ValueError('Non-TEST rows in TEST context')
        predictions=[]
        predictions = [{'station_code': row['station_code'], 'date': row['date'],
                        'PM_FAO56_V1': row['target']} for row in rows]
        for family, info in bundle.items():
            indices = [i for i, row in enumerate(rows) if row['features'][family] is not None]
            for record in predictions:
                record[family] = None
            if not indices:
                continue
            x = info['scaler'].transform(np.asarray(
                [rows[i]['features'][family] for i in indices], dtype=float))
            values = np.mean([model.predict(x) for model in info['models']], axis=0)
            if not np.all(np.isfinite(values)):
                raise ValueError('Nonfinite ANN TEST predictions: ' + family)
            for i, value in zip(indices, values, strict=True):
                predictions[i][family] = float(value)
        families=list(selection['selected'])
        common=[r for r in predictions if isinstance(r['PM_FAO56_V1'],(int,float))
                and math.isfinite(r['PM_FAO56_V1'])
                and all(r[f] is not None and math.isfinite(r[f]) for f in families)]
        if not common:
            raise ValueError('Empty ANN common TEST cohort')
        with (out/'test_predictions.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['station_code','date','PM_FAO56_V1',*families])
            writer.writeheader();writer.writerows(predictions)
        score=[]
        for family in families:
            eligible=[r for r in predictions if isinstance(r['PM_FAO56_V1'],(int,float))
                      and math.isfinite(r['PM_FAO56_V1'])
                      and r[family] is not None and math.isfinite(r[family])]
            for cohort_id,subset in [('own',eligible),('common_ann_5',common)]:
                grouped=defaultdict(list)
                for row in subset:
                    for group,key in _groups(row):grouped[group,key].append(row)
                for (group,key),members in sorted(grouped.items()):
                    m=metrics([r['PM_FAO56_V1'] for r in members],
                              [r[family] for r in members])
                    score.append({'family':family,'cohort_id':cohort_id,'group':group,'key':key,**m})
        with (out/'test_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(score[0]));writer.writeheader();writer.writerows(score)
        # Classical HGB comparison is read from the published and hash-verified run.
        hgb=Path(hgb_dir);hgb_manifest=json.loads((hgb/'manifest.json').read_text())
        if hgb_manifest['dataset_sha256']!=ann['dataset_sha256']:
            raise ValueError('HGB Dataset mismatch')
        hgb_entry=hgb_manifest['outputs']['test_predictions.csv']
        if _sha(hgb/hgb_entry['path'])!=hgb_entry['sha256']:
            raise ValueError('HGB output SHA mismatch')
        with (hgb/hgb_entry['path']).open(encoding='utf-8-sig',newline='') as stream:
            hgb_rows={(r['station_code'],r['date']):r for r in csv.DictReader(stream)}
        key='ML_HIST_GRADIENT_BOOSTING__FEATURE_SET_FULL'
        paired=[(r,hgb_rows[(r['station_code'],r['date'])]) for r in common
                if (r['station_code'],r['date']) in hgb_rows
                and hgb_rows[(r['station_code'],r['date'])].get(key) not in (None,'')]
        if len(paired)!=len(common):
            raise ValueError('ANN/HGB common cohort mismatch')
        comparison={'n':len(paired),'champion_family':selection['champion']['family'],
                    'ann':metrics([r['PM_FAO56_V1'] for r,_ in paired],
                                  [r[selection['champion']['family']] for r,_ in paired]),
                    'hgb':metrics([r['PM_FAO56_V1'] for r,_ in paired],
                                  [float(h[key]) for _,h in paired]),
                    'hgb_manifest_sha256':_sha(hgb/'manifest.json')}
        _write_json(out/'comparison.json',comparison)
        _write_checksums(out, ('test_predictions.csv','test_metrics.csv','comparison.json'))
        manifest.update(status='SUCCESS',ann_protocol_sha256=ann_sha,
                        development_manifest_sha256=_sha(dev/'manifest.json'),pipeline_sha=frozen['pipeline_sha'],dataset_sha256=ann['dataset_sha256'],dataset_id=frozen['dataset_id'],audit_source_repo_sha=frozen['audit_source_repo_sha'],source_repo_sha=frozen['source_repo_sha'],quality_post_id=frozen['quality_post_id'],
                        hgb_manifest_sha256=comparison['hgb_manifest_sha256'],
                        common_n=len(common),champion=selection['champion'],
                        outputs={name:_sha(out/name) for name in ('test_predictions.csv',
                            'test_metrics.csv','comparison.json','SHA256SUMS')})
        _write_json(out/'manifest.json',manifest)
        _journal(journal,'SUCCESS',manifest_sha256=_sha(out/'manifest.json'))
        return manifest
    except Exception as exc:
        for name in ('test_predictions.csv','test_metrics.csv','comparison.json','SHA256SUMS'):
            (out/name).unlink(missing_ok=True)
        manifest.update(status='FAILED',error=str(exc),outputs={})
        _write_json(out/'manifest.json',manifest)
        _journal(journal,'FAILED',reason=str(exc))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--development',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--ahc-manifest',required=True)
    parser.add_argument('--ml-protocol',required=True)
    parser.add_argument('--ann-protocol',required=True)
    parser.add_argument('--hgb-run',required=True)
    parser.add_argument('--output',required=True)
    a=parser.parse_args()
    evaluate(a.development,a.dataset,a.ahc_manifest,a.ml_protocol,
             a.ann_protocol,a.hgb_run,a.output)


if __name__=='__main__':main()
