"""Opt-in acceptance checks against immutable official publication bytes.

Set EXPLORER_DATA_DIR and EXPLORER_ML_SOURCE to run; CI has no scientific payload.
The independent CSV comparisons detect rounding, filtering and cohort mistakes.
"""
import csv
import json
import os
from pathlib import Path

import pytest

from web_app.backend.app import create_app
from web_app.backend.convert import sha
from test_api import ASGIClient

CHAMPION = 'ML_HIST_GRADIENT_BOOSTING__FEATURE_SET_FULL'


@pytest.fixture(scope='module')
def published():
    data, source = os.getenv('EXPLORER_DATA_DIR'), os.getenv('EXPLORER_ML_SOURCE')
    if not data or not source:
        pytest.skip('Official publication acceptance requires explicit local paths')
    source = Path(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    assert sha(source / 'manifest.json') == '05834edaebf3f990729ebb8b15ecd7511bb96e006bfab69f11408a250f432aab'
    for name, entry in manifest['outputs'].items():
        assert sha(source / name) == entry['sha256']
    rows = list(csv.DictReader((source / 'test_metrics.csv').open(encoding='utf-8-sig')))
    return ASGIClient(create_app(data)), rows


@pytest.mark.parametrize('station,year,group,key', [('',None,'global','all'),('IB11',None,'station','IB11'),('',2023,'year','2023'),('IB04',2023,'station_year','IB04:2023')])
@pytest.mark.parametrize('model', [CHAMPION,'HG_RS_AHC_VALIDATED_SELECTED','ML_LINEAR__FEATURE_SET_T'])
def test_metrics_exactly_equal_published_csv(published,station,year,group,key,model):
    client, source = published
    params = dict(station=station,models=model,cohort='common_all')
    if year: params['year'] = year
    response = client.get('/api/explorer/metrics', params=params)
    assert response.status_code == 200
    actual = response.json()['metrics']
    expected = [r for r in source if r['reference_id']=='PM_FAO56_V1' and r['cohort_id']=='common_all' and r['method_id']==model and r['group']==group and r['key']==key]
    assert len(actual)==len(expected)==1
    for field in ['mae','rmse','bias','r2']:
        assert actual[0][field] == (float(expected[0][field]) if expected[0][field] else None)
    assert actual[0]['n']==int(expected[0]['n'])
    assert actual[0]['cohort_sha256']==expected[0]['cohort_sha256']
    assert actual[0]['run_id']==expected[0]['run_id']


def test_champion_metadata_and_negative_flags(published):
    client,_=published
    catalog=client.get('/api/explorer/catalog').json()
    assert len(catalog['models'])==43
    champion=next(m for m in catalog['models'] if m['champion'])
    assert champion['id']==CHAMPION
    assert champion['candidate_id']=='mlc-95d2be33ea8867bcd84cb95281d7b7317e05fc02b4100e285a34df12e6e0647a'
    assert champion['negative_predictions']==0
    assert sum(m['negative_predictions'] for m in catalog['models'] if m.get('algorithm'))==1007
    assert catalog['audit']['manifest_sha256']=='05834edaebf3f990729ebb8b15ecd7511bb96e006bfab69f11408a250f432aab'
    assert champion['parameters']==dict(learning_rate=.1,max_iter=200,max_leaf_nodes=31,l2_regularization=0.0)


def test_cohort_filter_and_no_invented_metrics(published):
    client,_=published
    for cohort in ['own','common_all','common_ml_35']:
        r=client.get('/api/explorer/metrics',params=dict(models=CHAMPION,cohort=cohort)).json()['metrics'][0]
        assert r['cohort_id']==cohort and r['n']==15958
    assert client.get('/api/explorer/metrics',params=dict(station='IB11',group='month')).json()['metrics']==[]
    assert client.get('/api/explorer/metrics',params=dict(split='TRAIN')).json()['metrics']==[]
    assert client.get('/api/explorer/metrics',params=dict(station='bad')).status_code==422
    assert client.get('/api/explorer/predictions',params=dict(models='bad')).status_code==422


def test_predictions_sampling_and_negatives_preserved(published):
    client,_=published
    result=client.get('/api/explorer/predictions',params=dict(station='IB11',models=CHAMPION,max_points=100)).json()
    assert result['source_rows']==1822
    assert 0<len(result['points'])<=100
    assert {p['station_code'] for p in result['points']}=={'IB11'}
    assert result['affects_metrics'] is False
    source=Path(os.environ['EXPLORER_ML_SOURCE']) / 'test_predictions.csv'
    with source.open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            negative=next((m for m,v in row.items() if m.startswith('ML_LINEAR__') and v and float(v)<0),None)
            if negative:
                break
    assert negative
    linear=client.get('/api/explorer/predictions',params=dict(models=negative,station=row['station_code'],cohort='own',**{'from':row['date'],'to':row['date']})).json()
    assert linear['points'][0][negative]==float(row[negative])<0
