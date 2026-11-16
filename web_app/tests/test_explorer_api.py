"""Synthetic API contract checks run in CI without official scientific payloads."""
import json
import duckdb
from fastapi import FastAPI
import pytest

from web_app.backend.explorer import router
from web_app.backend.convert_explorer import package
from test_api import ASGIClient


@pytest.fixture
def explorer(tmp_path):
    models=[dict(id='PM_FAO56_V1',category='Referencia calculada'),dict(id='ML_A',category='Machine Learning'),dict(id='ML_B',category='Machine Learning')]
    for filename,value in [('explorer_models.json',models),('explorer_audit.json',dict(run_id='synthetic',source_artifact='synthetic-only',test_semantics='retrospective_recovery_evaluation',manifest_sha256='a'*64)),('explorer_selection.json',{})]:
        (tmp_path/filename).write_text(json.dumps(value),encoding='utf-8')
    db=duckdb.connect(config={'threads':1})
    db.execute('''CREATE TABLE fixture_metrics AS SELECT 'synthetic' run_id,'TEST' split,'PM_FAO56_V1' reference_id,
        'global' "group",'all' key,NULL::VARCHAR station_code,'common_all' cohort_id,'a' cohort_sha256,
        'ML_A' method_id,DATE '2023-01-01' start_date,DATE '2023-01-02' end_date,
        2::BIGINT n,0.123456789123::DOUBLE mae,0.876543210987::DOUBLE rmse,-0.03123456789::DOUBLE bias,-1.23456789::DOUBLE r2''')
    db.execute("INSERT INTO fixture_metrics SELECT run_id,split,reference_id,\"group\",key,station_code,'own','different',method_id,start_date,end_date,n,mae,rmse,bias,r2 FROM fixture_metrics")
    db.execute('COPY fixture_metrics TO ? (FORMAT PARQUET)',[str(tmp_path/'explorer_metrics.parquet')])
    db.execute('CREATE VIEW ahc_metrics AS SELECT * FROM fixture_metrics WHERE false')
    db.execute('''COPY (SELECT 'IB01' station_code, DATE '2023-01-01' date,0::DOUBLE PM_FAO56_V1,-0.25::DOUBLE ML_A,1::DOUBLE ML_B,'["negative_prediction:ML_A"]' scientific_flags
        UNION ALL SELECT 'IB01',DATE '2023-01-02',1,1.1,1.2,'[]') TO ? (FORMAT PARQUET)''',[str(tmp_path/'explorer_predictions.parquet')])
    app=FastAPI();app.include_router(router(tmp_path,db,[dict(code='IB01')]))
    yield ASGIClient(app)
    db.close()


def test_metrics_verbatim_and_comparability(explorer):
    result=explorer.get('/api/explorer/metrics').json()
    row=result['metrics'][0]
    assert result['comparable'] is True
    assert (row['n'],row['mae'],row['rmse'],row['bias'],row['r2'])==(2,0.123456789123,0.876543210987,-0.03123456789,-1.23456789)
    assert row['source_rows']==2
    assert explorer.get('/api/explorer/metrics',params=dict(cohort='own')).json()['metrics'][0]['cohort_sha256']=='different'


def test_no_fabricated_station_month_or_ml_train(explorer):
    result=explorer.get('/api/explorer/metrics',params=dict(station='IB01',group='month')).json()
    assert result['metrics']==[] and result['unavailable_reason']
    assert explorer.get('/api/explorer/metrics',params=dict(split='TRAIN')).json()['metrics']==[]
    assert explorer.get('/api/explorer/metrics',params=dict(cohort='forged')).status_code==422


def test_prediction_filtering_zero_and_negative(explorer):
    params=dict(models='ML_A',station='IB01',**{'from':'2023-01-01','to':'2023-01-01'})
    data=explorer.get('/api/explorer/predictions',params=params).json()
    assert data['source_rows']==data['returned_points']==1
    assert data['points'][0]['ML_A']==-.25
    assert data['points'][0]['PM_FAO56_V1']==0
    assert data['affects_metrics'] is False
    assert explorer.get('/api/explorer/predictions',params=dict(models='ML_A"; DROP TABLE x')).status_code==422


def test_converter_rejects_unapproved_source_before_output(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'manifest.json').write_text('{}')
    selection=tmp_path/'selection.json';selection.write_text('{}')
    target=tmp_path/'output'
    with pytest.raises(ValueError,match='Unapproved'):
        package(tmp_path,source,selection,target)
    assert not target.exists()
