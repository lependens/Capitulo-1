"""Scientific Explorer routes: filter published metrics, sample published predictions."""
import json
import math
from datetime import date
from typing import Literal
from fastapi import APIRouter, HTTPException, Query


def router(folder, db, stations):
    api = APIRouter(prefix='/api/explorer')
    load = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    models, audit, selection = (load(n) for n in ['explorer_models.json','explorer_audit.json','explorer_selection.json'])
    model_ids = {m['id'] for m in models}
    ml_ids = [m['id'] for m in models if m['category'] == 'Machine Learning']
    station_ids = {s['code'] for s in stations}
    for view, filename in [('explorer_metrics','explorer_metrics.parquet'),('explorer_predictions','explorer_predictions.parquet')]:
        path = str(folder / filename).replace("'", "''")
        db.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{path}')")
    db.execute("CREATE VIEW explorer_published AS SELECT * FROM explorer_metrics UNION ALL BY NAME SELECT * FROM ahc_metrics WHERE split IN ('TRAIN','VALIDATION')")

    def query(sql, values=()):
        cur = db.execute(sql, values)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    def validate(station, ids=()):
        if station and station not in station_ids:
            raise HTTPException(422, 'Unknown station')
        if any(m not in model_ids for m in ids):
            raise HTTPException(422, 'Unknown model')

    @api.get('/catalog')
    async def catalog():
        return dict(models=models, stations=stations, reference='PM_FAO56_V1', reference_meaning='Referencia calculada; no medición independiente de ET₀', cohorts=['common_all','common_ml_35','own'], splits=['TEST','TRAIN','VALIDATION','ALL'], test_period=['2021-01-01','2025-12-31'], audit=audit, selection=selection, data_recalculated=False)

    @api.get('/metrics')
    async def metrics(station: str | None = None, models: str = '', split: Literal['TEST','TRAIN','VALIDATION','ALL'] = 'TEST', cohort: str = 'common_all', group: Literal['global','station','year','month','season','season_year'] = 'global', year: int | None = Query(None,ge=2004,le=2025)):
        ids = list(filter(None, models.split(',')))
        validate(station, ids)
        if split in ('TRAIN','VALIDATION','ALL') and cohort not in ('own','common_7','common_selected'):
            if split != 'ALL':
                return dict(metrics=[], unavailable_reason='TRAIN/VALIDATION solo publican cohortes AHC: own, common_7 o common_selected. ML usa DEVELOPMENT rolling; no hay métricas ML TRAIN/VALIDATION.', comparable=False)
        if cohort not in ('own','common_all','common_ml_35','common_7','common_selected'):
            raise HTTPException(422,'Unknown published cohort')
        actual_group = ('station' if group == 'global' else 'station_year' if group == 'year' else None) if station else group
        if actual_group is None or (year and group not in ('global','year','station')):
            return dict(metrics=[], unavailable_reason='No existe ese agregado publicado. Cambie el grupo o la estación; WEB no recalcula métricas.', comparable=False)
        if year:
            actual_group = 'station_year' if station or group == 'station' else 'year'
        clauses, values = ['reference_id=?','cohort_id=?','"group"=?'], ['PM_FAO56_V1',cohort,actual_group]
        if station:
            clauses.append('station_code=?'); values.append(station)
        if year:
            if group == 'station' and not station:
                clauses.append('ends_with(key, ?)'); values.append(':' + str(year))
            else:
                clauses.append('key=?'); values.append(f'{station}:{year}' if station else str(year))
        if split != 'ALL':
            clauses.append('split=?'); values.append(split)
        if ids:
            clauses.append('method_id IN (' + ','.join('?' for _ in ids) + ')'); values.extend(ids)
        view = 'ahc_metrics' if cohort in ('common_7','common_selected') or split in ('TRAIN','VALIDATION') else 'explorer_published'
        rows = query('SELECT * FROM '+view+' WHERE '+' AND '.join(clauses)+' ORDER BY split,key,rmse,method_id', values)
        for row in rows:
            ml_publication = row.get('run_id') == audit['run_id']
            row['source_artifact'] = audit['source_artifact'] if ml_publication else 'ahc-20261024T172400Z/test'
            row['source_table'] = 'test_metrics.csv' if ml_publication else 'metrics.csv'
            row['test_semantics'] = audit['test_semantics'] if ml_publication else 'retrospective_locked'
            # Denominator is availability of preserved rows, never an accuracy calculation.
            conditions, params = [], []
            if ml_publication:
                row_station = row.get('station_code') if actual_group in ('station','station_year') else station
                if row_station: conditions.append('station_code=?'); params.append(row_station)
                if actual_group in ('year','station_year'): conditions.append('year(date)=?'); params.append(int(row['key'].split(':')[-1]))
                if actual_group == 'month': conditions.append('month(date)=?'); params.append(int(row['key']))
                if actual_group == 'global' or actual_group in ('station','year','station_year','month'):
                    row['source_rows'] = query('SELECT count(*) n FROM explorer_predictions WHERE '+(' AND '.join(conditions) or 'true'),params)[0]['n']
        signatures = {(r['split'],r['key'],r['cohort_sha256']) for r in rows}
        comparable = len(signatures) == 1 and bool(rows)
        return dict(metrics=rows, comparable=comparable, comparison_note='Mismas claves estación-fecha verificadas' if comparable else 'Compare solo filas del mismo split, grupo y SHA de cohorte; grupos temporales distintos no son una cohorte única.', data_recalculated=False, period_filter_applied=bool(year))

    @api.get('/predictions')
    async def predictions(models: str, station: str | None = None, year: int | None = Query(None,ge=2021,le=2025), cohort: Literal['common_all','common_ml_35','own'] = 'common_all', start: date | None = Query(None, alias='from'), end: date | None = Query(None, alias='to'), max_points: int = Query(1000,ge=50,le=3000)):
        ids = list(dict.fromkeys(filter(None,models.split(','))))
        validate(station, ids)
        if not ids or len(ids)>8:
            raise HTTPException(422,'Choose 1–8 models')
        if start and end and start>end:
            raise HTTPException(422,'from must be before to')
        members = list(model_ids) if cohort == 'common_all' else ml_ids if cohort == 'common_ml_35' else ids
        clauses = ['isfinite(try_cast(PM_FAO56_V1 AS DOUBLE))','PM_FAO56_V1>=0'] + [f'isfinite(try_cast("{m}" AS DOUBLE))' for m in members]
        values=[]
        for expression,value in [('station_code=?',station),('year(date)=?',year),('date>=?',start),('date<=?',end)]:
            if value is not None: clauses.append(expression); values.append(str(value) if isinstance(value,date) else value)
        condition=' AND '.join(clauses)
        count=query('SELECT count(*) n FROM explorer_predictions WHERE '+condition,values)[0]['n']
        stride=max(1,math.ceil(count/max_points))
        columns=','.join('"'+m+'"' for m in ids if m!='PM_FAO56_V1')
        points=query('WITH selected AS (SELECT *,row_number() OVER(ORDER BY date,station_code)-1 rn FROM explorer_predictions WHERE '+condition+') SELECT station_code,date,PM_FAO56_V1,scientific_flags'+(','+columns if columns else '')+f' FROM selected WHERE rn % {stride}=0 ORDER BY date,station_code',values)
        return dict(points=points,source_rows=count,returned_points=len(points),sampling=f'Una de cada {stride} filas, orden determinista',affects_metrics=False,reference='PM_FAO56_V1',cohort=cohort,test_semantics=audit['test_semantics'],source_artifact=audit['source_artifact'],manifest_sha256=audit['manifest_sha256'],note='Residual y distribución: representación visual de predicción − PM; no métricas nuevas. No clipping.')

    return api
