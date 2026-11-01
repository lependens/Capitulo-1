"""Verify published bytes and package them for the Scientific Explorer. No science imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import duckdb

from .convert import sha
from .catalog import DATASET_SHA, CSV_SHA, AHC_MODELS

MANIFEST_SHA = '05834edaebf3f990729ebb8b15ecd7511bb96e006bfab69f11408a250f432aab'
SELECTION_SHA = 'a41ee87956368d6c0e1fae2b39177ffe8c42cae7658477172c46725409f829c7'
CHAMPION = 'ML_HIST_GRADIENT_BOOSTING__FEATURE_SET_FULL'


def package(base, source, selection_path, output):
    base, source, selection_path, output = map(Path, (base, source, selection_path, output))
    if sha(source / 'manifest.json') != MANIFEST_SHA or sha(selection_path) != SELECTION_SHA:
        raise ValueError('Unapproved ML manifest or DEVELOPMENT selection')
    manifest = json.loads((source / 'manifest.json').read_text())
    selection = json.loads(selection_path.read_text())
    champion_candidate = selection['campaign_champion']['candidate_id']
    if champion_candidate != 'mlc-95d2be33ea8867bcd84cb95281d7b7317e05fc02b4100e285a34df12e6e0647a':
        raise ValueError('Unexpected frozen DEVELOPMENT champion')
    if manifest['status'] != 'SUCCESS' or manifest['dataset_sha256'] != DATASET_SHA:
        raise ValueError('ML publication is not SUCCESS on Dataset v1')
    for name, entry in manifest['outputs'].items():
        if Path(name).name != name or sha(source / name) != entry['sha256']:
            raise ValueError('ML source integrity failure: ' + name)
    info = json.loads((base / 'info.json').read_text(encoding='utf-8'))
    if info['schema_version'] != 'web-v0.2-ahc' or info['dataset_sha256'] != DATASET_SHA or info['source_sha256'] != CSV_SHA:
        raise ValueError('Expected approved WEB-V0.2 base')
    for name, digest in info['artifacts'].items():
        if Path(name).name != name or sha(base / name) != digest:
            raise ValueError('Base integrity failure: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    for name in info['artifacts']:
        shutil.copy2(base / name, output / name)
    artifacts = dict(info['artifacts'])
    with duckdb.connect() as db:
        for filename, target in [('test_metrics.csv', 'explorer_metrics.parquet'), ('test_predictions.csv', 'ml_predictions.parquet')]:
            db.execute('COPY (SELECT * FROM read_csv_auto(?, sample_size=-1)) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)', [str(source / filename), str(output / target)])
            count = db.execute('SELECT count(*) FROM read_parquet(?)', [str(output / target)]).fetchone()[0]
            if count != manifest['outputs'][filename]['rows']:
                raise ValueError('Row count mismatch: ' + filename)
            artifacts[target] = sha(output / target)
        # Baseline values are copied by their immutable station/date identities.
        baseline_ids = manifest['protocol']['baseline_benchmark']
        base_columns = ', '.join('a."' + m + '"' for m in baseline_ids if m != 'PM_FAO56_V1')
        db.execute('COPY (SELECT p.*, ' + base_columns + ' FROM read_parquet(?) p LEFT JOIN read_parquet(?) a USING(station_code,date)) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)', [str(output / 'ml_predictions.parquet'), str(base / 'ahc_predictions.parquet'), str(output / 'explorer_predictions.parquet')])
        # Verify membership against each producer hash; never invent a scientific cohort.
        ml_ids = [r['model_id'] for r in selection['selected']]
        for cohort, methods in [('common_all', baseline_ids + ml_ids), ('common_ml_35', ml_ids)]:
            condition = ' AND '.join('isfinite(try_cast("' + m + '" AS DOUBLE))' for m in ['PM_FAO56_V1'] + methods)
            keys = db.execute('SELECT station_code, cast(date AS VARCHAR) FROM read_parquet(?) WHERE ' + condition + ' ORDER BY station_code,date', [str(output / 'explorer_predictions.parquet')]).fetchall()
            digest = hashlib.sha256((json.dumps([list(k) for k in keys], sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()).hexdigest()
            expected = db.execute('SELECT DISTINCT cohort_sha256 FROM read_parquet(?) WHERE cohort_id=? AND "group"=\'global\' AND reference_id=\'PM_FAO56_V1\'', [str(output / 'explorer_metrics.parquet'), cohort]).fetchall()
            if expected != [(digest,)]:
                raise ValueError('Published cohort keys do not match joined representation: ' + cohort)
        artifacts['explorer_predictions.parquet'] = sha(output / 'explorer_predictions.parquet')
        metadata = list(csv.DictReader((source / 'model_metadata.csv').open(encoding='utf-8-sig')))
        for row in metadata:
            for key in ['features', 'parameters', 'fit_period']:
                row[key] = json.loads(row[key])
            row.pop('fitted_diagnostics', None)
            row.update(id=row['model_id'], label=row['algorithm'].replace('_', ' ').title() + ' · ' + row['feature_set'].replace('FEATURE_SET_', '').replace('_', '+'), category='Machine Learning', champion=row['candidate_id'] == champion_candidate)
        ahc = {m['id']:m for m in AHC_MODELS}
        for identifier in baseline_ids:
            item = ahc.get(identifier, {})
            base_id = item.get('base_method_id', identifier)
            inputs = {'PM_FAO56_V1':['Tcalc','TD','RH','Rs','wind','Ra','latitude_deg','altitude_m'], 'HS_RA_V1':['Tcalc','TD','Ra'], 'HG_RS_V1':['Tcalc','Rs'], 'VALIANTZAS_V1':['Tcalc','TD','RH','Ra']}
            formulas = {'HS_RA_V1':'0.0023 × 0.408 × Ra × (Tcalc+17.8) × √TD', 'HG_RS_V1':'0.0135 × 0.408 × Rs × (Tcalc+17.8)', 'VALIANTZAS_V1':'0.001861704 × Ra × TD^0.3 × (1.001−RHmedia/100)^0.2 × (Tcalc+17.8)'}
            formula = formulas.get(base_id,'FAO-56 Penman–Monteith')
            if 'AHC' in identifier:
                formula = 'a_station,group × (' + formula + ') · coefficient TRAIN congelado'
            metadata.append(dict(id=identifier, label=item.get('label', identifier), category='Referencia calculada' if identifier == 'PM_FAO56_V1' else 'Calibrado AHC' if 'AHC' in identifier else 'Determinista / empírico', features=inputs[base_id], formula=formula, champion=False, published=item))
        for row in metadata:
            identifier = row['id']
            row['negative_predictions'] = db.execute('SELECT count(*) FROM read_parquet(?) WHERE contains(scientific_flags, ?)', [str(output / 'explorer_predictions.parquet'), 'negative_prediction:' + identifier + '"']).fetchone()[0]
    audit = {key:manifest[key] for key in ['dataset_id','dataset_sha256','run_id','pipeline','protocol_sha256','protocol_file_sha256','development_selection_sha256','refitted_model_bundle_sha256','ahc_test_manifest_sha256','fit_period','stage']}
    audit['pipeline'] = {'git_sha': manifest['pipeline']['git_sha']}
    audit.update(quality_post=info['quality_post'], scientific_status='SUCCESS · published recovery manifest; verified by consumer')
    audit.update(manifest_sha256=MANIFEST_SHA, source_artifact='ml-v1-test-recovery-20261101T090828Z', source_hashes={name:e['sha256'] for name,e in manifest['outputs'].items()}, development_folds=manifest['protocol']['development_folds'], champion=selection['campaign_champion'], selection_status=selection['status'], test_semantics='retrospective_recovery_evaluation', reference='PM_FAO56_V1', science_recalculation_allowed=False)
    for name, value in [('explorer_models.json', metadata), ('explorer_audit.json', audit), ('explorer_selection.json', selection)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        artifacts[name] = sha(output / name)
    info.update(schema_version='web-v0.3-explorer', conversion_version='3.0.0', reference='PM_FAO56_V1 · referencia calculada', limitations=[v for v in info['limitations'] if 'Forecast y ML' not in v] + ['ML TEST: retrospective_recovery_evaluation; no holdout prospectivo virgen.', 'ANN moderna NOT STARTED. Forecast fuera del explorador.'], web_version='0.3.0', artifacts=artifacts, ml_manifest_sha256=MANIFEST_SHA, derived_sha256=artifacts['explorer_predictions.parquet'])
    (output / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['base', 'source', 'selection', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.base, args.source, args.selection, args.output), indent=2))
