"""Validate the locked pending IB09 handoff, offline and read-only.

This validates this handoff's deliberately restricted JSON Schema subset
(type, required, properties, additionalProperties, const). It does not
verify external official evidence or bind derived rows to source bytes.
"""
import csv, datetime, hashlib, json
from pathlib import Path

COLUMNS = tuple("Fecha TempMedia TempMax HorMinTempMax TempMin HorMinTempMin HumedadMedia HumedadMax HorMinHumMax HumedadMin HorMinHumMin VelViento DirViento VelVientoMax HorMinVelMax DirVientoVelMax Radiacion Precipitacion TempSuelo1 TempSuelo2 EtPMon PePMon Estacion".split())

class HandoffError(ValueError):
    pass

def _require(ok, message):
    if not ok: raise HandoffError(message)

def _schema_check(value, schema, path='$'):
    if 'const' in schema:
        expected=schema['const']
        _require(type(value) is type(expected) and value == expected, 'schema const mismatch: '+path)
    if schema.get('type') == 'object':
        _require(isinstance(value,dict), 'schema object: '+path)
        _require(set(value)==set(schema['required']), 'schema keys: '+path)
        for key, child in schema['properties'].items(): _schema_check(value[key],child,path+'.'+key)

def row_status(station, day):
    _require(station == 'IB09', 'station is not IB09')
    try: d=datetime.date.fromisoformat(day)
    except (TypeError,ValueError): raise HandoffError('invalid date')
    return 'CANDIDATE_ONLY' if datetime.date(2005,10,24)<=d<=datetime.date(2018,10,10) else 'OUTSIDE_OFFICIAL_PERIOD'

def validate_handoff(handoff, root, *, for_admission=False):
    schema_bytes=(Path(__file__).resolve().parent/'schema.json').read_bytes().replace(b'\r\n',b'\n')
    _require(hashlib.sha256(schema_bytes).hexdigest() == '6f02ea4a3333dbb956a3359c938ddc91433fd3fdf2de8d3cdaf6a98fe73a60af', 'schema hash mismatch')
    _schema_check(handoff,json.loads(schema_bytes))
    _require(not for_admission, 'admission blocked: official period unresolved and decision not effective')
    root=Path(root)
    for name in ('raw','catalog','ledger'):
        p=root / handoff[name+'_path']
        _require(hashlib.sha256(p.read_bytes()).hexdigest()==handoff[name+'_sha256'],name+' hash mismatch')
    for item in (handoff['quality_post'], {'path':handoff['policy']['prototype_path'], 'sha256':handoff['policy']['prototype_sha256']}):
        _require(hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'], 'referenced artifact hash mismatch')
    with (root/handoff['raw_path']).open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f); _require(tuple(reader.fieldnames or ())==COLUMNS,'raw schema mismatch'); rows=list(reader)
    _require(len(rows)==4423 and all(r['Estacion']=='IB09' and len(r)==23 and None not in r for r in rows),'raw station or schema mismatch')
    dates=[r['Fecha'] for r in rows]
    _require(len(set(dates))==len(dates) and dates==sorted(dates),'duplicate or unordered observation')
    inside={d for d in dates if row_status('IB09',d)=='CANDIDATE_ONLY'}
    outside=[d for d in dates if row_status('IB09',d)=='OUTSIDE_OFFICIAL_PERIOD']
    _require(len(inside)==4422 and outside==['2005-10-23'],'period mismatch')
    with (root/handoff['ledger_path']).open(encoding='utf-8',newline='') as f: gaps=[r for r in csv.DictReader(f) if r['station_code']=='IB09']
    _require(len(gaps)==313 and all(r['gap_status']=='CONFIRMED_NO_DATA' and r['last_query_result']=='ZERO_ROWS' and r['siar_records_returned']=='0' for r in gaps),'ledger status mismatch')
    gapdates=[r['observation_date'] for r in gaps]
    _require(len(gapdates)==len(set(gapdates)), 'duplicate ledger date')
    a=datetime.date(2005,10,24); b=datetime.date(2018,10,10)
    expected={(a+datetime.timedelta(days=i)).isoformat() for i in range((b-a).days+1)}
    _require(set(gapdates)==expected-inside, 'observation-ledger contradiction')
    return {'valid':True,'official_period_verification_status':'UNRESOLVED','v2_scientific_admission_status':'PENDING','decision_effective':False}

def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('--for-admission',action='store_true'); args=p.parse_args()
    here=Path(__file__).resolve().parent
    result=validate_handoff(json.loads((here/'handoff.json').read_text(encoding='utf-8')),args.root,for_admission=args.for_admission)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
