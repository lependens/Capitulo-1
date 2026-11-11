"""Offline IB09 V2 evidence inventory; reads preserved inputs only."""
import collections, csv, datetime, hashlib, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scientific_pipeline' / 'src'))
from siar_science.quality_ib09_v2 import ib09_v2_eligibility, IB09_RAW_SHA256
RAW = ROOT / 'datos_siar_baleares' / 'IB09_datos_completos.csv'
LEDGER = ROOT / 'data_quality/siar/post/SIAR-QA-POST-20261020T193140Z/siar_missing_days.csv'
METHODS = ('PM_FAO56_V1', 'HS_RA_V1', 'HG_RS_V1', 'VALIANTZAS_V1')

def inventory():
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != IB09_RAW_SHA256: raise ValueError('raw hash mismatch')
    with RAW.open(encoding='utf-8', newline='') as f: rows = list(csv.DictReader(f))
    with LEDGER.open(encoding='utf-8', newline='') as f: gaps = [r for r in csv.DictReader(f) if r['station_code'] == 'IB09']
    if any(r['gap_status'] != 'CONFIRMED_NO_DATA' or r['last_query_result'] != 'ZERO_ROWS' for r in gaps): raise ValueError('unresolved gap')
    dates = [r['Fecha'] for r in rows]
    if len(dates) != len(set(dates)) or dates != sorted(dates) or any(r['Estacion'] != 'IB09' for r in rows): raise ValueError('raw structural mismatch')
    start,end = datetime.date(2005,10,24),datetime.date(2018,10,10)
    expected = {(start+datetime.timedelta(days=n)).isoformat() for n in range((end-start).days+1)}
    official = [r for r in rows if r['Fecha'] in expected]
    gapdates = {r['observation_date'] for r in gaps}
    if expected - {r['Fecha'] for r in official} != gapdates or len(gaps)!=len(gapdates): raise ValueError('ledger mismatch')
    counts = collections.Counter(); year = collections.defaultdict(collections.Counter); reason = {m:collections.Counter() for m in METHODS}; common=0
    for r in official:
        values={}
        for k,v in r.items():
            if k in ('Fecha','Estacion'): continue
            try: values[k]=float(v) if v else None
            except ValueError: values[k]=None
        row={'station_code':'IB09','date':r['Fecha'],'source_data_sha':digest,'quality_status':'FAIL','observation_status':'OBSERVED','values':values}
        ok=[]
        for m in METHODS:
            result=ib09_v2_eligibility(row,{'latitude_deg':39+41/60+50/3600,'altitude_m':105},m,verified_raw_sha256=digest)
            reason[m].update(result)
            if not result: counts[m]+=1; year[r['Fecha'][:4]][m]+=1; ok.append(m)
        if len(ok)==len(METHODS): common+=1; year[r['Fecha'][:4]]['all_methods']+=1
        year[r['Fecha'][:4]]['observed']+=1
    for day in expected: year[day[:4]]['expected']+=1
    for day in gapdates: year[day[:4]]['confirmed_no_data']+=1
    return {'policy':'QUALITY-IB09-V2-PROTOTYPE','raw_sha256':digest,'ledger_sha256':hashlib.sha256(LEDGER.read_bytes()).hexdigest(),'official_period':[str(start),str(end)],'raw_rows':len(rows),'official_rows':len(official),'outside_dates':[d for d in dates if d not in expected],'confirmed_no_data':len(gaps),'method_eligible':dict(counts),'all_methods_intersection':common,'reasons':{m:dict(v) for m,v in reason.items()},'annual':{y:dict(v) for y,v in sorted(year.items())}}
if __name__ == '__main__': print(json.dumps(inventory(),ensure_ascii=False,indent=2,sort_keys=True))
