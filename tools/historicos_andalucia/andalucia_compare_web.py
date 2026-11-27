"""Compare captured SIAR web values with a local SIAR historical text.

No network access; no credentials; no changes to either source or V3.
The reference JSON must contain one mapping from ISO date to 13 strings in the
documented field order. CLI prints structured evidence to stdout.
"""
import argparse
import json
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from andalucia_stage import parse_file

FIELDS=('TempMedia','TempMax','HorMinTempMax','TempMin','HorMinTempMin','HumedadMedia','HumedadMax','HumedadMin','VelViento','DirViento','Radiacion','Precipitacion','EtPMon')

def compare(source, station, reference, dates, *, header_label=None, max_date=date(2020,6,5)):
    if len(dates)!=10 or len(set(dates))!=10: raise ValueError('exactly ten unique dates required')
    data=parse_file(source,station,2000,header_label=header_label,max_date=max_date)
    by_date={x['Fecha']:x for x in data['rows']}
    if set(reference)!=set(dates): raise ValueError('reference dates do not match the approved plan')
    results=[]
    for day in sorted(dates):
        if day not in by_date: raise ValueError('date missing or quarantined: '+day)
        values=reference[day]
        if not isinstance(values,list) or len(values)!=len(FIELDS): raise ValueError('wrong reference field count: '+day)
        original=by_date[day]; mismatches=[]; compared=[]; omitted=[]
        for field,display in zip(FIELDS,values):
            historic=original[field]; display=display.strip()
            if not historic or not display:
                omitted.append(field); continue
            if field.startswith('HorMin'):
                try:
                    hours,minutes=map(int,display.split(':')); matches=int(historic)==hours*100+minutes
                except ValueError as exc: raise ValueError('invalid hour') from exc
            else:
                try:
                    old=Decimal(historic); new=Decimal(display.replace(',','.'))
                    if not old.is_finite() or not new.is_finite(): raise ValueError('non-finite')
                except InvalidOperation as exc: raise ValueError('invalid number') from exc
                decimals=len(historic.partition('.')[2]); margin=Decimal(5)*Decimal(10)**(-decimals-1)+Decimal('0.005')
                matches=abs(old-new)<=margin
            compared.append(field)
            if not matches and field!='EtPMon': mismatches.append({'field':field,'historic':historic,'siar':display})
        # No empty sample can receive a pass.
        core=set(FIELDS)-{'EtPMon'}
        results.append({'date':day,'status':'REVIEW' if mismatches else ('PASS' if core.issubset(compared) else 'INCOMPLETE'), 'mismatches':mismatches,'compared':compared,'omitted':omitted,'historic_et0':original['EtPMon'],'siar_et0':values[-1]})
    return {'station_candidate':station,'source_sha256':data['source_sha256'],'reference_type':'user_supplied_siar_web_capture','results':results,'status':'PASS' if all(x['status']=='PASS' for x in results) else 'REVIEW'}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--station',required=True); p.add_argument('--reference-json',required=True); p.add_argument('--plan-json',required=True); p.add_argument('--header-label')
    a=p.parse_args(); plan=json.loads(Path(a.plan_json).read_text(encoding='utf-8')); reference=json.loads(Path(a.reference_json).read_text(encoding='utf-8'))
    expected_hash=None
    if isinstance(plan,list):
        entries=[x for x in plan if x.get('station_candidate')==a.station]
        if len(entries)!=1: raise ValueError('ambiguous or missing station in plan')
        entry=entries[0]; expected_hash=entry['sha256']
        from datetime import timedelta
        if len(entry['windows'])!=2: raise ValueError('station has no two-block plan')
        dates=[]
        for w in entry['windows']:
            start=date.fromisoformat(w['start']); end=date.fromisoformat(w['end'])
            if (end-start).days!=4: raise ValueError('invalid block length')
            dates.extend((start+timedelta(days=i)).isoformat() for i in range(5))
    else: dates=plan['dates']
    result=compare(a.source,a.station,reference,dates,header_label=a.header_label)
    if expected_hash and result['source_sha256']!=expected_hash: raise ValueError('source hash differs from plan')
    print(json.dumps(result,ensure_ascii=False,indent=2))

