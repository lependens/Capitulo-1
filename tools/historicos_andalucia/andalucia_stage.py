"""Conservative staging of labelled SIAR fixed-width exports; no production I/O."""
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

COLUMNS = ('Fecha','TempMedia','TempMax','HorMinTempMax','TempMin','HorMinTempMin','HumedadMedia','HumedadMax','HorMinHumMax','HumedadMin','HorMinHumMin','VelViento','DirViento','VelVientoMax','HorMinVelMax','DirVientoVelMax','Radiacion','Precipitacion','TempSuelo1','TempSuelo2','EtPMon','PePMon','Estacion')
HEADER = re.compile(r'^FECHA\s+DIA\s+(\S+?)TMax\b', re.I)
DATE = re.compile(r'^(\d{2})-(\d{2})-(\d{2})(?:\s+(.*))?$')

def parse_file(path, station, century):
    """Return rows and exceptions; never infer a station or century silently."""
    raw = Path(path).read_bytes()
    text = raw.decode('utf-8-sig', errors='strict')
    lines = text.splitlines()
    if not lines or not (match := HEADER.match(lines[0])):
        raise ValueError('station header not found')
    label = unicodedata.normalize('NFKD', match.group(1)).encode('ascii','ignore').decode().upper()
    if not re.fullmatch(r'[A-Z]{1,2}\d{2,3}', station) or label != station:
        raise ValueError(f'header {label} != confirmed station {station}')
    rows, exceptions = [], []
    for index, line in enumerate(lines[2:], start=3):
        if not line.strip():
            continue
        m = DATE.fullmatch(line.strip())
        if not m:
            exceptions.append({'line': index, 'reason':'unparsed_line', 'raw':line}); continue
        try:
            day, month, year = map(int, m.group(1,2,3))
            actual = date(century+year,month,day)
        except ValueError:
            exceptions.append({'line':index,'reason':'invalid_date','raw':line}); continue
        rest=m.group(4)
        if not rest:
            exceptions.append({'line':index,'reason':'date_only','date':actual.isoformat(),'raw':line}); continue
        parts=rest.split()
        if len(parts)!=14:
            exceptions.append({'line':index,'reason':'invalid_fields','date':actual.isoformat(),'raw':line}); continue
        try:
            ordinal=int(parts[0]); inferred=date(actual.year,1,1)+timedelta(days=ordinal-1)
            if ordinal < 1 or inferred.year != actual.year: raise ValueError()
        except ValueError:
            exceptions.append({'line':index,'reason':'invalid_day_of_year','date':actual.isoformat(),'raw':line}); continue
        if inferred != actual:
            exceptions.append({'line':index,'reason':'date_day_mismatch','date':actual.isoformat(),'day_candidate':inferred.isoformat(),'raw':line}); continue
        def timeval(value):
            m=re.fullmatch(r'(\d{1,2}):(\d{2})',value)
            if not m or int(m[1]) > 23 or int(m[2]) > 59: raise ValueError('invalid time')
            return str(int(m[1])*100+int(m[2]))
        try:
            for i in (1,3,5,6,7,8,9,10,11,12,13):
                if not math.isfinite(float(parts[i])): raise ValueError('non-finite value')
            row=dict.fromkeys(COLUMNS,'')
            row.update(Fecha=actual.isoformat(),Estacion=station,TempMax=parts[1],HorMinTempMax=timeval(parts[2]),TempMin=parts[3],HorMinTempMin=timeval(parts[4]),TempMedia=parts[5],HumedadMax=parts[6],HumedadMin=parts[7],HumedadMedia=parts[8],VelViento=parts[9],DirViento=parts[10],Radiacion=parts[11],Precipitacion=parts[12],EtPMon=parts[13])
        except ValueError:
            exceptions.append({'line':index,'reason':'invalid_value','date':actual.isoformat(),'raw':line}); continue
        rows.append((index,row))
    counts=Counter(r['Fecha'] for _,r in rows)
    blocked={value for item in exceptions for value in (item.get('date'),item.get('day_candidate')) if value}
    accepted=[]
    for index,row in rows:
        if counts[row['Fecha']] > 1: exceptions.append({'line':index,'reason':'duplicate_date','date':row['Fecha'],'raw':lines[index-1]})
        elif row['Fecha'] in blocked: exceptions.append({'line':index,'reason':'date_has_ambiguous_evidence','date':row['Fecha'],'raw':lines[index-1]})
        else: accepted.append(row)
    return {'source_sha256':hashlib.sha256(raw).hexdigest(),'station':station,'rows':accepted,'exceptions':exceptions}

