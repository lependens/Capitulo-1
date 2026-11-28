"""Read-only ET0 content classification; no source emission is inferred."""
import csv, io
from datetime import datetime
from decimal import Decimal, InvalidOperation

EXPECTED = ('Fecha','Kc','ET0','ETc','Pe','ETc - Pe')

def classify_csv(data: bytes) -> dict:
    try:
        raw=list(csv.reader(io.StringIO(data.decode('utf-8-sig'),newline=''),delimiter=';'))
    except (UnicodeError,csv.Error):
        return {'status':'failed','reason':'unreadable_csv','rows':0}
    if not raw or tuple(raw[0]) != EXPECTED:
        return {'status':'failed','reason':'unexpected_header','rows':max(0,len(raw)-1)}
    rows=raw[1:]
    if not rows:return {'status':'empty','reason':'no_rows','rows':0,'rows_with_et0':0}
    valid=0
    for row in rows:
        if len(row)!=len(EXPECTED):return {'status':'failed','reason':'row_width','rows':len(rows)}
        parsed=False
        for fmt in ('%d/%m/%Y','%Y-%m-%d','%d-%m-%Y','%Y/%m/%d'):
            try: datetime.strptime(row[0].strip(),fmt);parsed=True;break
            except ValueError: pass
        if not parsed:return {'status':'failed','reason':'invalid_date','rows':len(rows)}
        if row[2].strip():
            try:
                value=Decimal(row[2].strip().replace(',','.'))
                if value.is_finite() and value>=0:valid+=1
            except InvalidOperation:
                return {'status':'failed','reason':'invalid_et0','rows':len(rows)}
    return {'status':'useful' if valid==len(rows) else 'partial','reason':'et0_coverage','rows':len(rows),'rows_with_et0':valid}
