"""Colector experimental web de Extremadura, aislado de SIAR Sync V3."""
import csv, json, os, re, sqlite3, sys, time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo
from siar_web_descarga import retrieve, save_block

COLUMNS=('Fecha','TempMedia','TempMax','HorMinTempMax','TempMin','HorMinTempMin','HumedadMedia','HumedadMax','HorMinHumMax','HumedadMin','HorMinHumMin','VelViento','DirViento','VelVientoMax','HorMinVelMax','DirVientoVelMax','Radiacion','Precipitacion','TempSuelo1','TempSuelo2','EtPMon','PePMon','Estacion')
EXPECTED=('Fecha','Temp Media','Temp Max','Hora T','Temp Mín','Hora T','Hum Media','Hum Max','Hora Hum','Hum Min','Hora Hum','Vel Viento','Dir Viento','Vel V. Max','Hora Vel','Dir V. Vel Max','Rad','Precip','P. Efect','ET0')

def number(s):
    s=s.strip()
    if s in ('','-','--','---'): return ''
    try: n=Decimal(s.replace(',','.'))
    except InvalidOperation as e: raise ValueError('valor web no reconocido') from e
    if not n.is_finite(): raise ValueError('valor no finito')
    return str(n)

def hour(s):
    s=s.strip()
    if s in ('','-','--','---'): return ''
    m=re.fullmatch(r'(\d{1,2}):(\d{2})',s)
    if not m or int(m[1])>23 or int(m[2])>59: raise ValueError('hora web no reconocida')
    return str(int(m[1])*100+int(m[2]))

def normalize(station,headers,row):
    if len(headers)!=20 or len(row)!=20 or any(not h.startswith(e) for h,e in zip(headers,EXPECTED)): raise ValueError('cabeceras desconocidas')
    day=datetime.strptime(row[0],'%d/%m/%Y').date().isoformat()
    v=[hour(x) if i in (3,5,8,10,14) else number(x) for i,x in enumerate(row[1:],1)]
    return [day]+v[:17]+['','']+[v[18],v[17],station]

def export(db,root,station):
    result={}
    for (raw,) in db.execute('SELECT data FROM blocks WHERE station=? ORDER BY start',(station,)):
        data=json.loads(raw)
        for row in data['rows']:
            normalized=normalize(station,data['headers'],row); day=normalized[0]
            if day in result and result[day]!=normalized: raise ValueError('conflicto de datos')
            result[day]=normalized
    if not result: return
    dest=root/(station+'_datos_web_v3.csv'); temp=dest.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f); writer.writerow(COLUMNS); writer.writerows(result[k] for k in sorted(result))
        f.flush(); os.fsync(f.fileno())
    os.replace(temp,dest)

def main():
    root=Path('/home/josep/siar-web-extremadura-20261127'); root.mkdir(mode=0o700,exist_ok=True)
    stations=json.loads((Path(__file__).parent/'extremadura_estaciones.json').read_text(encoding='utf-8'))['stations']
    if len(stations)!=56 or len({x['code'] for x in stations})!=56: raise ValueError('catalogo incorrecto')
    deadline=datetime(2026,11,28,8,0,tzinfo=ZoneInfo('Europe/Madrid'))
    end=date(2026,11,26); first=date(2000,1,1)
    db=sqlite3.connect(root/'progreso.sqlite'); db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS blocks (station TEXT,start TEXT,end TEXT,captured TEXT,sha256 TEXT,data TEXT,missing TEXT, PRIMARY KEY(station,start,end))')
    db.execute('CREATE TABLE IF NOT EXISTS failures (station TEXT,start TEXT,end TEXT,attempted TEXT,error TEXT, PRIMARY KEY(station,start,end))');db.commit()
    from playwright.sync_api import sync_playwright
    consecutive=0
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        try:
            for st in stations: export(db,root,st['code'])
            while end>=first and datetime.now(ZoneInfo('Europe/Madrid'))<deadline:
                start=max(first,end-timedelta(days=30))
                for st in stations:
                    if datetime.now(ZoneInfo('Europe/Madrid'))>=deadline: break
                    code=st['code']; key=(code,start.isoformat(),end.isoformat())
                    if db.execute('SELECT 1 FROM blocks WHERE station=? AND start=? AND end=?',key).fetchone() or db.execute('SELECT 1 FROM failures WHERE station=? AND start=? AND end=?',key).fetchone(): continue
                    page=browser.new_page()
                    try:
                        headers,rows=retrieve(page,code,start,end)
                        for row in rows: normalize(code,headers,row)
                        count,missing=save_block(db,code,start,end,headers,rows)
                        export(db,root,code); consecutive=0
                        print(json.dumps({'station':code,'start':key[1],'end':key[2],'rows':count,'pending':missing}),flush=True)
                    except Exception as exc:
                        consecutive+=1
                        with db: db.execute('INSERT OR REPLACE INTO failures VALUES (?,?,?,?,?)',(*key,datetime.now(timezone.utc).isoformat(),type(exc).__name__))
                        print(json.dumps({'station':code,'start':key[1],'end':key[2],'status':'PENDING','error':type(exc).__name__}),flush=True)
                    finally: page.close()
                    if consecutive>=5: print('STOP: cinco fallos consecutivos',flush=True); return
                    time.sleep(15)
                end=start-timedelta(days=1)
        finally: browser.close();db.close()
if __name__=='__main__': main()

