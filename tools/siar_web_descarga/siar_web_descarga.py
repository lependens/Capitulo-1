"""Descarga experimental de la tabla diaria p�blica SIAR, sin API ni V3."""
import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

URL = 'https://servicio.mapa.gob.es/siarweb/consultaDatos/consultaDatos'
SCHEMA = {0:'fecha', 1:'tempmedia', 2:'tempmax', 3:'horat', 4:'tempm',
          5:'horat', 6:'hummedia', 7:'hummax', 9:'hummin', 11:'velviento',
          12:'dirviento', 16:'rad', 17:'precip', 19:'et0'}

def spans(first, last, days):
    while first <= last:
        end = min(last, first + timedelta(days=days-1))
        yield first, end
        first = end + timedelta(days=1)

def validate(headers, rows, start, end):
    normalized = [x.lower().replace(' ', '') for x in headers]
    if len(headers) <= max(SCHEMA) or any(not normalized[i].startswith(p) for i,p in SCHEMA.items()):
        raise ValueError('Esquema de SIAR no reconocido')
    found = {}
    for row in rows:
        if len(row) != len(headers): raise ValueError('N�mero de columnas incoherente')
        try: day = datetime.strptime(row[0], '%d/%m/%Y').date()
        except ValueError as exc: raise ValueError('Fecha no reconocida') from exc
        if not start <= day <= end or day in found: raise ValueError('Fecha duplicada o fuera del intervalo')
        found[day] = row
    if not found: raise ValueError('Sin filas verificables; consulta pendiente')
    missing = [x.isoformat() for x, _ in spans(start,end,1) if x not in found]
    return found, missing

def retrieve(page, station, start, end):
    page.goto(URL, wait_until='domcontentloaded', timeout=60000)
    page.get_by_role('link',name='Seleccionar estaciones *').click()
    page.get_by_role('textbox',name='Seleccionar por nombre corto').fill(station)
    page.get_by_role('button',name=re.compile(r'Buscar estaci')).click()
    page.locator('p.text-resultado').filter(has_text=re.compile(r'^(?:Inactiva\s+)?'+re.escape(station)+r'\s+-')).wait_for(timeout=30000)
    page.get_by_role('button',name=re.compile(r'Incluir selecci')).click()
    page.get_by_role('textbox',name='Fecha inicial *').fill(start.isoformat())
    page.get_by_role('textbox',name='Fecha final *').fill(end.isoformat())
    page.get_by_role('button',name='Realizar consulta').click()
    panel=page.get_by_role('tabpanel',name=re.compile(r'^'+re.escape(station)+r'\s+-'))
    panel.wait_for(timeout=60000)
    panel.locator('table tbody tr').first.wait_for(timeout=30000)
    headers=[x.text_content().strip() for x in panel.locator('table thead th').all()]
    rows=[[x.text_content().strip() for x in tr.locator('td').all()] for tr in panel.locator('table tbody tr').all()]
    validate(headers, rows, start, end)
    return headers, rows

def save_block(db, station, start, end, headers, rows):
    found, missing = validate(headers, rows, start, end)
    content=json.dumps({'headers':headers,'rows':rows}, ensure_ascii=False, separators=(',',':'))
    digest=hashlib.sha256(content.encode('utf-8')).hexdigest()
    with db:
        db.execute('INSERT INTO blocks VALUES (?,?,?,?,?,?,?)', (station,start.isoformat(),end.isoformat(),datetime.now(timezone.utc).isoformat(),digest,content,json.dumps(missing)))
    return len(found),len(missing)

def export(db, output, station):
    blocks=db.execute('SELECT data FROM blocks WHERE station=? ORDER BY start', (station,)).fetchall()
    if not blocks: return
    headers=None; allrows={}
    for (raw,) in blocks:
        data=json.loads(raw)
        if headers is None: headers=data['headers']
        if headers != data['headers']: raise ValueError('Cabeceras diferentes entre intervalos')
        for row in data['rows']:
            key=datetime.strptime(row[0],'%d/%m/%Y').date().isoformat()
            if key in allrows and allrows[key] != row: raise ValueError('Conflicto entre intervalos')
            allrows[key]=row
    path=output / (station+'_web.csv'); temp=output / (station+'_web.csv.tmp')
    with temp.open('w',encoding='utf-8',newline='') as out:
        writer=csv.writer(out); writer.writerow(['Estacion']+headers)
        writer.writerows([[station]+allrows[k] for k in sorted(allrows)])
    temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--station',required=True);p.add_argument('--start',type=date.fromisoformat,required=True);p.add_argument('--end',type=date.fromisoformat,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--days',type=int,default=31);p.add_argument('--pause',type=int,default=10);p.add_argument('--max-blocks',type=int,default=1)
    a=p.parse_args(); station=a.station.upper(); output=a.output.resolve()
    if not re.fullmatch(r'[A-Z]{1,2}\d{2,3}',station) or a.start>a.end or not 1<=a.days<=31 or a.pause<10 or a.max_blocks<1: p.error('argumentos no v�lidos')
    if any(output==Path(x) or output.is_relative_to(Path(x)) for x in ['/home/josep/siar-sync-v3','/home/josep/siar-sync-v3-data','/home/josep/siar-auditoria-web-input','/home/josep/siar-auditoria-web-results']): p.error('salida protegida')
    output.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(output/'progreso.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS blocks (station TEXT,start TEXT,end TEXT,captured TEXT,sha256 TEXT,data TEXT,missing TEXT, PRIMARY KEY(station,start,end))')
        done=0
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True)
            try:
                for start,end in spans(a.start,a.end,a.days):
                    if db.execute('SELECT 1 FROM blocks WHERE station=? AND start=? AND end=?',(station,start.isoformat(),end.isoformat())).fetchone(): continue
                    if done>=a.max_blocks: break
                    page=browser.new_page()
                    try:
                        headers,rows=retrieve(page,station,start,end)
                        count,missing=save_block(db,station,start,end,headers,rows)
                        export(db,output,station)
                        print(json.dumps({'station':station,'start':start.isoformat(),'end':end.isoformat(),'rows':count,'pending_dates':missing}),flush=True)
                    finally: page.close()
                    done+=1; time.sleep(a.pause)
            finally: browser.close()

if __name__=='__main__': main()

