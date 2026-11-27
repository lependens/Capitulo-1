"""SIAR web audit, isolated from V3. No authenticated SIAR API calls.

Requires: pip install playwright; playwright install chromium.
Run only in a dedicated directory, with an explicit source root and manifest.
Writes SQLite progress/evidence and never writes to input files or V3.
"""
import argparse
import hashlib
import json
import re
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

from andalucia_compare_web import compare
from andalucia_stage import parse_file

BASE='https://servicio.mapa.gob.es/siarweb/consultaDatos/consultaDatos'
MODIFY='Pulse este enlace para mostrar u ocultar el buscador que permite volver a establecer los filtros.'
FIELD_INDEX=(1,2,3,4,5,6,7,9,11,12,16,17,19)

def setup(db):
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS blocks (station TEXT, source_sha256 TEXT, start TEXT, end TEXT, result_json TEXT NOT NULL, PRIMARY KEY(station,source_sha256,start,end))')
    db.execute('CREATE TABLE IF NOT EXISTS audit (station TEXT, source_sha256 TEXT, plan_sha256 TEXT, status TEXT, result_json TEXT NOT NULL, PRIMARY KEY(station,source_sha256,plan_sha256))')
    db.commit()

def dates_of(start,end):
    a=date.fromisoformat(start); b=date.fromisoformat(end)
    if (b-a).days != 4: raise ValueError('each block must have exactly five days')
    return [(a+timedelta(days=i)).isoformat() for i in range(5)]

def plan_hash(entry):
    return hashlib.sha256(json.dumps(entry['windows'],sort_keys=True,separators=(',',':')).encode()).hexdigest()

def source_path(root, rel):
    root=Path(root).resolve(strict=True)
    # Plans may use Windows path separators; disallow absolute and traversal.
    parts=rel.replace('\\','/').split('/')
    if not parts or any(x in ('','.','..') or ':' in x for x in parts): raise ValueError('invalid source path')
    p=root.joinpath(*parts).resolve(strict=True)
    if not p.is_relative_to(root) or not p.is_file(): raise ValueError('source outside root')
    return p

def validate_plan(entry, root, header):
    station=entry['station_candidate']; digest=entry['sha256']
    if not re.fullmatch('[A-Z0-9]{2,12}',station) or not re.fullmatch('[a-f0-9]{64}',digest): raise ValueError('invalid identity/hash')
    source=source_path(root,entry['file']); actual=hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != digest: raise ValueError('source hash changed')
    if len(entry['windows'])!=2: raise ValueError('two windows required')
    dates=sum((dates_of(w['start'],w['end']) for w in entry['windows']),[])
    if len(set(dates))!=10: raise ValueError('overlapping dates')
    parsed=parse_file(source,station,2000,header_label=header,max_date=date(2020,6,5))
    if not set(dates).issubset({r['Fecha'] for r in parsed['rows']}): raise ValueError('sample includes a quarantined day')
    return source,dates

def get_block(page,station,start,end):
    # The selectors come from the visible public form. No hidden requests.
    page.goto(BASE,wait_until='domcontentloaded',timeout=60000)
    page.get_by_role('link',name='Seleccionar estaciones *').click()
    page.get_by_role('textbox',name='Seleccionar por nombre corto').fill(station)
    page.get_by_role('button',name='Buscar estaci�n').click()
    page.locator('p.text-resultado').filter(has_text=re.compile(r'^(?:Inactiva\s+)?'+re.escape(station)+r'\s+-')).wait_for(timeout=30000)
    page.get_by_role('button',name='Incluir selecci�n').click()
    page.get_by_role('textbox',name='Fecha inicial *').fill(start)
    page.get_by_role('textbox',name='Fecha final *').fill(end)
    page.get_by_role('button',name='Realizar consulta').click()
    page.get_by_role('tab',name=re.compile(r'^'+re.escape(station)+r'\s+-')).wait_for(timeout=60000)
    expected=dates_of(start,end); display=date.fromisoformat(start).strftime('%d/%m/%Y')
    page.get_by_role('cell',name=display,exact=True).wait_for(timeout=30000)
    panel=page.get_by_role('tabpanel',name=re.compile(r'^'+re.escape(station)+r'\s+-'))
    headers=[h.text_content().strip().lower().replace(' ','') for h in panel.locator('table thead th').all()]
    expected_prefix={0:'fecha',1:'tempmedia',2:'tempmax',3:'horat',4:'tempm',5:'horat',6:'hummedia',7:'hummax',9:'hummin',11:'velviento',12:'dirviento',16:'rad',17:'precip',19:'et0'}
    if len(headers)<=max(expected_prefix) or any(not headers[i].startswith(prefix) for i,prefix in expected_prefix.items()): raise ValueError('SIAR table schema changed')
    raw=panel.locator('table tbody tr').all()
    reference={}
    for row in raw:
        cells=[cell.text_content().strip() for cell in row.locator('td').all()]
        if not cells or not re.fullmatch(r'\d{2}/\d{2}/\d{4}',cells[0]): continue
        day=date(int(cells[0][6:]),int(cells[0][3:5]),int(cells[0][:2])).isoformat()
        if day in reference or len(cells)<=max(FIELD_INDEX): raise ValueError('duplicate or incomplete SIAR response')
        reference[day]=[cells[i] for i in FIELD_INDEX]
    if sorted(reference)!=expected: raise ValueError('SIAR returned missing, extra or invalid dates')
    return reference

def audit_one(entry, root, header, db, fetch, pause_seconds):
    source,dates=validate_plan(entry,root,header); station=entry['station_candidate']; digest=entry['sha256']; reference={}
    for w in entry['windows']:
        cached=db.execute('SELECT result_json FROM blocks WHERE station=? AND source_sha256=? AND start=? AND end=?',(station,digest,w['start'],w['end'])).fetchone()
        if cached: block=json.loads(cached[0])
        else:
            block=fetch(station,w['start'],w['end'])
            if sorted(block)!=dates_of(w['start'],w['end']): raise ValueError('block does not match requested dates')
            db.execute('INSERT INTO blocks VALUES (?,?,?,?,?)',(station,digest,w['start'],w['end'],json.dumps(block,ensure_ascii=False)));db.commit()
            if pause_seconds: time.sleep(pause_seconds)
        reference.update(block)
    result=compare(source,station,reference,dates,header_label=header)
    db.execute('INSERT INTO audit VALUES (?,?,?,?,?)',(station,digest,plan_hash(entry),result['status'],json.dumps(result,ensure_ascii=False))); db.commit()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',required=True);parser.add_argument('--plan',required=True);parser.add_argument('--inventory',required=True);parser.add_argument('--db',required=True);parser.add_argument('--max-stations',type=int,default=1);parser.add_argument('--pause-seconds',type=int,default=10);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.max_stations<1 or args.pause_seconds<3: parser.error('max-stations >= 1 and pause-seconds >= 3 required')
    root=Path(args.source_root).resolve(strict=True); dbpath=Path(args.db).resolve()
    protected=[root,Path('/home/josep/siar-sync-v3-data'),Path('/home/josep/siar-sync-v3')]
    if any(dbpath==p or dbpath.is_relative_to(p) for p in protected): parser.error('database must be outside sources and V3')
    plans=json.loads(Path(args.plan).read_text(encoding='utf-8')); inventory=json.loads(Path(args.inventory).read_text(encoding='utf-8'))
    metadata={x['file']:x for x in inventory if x.get('status')=='candidate_unverified'}
    if len(metadata)!=sum(x.get('status')=='candidate_unverified' for x in inventory): raise ValueError('duplicate source paths')
    dbpath.parent.mkdir(parents=True,exist_ok=True); db=sqlite3.connect(dbpath);setup(db)
    todo=[]
    for entry in plans:
        if entry.get('status')!='PENDING_SIAR': continue
        meta=metadata[entry['file']]
        if meta['candidate_station']!=entry['station_candidate'] or meta['sha256']!=entry['sha256']: raise ValueError('inventory/plan mismatch')
        validate_plan(entry,root,meta['header'])
        prior=db.execute('SELECT status FROM audit WHERE station=? AND source_sha256=? AND plan_sha256=?',(entry['station_candidate'],entry['sha256'],plan_hash(entry))).fetchone()
        if prior is None: todo.append((entry,meta['header']))
    if args.dry_run:
        print(json.dumps({'remaining_labelled':len(todo),'unsupported_files':sum(x.get('status')=='unidentified_or_other_format' for x in inventory),'network_calls':0})); return
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True); page=browser.new_page()
        try:
            for entry,header in todo[:args.max_stations]:
                try:
                    result=audit_one(entry,root,header,db,lambda st,a,b:get_block(page,st,a,b),args.pause_seconds)
                    print(json.dumps({'station':entry['station_candidate'],'status':result['status']},ensure_ascii=False),flush=True)
                except Exception as exc:
                    # Fail closed: do not write a false absence or loop on errors.
                    print(json.dumps({'station':entry['station_candidate'],'status':'STOPPED','error_type':type(exc).__name__}),flush=True)
                    break
        finally: browser.close(); db.close()
if __name__=='__main__': main()

