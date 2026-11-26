#!/usr/bin/env python3
"""Read-only AEMET dashboard data; never modifies captured files."""
import json, pathlib, subprocess, shutil, sys
sys.path.insert(0, '/opt/aemet-forecast')
from emissions import summarize
from backup_status import classify
ROOT=pathlib.Path('/srv/aemet-forecast/raw/runs')
def command(args):
 p=subprocess.run(args, text=True, capture_output=True, timeout=30)
 return {'returncode':p.returncode,'output':p.stdout.strip(),'error':p.stderr.strip()}
def report():
 files=sorted(ROOT.glob('*/*/*/run_*/*/*/metadata.json'))
 metadata=[]; forecasts=[]; errors=[]
 for p in files:
  try:
   m=json.loads(p.read_text()); metadata.append(m)
   if m.get('capture',{}).get('status') != 'complete': continue
   raw=(p.parent/'payload.raw').read_bytes()
   try: data=json.loads(raw.decode('utf-8-sig'))
   except UnicodeDecodeError: data=json.loads(raw.decode('latin-1'))
   for item in data:
    if str(item.get('id','')).zfill(5) != m.get('municipality',{}).get('id'): raise ValueError('payload municipality mismatch')
    for day in item.get('prediccion',{}).get('dia',[]):
     temp=day.get('temperatura',{}); humidity=day.get('humedadRelativa',{})
     forecasts.append({'capture_id':m.get('capture_id'),'station_code':m.get('station',{}).get('code'),'municipality_id':m.get('municipality',{}).get('id'),'municipality_name':m.get('municipality',{}).get('name'),'date':day.get('fecha'),'min_temp':temp.get('minima'),'max_temp':temp.get('maxima'),'min_humidity':humidity.get('minima'),'max_humidity':humidity.get('maxima')})
  except (OSError, ValueError, TypeError) as e: errors.append({'file':str(p),'error':str(e)})
 runs=sorted(p.name for p in ROOT.glob('*/*/*/run_*') if p.is_dir())
 j=command(['journalctl','-u','aemet-forecast-backup.service','-o','json','--no-pager'])
 events=[]
 if j['returncode']==0:
  for line in j['output'].splitlines():
   try: events.append(json.loads(line))
   except ValueError: pass
 backup=classify(runs,events)
 return {'captures':summarize(metadata),'forecasts':forecasts,'backup_evidence':backup,'backup_note':'Última evidencia del registro; no comprueba el remoto ahora.','errors':errors,'disk':shutil.disk_usage('/srv/aemet-forecast/raw')._asdict(),'capture_timer':command(['systemctl','show','aemet-forecast-capture.timer','-p','ActiveState','-p','NextElapseUSecRealtime','-p','LastTriggerUSec','--no-pager']),'backup_timer':command(['systemctl','show','aemet-forecast-backup.timer','-p','ActiveState','-p','NextElapseUSecRealtime','-p','LastTriggerUSec','--no-pager']),'capture_service':command(['systemctl','show','aemet-forecast-capture.service','-p','ActiveState','-p','Result','-p','ExecMainStatus','--no-pager']),'backup_service':command(['systemctl','show','aemet-forecast-backup.service','-p','ActiveState','-p','Result','-p','ExecMainStatus','--no-pager'])}
if __name__=='__main__': print(json.dumps(report(),ensure_ascii=False))
