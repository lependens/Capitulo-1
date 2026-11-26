#!/usr/bin/python3
"""Narrow SIAR Forecast control surface; no user-controlled command arguments."""
import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
A={'resume':['start','siar-forecast-capture.timer'],'pause':['stop','siar-forecast-capture.timer'],'capture':['start','--no-block','siar-forecast-capture.service'],'restart':['restart','--no-block','siar-forecast-capture.service'],'reconcile':['start','--no-block','siar-forecast-backup.service']}
A.update({'aemet-pause':['stop','aemet-forecast-capture.timer'],'aemet-resume':['start','aemet-forecast-capture.timer']})
if len(sys.argv)!=2:sys.exit(2)
a=sys.argv[1]
if a=='aemet-report':os.execv('/usr/bin/python3',['python3','-I','/opt/siar-forecast-aemet-read/report.py'])
if a=='logs':os.execv('/usr/bin/journalctl',['journalctl','-u','siar-forecast-capture.service','-u','siar-forecast-backup.service','-u','siar-forecast-quality.service','-n','120','--no-pager','-o','short-iso'])
if a=='preservation':
 root=Path('/srv/siar-forecast/raw/runs');state=Path('/var/lib/siar-forecast-backup/receipts');total=good=deferred=0
 for p in root.glob('*/*/*/run_*'):
  if not p.is_dir() or p.is_symlink():continue
  total+=1
  try:
   fp=hashlib.sha256((p/'run.json').read_bytes()+b'\0'+(p/'SHA256SUMS').read_bytes()).hexdigest(); receipt=json.loads((state/(p.name+'.json')).read_text())
   if receipt.get('fingerprint')==fp and receipt.get('checked_epoch',0)>=time.time()-604800:good+=1
   elif receipt.get('retry_after',0)>time.time():deferred+=1
  except (OSError,ValueError):pass
 print(json.dumps({'total':total,'verified_receipts':good,'pending_review':total-good,'retry_wait':deferred,'summary':str(good)+' verificados / '+str(total-good)+' pendientes de revision'}));sys.exit(0)
if a not in A:sys.exit(2)
if a in ('capture','restart'):
 p=subprocess.run(['/usr/bin/systemctl','is-active','--quiet','siar-forecast-capture.service'])
 if p.returncode==0:print('Ya hay una captura activa');sys.exit(3)
print('Solicitud aceptada',flush=True)
os.execv('/usr/bin/systemctl',['systemctl']+A[a])
