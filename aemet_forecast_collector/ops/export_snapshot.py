#!/usr/bin/env python3
"""Publish only sanitized AEMET dashboard data as an atomic read-only snapshot."""
import datetime, json, os, pathlib, sys, tempfile
sys.path.insert(0,'/opt/aemet-forecast')
from dashboard_data import report
DEST=pathlib.Path('/var/lib/aemet-forecast/export/status.json')
def main():
 data=report()
 data['generated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 data['schema']='aemet-dashboard-v1'
 data['source_notice']='Backup state is last journal evidence, not a live remote check.'
 fd,path=tempfile.mkstemp(prefix='.status-',dir=DEST.parent)
 try:
  os.fchmod(fd,0o640)
  with os.fdopen(fd,'w',encoding='utf-8') as f:
   json.dump(data,f,ensure_ascii=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
  os.replace(path,DEST)
 finally:
  if os.path.exists(path): os.unlink(path)
if __name__=='__main__': main()
