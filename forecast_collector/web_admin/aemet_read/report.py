#!/usr/bin/python3
import json,re,subprocess,sys
reader=['/usr/sbin/runuser','-u','aemet-forecast','--','/usr/bin/python3','/opt/siar-forecast-aemet-read/dashboard_data.py']
p=subprocess.run(reader,capture_output=True,text=True,timeout=40)
if p.returncode: sys.exit(1)
data=json.loads(p.stdout)
ids={x.get('run_id') for x in data.get('captures',[])}
states={x:{'state':'unknown'} for x in ids}
j=subprocess.run(['/usr/bin/journalctl','-u','aemet-forecast-backup.service','-o','json','--no-pager'],capture_output=True,text=True,timeout=30)
if j.returncode==0:
 for line in j.stdout.splitlines():
  try: e=json.loads(line); m=re.fullmatch(r'backup (verified|pending): (run_[A-Za-z0-9_.-]+)',e.get('MESSAGE',''))
  except (ValueError,TypeError): continue
  if m and m.group(2) in states: states[m.group(2)]={'state':m.group(1),'journal_realtime_us':e.get('__REALTIME_TIMESTAMP')}
data['backup_evidence']=states
data['backup_note']='Ultima evidencia registrada; no verifica el remoto ahora.'
print(json.dumps(data,ensure_ascii=False))
