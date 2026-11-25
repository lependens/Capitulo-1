import importlib.util,json,tempfile
from pathlib import Path
s=importlib.util.spec_from_file_location('queue',str(Path(__file__).resolve().parents[1]/'ops/reconcile_queue.py')); q=importlib.util.module_from_spec(s); s.loader.exec_module(q)
with tempfile.TemporaryDirectory() as d:
 root=Path(d)/'raw'; state=Path(d)/'state'; state.mkdir()
 def mk(name):
  p=root/'2026'/'11'/'25'/name;p.mkdir(parents=True); (p/'run.json').write_text(json.dumps({'run_id':name,'status':'complete'}));(p/'SHA256SUMS').write_text('test');return p
 a=mk('run_001');b=mk('run_002'); todo=q.pending(root,state); assert [p.name for p,f in todo]==['run_002','run_001']
 q.save(state,b.name,todo[0][1]); assert [p.name for p,f in q.pending(root,state)]==['run_001']
 (b/'SHA256SUMS').write_text('changed'); assert len(q.pending(root,state))==2
 print('queue checks passed: newest-first, receipt, changed manifest')

# Exercise bounded batches and failure retry without contacting the remote.
from unittest.mock import patch
with tempfile.TemporaryDirectory() as d:
 q.RAW=Path(d)/'raw';q.STATE=Path(d)/'receipts';q.STATE.mkdir()
 for i in range(6):
  p=q.RAW/'2026'/'11'/'25'/('run_%03d'%i);p.mkdir(parents=True);(p/'run.json').write_text(json.dumps({'run_id':p.name,'status':'complete'}));(p/'SHA256SUMS').write_text('x')
 class Result: returncode=0
 with patch.object(q.subprocess,'run',return_value=Result()) as run:
  assert q.main()==0 and run.call_count==4
  assert len(q.pending(q.RAW,q.STATE))==2
  assert q.main()==0 and run.call_count==6
  assert q.main()==0 and run.call_count==6
 print('batch and idempotence checks passed')

with tempfile.TemporaryDirectory() as d:
 q.RAW=Path(d)/'raw';q.STATE=Path(d)/'receipts';q.STATE.mkdir()
 for i in range(5):
  p=q.RAW/'2026'/'11'/'25'/('run_%03d'%i);p.mkdir(parents=True);(p/'run.json').write_text(json.dumps({'run_id':p.name,'status':'complete'}));(p/'SHA256SUMS').write_text('x')
 class Fail: returncode=1
 with patch.object(q.subprocess,'run',return_value=Fail()): assert q.main()==1
 left=q.pending(q.RAW,q.STATE);assert [p.name for p,f in left]==['run_000']
 print('failed runs deferred; older run remains eligible')
