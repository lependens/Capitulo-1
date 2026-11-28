#!/usr/bin/python3
"""Bounded, newest-first dispatch using durable verification receipts."""
import hashlib, json, os, re, subprocess, tempfile, time
from pathlib import Path
RAW = Path('/srv/siar-forecast/raw/runs')
STATE = Path('/var/lib/siar-forecast-backup/receipts')
RECON = '/opt/siar-forecast/bin/siar-forecast-reconcile'
ID = re.compile(r'^run_[A-Za-z0-9._-]+$')
MAX_RUNS = 4

def fingerprint(path):
    if path.is_symlink() or not path.is_dir(): raise ValueError('unsafe run')
    a=path/'run.json'; b=path/'SHA256SUMS'
    if a.is_symlink() or b.is_symlink(): raise ValueError('unsafe metadata')
    data=json.loads(a.read_text())
    if data.get('run_id') != path.name or data.get('status') not in ('complete','partial','failed'): raise ValueError('invalid metadata')
    return hashlib.sha256(a.read_bytes()+b'\0'+b.read_bytes()).hexdigest()

def pending(root, state):
    candidates=sorted(root.glob('[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/run_*'), reverse=True)
    out=[]
    for p in candidates:
        if not ID.fullmatch(p.name): raise ValueError('invalid id')
        for parent in (p.parent,p.parent.parent,p.parent.parent.parent):
            if parent.is_symlink(): raise ValueError('unsafe path')
        fp=fingerprint(p); receipt=state/(p.name+'.json')
        try: prior=json.loads(receipt.read_text())
        except (OSError,ValueError): prior={}
        if prior.get('retry_after', 0) > time.time(): continue
        if prior.get('fingerprint') != fp or prior.get('checked_epoch', 0) < time.time()-604800: out.append((p,fp))
    return out

def save(state,run_id,fp,failed=False):
    fd,tmp=tempfile.mkstemp(prefix='.receipt-',dir=state)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump({'fingerprint':None if failed else fp,'checked_epoch':time.time(),'retry_after':time.time()+21600 if failed else 0,'verified_at_utc':None if failed else time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,state/(run_id+'.json'))
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def main():
    STATE.mkdir(mode=0o700,parents=True,exist_ok=True)
    try: todo=pending(RAW,STATE)
    except (OSError,ValueError) as e: print('ERROR inventory',type(e).__name__,flush=True); return 1
    print('ELIGIBLE',len(todo),flush=True); failures=0
    for path,fp in todo[:MAX_RUNS]:
        print('RECONCILE',path.name,flush=True)
        try: result=subprocess.run([RECON,'--only-run-id',path.name],stdin=subprocess.DEVNULL,timeout=240,check=False)
        except (OSError,subprocess.TimeoutExpired) as e: print('ERROR',path.name,type(e).__name__,flush=True); save(STATE,path.name,fp,failed=True); failures+=1; continue
        if result.returncode or fingerprint(path)!=fp:
            print('ERROR reconcile_or_changed',path.name,flush=True); save(STATE,path.name,fp,failed=True); failures+=1; continue
        save(STATE,path.name,fp); print('RECEIPT',path.name,flush=True)
    print('ELIGIBLE_REMAINING',max(0,len(todo)-MAX_RUNS)+failures,flush=True)
    return 1 if failures else 0
if __name__=='__main__': raise SystemExit(main())
