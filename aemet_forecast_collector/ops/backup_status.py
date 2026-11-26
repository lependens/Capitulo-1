import json, pathlib, re, subprocess, sys
ROOT=pathlib.Path('/srv/aemet-forecast/raw/runs')
PAT=re.compile(r'^backup (verified|pending): (run_[A-Za-z0-9_.-]+)$')

def classify(run_ids, events):
    states={r:{'state':'unknown','verified_at_utc':None} for r in run_ids}
    for e in events:
        m=PAT.fullmatch(e.get('MESSAGE',''))
        if not m or m.group(2) not in states: continue
        states[m.group(2)]={'state':m.group(1),'journal_realtime_us':e.get('__REALTIME_TIMESTAMP')}
    return states

def main():
    runs=sorted(p.name for p in ROOT.glob('*/*/*/run_*') if p.is_dir())
    p=subprocess.run(['journalctl','-u','aemet-forecast-backup.service','--no-pager','-o','json'],capture_output=True,text=True)
    if p.returncode: raise RuntimeError('journal read failed')
    events=[json.loads(s) for s in p.stdout.splitlines()]
    states=classify(runs,events)
    print(json.dumps({'runs':states,'total':len(runs),'pending':sum(x['state']=='pending' for x in states.values()),'unknown':sum(x['state']=='unknown' for x in states.values()),'note':'Last journal evidence only; verified refers to verification at the recorded time.'},indent=2))
if __name__=='__main__':
    try: main()
    except Exception as e:
        print('MONITOR_UNKNOWN '+type(e).__name__,file=sys.stderr); sys.exit(1)
