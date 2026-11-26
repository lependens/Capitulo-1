import json, pathlib, subprocess, hashlib
from collections import Counter
ROOT=pathlib.Path('/srv/aemet-forecast/raw/runs')

def summarize(items):
    seen=set(); result=[]
    for m in sorted(items,key=lambda x:x.get('capture',{}).get('started_at_utc','')):
        src=m.get('source',{}); municipality=m.get('municipality',{}).get('id'); issue=src.get('issue_time_raw'); product=src.get('product'); key=(municipality,product,issue)
        status=m.get('capture',{}).get('status'); valid=status=='complete' and isinstance(issue,str) and bool(issue.strip())
        repeat=key in seen if valid else None
        if valid: seen.add(key)
        result.append({'run_id':m.get('run_id'),'capture_id':m.get('capture_id'),'status':status,'station_code':m.get('station',{}).get('code'),'station_name':m.get('station',{}).get('name'),'municipality_id':municipality,'municipality_name':m.get('municipality',{}).get('name'),'product':product,'issue_time_raw':issue,'captured_at_utc':m.get('capture',{}).get('started_at_utc'),'valid_times_raw':m.get('valid_times_raw',[]),'repeated_emission':repeat})
    return result

def main():
    items=[json.loads(p.read_text()) for p in ROOT.glob('*/*/*/run_*/*/*/metadata.json')]
    report={'captures':summarize(items),'runs_local':len(list(ROOT.glob('*/*/*/run_*'))),'note':'Emission identity is municipality, product and literal issue time; unknown when absent. Backup status requires remote verification.'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
