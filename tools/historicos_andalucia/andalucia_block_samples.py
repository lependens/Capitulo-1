"""Choose two reproducible 5-day windows from eligible labelled rows."""
from pathlib import Path
from datetime import date, timedelta
import json, random, sys
from andalucia_stage import parse_file

def select(dates,seed):
    dates=set(dates)
    starts=sorted(d for d in dates if all(d+timedelta(days=i) in dates for i in range(5)))
    if len(starts)<2: return []
    rng=random.Random(seed)
    first_half=starts[:max(1,len(starts)//2)]; second_half=starts[len(starts)//2:]
    rng.shuffle(first_half)
    for a in first_half:
        eligible=[b for b in second_half if (b-a).days >= 5]
        if eligible: return [a,rng.choice(eligible)]
    return []

def main(root,inventory,output):
    root=Path(root); items=json.loads(Path(inventory).read_text(encoding='utf-8')); out=[]
    completed={'AL01','AL02','AL03','AL04','AL05'}
    for item in items:
        if item.get('status')!='candidate_unverified' or item.get('candidate_station') in completed: continue
        data=parse_file(root/item['file'],item['candidate_station'],2000,header_label=item['header'],max_date=date(2020,6,5))
        if data['source_sha256'].lower()!=item['sha256'].lower(): raise ValueError('source changed: '+item['file'])
        starts=select([date.fromisoformat(r['Fecha']) for r in data['rows']],int(data['source_sha256'][:16],16))
        out.append({'file':item['file'],'station_candidate':item['candidate_station'],'sha256':data['source_sha256'],'windows':[{'start':d.isoformat(),'end':(d+timedelta(days=4)).isoformat()} for d in starts], 'status':'PENDING_SIAR' if len(starts)==2 else 'MANUAL_SAMPLE_REQUIRED'})
    Path(output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('files',len(out),'two_windows',sum(len(x['windows'])==2 for x in out),'manual',sum(not x['windows'] for x in out))
if __name__=='__main__': main(*sys.argv[1:])

