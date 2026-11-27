"""Select a deterministic ten-day sample for each labelled source, not a validation."""
from pathlib import Path
from datetime import date
from collections import Counter
import json,random,sys
from andalucia_stage import parse_file

def main(root, inventory_path, output_path):
    root=Path(root); inventory=json.loads(Path(inventory_path).read_text(encoding='utf-8')); out=[]
    for item in inventory:
        if item['status']!='candidate_unverified': continue
        data=parse_file(root/item['file'],item['candidate_station'],2000,header_label=item['header'],max_date=date(2020,6,5))
        if data['source_sha256'].lower()!=item['sha256'].lower(): raise ValueError('source changed: '+item['file'])
        rows=data['rows']; seed=int(data['source_sha256'][:16],16)
        selected=sorted(r['Fecha'] for r in random.Random(seed).sample(rows,min(10,len(rows))))
        out.append({'file':item['file'],'source_sha256':data['source_sha256'],'candidate_station':item['candidate_station'],'selected_dates':selected,'validation':'PENDING_SIAR','seed':'first 16 hex digits of source SHA256'})
    Path(output_path).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('files',len(out),'complete_samples',sum(len(x['selected_dates'])==10 for x in out),'status',dict(Counter(x['validation'] for x in out)))
if __name__=='__main__': main(*sys.argv[1:])

