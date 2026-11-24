"""Trace matching frozen bytes to earliest available saved Kaggle batch."""
import collections, hashlib, json, zipfile
from pathlib import Path

base = Path(__file__).resolve().parent.parent
target = base / 'et0-v2-retrospective-370251881-preservation/et0-v2-retrospective-370251881.zip'
batches = [
 ('369095500', base/'et0-v2-development-369095500-preservation/et0-v2-lote-369095500.zip'),
 ('369223074', base/'et0-v2-development-369223074-preservation/et0-v2-lote-369223074.zip'),
 ('369632717', base/'et0-v2-ann-e-369632717-preservation/et0-v2-ann-e-369632717.zip'),
 ('369691836', base/'et0-v2-ann-e-369691836-preservation/et0-v2-ann-e-369691836.zip'),
 ('369788169', base/'et0-v2-ann-e-369788169-preservation/et0-v2-ann-e-369788169.zip'),
 ('369953511', base/'et0-v2-ann-e-369953511-preservation/et0-v2-ann-e-369953511.zip'),
]
root='et0-v2-retrospective-run-20261116/frozen-run/'
prefix='et0-v2-official-20261111/'
records={}
with zipfile.ZipFile(target) as z:
 for info in z.infolist():
  if not info.filename.startswith(root): continue
  key=info.filename[len(root):]
  if '/candidates/' not in key or not key.endswith('.json'): continue
  records[key]={'size':info.file_size,'crc':info.CRC,'sha256':hashlib.sha256(z.read(info)).hexdigest(),'batch':None}
first = base/'et0-v2-lote01-368475953/kaggle-output/et0-v2-official-20261111'
for key, r in records.items():
 path=first/key
 if path.is_file() and path.stat().st_size==r['size'] and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']:
  r['batch']='368475953'
for batch,path in batches:
 with zipfile.ZipFile(path) as z:
  for info in z.infolist():
   if not info.filename.startswith(prefix): continue
   key=info.filename[len(prefix):]
   r=records.get(key)
   if r and r['batch'] is None and (info.file_size,info.CRC)==(r['size'],r['crc']):
    if hashlib.sha256(z.read(info)).hexdigest()==r['sha256']: r['batch']=batch
counts=collections.Counter((k.split('/')[0],v['batch'] or 'unmatched') for k,v in records.items())
out={'meaning':'earliest matching available batch, not necessarily creation batch', 'counts':{f'{a}:{b}':n for (a,b),n in sorted(counts.items())},'records':records}
(base/'et0-v2-retrospective-370251881-inspection/checkpoint-trace.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(out['counts'],indent=2))

