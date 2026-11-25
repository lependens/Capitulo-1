"""Check freeze → stage indices → cell manifests → model/checkpoint bytes in ZIP."""
import hashlib,json,os,zipfile,collections,sys
from pathlib import Path

archive=Path(sys.argv[1]) if len(sys.argv)==2 else sys.exit('Usage: python verify_et0_v2_freeze.py RETROSPECTIVE_ZIP')
root='et0-v2-retrospective-run-20261116/frozen-run/'
expected='97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd'
def digest(data): return hashlib.sha256(data).hexdigest()
counts=collections.Counter()
with zipfile.ZipFile(archive) as z:
 freeze_raw=z.read(root+'freeze.json')
 assert digest(freeze_raw)==expected
 freeze=json.loads(freeze_raw)
 assert freeze['status']=='FROZEN_BEFORE_RETROSPECTIVE_ACCESS'
 for stage,index_hash in freeze['stages'].items():
  index_raw=z.read(root+stage+'/index.json'); assert digest(index_raw)==index_hash, stage
  index=json.loads(index_raw); assert index['stage']==stage and index['mode']=='official'
  for cell,expected_cell in index['cells'].items():
   base=root+stage+'/'+cell+'/';raw=z.read(base+'frozen.json');assert digest(raw)==expected_cell,(stage,cell)
   item=json.loads(raw);assert item['stage']==stage and item['mode']=='official' and '__'.join(item['key'])==cell
   counts[stage+'_cells']+=1
   for name,info in item['models'].items():
    data=z.read(base+name);assert digest(data)==info['sha256'] and len(data)==info['bytes'],(stage,cell,name);counts[stage+'_models']+=1
   for name,sha in item['checkpoints'].items():
    data=z.read(base+'candidates/'+name);assert digest(data)==sha,(stage,cell,name);entry=json.loads(data)
    encoded=(json.dumps(entry['summary'],ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    assert digest(encoded)==entry['summary_sha256'],(stage,cell,name);counts[stage+'_checkpoints']+=1
 assert {s:counts[s+'_cells'] for s in ('AHC','ML','ANN_A','ANN_E')}=={'AHC':108,'ML':105,'ANN_A':105,'ANN_E':105}
 print(json.dumps({'status':'PASS','freeze_sha256':expected,'stage_sha256':freeze['stages'],'counts':dict(counts)},indent=2))

