"""Check retrospective pair membership hashes and unique metric keys."""
import collections,csv,gzip,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)==2 else sys.exit('Usage: python verify_et0_v2_cohorts.py DIRECTORY_WITH_CSV_FILES')
core=('FORMULA','AHC','ML','ANN_A','ANN_E')
methods=(*core,'ML_LINEAR','ML_RIDGE','ML_ELASTIC_NET','ML_TREE','ML_RANDOM_FOREST','ML_EXTRA_TREES','ML_HIST_GRADIENT_BOOSTING')
keys=collections.defaultdict(list)
with gzip.open(root/'predictions.csv.gz','rt',encoding='utf8',newline='') as f:
 for r in csv.DictReader(f):
  if not r['target'] or not all(r[m] for m in core):continue
  d=r['date']; st=r['station_code']; month=d[5:7]; season=('DJF','MAM','JJA','SON')[(int(month)%12)//3]
  groups=(('global','all'),('station',st),('year',d[:4]),('station_year',st+':'+d[:4]),('month',month),('season',season))
  for method in methods:
   if r[method]:
    for g,k in groups:keys[(r['family'],r['calendar'],g,k,method)].append([st,d])
seen=set(); n=0
with (root/'retrospective-metrics.csv').open(encoding='utf8',newline='') as f:
 for r in csv.DictReader(f):
  k=tuple(r[x] for x in ('family','calendar','group','key','method'))
  if k in seen:raise ValueError(('duplicate metric row',k))
  seen.add(k); raw=(json.dumps(keys[k],ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode(); actual=hashlib.sha256(raw).hexdigest()
  if actual!=r['common_keys_sha256']:raise ValueError(('cohort mismatch',k,actual,r['common_keys_sha256']))
  n+=1
print('COHORT_HASHES_AND_UNIQUE_KEYS_PASS',n)

