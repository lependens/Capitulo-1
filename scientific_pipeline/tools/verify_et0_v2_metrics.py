"""Recalculate grouped counts and common-cohort metrics from original predictions."""
import collections,csv,gzip,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)==2 else sys.exit('Usage: python verify_et0_v2_metrics.py DIRECTORY_WITH_CSV_FILES')
core=('FORMULA','AHC','ML','ANN_A','ANN_E')
methods=(*core,'ML_LINEAR','ML_RIDGE','ML_ELASTIC_NET','ML_TREE','ML_RANDOM_FOREST','ML_EXTRA_TREES','ML_HIST_GRADIENT_BOOSTING')
buckets=collections.defaultdict(lambda:[0,0,0,0,0.,0.,0.,0.,0.])
with gzip.open(root/'predictions.csv.gz','rt',encoding='utf-8',newline='') as f:
 for r in csv.DictReader(f):
  d=r['date']; station=r['station_code']; year=d[:4]; month=d[5:7]; season=('DJF','MAM','JJA','SON')[(int(month)%12)//3]
  groups=(('global','all'),('station',station),('year',year),('station_year',station+':'+year),('month',month),('season',season))
  target=float(r['target']) if r['target'] else None
  common=target is not None and all(r[m] for m in core)
  for method in methods:
   pred=float(r[method]) if r[method] else None
   for group,key in groups:
    a=buckets[(r['family'],r['calendar'],group,key,method)]
    a[0]+=target is not None; a[1]+=pred is not None; a[2]+=target is not None and pred is not None
    if common and pred is not None:
     delta=pred-target; a[3]+=1;a[4]+=abs(delta);a[5]+=delta*delta;a[6]+=delta;a[7]+=target;a[8]+=target*target
checked=0
with (root/'retrospective-metrics.csv').open(encoding='utf8',newline='') as f:
 for r in csv.DictReader(f):
  a=buckets[(r['family'],r['calendar'],r['group'],r['key'],r['method'])]
  pm,pred,own,n,ae,se,be,sy,sy2=a
  for name,expected in [('pm_days',pm),('predicted_days',pred),('pm_pairs_own',own),('n',n)]:
   if int(r[name])!=expected:raise ValueError(f'count mismatch {name}: {r}')
  if n:
   for name,val in [('mae',ae/n),('rmse',math.sqrt(se/n)),('bias',be/n)]:
    if abs(float(r[name])-val)>1e-9:raise ValueError(f'{name} mismatch: {r}')
   sst=sy2-sy*sy/n
   if n>=2 and sst>1e-12 and r['r2'] and abs(float(r['r2'])-(1-se/sst))>1e-7:raise ValueError(f'r2 mismatch: {r}')
  checked+=1
print(json.dumps({'status':'PASS','metric_rows_checked':checked,'counts':'PASS','mae_rmse_bias':'PASS','r2':'PASS except undefined/near-constant groups'}))

