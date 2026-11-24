"""Summarize numerical ranges without changing or filtering predictions."""
import csv,gzip,json,math,collections
from pathlib import Path
p=Path(__file__).parent/'predictions.csv.gz'
cols=('target','FORMULA','AHC','ML','ANN_A','ANN_E','ML_LINEAR','ML_RIDGE','ML_ELASTIC_NET','ML_TREE','ML_RANDOM_FOREST','ML_EXTRA_TREES','ML_HIST_GRADIENT_BOOSTING')
d={c:{'n':0,'negative':0,'nonfinite':0,'min':None,'max':None} for c in cols}
with gzip.open(p,'rt',encoding='utf8',newline='') as f:
 for r in csv.DictReader(f):
  for c in cols:
   if not r[c]: continue
   x=float(r[c]); s=d[c];s['n']+=1
   if not math.isfinite(x): s['nonfinite']+=1;continue
   s['negative']+=x<0;s['min']=x if s['min'] is None else min(s['min'],x);s['max']=x if s['max'] is None else max(s['max'],x)
out=json.dumps(d,indent=2,sort_keys=True)+'\n'
(Path(__file__).parent/'numeric-ranges.json').write_text(out,encoding='utf8')
print(out)

