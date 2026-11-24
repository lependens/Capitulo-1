"""Check published coverage ratios against their stated denominators."""
import csv, math, sys
from pathlib import Path
p=Path(sys.argv[1]) if len(sys.argv)==2 else sys.exit('Usage: python verify_et0_v2_coverage.py RETROSPECTIVE_METRICS_CSV')
n=0
with p.open(encoding='utf8',newline='') as f:
 for r in csv.DictReader(f):
  own=int(r['pm_pairs_own']); days=int(r['calendar_days']); pm=int(r['pm_days']); predicted=int(r['predicted_days'])
  for col,num,den in (('coverage_calendar',predicted,days),('coverage_pm_own',own,pm)):
   if den and (not r[col] or not math.isclose(float(r[col]),num/den,abs_tol=1e-12)):
    raise ValueError((col,r))
   if not den and r[col]: raise ValueError((col,r))
  n+=1
print('COVERAGE_RATIO_PASS',n)

