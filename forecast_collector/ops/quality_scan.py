"""Read-only classification of SIAR captures; outputs JSON to stdout."""
import argparse,glob,json,os,tempfile
from pathlib import Path
from siar_forecast_raw.quality import classify_csv

def scan(root):
 records=[]
 for filename in sorted(glob.glob(str(root/'runs/*/*/*/run_*/run.json'))):
  path=Path(filename); run=json.loads(path.read_text(encoding='utf-8')); captures=[]
  for c in run.get('captures',[]):
   station=c['station_code']; capture=c['capture_id']; csvfile=path.parent/station/capture/'forecast.csv'
   if c.get('status')=='failed': q={'status':'failed','reason':'capture_failed'}
   elif not csvfile.is_file(): q={'status':'failed','reason':'csv_missing'}
   else: q=classify_csv(csvfile.read_bytes())
   captures.append({'station':station,'capture_id':capture,'quality':q})
  records.append({'run_id':run['run_id'],'started_at_utc':run['started_at_utc'],'collector_status':run['status'],'captures':captures})
 return records
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/srv/siar-forecast/raw'));p.add_argument('--output',type=Path);a=p.parse_args();data=json.dumps(scan(a.root),ensure_ascii=False)
 if a.output:
  fd,tmp=tempfile.mkstemp(prefix='.quality-',dir=a.output.parent)
  try:
   with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(data);f.flush();os.fsync(f.fileno())
   os.chmod(tmp,0o640);os.replace(tmp,a.output)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
 else:print(data)
