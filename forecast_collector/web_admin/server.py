#!/usr/bin/python3
"""LAN-only SIAR Forecast administrator, served through TLS."""
import csv,datetime,io,ipaddress,json,os,re,ssl,subprocess,urllib.parse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

ROOT=Path('/srv/siar-forecast/raw/runs')
INDEX=Path('/var/lib/siar-forecast-quality/status.json')
AEMET_INDEX=Path('/var/lib/aemet-forecast/export/status.json')
LAN=ipaddress.ip_network('192.168.1.0/24')
HELPER='/usr/local/libexec/siar-forecast-web-control'
ORIGIN=os.environ.get('SIAR_WEB_ORIGIN','https://127.0.0.1:8091')
BIND=os.environ.get('SIAR_WEB_BIND','127.0.0.1')
PORT=int(os.environ.get('SIAR_WEB_PORT','8091'))
ACTIONS={'resume','pause','capture','restart','reconcile','aemet-pause','aemet-resume'}
ID=re.compile(r'^run_[A-Za-z0-9._-]+$'); STATION=re.compile(r'^IB(?:01|02|03|04|05|06|10|11)$')

def load():
 data=json.loads(INDEX.read_text(encoding='utf-8'))
 if not isinstance(data,list):raise ValueError('index')
 return data

def load_aemet(path=None,now=None):
 path=path or AEMET_INDEX
 data=json.loads(path.read_text(encoding='utf-8'))
 if data.get('schema')!='aemet-dashboard-v1' or not isinstance(data.get('captures'),list) or not isinstance(data.get('forecasts'),list):raise ValueError('invalid aemet export')
 try:
  generated=datetime.datetime.fromisoformat(data['generated_at_utc'].replace('Z','+00:00'))
  if generated.tzinfo is None:raise ValueError('timezone missing')
  age=((now or datetime.datetime.now(datetime.timezone.utc))-generated).total_seconds()
  data['source_age_seconds']=int(age)
  data['source_status']='current' if -300<=age<=900 else 'stale'
 except (KeyError,TypeError,ValueError,OverflowError):data['source_age_seconds']=None;data['source_status']='unknown'
 return data

def get_csv(run,station):
 if not ID.fullmatch(run) or not STATION.fullmatch(station):raise ValueError('invalid selector')
 records=[r for r in load() if r.get('run_id')==run]
 if len(records)!=1:raise ValueError('unknown run')
 captures=[c for c in records[0]['captures'] if c.get('station')==station]
 if len(captures)!=1:raise ValueError('unknown station')
 cap=captures[0].get('capture_id','')
 if not re.fullmatch(r'cap_[A-Za-z0-9._-]+',cap):raise ValueError('invalid capture')
 matches=list(ROOT.glob('*/*/*/'+run+'/'+station+'/'+cap+'/forecast.csv'))
 if len(matches)!=1:raise ValueError('missing csv')
 path=matches[0]
 if any(p.is_symlink() for p in (path,path.parent,path.parent.parent,path.parent.parent.parent)):raise ValueError('unsafe path')
 path.resolve(strict=True).relative_to(ROOT.resolve(strict=True))
 raw=path.read_bytes()
 if len(raw)>1000000:raise ValueError('csv too large')
 return raw

def execute(action):
 if action not in ACTIONS and action not in ('logs','preservation','aemet-report'):raise ValueError('invalid action')
 return subprocess.run(['sudo','-n',HELPER,action],capture_output=True,text=True,timeout=90,check=False)

class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def reply(self,code,data,kind='application/json; charset=utf-8'):
  if not isinstance(data,bytes):data=data.encode('utf-8')
  self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'");self.end_headers();self.wfile.write(data)
 def access(self):
  try:
   peer=ipaddress.ip_address(self.client_address[0])
   host=self.headers.get('Host','')
   ok=(peer.is_loopback or peer in LAN) and host==urllib.parse.urlsplit(ORIGIN).netloc
  except ValueError:ok=False
  if not ok:self.reply(403,'{}')
  return ok
 def do_GET(self):
  if not self.access():return
  u=urllib.parse.urlsplit(self.path)
  try:
   if u.path=='/':return self.reply(200,Path('/opt/siar-forecast/web/index.html').read_bytes(),'text/html; charset=utf-8')
   if u.path=='/app.js':return self.reply(200,Path('/opt/siar-forecast/web/app.js').read_bytes(),'text/javascript; charset=utf-8')
   if u.path=='/style.css':return self.reply(200,Path('/opt/siar-forecast/web/style.css').read_bytes(),'text/css; charset=utf-8')
   if u.path=='/combined.js':return self.reply(200,Path('/opt/siar-forecast/web/combined.js').read_bytes(),'text/javascript; charset=utf-8')
   if u.path=='/api/aemet':
    return self.reply(200,json.dumps(load_aemet(),ensure_ascii=False))
   if u.path=='/api/runs':return self.reply(200,json.dumps(load(),ensure_ascii=False))
   if u.path=='/api/system':
    p=subprocess.run(['systemctl','show','siar-forecast-capture.timer','siar-forecast-backup.service','siar-forecast-quality.service','--property=Id,ActiveState,Result,NextElapseUSecRealtime,LastTriggerUSec','--no-pager'],capture_output=True,text=True,timeout=10);v=os.statvfs(ROOT);return self.reply(200,json.dumps({'text':p.stdout,'free_bytes':v.f_bavail*v.f_frsize,'free_inodes':v.f_favail}))
   if u.path=='/api/csv':
    q=urllib.parse.parse_qs(u.query);return self.reply(200,get_csv(q.get('run',[''])[0],q.get('station',[''])[0]),'text/csv; charset=utf-8')
   if u.path=='/api/preservation':
    p=execute('preservation');return self.reply(200 if p.returncode==0 else 503,p.stdout if p.returncode==0 else '{}')
   if u.path=='/api/logs':
    p=execute('logs');return self.reply(200 if p.returncode==0 else 503,json.dumps({'text':p.stdout[-30000:]}))
   self.reply(404,'{}')
  except (OSError,ValueError,KeyError,json.JSONDecodeError,subprocess.SubprocessError):self.reply(503,json.dumps({'error':'Datos no disponibles'}))
 def do_POST(self):
  if not self.access():return
  if self.headers.get('Origin')!=ORIGIN or self.headers.get('X-SIAR-Request')!='1' or self.headers.get('Content-Length','0')!='0':return self.reply(403,'{}')
  action=urllib.parse.urlsplit(self.path).path.removeprefix('/api/action/')
  try:
   p=execute(action);self.reply(200 if p.returncode==0 else 409,json.dumps({'ok':p.returncode==0,'message':p.stdout[-1000:] or p.stderr[-1000:]}))
  except (ValueError,subprocess.SubprocessError):self.reply(400,'{}')
if __name__=='__main__':
 server=ThreadingHTTPServer((BIND,PORT),Handler)
 ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.minimum_version=ssl.TLSVersion.TLSv1_2;ctx.load_cert_chain('/etc/siar-forecast-web/server.crt','/etc/siar-forecast-web/server.key');server.socket=ctx.wrap_socket(server.socket,server_side=True);server.serve_forever()
