import base64,importlib.util,json,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from unittest.mock import patch
s=importlib.util.spec_from_file_location('web',str(Path(__file__).resolve().parents[1]/'web_admin/server.py'));w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name);w.PASSWORD=root/'password';w.PASSWORD.write_text('a'*40);w.INDEX=root/'status.json';w.INDEX.write_text('[]');w.ROOT=root/'raw';w.ROOT.mkdir();self.s=w.ThreadingHTTPServer(('127.0.0.1',0),w.Handler);threading.Thread(target=self.s.serve_forever,daemon=True).start();self.url='http://127.0.0.1:'+str(self.s.server_port);w.ORIGIN=self.url;self.auth='Basic '+base64.b64encode(('admin:'+('a'*40)).encode()).decode()
 def tearDown(self):self.s.shutdown();self.s.server_close();self.tmp.cleanup()
 def req(self,path,method='GET',headers=None):
  h={'Authorization':self.auth};h.update(headers or {});r=urllib.request.Request(self.url+path,method=method,headers=h,data=b'' if method=='POST' else None)
  try:
   with urllib.request.urlopen(r) as x:return x.status,x.read()
  except urllib.error.HTTPError as e:return e.code,e.read()
 def test_lan_without_auth(self):
  self.auth='';self.assertEqual(self.req('/api/runs')[0],200)
 def test_untrusted_host(self):
  self.assertEqual(self.req('/api/runs',headers={'Host':'evil.example'})[0],403)
 def test_outside_lan(self):
  handler=object.__new__(w.Handler);handler.client_address=('203.0.113.4',1234);handler.headers={'Host':w.urllib.parse.urlsplit(w.ORIGIN).netloc};handler.reply=lambda *x:None
  self.assertFalse(handler.access())
 def test_data(self):self.assertEqual(json.loads(self.req('/api/runs')[1]),[])
 def test_reject_traversal(self):self.assertEqual(self.req('/api/csv?run=../x&station=IB04')[0],503)
 def test_csrf(self):self.assertEqual(self.req('/api/action/pause','POST')[0],403)
 def test_invalid_action(self):
  headers={'Origin':w.ORIGIN,'X-SIAR-Request':'1'};self.assertEqual(self.req('/api/action/shell','POST',headers)[0],400)
 def test_whitelist(self):
  with self.assertRaises(ValueError):w.execute('shell')
 def test_free_space(self):
  code,data=self.req('/api/system');info=json.loads(data);self.assertEqual(code,200);self.assertGreater(info['free_bytes'],0)
if __name__=='__main__':unittest.main()
