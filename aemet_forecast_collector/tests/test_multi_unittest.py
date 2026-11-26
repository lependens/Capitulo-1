import json, pathlib, tempfile, unittest
from unittest.mock import patch
from aemet_forecast_collector.capture import capture_run
from aemet_forecast_collector.config import Config, Target
from aemet_forecast_collector.client import CaptureOutcome
class MultiCaptureTests(unittest.TestCase):
 def test_dedup_and_failed_municipality(self):
  calls=[]
  class FakeClient:
   def __init__(self,*a,**k): pass
   def capture(self,target):
    calls.append(target.municipality_id)
    if target.municipality_id=='07027': raise RuntimeError('secret must not be logged')
    return CaptureOutcome(complete=True,issue_time_raw='2026-11-26T00:00:00',artifacts={'payload.raw':b'example'})
  with tempfile.TemporaryDirectory() as root:
   targets=(Target('IB04','Son Ferriol','07040','Palma','municipal_capital'),Target('TEST','Prueba','07040','Palma','municipal_capital'),Target('IB02','Inca','07027','Inca','municipal_capital'))
   with patch('aemet_forecast_collector.capture.AemetClient',FakeClient),patch('aemet_forecast_collector.capture.time.sleep') as sleep:
    run=capture_run(Config(pathlib.Path(root),30,targets),api_key='synthetic')
   data=json.loads((run/'run.json').read_text())
   self.assertEqual(calls,['07040','07027']); sleep.assert_called_once_with(3)
   self.assertEqual(data['status'],'partial'); self.assertEqual([x['status'] for x in data['captures']],['complete','complete','failed'])
   self.assertEqual(len(list(run.glob('*/*/metadata.json'))),3)
   from aemet_forecast_collector.storage import verify_checksums
   self.assertEqual(verify_checksums(run),[])
class RecoveryTests(unittest.TestCase):
 def test_publish_failure_preserves_raw_stage(self):
  class FakeClient:
   def __init__(self,*a,**k): pass
   def capture(self,target): return CaptureOutcome(complete=True,issue_time_raw='2026-11-26T00:00:00',artifacts={'payload.raw':b'preserve-me'})
  with tempfile.TemporaryDirectory() as root:
   target=Target('IB04','Son Ferriol','07040','Palma','municipal_capital')
   with patch('aemet_forecast_collector.capture.AemetClient',FakeClient),patch('aemet_forecast_collector.capture.publish_run',side_effect=OSError('synthetic failure')):
    with self.assertRaises(OSError): capture_run(Config(pathlib.Path(root),30,(target,)),api_key='synthetic')
   saved=list(pathlib.Path(root).glob('runs/*/*/*/.run_*/*/*/payload.raw'))
   self.assertEqual(len(saved),1); self.assertEqual(saved[0].read_bytes(),b'preserve-me')
if __name__=='__main__': unittest.main()
