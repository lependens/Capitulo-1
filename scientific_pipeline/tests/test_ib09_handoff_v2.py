import copy, hashlib, importlib.util, json, pathlib, shutil, tempfile, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
DIR=ROOT/'data_quality/siar/ib09-handoff-v2'
spec=importlib.util.spec_from_file_location('ib09_handoff_validator',DIR/'validate_handoff.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class HandoffTests(unittest.TestCase):
 def setUp(self):
  self.h=json.loads((DIR/'handoff.json').read_text(encoding='utf-8'))
 def test_pending_is_valid_but_admission_is_blocked(self):
  self.assertFalse(v.validate_handoff(self.h,ROOT)['decision_effective'])
  with self.assertRaises(v.HandoffError): v.validate_handoff(self.h,ROOT,for_admission=True)
 def test_official_evidence_and_effective_decision_cannot_be_fabricated(self):
  for key,value in [('official_period_verification_status','VERIFIED'),('official_source_sha256','a'*64),('decision_effective',True),('station_code','IB01')]:
   h=copy.deepcopy(self.h);h[key]=value
   with self.subTest(key=key),self.assertRaises(v.HandoffError):v.validate_handoff(h,ROOT)
 def test_wrong_source_hash_rejected(self):
  h=copy.deepcopy(self.h);h['raw_sha256']='0'*64
  with self.assertRaises(v.HandoffError):v.validate_handoff(h,ROOT)
 def test_inclusive_boundaries_and_other_station(self):
  for d in ('2005-10-24','2018-10-10'):self.assertEqual(v.row_status('IB09',d),'CANDIDATE_ONLY')
  for d in ('2005-10-23','2018-10-11'):self.assertEqual(v.row_status('IB09',d),'OUTSIDE_OFFICIAL_PERIOD')
  with self.assertRaises(v.HandoffError):v.row_status('IB01','2012-01-01')
 def _corruption(self,source,mutate):
  with tempfile.TemporaryDirectory() as t:
   root=pathlib.Path(t)
   for key in ('raw','catalog','ledger'):
    p=pathlib.Path(self.h[key+'_path']);(root/p).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,root/p)
   for rel in (self.h['quality_post']['path'], self.h['policy']['prototype_path']):
    p2=pathlib.Path(rel);(root/p2).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p2,root/p2)
   p=root/self.h[source+'_path']; mutate(p); bad=p.read_bytes(); original=hashlib.sha256
   class HashProxy:
    def hexdigest(self):return self.value
   def digest(data=b''):
    if data==bad:
     x=HashProxy();x.value=self.h[source+'_sha256'];return x
    return original(data)
   with patch.object(v.hashlib,'sha256',side_effect=digest):
    with self.assertRaises(v.HandoffError):v.validate_handoff(self.h,root)
 def test_duplicate_observation_even_if_hash_check_bypassed(self):
  def mutate(p):
   lines=p.read_text(encoding='utf-8').splitlines();lines[3]=lines[2];p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
  self._corruption('raw',mutate)
 def test_ledger_observation_contradiction_even_if_hash_check_bypassed(self):
  def mutate(p):
   s=p.read_text(encoding='utf-8');s=s.replace(',IB09,2006-11-21,',',IB09,2006-11-20,',1);p.write_text(s,encoding='utf-8')
  self._corruption('ledger',mutate)
if __name__=='__main__': unittest.main()
