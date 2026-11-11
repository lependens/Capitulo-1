import copy,importlib.util,json,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
DIR=ROOT/'data_quality/siar/ib09-handoff-v2-development'
spec=importlib.util.spec_from_file_location('ib09_development_handoff',DIR/'validate_handoff.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class DevelopmentHandoffTests(unittest.TestCase):
 def setUp(self):self.h=json.loads((DIR/'handoff.json').read_text(encoding='utf-8'))
 def test_valid_evidence_and_development(self):
  self.assertEqual(v.validate(self.h,ROOT)['official_period_verification'],'UNRESOLVED')
  result=v.validate(self.h,ROOT,for_admission=True,scope='DEVELOPMENT_ONLY',official_evidence_contradiction=False)
  self.assertEqual(result['admission'],'APPROVED');self.assertFalse(result['campaign_launch_authorized']);self.assertEqual(result['dataset_row_binding'],'CONSUMER_REQUIRED')
 def test_scope_cannot_expand(self):
  for scope in (None,'TEST','ALL','PRODUCTION'):
   with self.subTest(scope=scope),self.assertRaises(v.HandoffError):v.validate(self.h,ROOT,for_admission=True,scope=scope,official_evidence_contradiction=False)
 def test_conflict_or_unknown_blocks(self):
  for conflict in (True,None):
   with self.subTest(conflict=conflict),self.assertRaises(v.HandoffError):v.validate(self.h,ROOT,for_admission=True,scope='DEVELOPMENT_ONLY',official_evidence_contradiction=conflict)
 def test_tampered_metadata_rejected(self):
  for k,val in [('raw_sha256','0'*64),('quality_v1_status','PASS'),('official_period_verification_status','VERIFIED'),('station_code','IB01'),('decision_effective',False)]:
   h=copy.deepcopy(self.h);h[k]=val
   with self.subTest(field=k),self.assertRaises(v.HandoffError):v.validate(h,ROOT)
 def test_pending_previous_handoff_still_rejects_admission(self):
  path=ROOT/'data_quality/siar/ib09-handoff-v2/validate_handoff.py';spec2=importlib.util.spec_from_file_location('pending_test',path);p=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(p)
  old=json.loads((path.parent/'handoff.json').read_text(encoding='utf-8'));self.assertTrue(p.validate_handoff(old,ROOT)['valid'])
  with self.assertRaises(p.HandoffError):p.validate_handoff(old,ROOT,for_admission=True)
if __name__=='__main__':unittest.main()
