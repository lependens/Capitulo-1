import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('guard',str(Path(__file__).resolve().parents[1]/'ops/capture_guard.py'));g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class GuardTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);(self.root/'tmp').mkdir();self.run=self.root/'runs/2026/11/26/run_test';self.run.mkdir(parents=True);(self.run/'run.json').write_text('{}');self.man=self.run/'SHA256SUMS';self.man.write_text(hashlib.sha256(b'{}').hexdigest()+'  run.json\n')
 def tearDown(self):self.t.cleanup()
 def test_valid(self):self.assertEqual(g.check(self.root,0,0),'verified_last_run')
 def test_space(self):
  with self.assertRaisesRegex(ValueError,'insufficient_free_space'):g.check(self.root,10**30,0)
 def test_corruption(self):
  (self.run/'run.json').write_text('changed')
  with self.assertRaisesRegex(ValueError,'hash_mismatch'):g.check(self.root,0,0)
 def test_traversal(self):
  self.man.write_text('a'*64+'  ../outside\n')
  with self.assertRaisesRegex(ValueError,'unsafe_manifest_path'):g.check(self.root,0,0)
if __name__=='__main__':unittest.main()
