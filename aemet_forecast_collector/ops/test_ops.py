import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ops', Path(__file__).with_name('ops.py'))
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


class OperationsTests(unittest.TestCase):
    def test_cloud_uses_persistent_aemet_config(self):
        from types import SimpleNamespace
        with patch.object(ops.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            ops.cloud('lsf', 'forecast-offsite-crypt:test')
        argv = run.call_args.args[0]
        self.assertEqual(Path(argv[argv.index('--config') + 1]), Path('/var/lib/aemet-forecast/rclone/rclone.conf'))
        self.assertIn('--ask-password=false', argv)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name)
        (self.raw / 'tmp').mkdir()
        self.run = self.raw / 'runs/2026/10/18/run_test'
        self.run.mkdir(parents=True)
        (self.run / 'run.json').write_text(json.dumps({
            'schema_version': 'aemet-raw-run-v1', 'run_id': 'run_test', 'status': 'partial'}))
        digest = hashlib.sha256((self.run / 'run.json').read_bytes()).hexdigest()
        (self.run / 'SHA256SUMS').write_text(digest + '  run.json\n')

    def test_valid_partial_run(self):
        ops.verify(self.run)

    def test_corruption_rejected(self):
        (self.run / 'run.json').write_text('{}')
        with self.assertRaises(ValueError):
            ops.verify(self.run)

    def test_unlisted_file_rejected(self):
        (self.run / 'extra.raw').write_bytes(b'raw')
        with self.assertRaises(ValueError):
            ops.verify(self.run)

    def test_path_traversal_rejected(self):
        (self.run / 'SHA256SUMS').write_text('0' * 64 + '  ../outside\n')
        with self.assertRaises(ValueError):
            ops.verify(self.run)

    def test_gate_blocks_capture_before_credential_read(self):
        with patch.object(ops, 'ETC', self.raw), patch.object(ops, 'credentials') as cred:
            with self.assertRaises(FileNotFoundError):
                ops.capture()
            cred.assert_not_called()

    def test_gate_blocks_backup_before_cloud(self):
        with patch.object(ops, 'ETC', self.raw), patch.object(ops, 'cloud') as cloud:
            with self.assertRaises(FileNotFoundError):
                ops.reconcile()
            cloud.assert_not_called()

    def test_gate_blocks_restore_before_cloud(self):
        with patch.object(ops, 'ETC', self.raw), patch.object(ops, 'cloud') as cloud:
            with self.assertRaises(FileNotFoundError):
                ops.restore('2026/10/18/run_test')
            cloud.assert_not_called()

    def test_restore_checks_marker_and_keeps_quarantine(self):
        import shutil
        def cloud(*args):
            target = Path(args[2])
            if args[1].endswith('/COMPLETE'):
                (target / 'COMPLETE').write_text(json.dumps({
                    'schema': 'aemet-offsite-complete-v1',
                    'manifest_sha256': hashlib.sha256((self.run / 'SHA256SUMS').read_bytes()).hexdigest()}))
            else:
                shutil.copytree(self.run, target, dirs_exist_ok=True)
        with patch.object(ops, 'RAW', self.raw), patch.object(ops, 'gate'), patch.object(ops, 'cloud', cloud):
            self.assertEqual(ops.restore('2026/10/18/run_test'), 0)
        self.assertEqual(len(list((self.raw / 'tmp').glob('restore-*/run_test/run.json'))), 1)

    def test_restore_without_complete_keeps_downloaded_data(self):
        import shutil
        def cloud(*args):
            if args[1].endswith('/COMPLETE'):
                raise RuntimeError('marker absent')
            shutil.copytree(self.run, Path(args[2]), dirs_exist_ok=True)
        with patch.object(ops, 'RAW', self.raw), patch.object(ops, 'gate'), patch.object(ops, 'cloud', cloud):
            with self.assertRaises(RuntimeError):
                ops.restore('2026/10/18/run_test')
        self.assertEqual(len(list((self.raw / 'tmp').glob('restore-*/run_test/run.json'))), 1)

    def test_failed_verification_never_uploads_complete(self):
        calls = []
        def cloud(*args):
            calls.append(args)
            if args[0] == 'check':
                raise RuntimeError('synthetic failure')
        with patch.object(ops, 'RAW', self.raw), patch.object(ops, 'gate'), patch.object(ops, 'cloud', cloud):
            self.assertEqual(ops.reconcile(), 1)
        self.assertEqual([x[0] for x in calls], ['copy', 'check'])
        self.assertTrue((self.run / 'run.json').exists())
        self.assertEqual(list((self.raw / 'tmp').iterdir()), [])

    def test_complete_uploaded_only_after_data_check(self):
        calls = []
        with patch.object(ops, 'RAW', self.raw), patch.object(ops, 'gate'), patch.object(ops, 'cloud', side_effect=lambda *a: calls.append(a)), patch.object(ops.subprocess, 'run', return_value=__import__('types').SimpleNamespace(returncode=0, stdout=b'')):
            self.assertEqual(ops.reconcile(), 0)
        self.assertEqual([x[0] for x in calls], ['copy', 'check', 'copy', 'check'])
        self.assertEqual(Path(calls[0][1]), self.run)
        self.assertTrue((Path(calls[2][1]) / 'COMPLETE').is_file())
        self.assertTrue(all('--immutable' in x for x in calls if x[0] == 'copy'))
        self.assertTrue(all('--download' in x for x in calls if x[0] == 'check'))


if __name__ == '__main__':
    unittest.main()
