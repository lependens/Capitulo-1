import copy
import json
import unittest
from pathlib import Path

import numpy as np

from siar_science.ann_modern import (validate_ann_contract, fold_candidate,
    summarize_folds, final_epochs)


class AnnModernTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.ann = json.loads((root/'experiments/ann-v1-modern-et0.json').read_text())
        self.ml_bytes = (root/'experiments/ml-v1-classical.json').read_bytes()
        self.ml = json.loads(self.ml_bytes)

    def test_contract_rejects_unfrozen_and_changed_science(self):
        draft = copy.deepcopy(self.ann)
        draft['status'] = 'DRAFT_NOT_FROZEN'
        with self.assertRaisesRegex(ValueError, 'not frozen'):
            validate_ann_contract(draft, self.ml, ml_bytes=self.ml_bytes)
        frozen = copy.deepcopy(self.ann)
        validate_ann_contract(frozen, self.ml, ml_bytes=self.ml_bytes)
        frozen['test'][0] = '2020-01-01'
        with self.assertRaisesRegex(ValueError, 'period mismatch'):
            validate_ann_contract(frozen, self.ml, ml_bytes=self.ml_bytes)

    def test_fold_never_uses_test_or_outer_validation_for_training(self):
        ann = copy.deepcopy(self.ann)
        ann['early_stopping'] = {'patience_epochs': 2, 'min_delta_rmse': 1e-6, 'max_epochs': 5}
        ann['seeds'] = [3, 5]
        ann['feature_sets'] = ['FEATURE_SET_T']
        rows = []
        for year in range(2004, 2013):
            for day in range(1, 9):
                date = f'{year}-01-{day:02}'
                x = year-2004+day/10
                rows.append({'station_code':'IB01','date':date,
                             'split':'DEVELOPMENT', 'target':2+x/10,
                             'features':{'FEATURE_SET_T':[x]}})
        rows.append({'station_code':'IB01','date':'2022-01-01','split':'TEST',
                     'target':1e9,'features':{'FEATURE_SET_T':[1e9]}})
        fold = {'id':'F1','train':['2004-01-01','2009-12-31'],
                'validation':['2010-01-01','2012-12-31']}
        a = fold_candidate(rows, fold, 'FEATURE_SET_T', [4], ann)
        altered = copy.deepcopy(rows)
        altered[-1]['target'] = -1e12
        b = fold_candidate(altered, fold, 'FEATURE_SET_T', [4], ann)
        self.assertEqual(a, b)
        self.assertEqual(a['n'], 24)
        self.assertEqual(len(a['seed_rmse']), 2)
        self.assertTrue(np.isfinite(a['rmse']))
        self.assertGreaterEqual(final_epochs([a],ann['seeds'])['3'],1)
        self.assertEqual(summarize_folds([a])['n'],24)


if __name__ == '__main__':
    unittest.main()

class AnnTestGateTests(unittest.TestCase):
    def test_invalid_development_freeze_never_loads_test(self):
        import tempfile
        from unittest.mock import patch
        from siar_science.ann_evaluate import evaluate
        root=Path(__file__).resolve().parents[1]
        ann=json.loads((root/'experiments/ann-v1-modern-et0.json').read_text())
        ann['status']='FROZEN'
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);dev=base/'development';dev.mkdir()
            (dev/'manifest.json').write_text(json.dumps({'status':'FAILED','test_access_count':0}))
            ann_path=base/'ann.json';ann_path.write_text(json.dumps(ann))
            with (patch('siar_science.ann_evaluate._sha',return_value=ann['dataset_sha256']),
                  patch('siar_science.ann_evaluate._context_for_split') as loader):
                with self.assertRaisesRegex(ValueError,'Development freeze'):
                    evaluate(dev,base/'dataset.json',base/'ahc.json',
                             root/'experiments/ml-v1-classical.json',ann_path,
                             base/'hgb',base/'test_run')
                loader.assert_not_called()
            manifest=json.loads((base/'test_run/manifest.json').read_text())
            self.assertEqual(manifest['status'],'FAILED')
            self.assertEqual(manifest['outputs'],{})
            self.assertNotIn('TEST_OPENED',(base/'test_run/journal.jsonl').read_text())

    def test_candidate_tie_uses_stability_and_parameter_count(self):
        from siar_science.ann_campaign import _compare_candidates, _parameter_count
        ml = json.loads((Path(__file__).resolve().parents[1]/"experiments/ml-v1-classical.json").read_text())
        self.assertEqual(_parameter_count(7, [32]), 289)
        a = {'family':'FEATURE_SET_T','widths':[32],
             'aggregate':{'weighted_mean_rmse':0.1,'weighted_sd_rmse':0.02}}
        b = {'family':'FEATURE_SET_T','widths':[64,32],
             'aggregate':{'weighted_mean_rmse':0.1+5e-13,'weighted_sd_rmse':0.03}}
        self.assertLess(_compare_candidates(a,b,ml),0)
        b['aggregate']['weighted_mean_rmse'] = 0.099
        self.assertGreater(_compare_candidates(a,b,ml),0)

    def test_checksum_inventory_matches_bytes(self):
        from siar_science.ann_campaign import _sha, _write_checksums
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'sample.csv').write_bytes(b'one,two\n1,2\n')
            _write_checksums(path, ('sample.csv',))
            self.assertEqual((path/'SHA256SUMS').read_text().strip(),
                             _sha(path/'sample.csv')+'  sample.csv')
