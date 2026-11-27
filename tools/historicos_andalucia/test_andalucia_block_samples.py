import unittest
from datetime import date,timedelta
from andalucia_block_samples import select

class SampleWindows(unittest.TestCase):
    def test_reproducible_nonoverlapping_and_complete(self):
        days=[date(2000,1,1)+timedelta(days=i) for i in range(100)]
        windows=select(days,123)
        self.assertEqual(windows,select(days,123)); self.assertEqual(len(windows),2)
        self.assertGreaterEqual((windows[1]-windows[0]).days,5)
        self.assertTrue(all(start+timedelta(days=i) in days for start in windows for i in range(5)))
    def test_no_two_windows_is_pending(self):
        self.assertEqual(select([date(2000,1,1)+timedelta(days=i) for i in range(5)],1),[])

if __name__=='__main__': unittest.main()

