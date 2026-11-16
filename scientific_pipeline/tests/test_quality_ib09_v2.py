import unittest
from siar_science.quality_ib09_v2 import ib09_v2_eligibility, IB09_RAW_SHA256

class IB09V2Test(unittest.TestCase):
    def setUp(self):
        self.row = {"station_code":"IB09", "source_data_sha":IB09_RAW_SHA256, "date":"2012-01-01", "observation_status":"OBSERVED", "quality_status":"FAIL", "values":{"TempMin":10., "TempMax":20., "HumedadMin":40., "HumedadMax":90., "HumedadMedia":65., "VelViento":1., "Radiacion":12.}}
        self.station = {"latitude_deg":39.7, "altitude_m":105.}
    def evaluate(self, method="HS_RA_V1", digest=IB09_RAW_SHA256):
        return ib09_v2_eligibility(self.row, self.station, method, verified_raw_sha256=digest)
    def test_inside_keeps_v1_status(self):
        self.assertEqual(self.evaluate(), [])
        self.assertEqual(self.row["quality_status"], "FAIL")
    def test_outside_is_excluded(self):
        self.row["date"] = "2005-10-23"
        self.assertEqual(self.evaluate(), ["OUTSIDE_OFFICIAL_PERIOD"])
    def test_wrong_sha_and_station_are_rejected(self):
        self.assertEqual(self.evaluate(digest="0"*64), ["unverified_ib09_snapshot"])
        self.row["station_code"]="IB01"
        self.assertEqual(self.evaluate(), ["ib09_v2_exception_not_applicable"])
    def test_missing_wind_only_blocks_pm(self):
        self.row["values"]["VelViento"] = None
        self.assertEqual(self.evaluate(), [])
        self.assertIn("missing_or_nonfinite:VelViento", self.evaluate("PM_FAO56_V1"))
    def test_unobserved_stays_blocked(self):
        self.row["observation_status"]="CONFIRMED_NO_DATA"
        self.assertIn("not_observed_in_official_period", self.evaluate())

if __name__ == "__main__": unittest.main()
