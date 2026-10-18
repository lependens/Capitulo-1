"""QC isolation and model-specific eligibility without real observational CSVs."""
import copy
import unittest
from siar_science.quality import eligibility, scientific_flags


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.row = {"date": "2024-07-06", "observation_status": "OBSERVED",
                    "quality_status": "PASS", "values": {
                        "TempMin": 12.3, "TempMax": 21.5, "TempMedia": 16.9,
                        "HumedadMin": 63, "HumedadMax": 84,
                        "VelViento": 2.078, "Radiacion": 22.07, "EtPMon": 3.9}}
        self.station = {"latitude_deg": 50.8, "altitude_m": 100}

    def test_default_siar_height_predicts_and_records_limitation(self):
        self.assertEqual(eligibility(self.row, self.station, "pm"), [])
        self.assertIn("wind_height_assumed_2m", scientific_flags(self.row, self.station))
        self.station["wind_height_m"] = 2
        self.assertNotIn("wind_height_assumed_2m", scientific_flags(self.row, self.station))

    def test_explicit_other_height_stops_only_pm(self):
        for height in (10, 0, -1, float("nan"), "2", True):
            with self.subTest(height=height):
                self.station["wind_height_m"] = height
                self.assertIn("STOP:wind_height_not_2m", eligibility(self.row, self.station, "pm"))
                self.assertEqual(eligibility(self.row, self.station, "hargreaves"), [])

    def test_radiation_and_humidity_only_affect_pm(self):
        for key, value in (("Radiacion", None), ("Radiacion", -1),
                           ("HumedadMin", None), ("HumedadMin", 101),
                           ("HumedadMin", 90), ("VelViento", -1)):
            with self.subTest(key=key, value=value):
                row = copy.deepcopy(self.row)
                row["values"][key] = value
                self.assertTrue(eligibility(row, self.station, "pm"))
                self.assertEqual(eligibility(row, self.station, "hargreaves"), [])
                self.assertTrue(scientific_flags(row, self.station))

    def test_reference_and_observed_mean_are_not_prediction_inputs(self):
        for key, value in (("TempMedia", None), ("TempMedia", 100),
                           ("EtPMon", None), ("EtPMon", -1)):
            with self.subTest(key=key, value=value):
                row = copy.deepcopy(self.row)
                row["values"][key] = value
                self.assertEqual(eligibility(row, self.station, "pm"), [])
                self.assertEqual(eligibility(row, self.station, "hargreaves"), [])
                flags = scientific_flags(row, self.station)
                if key == "TempMedia" and value == 100:
                    self.assertIn("temperature_order:TempMedia", flags)
                if key == "EtPMon" and value == -1:
                    self.assertIn("negative:EtPMon", flags)

    def test_temperature_extremes_required_by_each_method(self):
        for key, value in (("TempMin", None), ("TempMax", float("nan")),
                           ("TempMax", 10), ("TempMin", -237.3)):
            with self.subTest(key=key, value=value):
                row = copy.deepcopy(self.row)
                row["values"][key] = value
                for method in ("pm", "hargreaves"):
                    self.assertTrue(eligibility(row, self.station, method))

    def test_low_and_high_rs_rso_flag_without_changing_eligibility(self):
        for radiation, flag in ((0, "low_rs_rso:negative_cloud_factor"),
                                (100, "rs_above_rso")):
            self.row["values"]["Radiacion"] = radiation
            self.assertIn(flag, scientific_flags(self.row, self.station))
            self.assertEqual(eligibility(self.row, self.station, "pm"), [])

    def test_source_acceptance_shared_without_global_deletion(self):
        self.row["observation_status"] = "CONFIRMED_NO_DATA"
        for method in ("pm", "hargreaves"):
            self.assertIn("not_observed_in_official_period", eligibility(self.row, self.station, method))
        self.row["observation_status"] = "RECOVERED"
        self.row["quality_status"] = "WARN"
        self.assertEqual(eligibility(self.row, self.station, "pm"), [])

    def test_method_metadata_and_date_requirements(self):
        self.station["altitude_m"] = None
        self.assertIn("invalid_altitude", eligibility(self.row, self.station, "pm"))
        self.assertEqual(eligibility(self.row, self.station, "HS_RA_V1"), [])
        self.row["date"] = "not-a-date"
        self.assertIn("invalid_date", eligibility(self.row, self.station, "HS_RA_V1"))
        with self.assertRaises(ValueError):
            eligibility(self.row, self.station, "typo")

    def test_no_mutation_and_versioned_methods(self):
        before = copy.deepcopy((self.row, self.station))
        self.assertEqual(eligibility(self.row, self.station, "PM_FAO56_V1"), [])
        scientific_flags(self.row, self.station)
        self.assertEqual((self.row, self.station), before)

    def test_polar_night_pm_undefined_hs_zero_is_defined(self):
        self.row["date"] = "2024-12-21"
        self.station["latitude_deg"] = 89
        self.assertIn("clear_sky_radiation_not_positive", eligibility(self.row, self.station, "pm"))
        self.assertEqual(eligibility(self.row, self.station, "hargreaves"), [])


if __name__ == "__main__":
    unittest.main()
