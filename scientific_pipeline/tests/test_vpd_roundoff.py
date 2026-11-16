"""VPD numerical boundary and controlled snapshots of the three IB10 failures."""
from copy import deepcopy
import math
import unittest

from siar_science.atmosphere import (
    actual_vapour_pressure, mean_saturation_vapour_pressure,
    vapour_pressure_deficit, VPD_ABS_TOL_KPA, VPD_REL_TOL,
)
from siar_science.penman_monteith import PM_FAO56_V1_PARAMETERS
from siar_science.pipeline import calculate
from tests_support import dataset_fixture


class VpdRoundoffTests(unittest.TestCase):
    def test_equal_pressures(self):
        self.assertEqual(vapour_pressure_deficit(2, 2), 0.0)

    def test_ulp_and_within_tolerance_excess(self):
        for es, ea in ((1.0, math.nextafter(1.0, math.inf)), (2.0, 2.0 + 5e-13)):
            with self.subTest(es=es, ea=ea):
                self.assertEqual(vapour_pressure_deficit(es, ea), 0.0)

    def test_exact_absolute_boundary_and_next_float(self):
        self.assertEqual(vapour_pressure_deficit(0.0, VPD_ABS_TOL_KPA), 0.0)
        with self.assertRaises(ValueError):
            vapour_pressure_deficit(0.0, math.nextafter(VPD_ABS_TOL_KPA, math.inf))

    def test_material_excess_and_no_relative_tolerance(self):
        for es, ea in ((2.0, 2.0 + 1e-10), (1.0, 2.0), (1e6, 1e6 + 1e-6)):
            with self.subTest(es=es, ea=ea), self.assertRaises(ValueError):
                vapour_pressure_deficit(es, ea)

    def test_negative_and_nonfinite_pressures_rejected(self):
        for es, ea in ((1, -1e-16), (-1e-16, 0), (math.nan, 1), (1, math.inf)):
            with self.subTest(es=es, ea=ea), self.assertRaises(ValueError):
                vapour_pressure_deficit(es, ea)

    def test_positive_deficits_unchanged_including_below_tolerance(self):
        for es, ea in ((2.0, 1.3), (1.0, math.nextafter(1.0, -math.inf))):
            with self.subTest(es=es, ea=ea):
                self.assertEqual(vapour_pressure_deficit(es, ea), es - ea)
                self.assertGreater(vapour_pressure_deficit(es, ea), 0)

    def test_policy_is_recorded_in_pm_run_parameters(self):
        policy = PM_FAO56_V1_PARAMETERS["vpd_roundoff"]
        self.assertEqual(policy["abs_tol_kpa"], 1e-12)
        self.assertEqual(policy["rel_tol"], 0.0)
        self.assertEqual((VPD_ABS_TOL_KPA, VPD_REL_TOL), (1e-12, 0.0))

    def test_three_ib10_snapshots_integrate_without_source_mutation(self):
        # Minimal frozen values from the failed run, never opens canonical files.
        snapshots = (
            ("2022-12-13", 16.7, 22.19, 1.277, 3.765),
            ("2022-12-31", 7.33, 21.99, .273, 7.65),
            ("2023-03-08", 5.22, 20.31, .754, 15.74),
        )
        data = dataset_fixture()
        template = data["rows"][0]
        data["rows"] = []
        data["missing_days"] = []
        data["stations"]["SYNTH"].update(latitude_deg=39.551111111111105, altitude_m=30)
        for day, tmin, tmax, wind, radiation in snapshots:
            row = deepcopy(template)
            row["date"] = day
            row["values"].update(Fecha=day, TempMin=tmin, TempMax=tmax,
                HumedadMin=100.0, HumedadMax=100.0, VelViento=wind, Radiacion=radiation)
            data["rows"].append(row)
        original = deepcopy(data)
        result = calculate(data)
        self.assertEqual(data, original)
        for before, after in zip(original["rows"], result["rows"]):
            with self.subTest(day=before["date"]):
                values = before["values"]
                es = mean_saturation_vapour_pressure(values["TempMin"], values["TempMax"])
                ea = actual_vapour_pressure(values["TempMin"], values["TempMax"], 100, 100)
                self.assertGreater(ea, es)  # Reproduce the previous strict-guard failure.
                self.assertEqual(vapour_pressure_deficit(es, ea), 0.0)
                self.assertEqual(after["values"], before["values"])
                self.assertEqual(after["eligibility"]["pm"]["status"], "CALCULATED")
                self.assertTrue(math.isfinite(after["ET0_FAO56"]))
