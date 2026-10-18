"""Independent examples: FAO-56 chapters 3 and 4 (1998).

https://www.fao.org/4/X0490E/x0490e07.htm
https://www.fao.org/4/X0490E/x0490e08.htm
Published rounded values use explicit rounding tolerances, not CSV regression.
"""

from datetime import date
import math
import unittest

from siar_science.solar import (
    clear_sky_radiation, daylight_hours, day_of_year,
    extraterrestrial_radiation, inverse_relative_distance,
    solar_declination, sunset_hour_angle,
)
from siar_science.penman_monteith import (
    actual_vapour_pressure, atmospheric_pressure, fao56_daily,
    mean_saturation_vapour_pressure, net_longwave_radiation, net_radiation,
    penman_monteith_et0, psychrometric_constant, saturation_vapour_pressure,
    saturation_vapour_pressure_slope, vapour_pressure_deficit,
)
from siar_science.hargreaves import hargreaves_samani


class ScientificMathTests(unittest.TestCase):
    def test_calendar_leap_year(self):
        self.assertEqual(day_of_year(date(2024, 2, 29)), 60)
        self.assertEqual(day_of_year(date(2024, 12, 31)), 366)
        self.assertEqual(day_of_year(date(2023, 12, 31)), 365)

    def test_fao_example_8_and_9_geometry(self):
        self.assertAlmostEqual(inverse_relative_distance(246), 0.985, delta=0.0005)
        self.assertAlmostEqual(solar_declination(246), 0.120, delta=0.0005)
        self.assertAlmostEqual(sunset_hour_angle(-20, 246), 1.527, delta=0.001)
        self.assertAlmostEqual(extraterrestrial_radiation(-20, 246), 32.2, delta=0.05)
        self.assertAlmostEqual(daylight_hours(-20, 246), 11.7, delta=0.05)

    def test_polar_extension(self):
        self.assertEqual(extraterrestrial_radiation(90, 355), 0)
        self.assertEqual(daylight_hours(90, 172), 24)
        self.assertTrue(math.isfinite(extraterrestrial_radiation(90, 172)))

    def test_fao_example_2_atmosphere(self):
        self.assertAlmostEqual(atmospheric_pressure(1800), 81.8, delta=0.05)
        self.assertAlmostEqual(psychrometric_constant(atmospheric_pressure(1800)), 0.054, delta=0.0005)
        self.assertEqual(psychrometric_constant(100), 0.0665)

    def test_fao_example_3_saturation(self):
        self.assertAlmostEqual(saturation_vapour_pressure(24.5), 3.075, delta=0.001)
        self.assertAlmostEqual(saturation_vapour_pressure(15), 1.705, delta=0.001)
        self.assertAlmostEqual(mean_saturation_vapour_pressure(15, 24.5), 2.39, delta=0.005)
        self.assertGreater(mean_saturation_vapour_pressure(15, 24.5), saturation_vapour_pressure(19.75))

    def test_fao_examples_5_and_6_vapour(self):
        ea = actual_vapour_pressure(18, 25, 54, 82)
        es = mean_saturation_vapour_pressure(18, 25)
        self.assertAlmostEqual(ea, 1.70, delta=0.005)
        self.assertAlmostEqual(vapour_pressure_deficit(es, ea), 0.91, delta=0.005)

    def test_fao_example_18_slope_and_rso(self):
        self.assertAlmostEqual(saturation_vapour_pressure_slope(16.9), 0.122, delta=0.0005)
        self.assertAlmostEqual(clear_sky_radiation(41.09, 100), 30.90, delta=0.005)

    def test_fao_examples_11_and_12_radiation(self):
        self.assertAlmostEqual(net_longwave_radiation(19.1, 25.1, 2.1, 14.5, 18.8), 3.5, delta=0.05)
        self.assertAlmostEqual(net_radiation(19.1, 25.1, 2.1, 14.5, 18.8), 7.6, delta=0.10)

    def test_no_legacy_lower_cloud_floor_and_upper_cap(self):
        self.assertLess(net_longwave_radiation(15, 25, 1.5, 0, 20), 0)
        self.assertEqual(net_longwave_radiation(15, 25, 1.5, 20, 20),
                         net_longwave_radiation(15, 25, 1.5, 30, 20))

    def test_fao_example_18_daily_end_to_end(self):
        # Published 3.9 mm/day; inputs wind and Rs are rounded in FAO's sheet.
        result = fao56_daily(tmean_c=16.9, tmin_c=12.3, tmax_c=21.5,
                            rhmin_pct=63, rhmax_pct=84, wind_2m_m_s=2.078,
                            solar_radiation_mj_m2_day=22.07, latitude_deg=50.8,
                            elevation_m=100, day_of_year=187)
        self.assertAlmostEqual(result, 3.9, delta=0.05)

    def test_fao_example_20_hargreaves(self):
        # Lyon July: Tmax=26.6, Tmin=14.8, Ra=40.6; FAO reports 5.0 mm/day.
        self.assertAlmostEqual(hargreaves_samani(tmean_c=20.7, tmin_c=14.8,
                                                tmax_c=26.6, ra_mj_m2_day=40.6),
                               5.0, delta=0.05)

    def test_hargreaves_zero_range_and_units(self):
        self.assertEqual(hargreaves_samani(tmean_c=20, tmin_c=20, tmax_c=20, ra_mj_m2_day=30), 0)
        # Hand-verifiable: sqrt(range)=2, Tmean+17.8=20, Ra=.408^-1.
        self.assertAlmostEqual(hargreaves_samani(tmean_c=2.2, tmin_c=.2,
                                                tmax_c=4.2, ra_mj_m2_day=1/.408), .092)

    def test_negative_pm_not_silently_clipped(self):
        self.assertLess(penman_monteith_et0(tmean_c=15, wind_2m_m_s=0,
                                          net_radiation_mj_m2_day=-1, es_kpa=2,
                                          ea_kpa=2, delta_kpa_c=.1, gamma_kpa_c=.067), 0)

    def test_reject_missing_nonfinite_and_invalid_domains(self):
        cases = [
            (saturation_vapour_pressure, (float("nan"),)),
            (saturation_vapour_pressure, (None,)),
            (saturation_vapour_pressure, (-237.3,)),
            (atmospheric_pressure, (float("inf"),)),
            (psychrometric_constant, (0,)),
            (mean_saturation_vapour_pressure, (30, 20)),
            (actual_vapour_pressure, (15, 25, 80, 20)),
            (actual_vapour_pressure, (15, 25, 20, 101)),
            (vapour_pressure_deficit, (1, 2)),
            (extraterrestrial_radiation, (91, 100)),
            (extraterrestrial_radiation, (40, 0)),
            (extraterrestrial_radiation, (40, 367)),
            (extraterrestrial_radiation, (40, 1.5)),
            (net_longwave_radiation, (15, 25, 1, 0, 0)),
            (clear_sky_radiation, (-1, 100)),
        ]
        for function, arguments in cases:
            with self.subTest(function=function.__name__, arguments=arguments):
                with self.assertRaises(ValueError):
                    function(*arguments)
        with self.assertRaises(ValueError):
            hargreaves_samani(tmean_c=20, tmin_c=20, tmax_c=19, ra_mj_m2_day=30)
        with self.assertRaises(ValueError):
            penman_monteith_et0(tmean_c=15, wind_2m_m_s=-1,
                               net_radiation_mj_m2_day=5, es_kpa=2, ea_kpa=1,
                               delta_kpa_c=.1, gamma_kpa_c=.067)


if __name__ == "__main__":
    unittest.main()
