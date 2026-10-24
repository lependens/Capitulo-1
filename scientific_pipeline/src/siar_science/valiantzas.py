"""Martí et al. (2024), Eqs. 1b + 3; fixed MJ-to-mm factor 0.408.

DOI: 10.1016/j.agwat.2024.108903. This is the unit-explicit composition,
not a literal transcription of their Eq. 4 with Ra in MJ m-2 day-1.
"""
from .quality import finite

VALIANTZAS_V1 = "VALIANTZAS_V1"
VALIANTZAS_V1_PARAMETERS = {
    "source": "Marti et al. 2024, Eqs. 1b + 3; doi:10.1016/j.agwat.2024.108903",
    "radiation_coefficient": 0.338, "hargreaves_radiation_coefficient": 0.0135,
    "radiation_to_evaporation": 0.408, "expanded_coefficient": 0.001861704,
    "temperature_range_exponent": 0.3, "humidity_exponent": 0.2,
    "humidity_offset": 1.001, "temperature_offset_c": 17.8,
    "tmean_convention": "(Tmax+Tmin)/2", "rh_input": "HumedadMedia percent",
    "radiation": "Ra MJ m-2 day-1", "output": "mm/day",
    "local_calibration": False, "output_clipping": False,
}


def estimated_solar_radiation(*, tmin_c, tmax_c, rhmean_pct, ra_mj_m2_day):
    """Estimate Rs [MJ m-2 day-1]; RH is daily mean percent, not RHmin.

    Physical temperature domain is strictly above absolute zero. No local
    plausibility thresholds, RH floor, imputation or clear-sky clipping.
    """
    inputs = (tmin_c, tmax_c, rhmean_pct, ra_mj_m2_day)
    if not all(finite(v) for v in inputs):
        raise ValueError("Valiantzas inputs must be finite numbers")
    if tmin_c <= -273.15 or tmax_c < tmin_c:
        raise ValueError("Invalid temperature domain or extrema order")
    if not 0 <= rhmean_pct <= 100 or ra_mj_m2_day < 0:
        raise ValueError("RH must be in [0,100] and Ra nonnegative")
    return 0.338 * ra_mj_m2_day * (tmax_c-tmin_c)**0.3 * (1.001-rhmean_pct/100)**0.2


def valiantzas(*, tmin_c, tmax_c, rhmean_pct, ra_mj_m2_day):
    """Uncalibrated daily ET0 [mm/day]. Signed cold-regime output is retained."""
    rs = estimated_solar_radiation(tmin_c=tmin_c, tmax_c=tmax_c,
        rhmean_pct=rhmean_pct, ra_mj_m2_day=ra_mj_m2_day)
    value = 0.408 * 0.0135 * rs * ((tmax_c+tmin_c)/2 + 17.8)
    if not finite(value):
        raise ValueError("Nonfinite Valiantzas result")
    return value
