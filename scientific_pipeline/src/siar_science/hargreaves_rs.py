"""Hargreaves with measured global solar radiation, fixed uncalibrated v1.

Hargreaves & Allen (2003), doi:10.1061/(ASCE)0733-9437(2003)129:1(53).
Rs is expressed in MJ m-2 day-1; 0.408 converts it to equivalent mm/day.
"""
from .quality import finite

HG_RS_V1 = "HG_RS_V1"
HG_RS_V1_PARAMETERS = {
    "source": "Hargreaves and Allen 2003; doi:10.1061/(ASCE)0733-9437(2003)129:1(53)",
    "coefficient": 0.0135, "radiation_to_evaporation": 0.408,
    "expanded_coefficient": 0.005508, "temperature_offset_c": 17.8,
    "tmean_convention": "(Tmax+Tmin)/2", "radiation": "measured Radiacion MJ m-2 day-1",
    "output": "mm/day", "local_calibration": False, "output_clipping": False,
}


def hargreaves_rs(*, tmean_c, solar_radiation_mj_m2_day):
    """Return ET0 [mm/day] from mean temperature [°C] and measured Rs.

    No Ra substitution, temperature-range multiplier, humidity/wind inputs or
    calibration. Negative cold-regime predictions are retained, never clipped.
    """
    if not finite(tmean_c) or not finite(solar_radiation_mj_m2_day):
        raise ValueError("Hargreaves-Rs inputs must be finite numbers")
    if tmean_c <= -273.15 or solar_radiation_mj_m2_day < 0:
        raise ValueError("Temperature must exceed absolute zero and Rs be nonnegative")
    value = 0.0135 * 0.408 * solar_radiation_mj_m2_day * (tmean_c + 17.8)
    if not finite(value):
        raise ValueError("Nonfinite Hargreaves-Rs result")
    return value
