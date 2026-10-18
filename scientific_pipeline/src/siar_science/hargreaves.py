"""Uncalibrated Hargreaves–Samani with extraterrestrial radiation Ra.

FAO-56 Eq. 52: https://www.fao.org/4/X0490E/x0490e08.htm
Ra in MJ m⁻² day⁻¹ is converted to equivalent evaporation with 0.408.
"""

import math
from .atmosphere import _extremes, _temperature
from .solar import _finite


HS_RA_V1 = "HS_RA_V1"
HS_RA_V1_PARAMETERS = {
    "coefficient": 0.0023, "radiation_to_evaporation": 0.408,
    "temperature_offset_c": 17.8, "temperature_range_exponent": 0.5,
    "tmean_convention": "(Tmax+Tmin)/2", "radiation": "Ra MJ m-2 day-1",
    "output_clipping": False,
}


def hargreaves_samani(*, tmean_c: float, tmin_c: float, tmax_c: float,
                      ra_mj_m2_day: float) -> float:
    """Return ET0 [mm/day] from °C and Ra [MJ m⁻² day⁻¹].

    No calibration, Rs substitution, imputation, or output clipping. Cold
    means below -17.8°C produce signed output; QC may exclude that regime.
    """
    _extremes(tmin_c, tmax_c)
    _temperature(tmean_c)
    _finite("ra_mj_m2_day", ra_mj_m2_day)
    if not tmin_c <= tmean_c <= tmax_c or ra_mj_m2_day < 0:
        raise ValueError("mean must lie between extrema and Ra must be nonnegative")
    return 0.0023 * 0.408 * ra_mj_m2_day * (tmean_c + 17.8) * math.sqrt(tmax_c - tmin_c)
