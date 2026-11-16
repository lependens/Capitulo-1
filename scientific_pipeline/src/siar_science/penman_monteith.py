"""Pure daily FAO-56 grass-reference calculations; no imputation or clipping.

Sources: https://www.fao.org/4/X0490E/x0490e07.htm (meteorology),
https://www.fao.org/4/X0490E/x0490e08.htm (daily ET0).
Temperatures °C, humidity %, wind m/s at 2 m, pressure kPa, radiation
MJ m⁻² day⁻¹, elevation m, ET0 mm day⁻¹. None/NaN/inf are errors.
"""

import math
from .solar import _finite, clear_sky_radiation, extraterrestrial_radiation


# Compatibility reexports preserve the original public imports.
from .atmosphere import (
    _temperature, _extremes, atmospheric_pressure, psychrometric_constant,
    saturation_vapour_pressure, mean_saturation_vapour_pressure,
    saturation_vapour_pressure_slope, actual_vapour_pressure,
    vapour_pressure_deficit, VPD_ABS_TOL_KPA, VPD_REL_TOL,
)

PM_FAO56_V1 = "PM_FAO56_V1"
PM_FAO56_V1_PARAMETERS = {
    "albedo": 0.23, "soil_heat_flux_mj_m2_day": 0.0,
    "stefan_boltzmann_mj_k4_m2_day": 4.903e-9,
    "psychrometric_coefficient": 0.000665,
    "radiation_to_evaporation": 0.408, "kelvin_offset_longwave": 273.16,
    "kelvin_offset_aerodynamic": 273.0,
    "cloud_ratio_upper_cap": 1.0, "cloud_ratio_lower_floor": None,
    "tmean_convention": "(Tmax+Tmin)/2",
    "wind_policy": "SIAR 2m convention; explicit other height STOP",
    "vpd_roundoff": {"abs_tol_kpa": VPD_ABS_TOL_KPA, "rel_tol": VPD_REL_TOL,
                     "policy": "ea>es within tolerance gives VPD=0; pressures unchanged; otherwise error"},
}


def net_longwave_radiation(tmin_c: float, tmax_c: float, ea_kpa: float,
                          solar_radiation_mj_m2_day: float,
                          clear_sky_radiation_mj_m2_day: float) -> float:
    """Rnl [MJ m⁻² day⁻¹], Eq. 39, sigma=4.903e-9, Kelvin=T+273.16.

    Rs/Rso is capped at 1 only, as FAO-56 states. There is no lower
    cloud-factor floor (legacy PM2 floors that factor at 0.2). Below Rs/Rso=0.35/1.35 the cloud factor, and potentially Rnl,
    becomes negative. Preserve this diagnostic result; QC must flag very
    low ratios rather than silently changing the formula. Rso=0 is undefined.
    """
    _extremes(tmin_c, tmax_c)
    for name, value in (("ea_kpa", ea_kpa), ("Rs", solar_radiation_mj_m2_day),
                        ("Rso", clear_sky_radiation_mj_m2_day)):
        _finite(name, value)
    if ea_kpa < 0 or solar_radiation_mj_m2_day < 0 or clear_sky_radiation_mj_m2_day <= 0:
        raise ValueError("ea and Rs must be nonnegative; Rso must be positive")
    ratio = min(solar_radiation_mj_m2_day / clear_sky_radiation_mj_m2_day, 1)
    return (4.903e-9 * ((tmax_c + 273.16) ** 4 + (tmin_c + 273.16) ** 4) / 2
            * (0.34 - 0.14 * math.sqrt(ea_kpa)) * (1.35 * ratio - 0.35))


def net_radiation(tmin_c: float, tmax_c: float, ea_kpa: float,
                  solar_radiation_mj_m2_day: float,
                  clear_sky_radiation_mj_m2_day: float) -> float:
    """Rn [MJ m⁻² day⁻¹], Eqs. 38–40, reference albedo fixed at 0.23."""
    longwave = net_longwave_radiation(tmin_c, tmax_c, ea_kpa,
                                      solar_radiation_mj_m2_day, clear_sky_radiation_mj_m2_day)
    return 0.77 * solar_radiation_mj_m2_day - longwave


def penman_monteith_et0(*, tmean_c: float, wind_2m_m_s: float,
                        net_radiation_mj_m2_day: float, es_kpa: float,
                        ea_kpa: float, delta_kpa_c: float,
                        gamma_kpa_c: float) -> float:
    """Daily Eq. 6 ET0 [mm/day], G=0; output retains possible negatives.

    This low-level equation accepts explicit Tmean. The canonical daily
    baseline should supply (Tmax+Tmin)/2 per FAO-56 Eq. 9.
    """
    _temperature(tmean_c)
    for name, value in (("wind_2m_m_s", wind_2m_m_s),
                        ("net_radiation_mj_m2_day", net_radiation_mj_m2_day),
                        ("delta_kpa_c", delta_kpa_c), ("gamma_kpa_c", gamma_kpa_c)):
        _finite(name, value)
    if wind_2m_m_s < 0 or delta_kpa_c <= 0 or gamma_kpa_c <= 0:
        raise ValueError("wind must be nonnegative; delta and gamma positive")
    vpd = vapour_pressure_deficit(es_kpa, ea_kpa)
    return ((0.408 * delta_kpa_c * net_radiation_mj_m2_day
             + gamma_kpa_c * 900 / (tmean_c + 273) * wind_2m_m_s * vpd)
            / (delta_kpa_c + gamma_kpa_c * (1 + 0.34 * wind_2m_m_s)))


def fao56_daily(*, tmean_c: float, tmin_c: float, tmax_c: float,
                rhmin_pct: float, rhmax_pct: float, wind_2m_m_s: float,
                solar_radiation_mj_m2_day: float, latitude_deg: float,
                elevation_m: float, day_of_year: int) -> float:
    """Compute daily equation ET0 [mm/day] with explicit mean for compatibility.

    PM_FAO56_V1 callers MUST supply (Tmax+Tmin)/2, per Eq. 9. The explicit
    mean remains for historical sensitivity uses; those are not PM_FAO56_V1.
    Original SIAR TempMedia remains unchanged. No wind conversion occurs.
    """
    _extremes(tmin_c, tmax_c)
    _temperature(tmean_c)
    if not tmin_c <= tmean_c <= tmax_c:
        raise ValueError("tmean_c must lie between temperature extremes")
    es = mean_saturation_vapour_pressure(tmin_c, tmax_c)
    ea = actual_vapour_pressure(tmin_c, tmax_c, rhmin_pct, rhmax_pct)
    ra = extraterrestrial_radiation(latitude_deg, day_of_year)
    rso = clear_sky_radiation(ra, elevation_m)
    rn = net_radiation(tmin_c, tmax_c, ea, solar_radiation_mj_m2_day, rso)
    return penman_monteith_et0(tmean_c=tmean_c, wind_2m_m_s=wind_2m_m_s,
                               net_radiation_mj_m2_day=rn, es_kpa=es, ea_kpa=ea,
                               delta_kpa_c=saturation_vapour_pressure_slope(tmean_c),
                               gamma_kpa_c=psychrometric_constant(atmospheric_pressure(elevation_m)))
