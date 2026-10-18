"""FAO-56 atmosphere and vapour equations (7, 8, 11–13, 17).

Pure functions; °C, kPa, kPa/°C, humidity %, elevation m.
Source: https://www.fao.org/4/X0490E/x0490e07.htm
"""
import math
from .solar import _finite


def _temperature(value: float) -> None:
    _finite("temperature_c", value)
    if value <= -237.3:
        raise ValueError("temperature_c must exceed -237.3 (vapour equation domain)")


def _extremes(tmin_c: float, tmax_c: float) -> None:
    _temperature(tmin_c)
    _temperature(tmax_c)
    if tmin_c > tmax_c:
        raise ValueError("tmin_c must not exceed tmax_c")


def atmospheric_pressure(elevation_m: float) -> float:
    """P [kPa] from altitude [m], FAO-56 Eq. 7."""
    _finite("elevation_m", elevation_m)
    base = (293 - 0.0065 * elevation_m) / 293
    if base <= 0:
        raise ValueError("elevation exceeds atmospheric pressure equation domain")
    return 101.3 * base ** 5.26


def psychrometric_constant(pressure_kpa: float) -> float:
    """Gamma [kPa °C⁻¹], fixed FAO-56 Eq. 8 coefficient 0.000665."""
    _finite("pressure_kpa", pressure_kpa)
    if pressure_kpa <= 0:
        raise ValueError("pressure_kpa must be positive")
    return 0.000665 * pressure_kpa


def saturation_vapour_pressure(temperature_c: float) -> float:
    """e°(T) [kPa], FAO-56 Eq. 11."""
    _temperature(temperature_c)
    return 0.6108 * math.exp(17.27 * temperature_c / (temperature_c + 237.3))


def mean_saturation_vapour_pressure(tmin_c: float, tmax_c: float) -> float:
    """es [kPa] from temperature extremes (never e°(Tmean)), Eq. 12."""
    _extremes(tmin_c, tmax_c)
    return (saturation_vapour_pressure(tmin_c) + saturation_vapour_pressure(tmax_c)) / 2


def saturation_vapour_pressure_slope(tmean_c: float) -> float:
    """Delta [kPa °C⁻¹] evaluated at mean temperature, Eq. 13."""
    return 4098 * saturation_vapour_pressure(tmean_c) / (tmean_c + 237.3) ** 2


def actual_vapour_pressure(tmin_c: float, tmax_c: float,
                          rhmin_pct: float, rhmax_pct: float) -> float:
    """ea [kPa] from paired RH extremes, FAO-56 Eq. 17."""
    _extremes(tmin_c, tmax_c)
    _finite("rhmin_pct", rhmin_pct)
    _finite("rhmax_pct", rhmax_pct)
    if not 0 <= rhmin_pct <= rhmax_pct <= 100:
        raise ValueError("RH must satisfy 0 <= rhmin_pct <= rhmax_pct <= 100")
    return (saturation_vapour_pressure(tmin_c) * rhmax_pct / 100
            + saturation_vapour_pressure(tmax_c) * rhmin_pct / 100) / 2


def vapour_pressure_deficit(es_kpa: float, ea_kpa: float) -> float:
    """es − ea [kPa]; reject inconsistent pressures rather than clip."""
    _finite("es_kpa", es_kpa)
    _finite("ea_kpa", ea_kpa)
    if not 0 <= ea_kpa <= es_kpa:
        raise ValueError("vapour pressures must satisfy 0 <= ea <= es")
    return es_kpa - ea_kpa


