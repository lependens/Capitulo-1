"""FAO-56 daily solar geometry (equations 21–25, 34 and 37).

Source: https://www.fao.org/4/X0490E/x0490e07.htm
Radiation is MJ m⁻² day⁻¹; latitude input is decimal degrees, north positive.
The FAO approximation retains denominator 365 also in leap years.
"""

from datetime import date
import math


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _julian(day: int) -> None:
    if isinstance(day, bool) or not isinstance(day, int) or not 1 <= day <= 366:
        raise ValueError("day_of_year must be an integer in [1, 366]")


def day_of_year(day: date) -> int:
    """Return calendar ordinal within year (1–366), including leap days."""
    if not isinstance(day, date):
        raise ValueError("day must be a datetime.date")
    return day.timetuple().tm_yday


def inverse_relative_distance(day_of_year: int) -> float:
    """Dimensionless inverse relative Earth–Sun distance, FAO-56 Eq. 23."""
    _julian(day_of_year)
    return 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)


def solar_declination(day_of_year: int) -> float:
    """Solar declination in radians, FAO-56 Eq. 24."""
    _julian(day_of_year)
    return 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)


def sunset_hour_angle(latitude_deg: float, day_of_year: int) -> float:
    """Sunset angle [rad], with explicit polar day/night extension.

    Polar day/night clamp the arccos argument to [-1,1]; the FAO radiation
    approximation has limited winter validity beyond 55° latitude.
    """
    _finite("latitude_deg", latitude_deg)
    if not -90 <= latitude_deg <= 90:
        raise ValueError("latitude_deg must be in [-90, 90]")
    declination = solar_declination(day_of_year)
    argument = -math.tan(math.radians(latitude_deg)) * math.tan(declination)
    return math.acos(max(-1.0, min(1.0, argument)))


def daylight_hours(latitude_deg: float, day_of_year: int) -> float:
    """Maximum daylight duration [h day⁻¹], FAO-56 Eq. 34."""
    return 24 / math.pi * sunset_hour_angle(latitude_deg, day_of_year)


def extraterrestrial_radiation(latitude_deg: float, day_of_year: int) -> float:
    """Daily extraterrestrial radiation Ra [MJ m⁻² day⁻¹], Eq. 21."""
    angle = sunset_hour_angle(latitude_deg, day_of_year)
    latitude = math.radians(latitude_deg)
    declination = solar_declination(day_of_year)
    ra = (24 * 60 / math.pi * 0.0820 * inverse_relative_distance(day_of_year)
          * (angle * math.sin(latitude) * math.sin(declination)
             + math.cos(latitude) * math.cos(declination) * math.sin(angle)))
    return max(0.0, ra)  # remove floating-point roundoff at polar night


def clear_sky_radiation(ra_mj_m2_day: float, elevation_m: float) -> float:
    """Clear-sky Rso [MJ m⁻² day⁻¹], FAO-56 Eq. 37."""
    _finite("ra_mj_m2_day", ra_mj_m2_day)
    _finite("elevation_m", elevation_m)
    if ra_mj_m2_day < 0 or 0.75 + 2e-5 * elevation_m <= 0:
        raise ValueError("Ra must be nonnegative and clear-sky coefficient positive")
    return (0.75 + 2e-5 * elevation_m) * ra_mj_m2_day
