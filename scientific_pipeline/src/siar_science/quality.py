"""Per-variable QC and per-method eligibility; observations are never modified.

QC flags are diagnostics, not global row deletion. Eligibility uses only a
method's inputs and computational domains. PM_FAO56_V1 and HS_RA_V1 derive
Tmean from extremes, so SIAR TempMedia and EtPMon are not prediction inputs.
Wind defaults to the documented SIAR 2 m convention, explicitly flagged; an
explicit contradictory sensor height stops PM, with no inferred conversion.
"""
from datetime import date
import math
from .solar import clear_sky_radiation, extraterrestrial_radiation

TEMPS = ("TempMedia", "TempMin", "TempMax")
EXTREMES = ("TempMin", "TempMax")
HUMIDITY = ("HumedadMin", "HumedadMax")
PM_INPUTS = EXTREMES + HUMIDITY + ("VelViento", "Radiacion")


def finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _day(row):
    try:
        return date.fromisoformat(row["date"])
    except (KeyError, TypeError, ValueError):
        return None


def _radiation_ratio(row, station):
    values = row["values"]
    rs, lat, z = values.get("Radiacion"), station.get("latitude_deg"), station.get("altitude_m")
    day = _day(row)
    if (day is None or not finite(rs) or rs < 0 or not finite(lat)
            or not -90 < lat < 90 or not finite(z) or not -500 <= z <= 9000):
        return None
    rso = clear_sky_radiation(extraterrestrial_radiation(lat, day.timetuple().tm_yday), z)
    return rs / rso if rso > 0 else None


def eligibility(row, station, method):
    """Return calculation-blocking reasons for pm/hargreaves or versioned IDs.

    Missing reference EtPMon, missing or inconsistent TempMedia, and optional
    variables never block either prediction. Plausibility warnings live in
    scientific_flags; undefined mathematical domains block the relevant method.
    """
    aliases = {"pm": "pm", "PM_FAO56_V1": "pm",
               "hargreaves": "hargreaves", "HS_RA_V1": "hargreaves"}
    if method not in aliases:
        raise ValueError("Unknown scientific method: " + str(method))
    method = aliases[method]
    reasons = []
    if row.get("observation_status") not in ("OBSERVED", "RECOVERED"):
        reasons.append("not_observed_in_official_period")
    if row.get("quality_status") not in ("PASS", "WARN"):
        reasons.append("source_quality_not_accepted")
    if _day(row) is None:
        reasons.append("invalid_date")
    values = row["values"]
    for key in PM_INPUTS if method == "pm" else EXTREMES:
        value = values.get(key)
        if not finite(value):
            reasons.append("missing_or_nonfinite:" + key)
        elif key in EXTREMES and value <= -237.3:
            reasons.append("temperature_domain:" + key)
        elif key in HUMIDITY and not 0 <= value <= 100:
            reasons.append("humidity_out_of_range:" + key)
        elif key in ("VelViento", "Radiacion") and value < 0:
            reasons.append("negative:" + key)
    if all(finite(values.get(k)) for k in EXTREMES) and values["TempMin"] > values["TempMax"]:
        reasons.append("temperature_extremes_order")
    if method == "pm" and all(finite(values.get(k)) for k in HUMIDITY):
        if values["HumedadMin"] > values["HumedadMax"]:
            reasons.append("humidity_order")
    lat = station.get("latitude_deg")
    if not finite(lat) or not -90 < lat < 90:
        reasons.append("invalid_latitude")
    if method == "pm":
        z = station.get("altitude_m")
        if not finite(z) or not -500 <= z <= 9000:
            reasons.append("invalid_altitude")
        height = station.get("wind_height_m")
        if height is not None and (not finite(height) or height != 2):
            reasons.append("STOP:wind_height_not_2m")
        if (_day(row) is not None and finite(lat) and -90 < lat < 90
                and extraterrestrial_radiation(lat, _day(row).timetuple().tm_yday) <= 0):
            reasons.append("clear_sky_radiation_not_positive")
    return reasons


def scientific_flags(row, station):
    """Return non-mutating per-variable diagnostics, independent of eligibility.

    The (-80,60) temperature review range is descriptive, never a sigma rule
    or an implicit eligibility mask. Very low Rs/Rso preserves signed Rnl
    under FAO Eq.39; the calculation is flagged rather than floored or erased.
    """
    values, flags = row["values"], []
    for key in TEMPS + HUMIDITY + ("VelViento", "Radiacion", "EtPMon"):
        value = values.get(key)
        if not finite(value):
            flags.append("missing_or_nonfinite:" + key)
            continue
        if key in TEMPS and not -80 <= value <= 60:
            flags.append("temperature_review:" + key)
        if key in HUMIDITY and not 0 <= value <= 100:
            flags.append("humidity_out_of_range:" + key)
        if key in ("VelViento", "Radiacion", "EtPMon") and value < 0:
            flags.append("negative:" + key)
    if all(finite(values.get(k)) for k in EXTREMES):
        lo, hi = values["TempMin"], values["TempMax"]
        if lo > hi:
            flags.append("temperature_extremes_order")
        if finite(values.get("TempMedia")) and not lo <= values["TempMedia"] <= hi:
            flags.append("temperature_order:TempMedia")
        if (lo + hi) / 2 < -17.8:
            flags.append("hargreaves_negative_temperature_factor")
    if all(finite(values.get(k)) for k in HUMIDITY) and values["HumedadMin"] > values["HumedadMax"]:
        flags.append("humidity_order")
    height = station.get("wind_height_m")
    if height is None:
        flags.append("wind_height_assumed_2m")
    elif not finite(height) or height != 2:
        flags.append("STOP:wind_height_not_2m")
    ratio = _radiation_ratio(row, station)
    if ratio is not None:
        if ratio < 0.35 / 1.35:
            flags.append("low_rs_rso:negative_cloud_factor")
        elif ratio < 0.3:
            flags.append("low_rs_rso:review")
        if ratio > 1:
            flags.append("rs_above_rso")
    return flags
