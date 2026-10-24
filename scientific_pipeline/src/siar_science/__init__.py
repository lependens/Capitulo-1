"""Reproducible observational SIAR science; no raw edits."""

from .hargreaves import HS_RA_V1, HS_RA_V1_PARAMETERS, hargreaves_samani
from .penman_monteith import PM_FAO56_V1, PM_FAO56_V1_PARAMETERS, fao56_daily
from .valiantzas import VALIANTZAS_V1, VALIANTZAS_V1_PARAMETERS, valiantzas
from .hargreaves_rs import HG_RS_V1, HG_RS_V1_PARAMETERS, hargreaves_rs

__all__ = ["fao56_daily", "hargreaves_samani", "PM_FAO56_V1", "HS_RA_V1",
           "PM_FAO56_V1_PARAMETERS", "HS_RA_V1_PARAMETERS", "VALIANTZAS_V1",
           "VALIANTZAS_V1_PARAMETERS", "valiantzas", "HG_RS_V1", "HG_RS_V1_PARAMETERS", "hargreaves_rs"]
__version__ = "0.1.0"
