"""Display metadata for the closed four-baseline publication. No model code."""

MODELS = [
    dict(id="EtPMon", label="EtPMon · referencia SIAR", color="#243b53", reference=True,
         inputs="Referencia calculada por SIAR; no es medición instrumental independiente"),
    dict(id="PM_FAO56_V1", label="Penman–Monteith", color="#007e87", reference=False,
         inputs="Temperaturas, Rs medida, humedad, viento, presión/altitud y geometría"),
    dict(id="HS_RA_V1", label="Hargreaves–Samani · Ra", color="#ca7730", reference=False,
         inputs="Temperaturas y radiación extraterrestre geométrica"),
    dict(id="VALIANTZAS_V1", label="Valiantzas", color="#8757a6", reference=False,
         inputs="Temperaturas, Ra y humedad media"),
    dict(id="HG_RS_V1", label="Hargreaves · Rs", color="#3b72ba", reference=False,
         inputs="Temperaturas y radiación solar medida"),
]
METHOD_IDS = dict(ET0_FAO56="PM_FAO56_V1", ET0_Hargreaves="HS_RA_V1",
                  ET0_Valiantzas="VALIANTZAS_V1", ET0_Hargreaves_Rs="HG_RS_V1")
SUFFIXES = dict(PM_FAO56_V1="PM", HS_RA_V1="HS", VALIANTZAS_V1="VAL", HG_RS_V1="HGRS")
DATASET_SHA = "0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb"
CSV_SHA = "f55e4ed12691b84dbca4e090c49da538d5e7fad4f2ef371e1307a649cfb1af44"
LIMITATIONS = [
    "EtPMon es una referencia calculada SIAR, no ET₀ instrumental independiente ni lisímetro.",
    "IB09 permanece preservada y excluida de evaluación por Quality FAIL.",
    "IB101 tiene un registro corto.",
    "UNITS_PROVENANCE_PARTIAL; procedencia instrumental por estación y periodo no exhaustiva; viento a 2 m por convención SIAR.",
    "AHC temporal ya está validado científicamente pero se integra en WEB-V0.2; Forecast y ML todavía no integrados.",
]
