# ET₀ V2: contrato de inferencia — borrador

La inferencia reutiliza una celda identificada por estación, familia y calendario. Las rutas originales están dentro del ZIP conservado en `/srv/siar-project/scientific-artifacts/et0-v2/original/retrospective-20261116/et0-v2-retrospective-370251881.zip`, bajo `et0-v2-retrospective-run-20261116/frozen-run/{ML,ANN_A,ANN_E}/{estación}__{familia}__{calendario}/`. Cada celda incluye `frozen.json` y archivos `.joblib`. No usar una celda de otra estación ni extrapolar a categorías sin soporte.

Orden de entradas: HS_RA_V1 `[Tmin, Tmax, Ra]`; HG_RS_V1 `[(Tmin+Tmax)/2, Rs]`; VALIANTZAS_V1 `[Tmin, Tmax, RHmean, Ra]`. Temperaturas en °C, radiación y Ra en MJ/m²/día y humedad relativa en %. Añadir al final los indicadores de calendario definidos por `et0_v2_support.design_vector`: ninguno para annual; DJF/MAM/JJA/SON para seasonal; 01–12 para monthly. La elegibilidad y los valores ausentes deben evaluarse con el pipeline científico; no sustituirlos por ceros. La salida es ET₀ estimada en mm/día.

ML usa `model.predict(x)` y conserva su preprocesamiento dentro del objeto. ANN usa `model['scale'].transform(x)` seguido de la media aritmética de las salidas de `model['models']`. Las dependencias registradas fueron Python 3.13.18, NumPy 2.1.3, SciPy 1.16.3, joblib 1.6.0, threadpoolctl 3.7.0 y scikit-learn 1.9.1; el wheel de scikit-learn se verificó por hash.

## Comprobación aislada

Para IB01, HG_RS_V1, annual, 2020-01-01 se utilizó el Dataset v1 con SHA `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`. La entrada fue `[(2.65+15.84)/2, 4.922]`. Con scikit-learn 1.9.1 las tres predicciones coincidieron exactamente con el CSV retrospectivo:

| Modelo | SHA256 joblib | Predicción mm/día |
|---|---|---:|
| Elastic Net | `c2c918b0b7691b0ef7214f061ab46a17f1694c928b595e0c89aa7929807e6fb7` | 0.48400743995047346 |
| ANN A | `2e344fda9266be437e86d161efd92729542de8a380b694859acf9aef23420ca9` | 0.854429475184935 |
| ANN E | `8769bd6352421aff0c2d32f9b7c5ee2e8b2c27bb387dba1e9968af69f1e0930d` | 0.8546901355262677 |

Es una comprobación representativa de carga y cálculo, no una validación de todas las celdas ni autorización para un servicio productivo.

