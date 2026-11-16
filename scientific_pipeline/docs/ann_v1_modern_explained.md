# ANN-V1-MODERN-ET0: guía del experimento y sus resultados

Esta guía explica el primer experimento moderno de redes neuronales del proyecto ET0. Está pensada para una persona que conoce el proyecto, pero no necesita dominar aprendizaje automático. El contrato técnico congelado permanece en [ann_v1_modern_methodology.md](ann_v1_modern_methodology.md) y en [`ann-v1-modern-et0.json`](../experiments/ann-v1-modern-et0.json).

La [presentación ampliada de ANN-V1](assets/ann_v1_modern_et0_presentation.pptx) explica el experimento en 13 diapositivas. También está disponible su [vista previa](assets/ann_v1_modern_et0_presentation.png). Para una síntesis breve, véase la [diapositiva resumen](assets/ann_v1_modern_et0_summary.pptx) y su [vista previa](assets/ann_v1_modern_et0_summary.png).

La [revisión metodológica y propuesta de V2](et0_methodology_review_v2.md) compara los métodos calibrados, documenta la recuperación de TEST del ML clásico y recoge las decisiones pendientes.

## 1. La pregunta científica

El proyecto quiere estimar la evapotranspiración de referencia diaria, ET0. En este experimento, la red neuronal aprende a reproducir `PM_FAO56_V1`, el cálculo diario de Penman-Monteith FAO-56 implementado por el pipeline científico.

Esto fija el significado de todas las métricas:

- El objetivo de entrenamiento es `PM_FAO56_V1`.
- Un RMSE menor significa que el modelo reproduce mejor ese cálculo en fechas que no usó para ajustar sus pesos.
- La comparación principal enfrenta la ANN con `ML_HIST_GRADIENT_BOOSTING__FEATURE_SET_FULL`, el champion de ML clásico, usando exactamente las mismas filas y el mismo objetivo.
- Las variantes AHC y los baselines clásicos aportan contexto, pero no sustituyen la comparación pareada con HGB porque sus protocolos y requisitos de entrada no son idénticos.

El experimento no usa una medición instrumental independiente de ET0 como verdad de terreno. Por tanto, una ANN más precisa aproxima mejor la formulación PM adoptada; no demuestra que estime mejor la ET0 física real.

## 2. Lugar dentro del pipeline

El orden científico es:

1. Dataset Observational v1 auditado y cerrado.
2. Cálculo de `PM_FAO56_V1` como referencia interna.
3. Baselines `HS_RA_V1`, `HG_RS_V1` y `VALIANTZAS_V1`.
4. Ajustes AHC con separación temporal.
5. Modelos ML clásicos, cuyo champion fue HGB FULL.
6. ANN modernas con el mismo target, familias de inputs y cortes temporales del contrato ML.

La ANN no reconstruye el dataset, no modifica PM y no recalcula AHC. Consume resultados ya identificados por sus hashes.

## 3. Datos, estaciones y elegibilidad

La fuente es `siar-balears-observational-v1`, SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`. El histórico cubre 2004–2025 y conserva calidad, procedencia y ausencias por observación.

La campaña utiliza nueve estaciones primarias:

`IB01`, `IB02`, `IB03`, `IB04`, `IB05`, `IB06`, `IB08`, `IB10` e `IB11`.

Tres estaciones del Dataset v1 no forman parte de la comparación principal:

- `IB07`: cerró en 2012. Su histórico alcanza la primera ventana de validación, pero no el periodo TEST común.
- `IB09`: conserva su estado Quality FAIL.
- `IB101`: solo aporta una serie corta dentro de TRAIN.

La elegibilidad se calcula por familia de inputs. Una fila puede ser válida para una familia sencilla y no para FULL si falta humedad, radiación medida o viento. No se imputan variables meteorológicas y no se recortan predicciones. En TEST, la cohorte pareada ANN FULL–HGB FULL contiene 15.958 filas y cubre el 99,875 % de la cohorte de referencia compatible.

## 4. Qué recibe cada familia de inputs

Todas las familias reciben un contexto común: latitud, altitud, radiación extraterrestre calculada `Ra`, y dos variables cíclicas que representan el día del año. Los nombres T, T+RH y similares describen las variables meteorológicas añadidas a ese contexto común.

| Familia | Variables meteorológicas añadidas | Pregunta que permite estudiar |
|---|---|---|
| `FEATURE_SET_T` | Temperatura media de extremos (`Tcalc`) y amplitud térmica (`TD`) | Qué se puede aproximar con temperatura y contexto astronómico |
| `FEATURE_SET_T_RH` | T + humedad media, mínima y máxima | Cuánto aporta la humedad sin radiación medida |
| `FEATURE_SET_T_RS` | T + radiación solar medida (`Radiacion`) | Cuánto aporta la radiación medida sin humedad |
| `FEATURE_SET_T_RH_RS` | T + humedad + radiación medida | Efecto conjunto de humedad y radiación |
| `FEATURE_SET_FULL` | T + humedad + radiación medida + viento | Máxima información comparable con HGB FULL |

`PM_FAO56_V1`, `EtPMon`, otros modelos, residuos y el código de estación quedan fuera de los predictores. Así se evita que la red copie directamente el target o aprenda identificadores de estación.

## 5. Validación temporal

El experimento separa dos periodos:

- **DEVELOPMENT 2004–2020**: selección de arquitectura y ajuste final.
- **TEST 2021–2025**: evaluación única de las configuraciones ya congeladas.

Dentro de DEVELOPMENT se aplican cuatro ventanas crecientes:

| Fold | TRAIN | VALIDATION |
|---|---|---|
| F1 | 2004–2009 | 2010–2012 |
| F2 | 2004–2012 | 2013–2015 |
| F3 | 2004–2015 | 2016–2018 |
| F4 | 2004–2018 | 2019–2020 |

Cada fold entrena con años anteriores y valida en años posteriores. Nunca usa el futuro para ajustar una predicción del pasado.

La selección minimiza el RMSE medio de los cuatro folds, ponderado por el número de observaciones elegibles. Si dos candidatos empatan dentro de la tolerancia declarada, el protocolo prefiere menor dispersión entre folds y después menor complejidad.

El escalado `StandardScaler` se ajusta solamente con TRAIN. El 15 % final de las fechas de TRAIN actúa como validación interna cronológica para elegir el número de épocas. La VALIDATION externa del fold no controla el early stopping. Después, el modelo se reinicia y se ajusta sobre todo el TRAIN del fold durante el número de épocas seleccionado.

TEST tiene estado `retrospective_locked`. Ya se había usado en investigaciones ML anteriores y no debe describirse como un holdout prospectivo virgen. Para la ANN, configuración, selección, pesos y hashes quedaron congelados antes de abrirlo. El journal final registra `test_access_count = 1`; TEST no cambió el champion.

## 6. Arquitecturas y entrenamiento

La rejilla contiene tres perceptrones multicapa:

- `MLP [32]`: una capa oculta de 32 neuronas.
- `MLP [64,32]`: dos capas ocultas.
- `MLP [64,32,16]`: tres capas ocultas.

Las tres usan activación ReLU, optimizador Adam, penalización L2 `alpha = 0,0001`, learning rate inicial `0,001` y batch de 256 observaciones. El máximo es 200 épocas, con paciencia de 15 épocas y mejora mínima de RMSE de `0,0001`.

Cada configuración usa tres semillas fijas: `20261101`, `20261102` y `20261103`. El pipeline no elige la semilla que mejor sale. Promedia las tres predicciones y evalúa ese ensemble.

### Conteo exacto

Hay dos formas útiles de contar la campaña:

- **15 configuraciones candidatas**: 5 familias × 3 arquitecturas.
- **180 evaluaciones fold–seed**: 15 candidatas × 4 folds × 3 semillas.

Cada evaluación fold–seed contiene dos inicializaciones independientes: una para elegir épocas dentro de TRAIN y otra para reajustar en todo el TRAIN del fold. Esto suma 360 redes ajustadas durante la validación de desarrollo. Después se conserva una arquitectura por familia y se ajustan tres semillas sobre DEVELOPMENT completo: 5 familias × 3 semillas = 15 modelos finales. En total hubo 375 inicializaciones entrenadas, pero el bundle final conserva 15 modelos.

El champion ANN es un ensemble de tres modelos FULL con arquitectura `[32]`; su predicción diaria es la media de las tres semillas.

## 7. Entorno real de ejecución

La campaña oficial se ejecutó en el servidor Linux local `iaserver`, sobre CPU, desde el commit fusionado `754adceee12cc423ee06eb17b49e36429d25b70d`. El proceso fijó `OPENBLAS_NUM_THREADS=1` y `OMP_NUM_THREADS=1` para controlar el paralelismo.

El manifiesto DEVELOPMENT registra:

- Python `3.14.4`.
- NumPy `2.5.4`.
- scikit-learn `1.9.1`.
- Runtime DEVELOPMENT `1.496,78 s`, aproximadamente 24 min 57 s.

El log de DEVELOPMENT registra un pico de memoria residente de 709.528 KiB (aproximadamente 693 MiB). El modelo y las cinco familias cabían holgadamente en el servidor, por lo que esta campaña ANN no necesitó GPU ni Kaggle. El modelo usa `MLPRegressor` de scikit-learn; no depende de TensorFlow o PyTorch. El total de RAM y el modelo exacto de CPU del host no forman parte del manifiesto preservado, por lo que no se presentan como evidencia canónica.

## 8. Champion y resultados TEST

DEVELOPMENT seleccionó `FEATURE_SET_FULL / MLP [32]` con RMSE medio ponderado `0,074962 mm/día`. La arquitectura más sencilla ganó a las alternativas profundas bajo el criterio congelado.

Resultados sobre la cohorte común ANN FULL–HGB FULL:

| Métrica | ANN FULL [32] | HGB FULL | Diferencia ANN − HGB |
|---|---:|---:|---:|
| n | 15.958 | 15.958 | 0 |
| MAE (mm/día) | 0,037632 | 0,047045 | −0,009413 |
| RMSE (mm/día) | 0,051312 | 0,064104 | −0,012791 |
| bias (mm/día) | −0,022449 | −0,000966 | −0,021484 |
| R² predictivo | 0,999100 | 0,998595 | +0,000505 |

La ANN reduce el RMSE un 19,95 % y el MAE un 20,01 % frente a HGB FULL. También presenta un sesgo negativo mayor: en promedio queda `0,02245 mm/día` por debajo de PM. Esta diferencia debe conservarse junto a la mejora de error absoluto.

La cobertura es 99,875 %. No hay predicciones negativas ni no finitas.

### Estabilidad temporal y espacial

- Mejor año: 2024, RMSE `0,045078 mm/día`.
- Peor año: 2022, RMSE `0,058756 mm/día`.
- Mejor estación: IB04, RMSE `0,040756 mm/día`.
- Peor estación: IB02, RMSE `0,065308 mm/día`.
- Mejor temporada: SON, RMSE `0,042999 mm/día`.
- Peor temporada: JJA, RMSE `0,064226 mm/día`.

El resultado no depende de un único año o estación, aunque verano, IB02, IB03 e IB01 muestran más error y un sesgo negativo más claro.

## 9. Interpretación para no especialistas

FULL mejora porque aporta las variables que la propia formulación PM utiliza o resume: temperatura, humedad, radiación y viento, además del contexto espacial y astronómico. La red puede representar interacciones no lineales entre ellas y aproximar con mucha precisión la superficie matemática de PM.

Esto no demuestra causalidad. Tampoco significa que FULL haya descubierto una ley física nueva. La ANN aprende una función flexible que imita un cálculo físico ya definido.

Frente a HGB, la ANN consigue menor MAE y RMSE con una arquitectura compacta. A cambio, exige escalado, entrenamiento iterativo, control de semillas y más cuidado para reproducir y auditar el ajuste. HGB conserva ventajas de simplicidad operativa e interpretación relativa. La ANN añade valor como emulador de PM cuando están disponibles todos los inputs, pero no reemplaza la formulación FAO-56 ni valida la ET0 real.

## 10. Limitaciones

- El target es PM calculado, no una ET0 instrumental independiente.
- TEST es retrospectivo y ya había sido usado por la campaña ML clásica.
- El resultado se refiere a nueve estaciones de Baleares y a la disponibilidad observada entre 2004 y 2025.
- FULL depende de humedad, radiación medida y viento; una ausencia impide esa predicción diaria.
- La ANN tiene menor interpretabilidad y mayor coste de entrenamiento que los modelos clásicos.
- El sesgo negativo es pequeño en magnitud, pero claramente mayor que el de HGB.
- Las métricas no prueban rendimiento agronómico, balance hídrico parcelario ni transferencia directa a otra región.
- Los resultados AHC publicados siguen su propio protocolo; no deben presentarse como si hubieran participado en la misma selección ANN.

## 11. Próximos pasos

1. Validar prospectivamente los modelos congelados con datos 2026+ sin rediseñarlos con esos mismos datos.
2. Incorporar las predicciones ANN a una futura versión web como artefactos derivados, sin recalcular ciencia en la interfaz.
3. Estudiar transferencia territorial con metadatos, clima y sensores comparables.
4. Evaluar nuevas ANN solo mediante contratos versionados y TEST independientes.
5. Considerar uso operativo únicamente después de validar disponibilidad de inputs, mantenimiento y comportamiento fuera de distribución.

## 12. Trazabilidad canónica

- Protocolo ANN SHA256: `77c0b28963871ca786f90caebb088c59e07d3d64e941cec02e72e1bba1b5d53a`.
- Pipeline de ejecución: `754adceee12cc423ee06eb17b49e36429d25b70d`.
- Manifest TEST SHA256: `1978c8d018b57686b7e2d7fd9457ad2f9399a115e65789a13abd0f0c9359862f`.
- Artefacto canónico: `/srv/siar-project/scientific-artifacts/ann-v1-modern-20261102T065642Z/`.
- Receipt de ingestión: `science-ann-v1-modern-20261102T065642Z_447f9498.json`, estado `published-verified`.
- Snapshot: `backup_20261102T171546Z_40b4cd67`, estado `REMOTE_VERIFIED`.

Esta actualización documenta los resultados preservados. No reabre TEST, no reentrena modelos y no modifica raw, Quality, Dataset v1, PM, AHC o los artefactos científicos.
