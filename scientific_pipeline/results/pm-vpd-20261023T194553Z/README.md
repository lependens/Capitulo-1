# Primera ejecución oficial PM_FAO56_V1 y HS_RA_V1

PM-VPD-FLOAT-TOLERANCE-AND-BASELINES = **PASS**. Ejecución sin calibración,
AHC, Valiantzas, ML o forecast. Inicio UTC: `2026-10-23T19:48:53.843799Z`;
fin UTC: `2026-10-23T19:49:44.813948Z`.

## Identidades y corrección numérica

- PR del fix: [#30](https://github.com/lependens/Capitulo-1/pull/30).
- Pipeline/merge: `39c101e5999245eb15e3c6c1c867d90b4ac44468`; checkout limpio y paquete coincidente.
- Publicación canónica (`source_repo_sha`): `2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9`.
- Audit (`audit_source_repo_sha`): `18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865`.
- Dataset: `siar-balears-observational-v1`.
- Dataset SHA256 sin cambios: `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`.
- POST corregido: `SIAR-QA-POST-20261020T193140Z`.
- Paquete POST SHA256: `a2d1293fcc54254769dbc80c7105b08b4b03b6a14392df0a4e5ef19b4ef51d07`.
- Handoff autorizado SHA256: `2b47f9802edbc33d71916701df57237092f563f0fe527fbfe401faf522a724ed`.
- Manifest del resultado SHA256: `872ce9dfa01102fe68e27cde2c63b22769f8963c6e569b582cebf6e7cef8b1e4`.
- Configuración SHA256: `f57604749e986cfd0286fa83d726acdfe86e684c2a467b602c5f45444c75a53c`.

La tolerancia VPD es absoluta **1e-12 kPa**, relativa **0.0**, aplicada solo
cuando ea>es y ambas presiones son finitas y no negativas. Dentro del margen
VPD=0; excesos mayores y presiones negativas siguen siendo errores. Los VPD
positivos permanecen exactamente es−ea. No cambia humedad, temperaturas,
es, ea, EtPMon ni ecuaciones FAO-56. La política figura en los parámetros del
manifest. El margen es nueve órdenes menor que una diferencia de 0.001 kPa.

Pasan 62 tests y CI Python 3.11/3.12/3.13. La comparación completa antes del
merge y la comprobación del rerun oficial demuestran exactamente tres cambios:
IB10/2022-12-13, IB10/2022-12-31 e IB10/2023-03-08 pasan de fallo a cálculo PM.
Las otras 75.493 filas son idénticas; HS, valores fuente y elegibilidad no cambian.

## Dataset, cobertura y métricas globales

12 estaciones, 75.496 filas preservadas, 75.495 observaciones en periodo oficial,
2.345 CONFIRMED_NO_DATA, 1 RECOVERED, 0 PENDING_RETRY y 1 fila exterior:
IB09/2005-10-23. No se crea meteorología para ausencias.

PM elegible/calculado: **70.896**, fallos **0**; HS elegible/calculado:
**70.979**, fallos **0**. HS cubre 83 observaciones adicionales. Son recuentos
de predicción; la evaluación además requiere EtPMon admisible. PM evalúa
70.861 pares y HS 70.912. La intersección evaluable es **70.861**; coincide
con la cohorte propia PM. La cifra orientativa 70.899 no era elegibilidad:
los tres casos reparados ya estaban incluidos en 70.896 elegibles.

Error=predicción−EtPMon; MAE, RMSE y bias en mm/día. R²=1−SSE/SST.
Los 47 pares con EtPMon=0 se conservan en ambas evaluaciones. EtPMon es una
referencia **calculada SIAR**, sin independencia instrumental frente a PM.
El acuerdo elevado con PM describe similitud entre cálculos, no exactitud
validada frente a evapotranspiración medida.

| Grupo | Método | Cohorte | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| all | PM | own | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | HS-Ra | own | 70912 | 0.712895 | 0.874488 | 0.461359 | 0.746479 |
| all | PM | paired | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | HS-Ra | paired | 70861 | 0.712850 | 0.874365 | 0.461603 | 0.746552 |

## Estaciones y sesgos

| Estación | Método | Cohorte | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| IB01 | PM | own | 7702 | 0.010304 | 0.025524 | 0.000871 | 0.999790 |
| IB01 | HS-Ra | own | 7702 | 0.645967 | 0.791372 | 0.443881 | 0.798044 |
| IB02 | PM | own | 7693 | 0.010308 | 0.028500 | 0.000470 | 0.999770 |
| IB02 | HS-Ra | own | 7694 | 0.618015 | 0.766099 | 0.390309 | 0.833726 |
| IB03 | PM | own | 7864 | 0.009345 | 0.026640 | -0.002006 | 0.999781 |
| IB03 | HS-Ra | own | 7864 | 0.579051 | 0.713781 | 0.407356 | 0.842765 |
| IB04 | PM | own | 7707 | 0.010728 | 0.024717 | -0.003116 | 0.999801 |
| IB04 | HS-Ra | own | 7712 | 0.640949 | 0.773506 | 0.552768 | 0.804777 |
| IB05 | PM | own | 7826 | 0.010162 | 0.022719 | 0.003529 | 0.999829 |
| IB05 | HS-Ra | own | 7826 | 0.751890 | 0.897449 | 0.690638 | 0.733325 |
| IB06 | PM | own | 7859 | 0.013375 | 0.026690 | -0.005400 | 0.999761 |
| IB06 | HS-Ra | own | 7861 | 0.886503 | 1.042291 | 0.769361 | 0.635006 |
| IB07 | PM | own | 2928 | 0.012568 | 0.020656 | 0.001829 | 0.999825 |
| IB07 | HS-Ra | own | 2934 | 0.918057 | 1.055976 | 0.889596 | 0.542991 |
| IB08 | PM | own | 7050 | 0.013412 | 0.026199 | -0.003996 | 0.999733 |
| IB08 | HS-Ra | own | 7055 | 0.714255 | 0.878765 | 0.518395 | 0.699287 |
| IB09 | PM | own | 0 | — | — | — | — |
| IB09 | HS-Ra | own | 0 | — | — | — | — |
| IB10 | PM | own | 6975 | 0.013374 | 0.030263 | -0.002032 | 0.999651 |
| IB10 | HS-Ra | own | 6991 | 0.783888 | 0.942980 | 0.715225 | 0.661168 |
| IB101 | PM | own | 243 | 0.057131 | 0.087134 | -0.047892 | 0.998038 |
| IB101 | HS-Ra | own | 245 | 0.573520 | 0.808641 | -0.357356 | 0.831199 |
| IB11 | PM | own | 7014 | 0.011371 | 0.040013 | -0.002560 | 0.999407 |
| IB11 | HS-Ra | own | 7028 | 0.728441 | 0.948965 | -0.541424 | 0.666481 |

El mayor MAE y RMSE PM por estación corresponden a IB101: 0.057131 y
0.087134 mm/día, bias −0.047892, con solo 243 pares. IB11 tiene el siguiente
RMSE PM (0.040013), seguido de IB10 (0.030263). Las diferencias PM permanecen
pequeñas frente a EtPMon en las cohortes analizadas.

HS presenta los mayores MAE/RMSE en IB07 (0.918057/1.055976) y en IB06
(0.886503/1.042291). Sus sesgos son positivos, +0.889596 y +0.769361 mm/día.
IB11 contrasta con sesgo negativo −0.541424; IB101 también es negativo,
−0.357356. Los desgloses mantienen sus cohortes y tamaños: no se comparan
periodos diferentes como si fueran una intersección pareada.

IB09 tiene n=0 para ambos métodos porque el contrato heredado aplica Quality
FAIL de estación a sus 4.423 filas. Se conservan todas en el dataset, incluidas
4.422 observaciones dentro de periodo oficial. Esta limitación de cobertura
no se cambia durante el fix numérico.

## Periodos y anomalías descriptivas

Entre estaciones-año con n≥30, el mayor RMSE PM es IB11/2020: 0.126739
mm/día (n=366), bias +0.002589. El mayor residuo diario PM es IB11/2020-12-31:
PM=2.397795896 y EtPMon=0. También hay EtPMon=0 entre los mayores residuos
en IB11/2016-12-31, IB11/2012-12-31 e IB02/2020-12-31. Se conservan esos
ceros; esta observación descriptiva no demuestra su causa ni autoriza
reinterpretarlos, imputarlos o excluirlos.

HS tiene su mayor RMSE de estación-año con n≥30 en IB06/2025 (1.368230,
bias +1.147360, n=365), IB06/2024 (1.298751, +1.099274, n=366) e IB10/2023
(1.294469, +1.106406, n=365). El umbral n≥30 se usa solo para esta lista
descriptiva; todas las métricas, incluidas muestras menores, permanecen
publicadas. No se estima ni ajusta ningún coeficiente con estos resultados.

La cohorte pareada presenta estos resultados por estación climática:

| Grupo | Método | Cohorte | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| DJF | PM | paired | 17177 | 0.010002 | 0.047189 | 0.004036 | 0.991697 |
| DJF | HS-Ra | paired | 17177 | 0.524262 | 0.624833 | 0.268816 | -0.455772 |
| JJA | PM | paired | 18058 | 0.011950 | 0.018286 | -0.004524 | 0.999615 |
| JJA | HS-Ra | paired | 18058 | 0.907931 | 1.093383 | 0.647642 | -0.375204 |
| MAM | PM | paired | 17597 | 0.016344 | 0.023236 | -0.005691 | 0.999592 |
| MAM | HS-Ra | paired | 17597 | 0.721016 | 0.882316 | 0.466231 | 0.412392 |
| SON | PM | paired | 18029 | 0.007906 | 0.012805 | 0.000091 | 0.999852 |
| SON | HS-Ra | paired | 18029 | 0.689163 | 0.821970 | 0.454423 | 0.391004 |

HS tiene R² predictivo negativo en DJF y JJA: −0.455772 y −0.375204.
En esos subconjuntos SSE supera SST; el R² global 0.746552 no representa
uniformemente el rendimiento estacional. La mayor RMSE estacional HS es JJA
(1.093383), con bias +0.647642. No se usa esta descripción para calibrar.

## Artefactos, integridad y reproducción

`baselines/manifest.json` y `baselines/metrics.json` son copias de bytes
exactos del run. `result-summary.json`, `artifact-index.json`,
`impact-before-merge.json`, `dataset-build-manifest.json` y `preflight.json`
preservan la comprobación de identidades y hashes. `SHA256SUMS.txt` protege
el contenido versionado; la regla local `* -text` conserva sus bytes al hacer
checkout en Windows.

Las métricas completas contienen 1.436 registros: cohortes propias y pareadas
por global, estación, año, mes, estación-año, DJF/MAM/JJA/SON y temporada-año.
Diciembre pertenece al año siguiente en DJF. No se promedian RMSE de estaciones
para construir el global.

Los archivos masivos no se versionan. Conservación local del dataset y del run:

- Dataset validado: `C:/Users/Josep/.codex/.chatgpt-projects/g-p-6ac0c6c1d8908191b03f324e51c278f1/scientific-results/dataset-v1-20261021T055924Z/dataset.json`.
- Run completo: `C:/Users/Josep/.codex/.chatgpt-projects/g-p-6ac0c6c1d8908191b03f324e51c278f1/scientific-results/pm-vpd-20261023T194553Z/baselines/`.
- CSV de métricas con identidades: `C:/Users/Josep/.codex/.chatgpt-projects/g-p-6ac0c6c1d8908191b03f324e51c278f1/scientific-results/pm-vpd-20261023T194553Z/metrics-aggregated.csv`.

El índice contiene también hashes de outputs completos omitidos de esta copia
compacta (predicciones y CSV agregado). Las rutas relativas de outputs del
manifest original se resuelven en el directorio del **run completo**.

Con el paquete del SHA de pipeline instalado y el dataset preservado, ejecutar
`python -m siar_science run-baselines --dataset <dataset.json> --output <directorio-nuevo> --pipeline-sha 39c101e5999245eb15e3c6c1c867d90b4ac44468`.
Nunca sobrescribir una ejecución anterior. Para reconstruir el dataset desde
fuentes, materializar los blobs Git exactos de source_repo_sha y el handoff
autorizado, conservar las rutas relativas y verificar cada hash. El paquete
JSON POST tuvo conversión CRLF en checkout Windows; las entradas utilizadas
fueron sus bytes canónicos Git, sin normalización dentro del builder.

## Limitaciones retenidas

EtPMon calculado SIAR; `UNITS_PROVENANCE_PARTIAL`; viento a 2 m por convención
SIAR, sin verificación instrumental exhaustiva estación×periodo; cobertura IB09
excluida por Quality FAIL; elegibilidad independiente por método; intervalos
de cobertura desiguales y muestra corta IB101. No hay imputación, calibración
ni tuning. No hay predicciones negativas en este run; su política sigue siendo
conservarlas con flags. El JSON del dataset incluye rutas absolutas de
provenance: reconstruirlo en otra ubicación puede cambiar su hash de bytes.
Los intentos FAILED y sus blocked-run.json permanecen intactos fuera de Git.
