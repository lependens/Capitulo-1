# Primera comparación oficial PM / HS-Ra / VALIANTZAS_V1

**VALIANTZAS-V1-AND-CLASSICAL-BASELINES-EXPORT = PASS** (2026-10-24).
Sin calibración, AHC, ML o forecast. Inicio comparación UTC `2026-10-24T03:46:08.837611Z`;
fin `2026-10-24T03:47:14.584551Z`. Se conserva la ejecución PM/HS anterior.

## Fórmula y revisión aprobada

`VALIANTZAS_V1`: composición **Martí et al. (2024), Ecs.1b+3**, con factor
energético fijo 0.408, coeficiente expandido **0.001861704**, factor
`(1.001−HumedadMedia/100)^0.2`, Tcalc=(Tmax+Tmin)/2 y Ra en MJ/m²/día.
Salida mm/día. [Auditoría y fuentes primarias](../../docs/valiantzas_v1.md).
No es una transcripción literal de la Ec.4 impresa ni reproducción certificada
del código autoral. Frente a legacy cambia 1→1.001 y TempMedia→Tcalc, conservando
los valores originales y sin ajustar coeficientes contra EtPMon.

## Identidad y reproducibilidad

- Código: [PR #33](https://github.com/lependens/Capitulo-1/pull/33); merge/pipeline `4ac52e25c803cb2301316c5cb1353bf30cdb427a`.
- Productor PM/HS previo `39c101e5999245eb15e3c6c1c867d90b4ac44468`; importado, sin recalcular ni alterar predicciones/elegibilidad.
- Dataset `siar-balears-observational-v1`, SHA256 **`0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`**, sin cambios.
- Publicación canónica/source_repo_sha `2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9`.
- Snapshot auditado/audit_source_repo_sha `18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865`.
- POST vigente `SIAR-QA-POST-20261020T193140Z`; paquete SHA256 `a2d1293fcc54254769dbc80c7105b08b4b03b6a14392df0a4e5ef19b4ef51d07`.
- Handoff corregido SHA256 `2b47f9802edbc33d71916701df57237092f563f0fe527fbfe401faf522a724ed`.
- Result manifest SHA256 **`7ab3ff1f168fbdb4a32984e413ad5a2fd6a6af8b9ccde4d95c08bad2f38926f6`**.
- Manifest cálculo VAL SHA256 `4a4beee832859863d4421880276fa4697169e061d1520ad3cfc2496f17b0de44`.
- Configuración SHA256 `1c7124ab3435aa3d958bc816ca0879cbac432884faf64ec44f873be330505b90`.
- 68 tests PASS; [CI Python 3.11/3.12/3.13 PASS](https://github.com/lependens/Capitulo-1/actions/runs/40058668854).

Checkout productor limpio, paquete coincidente con checkout. Verificados fuentes
12/12, catálogo, paquete y tres informes POST, handoff, dataset, inputs de
predicciones y outputs. Ambas identidades audit/canonical permanecen separadas
del SHA del pipeline. Ver [result-summary.json](result-summary.json),
[índice de artefactos](artifact-index.json), [manifest comparación](comparison/manifest.json)
y [manifest VAL](valiantzas/manifest.json). Los manifiestos se copian sin
cambiar bytes; sus paths se resuelven contra el run completo original.

## Cobertura y cohortes

12 estaciones; 75.496 filas raw preservadas, 75.495 en periodo oficial, 1
RECOVERED, 2.345 CONFIRMED_NO_DATA, 0 PENDING_RETRY. Única fila exterior:
IB09/2005-10-23. El ledger no se convierte en meteorología fabricada.

| Método | Elegible/calculado | Evaluable propia | Fallos |
|---|---:|---:|---:|
| HS_RA_V1 | 70979 | 70912 | 0 |
| PM_FAO56_V1 | 70896 | 70861 | 0 |
| VALIANTZAS_V1 | 70944 | 70878 | 0 |

HS predice 83 filas más que PM; VAL predice 48 más que PM y 35 menos que HS.
Las máscaras se definen por inputs de cada modelo y estados Quality, no por
los errores resultantes. Los recuentos evaluables añaden referencia admisible.

| Estación | Predicciones PM | HS-Ra | VAL |
|---|---:|---:|---:|
| IB01 | 7704 | 7704 | 7704 |
| IB02 | 7696 | 7697 | 7696 |
| IB03 | 7870 | 7870 | 7870 |
| IB04 | 7711 | 7716 | 7711 |
| IB05 | 7830 | 7830 | 7830 |
| IB06 | 7860 | 7862 | 7860 |
| IB07 | 2929 | 2943 | 2943 |
| IB08 | 7055 | 7083 | 7079 |
| IB09 | 0 | 0 | 0 |
| IB10 | 6978 | 6995 | 6983 |
| IB101 | 246 | 248 | 248 |
| IB11 | 7017 | 7031 | 7020 |

La comparación principal es **PM_HS_VAL**: los tres métodos sobre exactamente
las mismas claves estación-fecha. PM_HS mantiene la intersección anterior.
Los tamaños y hashes de cada cohorte constan en el manifest.

PM∩HS=70.861; PM∩VAL=70.860; HS∩VAL=70.878; triple=70.860 pares.
La única observación PM elegible sin VAL es IB11/2014-02-08: falta
HumedadMedia. No se sustituye por RHmin/RHmax ni se excluye de PM.

## Métricas globales

Error=predicción−EtPMon. MAE/RMSE/bias en mm/día; R²=1−SSE/SST.
EtPMon es **referencia calculada SIAR, no ground truth independiente**.
Los pares con EtPMon=0 se conservan; no se utilizan AARE ni Pearson².
Global agrupa pares; no promedia métricas de estaciones.

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| all | own | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | own | HS-Ra | 70912 | 0.712895 | 0.874488 | 0.461359 | 0.746479 |
| all | own | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | PM_HS | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS | HS-Ra | 70861 | 0.712850 | 0.874365 | 0.461603 | 0.746552 |
| all | PM_VAL | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_VAL | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |
| all | HS_VAL | HS-Ra | 70878 | 0.712850 | 0.874422 | 0.461455 | 0.746515 |
| all | HS_VAL | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | PM_HS_VAL | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS_VAL | HS-Ra | 70860 | 0.712843 | 0.874359 | 0.461627 | 0.746559 |
| all | PM_HS_VAL | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |

## Estaciones, comparación principal

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| IB01 | PM_HS_VAL | PM | 7702 | 0.010304 | 0.025524 | 0.000871 | 0.999790 |
| IB01 | PM_HS_VAL | HS-Ra | 7702 | 0.645967 | 0.791372 | 0.443881 | 0.798044 |
| IB01 | PM_HS_VAL | VAL | 7702 | 0.389793 | 0.518879 | 0.128608 | 0.913179 |
| IB02 | PM_HS_VAL | PM | 7693 | 0.010308 | 0.028500 | 0.000470 | 0.999770 |
| IB02 | PM_HS_VAL | HS-Ra | 7693 | 0.617983 | 0.766085 | 0.390247 | 0.833749 |
| IB02 | PM_HS_VAL | VAL | 7693 | 0.383314 | 0.523214 | 0.045858 | 0.922452 |
| IB03 | PM_HS_VAL | PM | 7864 | 0.009345 | 0.026640 | -0.002006 | 0.999781 |
| IB03 | PM_HS_VAL | HS-Ra | 7864 | 0.579051 | 0.713781 | 0.407356 | 0.842765 |
| IB03 | PM_HS_VAL | VAL | 7864 | 0.352798 | 0.474402 | 0.050166 | 0.930543 |
| IB04 | PM_HS_VAL | PM | 7707 | 0.010728 | 0.024717 | -0.003116 | 0.999801 |
| IB04 | PM_HS_VAL | HS-Ra | 7707 | 0.641011 | 0.773600 | 0.552983 | 0.804690 |
| IB04 | PM_HS_VAL | VAL | 7707 | 0.402768 | 0.540039 | 0.268279 | 0.904821 |
| IB05 | PM_HS_VAL | PM | 7826 | 0.010162 | 0.022719 | 0.003529 | 0.999829 |
| IB05 | PM_HS_VAL | HS-Ra | 7826 | 0.751890 | 0.897449 | 0.690638 | 0.733325 |
| IB05 | PM_HS_VAL | VAL | 7826 | 0.467555 | 0.611855 | 0.390281 | 0.876046 |
| IB06 | PM_HS_VAL | PM | 7859 | 0.013375 | 0.026690 | -0.005400 | 0.999761 |
| IB06 | PM_HS_VAL | HS-Ra | 7859 | 0.886545 | 1.042360 | 0.769372 | 0.634944 |
| IB06 | PM_HS_VAL | VAL | 7859 | 0.436822 | 0.569986 | 0.283012 | 0.890843 |
| IB07 | PM_HS_VAL | PM | 2928 | 0.012568 | 0.020656 | 0.001829 | 0.999825 |
| IB07 | PM_HS_VAL | HS-Ra | 2928 | 0.918401 | 1.055986 | 0.889959 | 0.543047 |
| IB07 | PM_HS_VAL | VAL | 2928 | 0.721118 | 0.866246 | 0.704921 | 0.692505 |
| IB08 | PM_HS_VAL | PM | 7050 | 0.013412 | 0.026199 | -0.003996 | 0.999733 |
| IB08 | PM_HS_VAL | HS-Ra | 7050 | 0.714556 | 0.879037 | 0.518884 | 0.699162 |
| IB08 | PM_HS_VAL | VAL | 7050 | 0.470753 | 0.613673 | 0.301443 | 0.853380 |
| IB09 | PM_HS_VAL | PM | 0 | — | — | — | — |
| IB09 | PM_HS_VAL | HS-Ra | 0 | — | — | — | — |
| IB09 | PM_HS_VAL | VAL | 0 | — | — | — | — |
| IB10 | PM_HS_VAL | PM | 6975 | 0.013374 | 0.030263 | -0.002032 | 0.999651 |
| IB10 | PM_HS_VAL | HS-Ra | 6975 | 0.783918 | 0.942653 | 0.715175 | 0.661390 |
| IB10 | PM_HS_VAL | VAL | 6975 | 0.467428 | 0.638071 | 0.379028 | 0.844856 |
| IB101 | PM_HS_VAL | PM | 243 | 0.057131 | 0.087134 | -0.047892 | 0.998038 |
| IB101 | PM_HS_VAL | HS-Ra | 243 | 0.560751 | 0.785298 | -0.342808 | 0.840603 |
| IB101 | PM_HS_VAL | VAL | 243 | 0.553016 | 0.722567 | -0.426124 | 0.865052 |
| IB11 | PM_HS_VAL | PM | 7013 | 0.011372 | 0.040016 | -0.002560 | 0.999407 |
| IB11 | PM_HS_VAL | HS-Ra | 7013 | 0.728119 | 0.948769 | -0.541245 | 0.666643 |
| IB11 | PM_HS_VAL | VAL | 7013 | 0.650529 | 0.825612 | -0.548084 | 0.747570 |

- PM: menor RMSE por estación IB07 (0.020656, n=2928); mayor IB101 (0.087134, n=243). Comparaciones descriptivas; periodos y tamaños difieren entre estaciones.
- HS-Ra: menor RMSE por estación IB03 (0.713781, n=7864); mayor IB07 (1.055986, n=2928). Comparaciones descriptivas; periodos y tamaños difieren entre estaciones.
- VAL: menor RMSE por estación IB03 (0.474402, n=7864); mayor IB07 (0.866246, n=2928). Comparaciones descriptivas; periodos y tamaños difieren entre estaciones.

IB09 tiene n=0 por Quality FAIL a nivel estación: se retienen las 4.423 filas,
incluidas 4.422 oficiales excluidas científicamente. IB101 tiene solo 249
observaciones fuente; el n evaluable es menor y limita generalización.

VAL presenta bias +0.704921 mm/día en IB07 y −0.548084 en IB11; el menor
RMSE por estación es IB03 (0.474402) y el mayor IB07 (0.866246). Los sesgos
de signo contrario no justifican introducir ajustes regionales en este baseline.

## Temporadas, comparación principal

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| DJF | PM_HS_VAL | PM | 17176 | 0.010002 | 0.047190 | 0.004036 | 0.991695 |
| DJF | PM_HS_VAL | HS-Ra | 17176 | 0.524220 | 0.624780 | 0.268904 | -0.455830 |
| DJF | PM_HS_VAL | VAL | 17176 | 0.321979 | 0.422152 | 0.083343 | 0.335348 |
| JJA | PM_HS_VAL | PM | 18058 | 0.011950 | 0.018286 | -0.004524 | 0.999615 |
| JJA | PM_HS_VAL | HS-Ra | 18058 | 0.907931 | 1.093383 | 0.647642 | -0.375204 |
| JJA | PM_HS_VAL | VAL | 18058 | 0.625376 | 0.804280 | 0.294701 | 0.255889 |
| MAM | PM_HS_VAL | PM | 17597 | 0.016344 | 0.023236 | -0.005691 | 0.999592 |
| MAM | PM_HS_VAL | HS-Ra | 17597 | 0.721016 | 0.882316 | 0.466231 | 0.412392 |
| MAM | PM_HS_VAL | VAL | 17597 | 0.449883 | 0.590385 | 0.120277 | 0.736906 |
| SON | PM_HS_VAL | PM | 18029 | 0.007906 | 0.012805 | 0.000091 | 0.999852 |
| SON | PM_HS_VAL | HS-Ra | 18029 | 0.689163 | 0.821970 | 0.454423 | 0.391004 |
| SON | PM_HS_VAL | VAL | 18029 | 0.419812 | 0.547468 | 0.171941 | 0.729841 |

Temporadas meteorológicas del hemisferio norte DJF/MAM/JJA/SON; diciembre
pertenece al año de enero/febrero. Todos los desgloses de estación, año, mes,
temporada, estación-año y temporada-año están en [metrics.json](comparison/metrics.json)
y [CSV de métricas agregadas](metrics-aggregated.csv). No se extrapolan los
resultados de una cohorte a las filas no elegibles ni a ET medida independiente.

VAL alcanza su mayor RMSE en JJA (0.804280 mm/día; bias +0.294701) y el menor
en DJF (0.422152; bias +0.083343). Su R² global 0.877055 incorpora variación
entre temporadas: en JJA es 0.255889 y en DJF 0.335348. HS tiene R² predictivo
negativo en DJF/JJA: SSE supera SST en esas subcohortes; no es Pearson² ni un
fallo de serialización. Las métricas estacionales usan su propia SST.

## Periodos de mayor desviación y extremos

Cinco mayores RMSE estación-año por método en cohorte triple, con n≥30
(regla descriptiva de presentación, no exclusión en métricas):

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| IB11:2020 | PM_HS_VAL | PM | 366 | 0.018966 | 0.126739 | 0.002589 | 0.995143 |
| IB101:2009 | PM_HS_VAL | PM | 217 | 0.046974 | 0.064865 | -0.043148 | 0.998833 |
| IB11:2016 | PM_HS_VAL | PM | 362 | 0.013840 | 0.059858 | -0.001145 | 0.998541 |
| IB11:2012 | PM_HS_VAL | PM | 349 | 0.013917 | 0.058713 | -0.000140 | 0.998715 |
| IB02:2020 | PM_HS_VAL | PM | 366 | 0.011646 | 0.056690 | 0.002538 | 0.999064 |
| IB06:2025 | PM_HS_VAL | HS-Ra | 363 | 1.185443 | 1.370938 | 1.149692 | 0.322245 |
| IB06:2024 | PM_HS_VAL | HS-Ra | 366 | 1.137159 | 1.298751 | 1.099274 | 0.375699 |
| IB10:2023 | PM_HS_VAL | HS-Ra | 363 | 1.129974 | 1.295859 | 1.106961 | 0.215471 |
| IB10:2015 | PM_HS_VAL | HS-Ra | 365 | 1.103422 | 1.256479 | 1.073615 | 0.466881 |
| IB05:2025 | PM_HS_VAL | HS-Ra | 365 | 1.056275 | 1.211205 | 1.051189 | 0.494232 |
| IB11:2007 | PM_HS_VAL | VAL | 359 | 0.859725 | 1.048355 | -0.800114 | 0.594729 |
| IB07:2012 | PM_HS_VAL | VAL | 318 | 0.825633 | 0.998361 | 0.820190 | 0.600196 |
| IB11:2009 | PM_HS_VAL | VAL | 363 | 0.777823 | 0.960117 | -0.686248 | 0.697553 |
| IB11:2008 | PM_HS_VAL | VAL | 364 | 0.776438 | 0.954605 | -0.711755 | 0.656773 |
| IB11:2010 | PM_HS_VAL | VAL | 364 | 0.761433 | 0.945509 | -0.666277 | 0.681147 |

Diez mayores errores absolutos VAL con inputs preservados:

| Estación | Fecha | VAL | EtPMon | VAL−EtPMon | Tmin | Tmax | RHmedia | Viento |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IB05 | 2010-07-15 | 5.601745 | 2.144274 | 3.457471 | 18.71 | 32.42 | 72.106 | 1.134 |
| IB02 | 2009-01-24 | 1.497938 | 4.923190 | -3.425252 | 10.94 | 17.06 | 42.39 | 9.27 |
| IB06 | 2009-01-24 | 1.605917 | 5.029029 | -3.423111 | 11.81 | 18.66 | 41.77 | 8.89 |
| IB04 | 2013-07-13 | 5.788719 | 2.399152 | 3.389567 | 20.27 | 32.2 | 63.18 | 0.489 |
| IB10 | 2013-07-13 | 5.916260 | 2.683391 | 3.232869 | 20.27 | 32.2 | 58.93 | 0.52 |
| IB10 | 2019-07-08 | 7.297735 | 4.087784 | 3.209951 | 21.5 | 37.85 | 51.84 | 1.123 |
| IB07 | 2006-06-20 | 6.517318 | 3.316477 | 3.200840 | 22.02 | 35.57 | 62.8 | 0.618 |
| IB06 | 2013-03-30 | 2.748749 | 5.911233 | -3.162484 | 16.43 | 20.09 | 43.5 | 5.833 |
| IB07 | 2009-08-12 | 5.664709 | 2.627130 | 3.037579 | 18.84 | 33.58 | 61.0 | 0.201 |
| IB10 | 2021-08-10 | 6.220845 | 3.205778 | 3.015067 | 19.82 | 35.54 | 54.31 | 0.19 |

Los errores elevados no se corrigen por tuning ni se eliminan. La fórmula
estima radiación a partir de TD y RH media, y no representa directamente
viento/advección ni Rs observada. Los inputs y diagnósticos de cada extremo
permiten revisar estos límites sin atribuir causalidad a una sola variable.

El máximo exceso VAL es IB05/2010-07-15 (+3.457471): Rs estimada
23.452549 frente a Radiacion observada 3.913 MJ/m²/día, con TD=13.71 °C.
En IB02/2009-01-24, VAL subestima −3.425252 y el viento observado es
9.27 m/s, variable ausente de la fórmula. Estos contrastes explicitan límites
de los inputs del modelo; no prueban por sí solos error del sensor o causalidad.
El mayor RMSE VAL estación-año es IB11:2007 (1.048355, bias −0.800114,
n=359), seguido de IB07:2012 (0.998361, bias +0.820190, n=318).

| Método | Mínimo ET0 | Máximo ET0 | Negativas | EtPMon=0 evaluadas |
|---|---:|---:|---:|---:|
| PM_FAO56_V1 | 0.376277 | 8.315602 | 0 | 47 |
| HS_RA_V1 | 0.383345 | 9.487610 | 0 | 47 |
| VALIANTZAS_V1 | 0.245836 | 8.429273 | 0 | 47 |

## CSV combinado y ubicación completa

CSV completo (fuera de Git), **75496 filas**, SHA256:
`66486b4138b7194ded3cd53f4d4d31f08e53ce3051d7c17ee43b30cdc957ed5f`.

```text
C:\Users\Josep\.codex\.chatgpt-projects\g-p-6ac0c6c1d8908191b03f324e51c278f1\scientific-results\classical-20261024T034529Z\comparison\classical-baselines.csv
```

Se verificó por roundtrip cada celda fuente/predicción, claves, nulos, ceros,
elegibilidad, evaluación, flags y provenance. Todas las 23 columnas fuente
se conservan, además de ET0_PM_FAO56_V1 / ET0_HS_RA_V1 / ET0_VALIANTZAS_V1.
Formato UTF-8 BOM, coma, decimal punto, date ISO, ausencias vacías, booleanos
true/false, flags/provenance JSON entrecomillados, sin redondeo. En Excel usar
Datos→Desde texto/CSV con coma; en Power Query, tipos decimales con locale
inglés. pandas: `read_csv(path, encoding="utf-8-sig")`.

Los JSON masivos están en `valiantzas/predictions.json` y
`comparison/predictions.json`, relativos al mismo directorio de ejecución que
contiene el CSV. Se preservan fuera de Git; este directorio versiona métricas,
manifiestos, resumen e índice con hashes. No se han movido/copiado los JSON
grandes anteriores. No se certifica backup independiente nuevo en esta misión.

Para reproducir: checkout limpio del pipeline SHA indicado, mismo dataset
de bytes verificadas y comandos `calculate-valiantzas` seguido de
`evaluate-classical`, importando el PM/HS oficial previo y VAL recién calculado.
Usar directorios nuevos; no sobrescribir. Los comandos completos y paths de
inputs están en los manifests y [README de interfaz](../../README.md).

## Limitaciones retenidas

- EtPMon calculated SIAR reference; not independent ground truth
- UNITS_PROVENANCE_PARTIAL
- Wind 2 m / SIAR convention; station_period_provenance not exhaustively verified
- All 4423 IB09 source rows retained but excluded by unchanged Quality FAIL policy
- IB101 short period: 249 source observations
- Empirical radiation from RHmean and temperature; extrapolation/advection/cold/saturation limitations
- Reference evapotranspiration assumes well-watered reference conditions; not actual crop ET
- Unit-explicit composition of Marti Eqs.1b+3, not literal Eq.4 or certified author-code reproduction
- No imputation, local calibration, learned coefficients, AHC, ML or forecast
- Large artifacts preserved locally outside Git; no independent backup certification in this mission

Raw, Quality, catálogo, legacy y Dataset v1: **sin modificaciones**.
No se calibran constantes ni se cambia elegibilidad para mejorar métricas.
