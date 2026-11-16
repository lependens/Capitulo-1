# Primera comparación oficial de cuatro baselines clásicos

**HARGREAVES-RS-V1-AND-FOUR-BASELINES = PASS**, 2026-10-24.
HG_RS_V1 se ejecuta sin calibración; PM/HS/VAL se importan de sus resultados
oficiales preservados. Inicio comparación `2026-10-24T04:23:44.061929Z`;
fin `2026-10-24T04:25:23.272084Z`. No se modifica Dataset v1.

## Fórmula y distinción científica

`HG_RS_V1 = 0.0135 × 0.408 × Radiacion × (Tcalc+17.8)`, con
Tcalc=(Tmax+Tmin)/2, Rs medida en MJ/m²/día y salida mm/día. Coeficiente
expandido **0.005508**. Sin Ra, √TD, humedad, viento o calibración local.
El legacy usaba TempMedia SIAR; v1 conserva esa columna y usa Tcalc explícita.
[Contrato, fuentes primarias y unidades](../../docs/hargreaves_rs_v1.md).
No se sustituye Ra por Rs en HS ni se cambian PM, HS o VAL.

## Identidades y controles

- [PR #35](https://github.com/lependens/Capitulo-1/pull/35); pipeline/merge `a0b440fa3ab92e4593be521e4e819c6421327656`, limpio y paquete coincidente.
- Productor upstream de tres métodos `4ac52e25c803cb2301316c5cb1353bf30cdb427a`; conserva a su vez los productores PM/HS y VAL originales.
- Dataset `siar-balears-observational-v1`, SHA256 `0ddb10941710f6d3a945eae94fe6b9eca36ae4c12f28203e2364862920a49bbb`, **sin cambios**.
- source_repo_sha `2c4874c73b6f818793cd2e0af42eeeb4cea9a6e9`: publicación canónica.
- audit_source_repo_sha `18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865`: snapshot auditado.
- POST `SIAR-QA-POST-20261020T193140Z`; paquete SHA256 `a2d1293fcc54254769dbc80c7105b08b4b03b6a14392df0a4e5ef19b4ef51d07`.
- Handoff corregido SHA256 `2b47f9802edbc33d71916701df57237092f563f0fe527fbfe401faf522a724ed`.
- Manifest comparación SHA256 **`78d249f3736ff0aeefbdd4599b4f2e77ded0c76c21bff26f274664e282ebee62`**.
- Manifest cálculo HG-Rs SHA256 `acafeae48a41aec1cba1209874b0005db68c591325ae0a54e0a3d982f0ad3890`.
- Configuración SHA256 `d24e7fe1486ca312ac2af1a8d32dfb2eb8d1bd5ea638ac70998462bd5a1a232c`.
- 75 tests y [CI Python 3.11/3.12/3.13 PASS](https://github.com/lependens/Capitulo-1/actions/runs/40062004170).

Verificados hashes de las 12 fuentes, catálogo, POST/3 informes y handoff;
dataset antes/después; inputs de predicciones; outputs y cada celda del CSV.
Predicciones/elegibilidad PM/HS/VAL, sus métricas y hashes de cohortes son
exactamente los anteriores. Ver [regresión previa](regression-before-merge.json),
[result-summary.json](result-summary.json), [manifest comparación](comparison/manifest.json),
[manifest HG-Rs](hargreaves-rs/manifest.json) e [índice de hashes](artifact-index.json).
Los manifests nativos se copian sin modificar bytes y sus rutas se resuelven
contra el run completo, no contra esta copia compacta Git.

## Dataset, cobertura y cohortes

12 estaciones, 75.496 filas preservadas, 75.495 dentro de periodo,
1 RECOVERED, 2.345 CONFIRMED_NO_DATA, 0 PENDING_RETRY; única fila exterior
IB09/2005-10-23. No se fabrican observaciones para los huecos.

| Método | Elegible/calculado | Evaluables propios | Fallos |
|---|---:|---:|---:|
| HG_RS_V1 | 70978 | 70912 | 0 |
| HS_RA_V1 | 70979 | 70912 | 0 |
| PM_FAO56_V1 | 70896 | 70861 | 0 |
| VALIANTZAS_V1 | 70944 | 70878 | 0 |

HG-Rs admite 82 predicciones adicionales respecto a PM:
no exige viento/humedad. Tiene 1 menos que HS-Ra
porque requiere Rs medida. Los motivos independientes están en resultados;
un dato ausente no se sustituye por radiación estimada ni por cero.

| Estación | PM | HS-Ra | VAL | HG-Rs |
|---|---:|---:|---:|---:|
| IB01 | 7704 | 7704 | 7704 | 7704 |
| IB02 | 7696 | 7697 | 7696 | 7697 |
| IB03 | 7870 | 7870 | 7870 | 7870 |
| IB04 | 7711 | 7716 | 7711 | 7716 |
| IB05 | 7830 | 7830 | 7830 | 7830 |
| IB06 | 7860 | 7862 | 7860 | 7862 |
| IB07 | 2929 | 2943 | 2943 | 2943 |
| IB08 | 7055 | 7083 | 7079 | 7083 |
| IB09 | 0 | 0 | 0 | 0 |
| IB10 | 6978 | 6995 | 6983 | 6994 |
| IB101 | 246 | 248 | 248 | 248 |
| IB11 | 7017 | 7031 | 7020 | 7031 |

Comparación principal **PM_HS_VAL_HGRS**, n=70860; todos sobre las mismas
claves estación-fecha con EtPMon admisible. También se conservan cohortes propias,
los seis pares y cuatro triples, con hashes explícitos y métricas por modelo.
PM_HS_VAL conserva su significado anterior; los comandos de dos/tres modelos
y sus exportaciones se mantienen. No se usa una máscara global destructiva.

## Métricas globales

Error=predicción−EtPMon, MAE/RMSE/bias en mm/día, R²=1−SSE/SST; ceros de
referencia incluidos, R² indefinido como null. EtPMon es **referencia calculada
SIAR**, no ground truth independiente; comparte Rs con PM y HG-Rs. El acuerdo
entre fórmulas no certifica exactitud frente a evapotranspiración medida.

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| all | own | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | own | HS-Ra | 70912 | 0.712895 | 0.874488 | 0.461359 | 0.746479 |
| all | own | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | own | HG-Rs | 70912 | 0.471178 | 0.565634 | 0.275611 | 0.893934 |
| all | PM_HS | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS | HS-Ra | 70861 | 0.712850 | 0.874365 | 0.461603 | 0.746552 |
| all | PM_VAL | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_VAL | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |
| all | HS_VAL | HS-Ra | 70878 | 0.712850 | 0.874422 | 0.461455 | 0.746515 |
| all | HS_VAL | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | PM_HS_VAL | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS_VAL | HS-Ra | 70860 | 0.712843 | 0.874359 | 0.461627 | 0.746559 |
| all | PM_HS_VAL | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |
| all | PM_HGRS | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HGRS | HG-Rs | 70861 | 0.471174 | 0.565638 | 0.275791 | 0.893933 |
| all | HS_HGRS | HS-Ra | 70912 | 0.712895 | 0.874488 | 0.461359 | 0.746479 |
| all | HS_HGRS | HG-Rs | 70912 | 0.471178 | 0.565634 | 0.275611 | 0.893934 |
| all | VAL_HGRS | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | VAL_HGRS | HG-Rs | 70878 | 0.471150 | 0.565619 | 0.275694 | 0.893938 |
| all | PM_HS_HGRS | PM | 70861 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS_HGRS | HS-Ra | 70861 | 0.712850 | 0.874365 | 0.461603 | 0.746552 |
| all | PM_HS_HGRS | HG-Rs | 70861 | 0.471174 | 0.565638 | 0.275791 | 0.893933 |
| all | PM_VAL_HGRS | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_VAL_HGRS | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |
| all | PM_VAL_HGRS | HG-Rs | 70860 | 0.471171 | 0.565637 | 0.275804 | 0.893935 |
| all | HS_VAL_HGRS | HS-Ra | 70878 | 0.712850 | 0.874422 | 0.461455 | 0.746515 |
| all | HS_VAL_HGRS | VAL | 70878 | 0.455990 | 0.609080 | 0.168851 | 0.877013 |
| all | HS_VAL_HGRS | HG-Rs | 70878 | 0.471150 | 0.565619 | 0.275694 | 0.893938 |
| all | PM_HS_VAL_HGRS | PM | 70860 | 0.011540 | 0.028298 | -0.001565 | 0.999735 |
| all | PM_HS_VAL_HGRS | HS-Ra | 70860 | 0.712843 | 0.874359 | 0.461627 | 0.746559 |
| all | PM_HS_VAL_HGRS | VAL | 70860 | 0.455952 | 0.608984 | 0.168919 | 0.877055 |
| all | PM_HS_VAL_HGRS | HG-Rs | 70860 | 0.471171 | 0.565637 | 0.275804 | 0.893935 |

## Estaciones sobre la cohorte de cuatro

Sobre los 70.860 pares comunes, HG-Rs tiene menor RMSE que VAL
(0.565637 frente a 0.608984 mm/día), pero mayor MAE (0.471171 frente a
0.455952) y bias positivo (0.275804 frente a 0.168919). Frente a HS-Ra,
HG-Rs reduce MAE/RMSE en esta cohorte. No se declara un ganador universal:
estas métricas describen distintos aspectos del error frente a SIAR.

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| IB01 | PM_HS_VAL_HGRS | PM | 7702 | 0.010304 | 0.025524 | 0.000871 | 0.999790 |
| IB01 | PM_HS_VAL_HGRS | HS-Ra | 7702 | 0.645967 | 0.791372 | 0.443881 | 0.798044 |
| IB01 | PM_HS_VAL_HGRS | VAL | 7702 | 0.389793 | 0.518879 | 0.128608 | 0.913179 |
| IB01 | PM_HS_VAL_HGRS | HG-Rs | 7702 | 0.414309 | 0.492389 | 0.233563 | 0.921817 |
| IB02 | PM_HS_VAL_HGRS | PM | 7693 | 0.010308 | 0.028500 | 0.000470 | 0.999770 |
| IB02 | PM_HS_VAL_HGRS | HS-Ra | 7693 | 0.617983 | 0.766085 | 0.390247 | 0.833749 |
| IB02 | PM_HS_VAL_HGRS | VAL | 7693 | 0.383314 | 0.523214 | 0.045858 | 0.922452 |
| IB02 | PM_HS_VAL_HGRS | HG-Rs | 7693 | 0.378492 | 0.466333 | 0.113647 | 0.938397 |
| IB03 | PM_HS_VAL_HGRS | PM | 7864 | 0.009345 | 0.026640 | -0.002006 | 0.999781 |
| IB03 | PM_HS_VAL_HGRS | HS-Ra | 7864 | 0.579051 | 0.713781 | 0.407356 | 0.842765 |
| IB03 | PM_HS_VAL_HGRS | VAL | 7864 | 0.352798 | 0.474402 | 0.050166 | 0.930543 |
| IB03 | PM_HS_VAL_HGRS | HG-Rs | 7864 | 0.365303 | 0.449531 | 0.122981 | 0.937635 |
| IB04 | PM_HS_VAL_HGRS | PM | 7707 | 0.010728 | 0.024717 | -0.003116 | 0.999801 |
| IB04 | PM_HS_VAL_HGRS | HS-Ra | 7707 | 0.641011 | 0.773600 | 0.552983 | 0.804690 |
| IB04 | PM_HS_VAL_HGRS | VAL | 7707 | 0.402768 | 0.540039 | 0.268279 | 0.904821 |
| IB04 | PM_HS_VAL_HGRS | HG-Rs | 7707 | 0.462890 | 0.535221 | 0.344153 | 0.906511 |
| IB05 | PM_HS_VAL_HGRS | PM | 7826 | 0.010162 | 0.022719 | 0.003529 | 0.999829 |
| IB05 | PM_HS_VAL_HGRS | HS-Ra | 7826 | 0.751890 | 0.897449 | 0.690638 | 0.733325 |
| IB05 | PM_HS_VAL_HGRS | VAL | 7826 | 0.467555 | 0.611855 | 0.390281 | 0.876046 |
| IB05 | PM_HS_VAL_HGRS | HG-Rs | 7826 | 0.520383 | 0.609127 | 0.430502 | 0.877149 |
| IB06 | PM_HS_VAL_HGRS | PM | 7859 | 0.013375 | 0.026690 | -0.005400 | 0.999761 |
| IB06 | PM_HS_VAL_HGRS | HS-Ra | 7859 | 0.886545 | 1.042360 | 0.769372 | 0.634944 |
| IB06 | PM_HS_VAL_HGRS | VAL | 7859 | 0.436822 | 0.569986 | 0.283012 | 0.890843 |
| IB06 | PM_HS_VAL_HGRS | HG-Rs | 7859 | 0.449633 | 0.534826 | 0.229501 | 0.903894 |
| IB07 | PM_HS_VAL_HGRS | PM | 2928 | 0.012568 | 0.020656 | 0.001829 | 0.999825 |
| IB07 | PM_HS_VAL_HGRS | HS-Ra | 2928 | 0.918401 | 1.055986 | 0.889959 | 0.543047 |
| IB07 | PM_HS_VAL_HGRS | VAL | 2928 | 0.721118 | 0.866246 | 0.704921 | 0.692505 |
| IB07 | PM_HS_VAL_HGRS | HG-Rs | 2928 | 0.588732 | 0.723812 | 0.429917 | 0.785313 |
| IB08 | PM_HS_VAL_HGRS | PM | 7050 | 0.013412 | 0.026199 | -0.003996 | 0.999733 |
| IB08 | PM_HS_VAL_HGRS | HS-Ra | 7050 | 0.714556 | 0.879037 | 0.518884 | 0.699162 |
| IB08 | PM_HS_VAL_HGRS | VAL | 7050 | 0.470753 | 0.613673 | 0.301443 | 0.853380 |
| IB08 | PM_HS_VAL_HGRS | HG-Rs | 7050 | 0.529841 | 0.630204 | 0.342773 | 0.845374 |
| IB09 | PM_HS_VAL_HGRS | PM | 0 | — | — | — | — |
| IB09 | PM_HS_VAL_HGRS | HS-Ra | 0 | — | — | — | — |
| IB09 | PM_HS_VAL_HGRS | VAL | 0 | — | — | — | — |
| IB09 | PM_HS_VAL_HGRS | HG-Rs | 0 | — | — | — | — |
| IB10 | PM_HS_VAL_HGRS | PM | 6975 | 0.013374 | 0.030263 | -0.002032 | 0.999651 |
| IB10 | PM_HS_VAL_HGRS | HS-Ra | 6975 | 0.783918 | 0.942653 | 0.715175 | 0.661390 |
| IB10 | PM_HS_VAL_HGRS | VAL | 6975 | 0.467428 | 0.638071 | 0.379028 | 0.844856 |
| IB10 | PM_HS_VAL_HGRS | HG-Rs | 6975 | 0.543850 | 0.629205 | 0.432119 | 0.849137 |
| IB101 | PM_HS_VAL_HGRS | PM | 243 | 0.057131 | 0.087134 | -0.047892 | 0.998038 |
| IB101 | PM_HS_VAL_HGRS | HS-Ra | 243 | 0.560751 | 0.785298 | -0.342808 | 0.840603 |
| IB101 | PM_HS_VAL_HGRS | VAL | 243 | 0.553016 | 0.722567 | -0.426124 | 0.865052 |
| IB101 | PM_HS_VAL_HGRS | HG-Rs | 243 | 0.490934 | 0.570987 | 0.265011 | 0.915732 |
| IB11 | PM_HS_VAL_HGRS | PM | 7013 | 0.011372 | 0.040016 | -0.002560 | 0.999407 |
| IB11 | PM_HS_VAL_HGRS | HS-Ra | 7013 | 0.728119 | 0.948769 | -0.541245 | 0.666643 |
| IB11 | PM_HS_VAL_HGRS | VAL | 7013 | 0.650529 | 0.825612 | -0.548084 | 0.747570 |
| IB11 | PM_HS_VAL_HGRS | HG-Rs | 7013 | 0.551287 | 0.654902 | 0.188828 | 0.841167 |

- PM: menor RMSE estación IB07 (0.020656, n=2928); mayor IB101 (0.087134, n=243).
- HS-Ra: menor RMSE estación IB03 (0.713781, n=7864); mayor IB07 (1.055986, n=2928).
- VAL: menor RMSE estación IB03 (0.474402, n=7864); mayor IB07 (0.866246, n=2928).
- HG-Rs: menor RMSE estación IB03 (0.449531, n=7864); mayor IB07 (0.723812, n=2928).

Comparaciones descriptivas: las estaciones tienen diferentes periodos/tamaños.

Los bias HG-Rs son positivos en todas las estaciones evaluables; IB10
(+0.432119), IB05 (+0.430502) e IB07 (+0.429917 mm/día) destacan. El mejor
RMSE es IB03 (0.449531) y el peor IB07 (0.723812). IB11 tiene bias +0.188828,
mientras VAL en esa misma cohorte presenta −0.548084: no se corrigen con
coeficientes por estación ni se interpreta la diferencia como prueba causal.
IB09 conserva 4.423 filas, pero Quality FAIL excluye todas del cálculo, incluidas
4.422 oficiales. IB101 tiene 249 observaciones fuente y tamaño evaluable menor.
No se cambian Quality o umbrales para admitir más filas o reducir errores.

## Temporadas sobre la cohorte de cuatro

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| DJF | PM_HS_VAL_HGRS | PM | 17176 | 0.010002 | 0.047190 | 0.004036 | 0.991695 |
| DJF | PM_HS_VAL_HGRS | HS-Ra | 17176 | 0.524220 | 0.624780 | 0.268904 | -0.455830 |
| DJF | PM_HS_VAL_HGRS | VAL | 17176 | 0.321979 | 0.422152 | 0.083343 | 0.335348 |
| DJF | PM_HS_VAL_HGRS | HG-Rs | 17176 | 0.412469 | 0.510303 | 0.119214 | 0.028792 |
| JJA | PM_HS_VAL_HGRS | PM | 18058 | 0.011950 | 0.018286 | -0.004524 | 0.999615 |
| JJA | PM_HS_VAL_HGRS | HS-Ra | 18058 | 0.907931 | 1.093383 | 0.647642 | -0.375204 |
| JJA | PM_HS_VAL_HGRS | VAL | 18058 | 0.625376 | 0.804280 | 0.294701 | 0.255889 |
| JJA | PM_HS_VAL_HGRS | HG-Rs | 18058 | 0.556986 | 0.646764 | 0.464055 | 0.518812 |
| MAM | PM_HS_VAL_HGRS | PM | 17597 | 0.016344 | 0.023236 | -0.005691 | 0.999592 |
| MAM | PM_HS_VAL_HGRS | HS-Ra | 17597 | 0.721016 | 0.882316 | 0.466231 | 0.412392 |
| MAM | PM_HS_VAL_HGRS | VAL | 17597 | 0.449883 | 0.590385 | 0.120277 | 0.736906 |
| MAM | PM_HS_VAL_HGRS | HG-Rs | 17597 | 0.425708 | 0.508057 | 0.241048 | 0.805166 |
| SON | PM_HS_VAL_HGRS | PM | 18029 | 0.007906 | 0.012805 | 0.000091 | 0.999852 |
| SON | PM_HS_VAL_HGRS | HS-Ra | 18029 | 0.689163 | 0.821970 | 0.454423 | 0.391004 |
| SON | PM_HS_VAL_HGRS | VAL | 18029 | 0.419812 | 0.547468 | 0.171941 | 0.729841 |
| SON | PM_HS_VAL_HGRS | HG-Rs | 18029 | 0.485516 | 0.581799 | 0.270354 | 0.694895 |

HG-Rs tiene menor RMSE en MAM (0.508057, bias
0.241048) y mayor en JJA (0.646764,
bias 0.464055). Temporadas boreales DJF/MAM/JJA/SON,
diciembre pertenece al siguiente año de temporada. R² usa SST de cada subcohorte;
puede ser negativo sin constituir un fallo matemático. El valor global incluye
variación estacional y no sustituye estos desgloses.

Global/estación/año/mes/temporada/estación-año/temporada-año completos en
[metrics.json](comparison/metrics.json) y [metrics-aggregated.csv](metrics-aggregated.csv).

## Periodos y extremos

La mejora global de RMSE respecto a VAL no ocurre en todas las temporadas:
HG-Rs en JJA tiene RMSE 0.646764 frente a VAL 0.804280; en DJF HG-Rs
0.510303 frente a VAL 0.422152. Su R² DJF es solo 0.028792, pese al R² global
0.893935; la variación estacional no debe ocultar esta limitación.

Cinco mayores RMSE estación-año por método sobre cohorte de cuatro, n≥30
solo como regla descriptiva de presentación, sin filtrar las métricas:

| Grupo | Cohorte | Método | n | MAE | RMSE | bias/MBE | R² predictivo |
|---|---|---|---:|---:|---:|---:|---:|
| IB11:2020 | PM_HS_VAL_HGRS | PM | 366 | 0.018966 | 0.126739 | 0.002589 | 0.995143 |
| IB101:2009 | PM_HS_VAL_HGRS | PM | 217 | 0.046974 | 0.064865 | -0.043148 | 0.998833 |
| IB11:2016 | PM_HS_VAL_HGRS | PM | 362 | 0.013840 | 0.059858 | -0.001145 | 0.998541 |
| IB11:2012 | PM_HS_VAL_HGRS | PM | 349 | 0.013917 | 0.058713 | -0.000140 | 0.998715 |
| IB02:2020 | PM_HS_VAL_HGRS | PM | 366 | 0.011646 | 0.056690 | 0.002538 | 0.999064 |
| IB06:2025 | PM_HS_VAL_HGRS | HS-Ra | 363 | 1.185443 | 1.370938 | 1.149692 | 0.322245 |
| IB06:2024 | PM_HS_VAL_HGRS | HS-Ra | 366 | 1.137159 | 1.298751 | 1.099274 | 0.375699 |
| IB10:2023 | PM_HS_VAL_HGRS | HS-Ra | 363 | 1.129974 | 1.295859 | 1.106961 | 0.215471 |
| IB10:2015 | PM_HS_VAL_HGRS | HS-Ra | 365 | 1.103422 | 1.256479 | 1.073615 | 0.466881 |
| IB05:2025 | PM_HS_VAL_HGRS | HS-Ra | 365 | 1.056275 | 1.211205 | 1.051189 | 0.494232 |
| IB11:2007 | PM_HS_VAL_HGRS | VAL | 359 | 0.859725 | 1.048355 | -0.800114 | 0.594729 |
| IB07:2012 | PM_HS_VAL_HGRS | VAL | 318 | 0.825633 | 0.998361 | 0.820190 | 0.600196 |
| IB11:2009 | PM_HS_VAL_HGRS | VAL | 363 | 0.777823 | 0.960117 | -0.686248 | 0.697553 |
| IB11:2008 | PM_HS_VAL_HGRS | VAL | 364 | 0.776438 | 0.954605 | -0.711755 | 0.656773 |
| IB11:2010 | PM_HS_VAL_HGRS | VAL | 364 | 0.761433 | 0.945509 | -0.666277 | 0.681147 |
| IB07:2012 | PM_HS_VAL_HGRS | HG-Rs | 318 | 0.745262 | 0.908825 | 0.650069 | 0.668692 |
| IB10:2023 | PM_HS_VAL_HGRS | HG-Rs | 363 | 0.705014 | 0.781106 | 0.690257 | 0.714955 |
| IB07:2011 | PM_HS_VAL_HGRS | HG-Rs | 306 | 0.645107 | 0.779097 | 0.543010 | 0.744642 |
| IB07:2010 | PM_HS_VAL_HGRS | HG-Rs | 363 | 0.639132 | 0.777459 | 0.439866 | 0.770213 |
| IB07:2009 | PM_HS_VAL_HGRS | HG-Rs | 363 | 0.629778 | 0.769042 | 0.448535 | 0.765534 |

Diez mayores errores absolutos HG-Rs con valores originales:

| Estación | Fecha | HG-Rs | EtPMon | Error | Tmin | Tmax | Rs | Viento |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IB06 | 2009-01-24 | 2.014262 | 5.029029 | -3.014767 | 11.81 | 18.66 | 11.07 | 8.89 |
| IB02 | 2009-01-24 | 2.028288 | 4.923190 | -2.894902 | 10.94 | 17.06 | 11.58 | 9.27 |
| IB11 | 2015-11-21 | 0.965573 | 3.799494 | -2.833921 | 13.65 | 21.38 | 4.964 | 5.655 |
| IB06 | 2021-12-10 | 0.769803 | 3.569361 | -2.799558 | 12.26 | 19.87 | 4.127 | 5.695 |
| IB11 | 2017-12-11 | 0.268180 | 2.936210 | -2.668030 | 13.38 | 18.55 | 1.442 | 5.656 |
| IB11 | 2022-11-22 | 0.507213 | 3.160634 | -2.653421 | 14.66 | 18.9 | 2.663 | 3.692 |
| IB02 | 2019-07-07 | 5.630760 | 8.238383 | -2.607623 | 27.1 | 42.15 | 19.5 | 2.414 |
| IB08 | 2013-02-03 | 0.221471 | 2.819804 | -2.598333 | 6.557 | 10.68 | 1.522 | 5.232 |
| IB11 | 2006-12-06 | 0.235291 | 2.827757 | -2.592467 | 10.95 | 17.93 | 1.325 | 6.316 |
| IB02 | 2013-11-05 | 1.000609 | 3.583567 | -2.582958 | 17.79 | 23.98 | 4.696 | 4.552 |

Los extremos se conservan sin tuning. HG-Rs depende de Rs y temperatura, pero
omite términos explícitos de viento/humedad/advección: el contraste de inputs
permite identificar límites sin atribuir causalidad ni declarar sensor erróneo
por una diferencia entre fórmulas. No se recortan ni se ajustan parámetros.

El mayor error absoluto HG-Rs es IB06/2009-01-24: −3.014767 mm/día,
con viento 8.89 m/s. IB02 en la misma fecha presenta −2.894902 y 9.27 m/s.
Son contrastes relevantes para una fórmula sin viento, conservados sin
atribuir causalidad exclusiva. El mayor RMSE HG-Rs estación-año es IB07:2012
(0.908825; bias +0.650069; n=318), seguido por IB10:2023 (0.781106;
bias +0.690257; n=363).

| Método | ET0 mínimo | ET0 máximo | Negativas | EtPMon=0 evaluados |
|---|---:|---:|---:|---:|
| PM_FAO56_V1 | 0.376277 | 8.315602 | 0 | 47 |
| HS_RA_V1 | 0.383345 | 9.487610 | 0 | 47 |
| VALIANTZAS_V1 | 0.245836 | 8.429273 | 0 | 47 |
| HG_RS_V1 | 0.028024 | 7.825254 | 0 | 47 |

## CSV y reproducción

CSV combinado **75496 filas**, roundtrip completo PASS; SHA256
`f55e4ed12691b84dbca4e090c49da538d5e7fad4f2ef371e1307a649cfb1af44`:

```text
C:\Users\Josep\.codex\.chatgpt-projects\g-p-6ac0c6c1d8908191b03f324e51c278f1\scientific-results\four-classical-20261024T042250Z\comparison\classical-baselines.csv
```

Preserva 23 columnas fuente, cuatro predicciones, eligible/evaluated/reasons
PM/HS/VAL/HGRS, estados, flags, provenance, dataset_id/SHA, ambos SHA
audit/canonical, POST ID, pipeline SHA y timestamp. UTF-8 BOM, coma, decimal
punto, date ISO, ausencias vacías, booleanos true/false, JSON entrecomillado,
sin redondeo. Excel: Datos→Desde texto/CSV con coma; Power Query: tipos
decimales con locale inglés; pandas: `read_csv(path,encoding="utf-8-sig")`.

Reproducción con checkout limpio del pipeline indicado y mismo dataset:

```powershell
python -m siar_science calculate-hargreaves-rs --dataset DATASET_V1 --output NUEVO_RUN_HGRS --pipeline-sha a0b440fa3ab92e4593be521e4e819c6421327656
python -m siar_science evaluate-four --dataset DATASET_V1 --predictions TRES_OFICIALES/predictions.json --predictions NUEVO_RUN_HGRS/predictions.json --output NUEVA_COMPARACION --pipeline-sha a0b440fa3ab92e4593be521e4e819c6421327656
```

Los paths de entrada exactos y hashes constan en manifests/índice. Crear
destinos nuevos. `run-four-baselines` permite una reproducción calculando los
cuatro, sin reemplazar los outputs anteriores. Los JSON masivos están en
`hargreaves-rs/predictions.json` y `comparison/predictions.json`, relativos al
directorio completo de esta ejecución. Permanecen fuera de Git junto al CSV;
se versionan métricas/manifests/resumen/índice. No se mueven ni sobrescriben
ejecuciones anteriores y no se certifica backup independiente nuevo.

## Limitaciones retenidas

- EtPMon calculated SIAR reference; no independent ground truth; shares measured Rs with PM and HG-Rs
- UNITS_PROVENANCE_PARTIAL
- Wind height 2 m / SIAR convention; station-period provenance not exhaustively verified
- All 4423 IB09 source rows retained but excluded by unchanged Quality FAIL policy
- IB101 short period: 249 source observations
- HG-Rs depends on measured Rs quality; empirical equation lacks explicit wind/humidity/advection terms
- Well-watered reference ET, not actual crop ET; cold signed outputs retained without clipping
- No local calibration, imputation, AHC, ML or forecast
- Large artifacts local outside Git; no new independent backup certification

Raw, Quality, catálogo, legacy y Dataset v1: **sin modificaciones**.
