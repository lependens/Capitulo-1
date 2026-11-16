# AHC v1 — protocolo bloqueado antes del análisis de ajuste

**Cierre vigente:** AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION PASS, PR #38/#39; 86 tests/CI PASS, pipeline 38503dbe8422982deeb3680bbc3e82fd950da4e1. El protocolo preregistrado siguiente se conserva; no se reescribe después de TEST. [Resultados](../results/ahc-20261024T172400Z/README.md).

## Referencia y pregunta científica

La misión estudia cuánto mejora una **escala multiplicativa por estación** la
emulación diaria de PM_FAO56_V1 por HS_RA_V1, HG_RS_V1 y VALIANTZAS_V1.
No se ajusta PM. PM y EtPMon son referencias calculadas, no ET medida.
La referencia principal es PM_FAO56_V1, versión científica explícita;
EtPMon se reserva a sensibilidad sobre las mismas parejas TRAIN y evaluación
secundaria. No se elige referencia por el error TEST.

Se fija **OLS por el origen**: a=Σ(x·PM)/Σ(x²), ET_ajustada=a·x. Es el
minimizador analítico de SSE de TRAIN dentro de esta familia, sin intercepto.
No es una selección retrospectiva de parámetros para ganar contra TEST.
El multiplicador a es adimensional. Para HS el coeficiente físico equivalente
es 0.0023·a; no confundirlo con el multiplicador. En HG-Rs/VAL se informa a
como escala empírica del método completo, conservando sus constantes y unidades.

## Legado y estimandos distintos

| Estimador | Fórmula | Pregunta |
|---|---|---|
| MEAN_RATIO legacy | media(PM/x), x>0 | Corrección proporcional diaria promedio |
| RATIO_SUMS | ΣPM/Σx | Igualar el total/promedio del periodo de ajuste |
| OLS0, principal v1 | Σx·PM/Σx² | Reducir error cuadrático diario en TRAIN |

No son intercambiables. Sobre x>0 son medias de PM/x con pesos 1, x y x²,
respectivamente (derivación algebraica). Con x=0, OLS aporta cero a sus sumas
y conserva el residual irreducible; RATIO_SUMS conserva y en su numerador,
por lo que esa equivalencia de cocientes no se extiende a tales filas.
No se atribuye la media de cocientes
a la regresión descrita por Martí. Los otros dos estimandos se publican como
sensibilidad TRAIN, **sin competir por TEST ni cambiar la pregunta primaria**.

`legacy/et0/calculate_ahc*.py` divide `ET0_calc` (PM legacy) por el empírico,
no por EtPMon. Usa nanmean sin guardia de infinitos/cero y, en la segunda
versión, dropna común a todos los modelos. La tabla General promedia estaciones
sin ponderar n. Coeficientes y evaluación usan todo el histórico: resultados
descriptivos dentro de muestra. Se conservan scripts/tablas, sin reutilizar
sus filtros, rutas, ecuaciones modificadas, métricas o atribuciones causales.

## Soporte bibliográfico verificado

- [Martí et al. (2017), DOI 10.18178/ijiet.2017.7.1.842](https://www.ijiet.org/vol7/842-JR180.pdf), §II.C, pp.62–63, Ec.6: PM=a·HG por regresión por origen; AHC físico=a·0.0023. No se heredan Pearson² ni filtrado global de esa propuesta docente.
- [Maestre-Valero et al. (2013), DOI 10.5424/sjar/2013114-3869](https://sjar.revistas.csic.es/index.php/sjar/article/view/3869), pp.1141–1142: calibración anual/mensual minimizando MBE y regionalización con estaciones excluidas. El cociente de sumas es la solución algebraica para una única escala; su agregación mensual no es nuestro experimento diario.
- [Moratiel et al. (2020), DOI 10.5194/nhess-20-859-2020](https://nhess.copernicus.org/articles/20/859/2020/nhess-20-859-2020.html), §2.3.4: mínimos cuadrados y análisis anual/estacional. Un título que dice temporal no demuestra por sí mismo TRAIN→TEST.
- [Martí et al. (2015), DOI 10.1016/j.jhydrol.2015.07.054](https://www.sciencedirect.com/science/article/abs/pii/S0022169415005740): parametrización/regionalización y variación anual. El resumen/manuscrito indexado no prueba la fórmula exacta de su ajuste local; no se presenta como evidencia de mean-ratio.
- [Shiri et al. (2015), DOI 10.1016/j.compag.2015.07.010](https://www.sciencedirect.com/science/article/abs/pii/S0168169915002057): necesidad de patrones de test no utilizados en calibración y sensibilidad a particiones. Nuestra partición es global, cronológica, sin random split.
- [Martí et al. (2015), DOI 10.1016/j.agwat.2014.10.028](https://www.sciencedirect.com/science/article/abs/pii/S0378377414003485): modelar referencias calculadas no equivale a validar ET medida. No se llama ground truth a PM o SIAR.
- [Gavilán et al. (2006), Regional calibration, DOI 10.1016/j.agwat.2005.05.001](https://www.sciencedirect.com/science/article/pii/S0378377405002088): regionalización en ambiente semiárido y validación en otras estaciones. No se transfieren automáticamente sus coeficientes a Baleares ni a IB101.

## Protocolo y partición temporal

[Configuración congelada](../experiments/ahc-v1.json), bloqueada el
2026-10-24T05:29:34Z a partir de bibliografía y disponibilidad exclusivamente:

| Partición | Fechas globales |
|---|---|
| TRAIN | 2004-01-01–2015-12-31 |
| VALIDATION | 2016-01-01–2020-12-31 |
| TEST | 2021-01-01–2025-12-31 |

Todas las estaciones comparten cortes. IB07 cerró en 2012, sin validación/test;
IB09 mantiene Quality FAIL; IB101 tiene solo 249 observaciones 2008–2009:
sin ajuste validado robusto ni extensión ficticia. Las nueve activas tienen
presencia en las tres particiones. No se mueven fechas para obtener más datos.

Los baselines de todo el histórico ya se examinaron: TEST es **retrospective_locked**,
fuera del ajuste y selección AHC desde este bloqueo, no un test nunca visto.
Una confirmación genuinamente prospectiva exigirá otro dataset posterior.
La estabilidad para decidir se estudia solo en TRAIN, bloques 2004–2009 y
2010–2015 y coeficientes anuales. Los bloques 2004–2010, 2011–2015, 2016–2020,
2021–2025 se describen **después** del TEST, sin alimentar selección/refit.
Esto evita que la exploración de estabilidad use el futuro como tuning.

`protocol_sha256` identifica el JSON canónico ordenado de la configuración;
`protocol_file_sha256` conserva además el hash de bytes del archivo leído.
La política LF del directorio experiments estabiliza esos bytes en checkout.
Dataset, fuentes, POST y artefactos se validan por sus hashes de bytes sin
normalización. El commit inicial del protocolo precede a todos los ajustes;
los mínimos por año/grupo y guardias de cobertura se completan antes de la
primera calibración, con datos sintéticos y disponibilidad únicamente.

## Dominio, soporte y STOP predeclarados

Parejas específicas por método: modelo y PM calculados, finitos/no negativos,
Quality PASS/WARN y observación oficial. OLS incluye x=0; se exige Σx²>0.
MEAN_RATIO excluye solo x=0 del cociente, con recuento; no epsilon ni clipping.
La evaluación conserva ceros de referencia y usa el mismo n para original/ajustado.

Ajuste anual robusto: n≥900 y ≥3 años con ≥300 parejas y ≥45 en cada temporada.
Se usa toda pareja TRAIN admisible cuando se cumple soporte; los años parciales
no se borran. Grupo estacional: n≥180 y ≥3 años con ≥45 parejas cada uno;
mensual: n≥90 y ≥3 años con ≥20 parejas cada uno.
Sin coeficiente: null con motivo, sin fallback regional ni imputación.

Los umbrales siguientes son criterios pragmáticos preregistrados de revisión,
no constantes universales bibliográficas. Antes de VALIDATION se detiene por
CV muestral de coeficientes entre ≥2 bloques robustos >20 % o max/min>1.5.
Se comparan PM vs EtPMon en las mismas parejas TRAIN: diferencia relativa
del coeficiente >5 % o diferencia media de predicciones ajustadas >0.10 mm/día
en estaciones de soporte robusto exige STOP. Las series cortas se describen
sin declararlas estables. Se repite el gate de bloques completos después de TEST,
antes de emitir climatología operativa; nunca se reajusta para resolver un STOP.

## Complejidad anual, estacional y mensual

Se investiga una alternativa solo si el rango de coeficientes estacionales
TRAIN robustos / coeficiente anual ≥15 % en algún grupo del método.
Los candidatos se ajustan enteramente en TRAIN y se congelan. VALIDATION
selecciona por RMSE agrupado sobre las **mismas claves**: mejora relativa ≥2 %
respecto a anual y sin degradar >5 % ninguna estación con n≥300. Mensual solo
se elige si también mejora ≥2 % frente a estacional cuando esta es admisible.
En empates o falta de soporte se conserva anual. No refit TRAIN+VALIDATION.

TEST publica el benchmark anual de siete métodos predeclarado (PM y tres
pares original/AHC_VALIDATED) y, separadamente, como máximo una granularidad
seleccionada por método. No se comparan todos los candidatos en TEST para
volver a elegir. Ninguna predicción TRAIN ajustada se etiqueta fuera de muestra.

## Climatología y cierre

AHC_CLIMATOLOGY se calcula tras TEST, anual con todo el histórico hasta 2025.
Su coeficiente no sustituye el validado ni demuestra rendimiento out-of-sample.
IB101 puede recibir una cifra **descriptiva de serie corta**, sin certificación
robusta ni elegibilidad operativa; IB09 carece de parejas aceptadas. No se
recalculan raw/Quality/Dataset o los cuatro baselines. Contratos ML y vista web
se documentan sin entrenar modelos ni construir web/ANN.
