# Auditoría científica del legado ET₀

> **Evidencia histórica preservada — actualización 23/10/2026.** Este informe describe la investigación offline y sus decisiones fechadas; no es el estado operativo actual del dataset. Las esperas de POST/ejecución pertenecen a esa etapa. Dataset v1 y la ejecución oficial PM/HS ya están PASS, con [resultados posteriores](../results/pm-vpd-20261023T194553Z/README.md). No se reescriben cifras forenses ni resultados legacy.

Fecha: 2026-10-18. Base de investigación: main `6f1dd95c415730ec29843055d3a32a7ac5a636ad`.

## Alcance y estado

Se partió de `docs_1.2_Tratamiento de datos y cálculo de estimaciones.md`, se contrastó con `legacy/et0/`, después con `legacy/data/siar-derived/` y finalmente con `docs_1.3_Análisis errores.md`. No se ejecutaron los scripts interactivos legacy ni se modificaron observaciones, raw, legacy o metadatos. Las comprobaciones numéricas leen exclusivamente los CSV históricos derivados. No constituyen evaluación nueva del Dataset Observational v1 ni resultados finales.

**Actualización de decisión 00, 2026-10-19:** forense **RESOLVED_FOR_V1**. Se acepta Radiacion en MJ m⁻² día⁻¹, VelViento en m/s y uso directo como viento a 2 m según convención SIAR. `UNITS_PROVENANCE_PARTIAL` y `station_period_provenance_not_exhaustively_verified` son limitaciones registradas, no STOP. Las conclusiones preliminares BLOCKED_UNITS/WIND_HEIGHT_UNVERIFIED de la fase anterior quedan superadas por esta decisión; no cambia ninguna cifra histórica. La falta de fichas instrumentales exhaustivas no impide v1. Una evidencia positiva de altura distinta de 2 m sí detiene el cálculo PM afectado hasta resolverla explícitamente. Ver [supuestos v1](assumptions.md). La preparación integrada solo se declarará READY_FOR_DATASET_V1 tras verificar pruebas/CI y completar la revisión; no implica publicación ni ejecución del dataset definitivo.

## Método documentado frente al implementado

### Penman–Monteith diario

`docs_1.2`, apartado 1.2.3, documenta el cociente diario FAO-56 con 0.408, 900, 273, 0.34, G=0, T media en °C, u₂ en m/s a 2 m, radiación neta en MJ m⁻² día⁻¹ y presiones en kPa. No documenta por completo todas las decisiones intermedias.

`legacy/et0/PM2.py` implementa ese cociente en `penman_monteith_ET0`, con las siguientes decisiones adicionales:

- e°(T)=0.611 exp(17.27T/(T+237.3)); eₛ es la media de e°(Tmax) y e°(Tmin).
- eₐ=[e°(Tmax) RHmin + e°(Tmin) RHmax]/200. Usa RH extrema, no `HumedadMedia`.
- Δ=4098 e°(Tmedia)/(Tmedia+237.3)².
- P=101.3[(293−0.0065z)/293]^5.26; λ=2.501−0.002361T; γ=0.00163P/λ. Esta γ variable con T es una decisión histórica que debe distinguirse de adoptar γ=0.000665P en una especificación FAO-56 nueva.
- Ra usa Gsc=0.0820 MJ m⁻² min⁻¹, año de 365 días y el día del año civil; Rso=(0.75+2·10⁻⁵z)Ra.
- Rns=0.77Rs. Rnl usa σ=4.903·10⁻⁹, temperaturas +273.15 K, emisividad 0.34−0.14√eₐ y factor de nubosidad `clip(1.35Rs/Rso−0.35, 0.2, 1.0)`.
- Si eₐ≤0 lo reemplaza por 0.0001; si Rs≤0 o Rso≤0 fuerza factor de nubosidad=1. Si el denominador es cero devuelve ET₀=0. Estas sustituciones silenciosas no equivalen a validación física.
- `VelViento` pasa directamente como u₂, sin conversión de altura o unidad ni evidencia almacenada sobre ellas. `Radiacion` pasa directamente como Rs suponiendo MJ m⁻² día⁻¹.
- No trunca ET₀ negativa. `diff` significa EtPMon−estimación; una diferencia negativa indica sobreestimación respecto a SIAR.

La variante `ET0_sun` sustituye Rs por Rso tanto en onda corta como en onda larga. Es un escenario de cielo claro, no una observación ni una sustitución inocua para Rs ausente. No forma parte de los dos baselines prioritarios nuevos.

### Hargreaves–Samani y otras variantes

| Método | Documento 1.2 | Implementación histórica efectiva | Unidades y variables asumidas |
|---|---|---|---|
| Hargreaves–Samani con Ra | 0.0023·0.408·Ra·(T+17.8)·√(Tmax−Tmin) | Misma expresión en `et0alternativos.py` (`ET0_harg`) y `et0alternativo2.py` (`ET0_harg_ra`) | Ra en MJ m⁻² día⁻¹; conversión 0.408 a equivalente de evaporación; T en °C; salida mm/día |
| Hargreaves con Rs medida | No figura en la fórmula 1.2.4; sí aparece en resultados 1.3 | `et0alternativo2.py`: 0.0135·0.408·Rs·(T+17.8), `ET0_harg_rs` | Rs medida en MJ m⁻² día⁻¹; no es la misma variante que HS con Ra |
| Variante etiquetada Valiantzas | 0.408·0.0135·0.338·Ra·(Tmax−Tmin)^0.3·(1−RH/100)^0.2·(T+17.8) | Misma expresión en ambas versiones; RH=`HumedadMedia` | RH en %, Ra en MJ m⁻² día⁻¹; sin tratamiento explícito de dominios inválidos |

El nombre `variants_et0.py` referido por docs_1.2 no existe en `legacy/et0/`; las versiones preservadas se llaman `et0alternativos.py` y `et0alternativo2.py`. El script de errores documentado como `calculate_errors_et0.py` tampoco está con ese nombre: existen `calculate_errors.py` y `calculate_errors2.py`.

Los comentarios de Valiantzas reconocen una interpretación no verificada de una errata del TFG (`1.001` frente a `1 −`) y de la columna RH. La coincidencia entre texto y programa no resuelve esa incertidumbre metodológica. Valiantzas queda como contexto histórico, fuera de los baselines prioritarios nuevos; no se presenta como método validado aquí.

## NaN, dominios y filtros

`filtra.py` elimina filas con un NaN en cualquiera de trece variables: temperaturas, humedades, viento y sus direcciones/extremos, radiación, precipitación y **EtPMon**. Por tanto, puede eliminar una fila utilizable para HS por ausencia de una variable que HS no necesita. No exige `TempSuelo1/2` ni `PePMon`.

Luego aplica ±3 desviaciones estándar muestrales (`pandas.std`, ddof=1), **secuencialmente**, en ese mismo orden; vuelve a calcular media y desviación con la muestra ya reducida por las columnas anteriores. No es una máscara simultánea construida sobre una muestra fija. Incluye precipitación, direcciones angulares y el propio valor objetivo EtPMon. No justifica que una cola meteorológica real sea un error de sensor. Las estadísticas se estiman con la serie completa, sin partición temporal: usarlas como preprocesamiento de validación futura incorporaría información del periodo de evaluación.

`PM2.py` aplica otro `dropna` sobre TempMax, TempMin, TempMedia, VelViento, Radiacion, HumedadMax, HumedadMin, EtPMon y Fecha. Las excepciones dejan al menos `ET0_calc` en NaN y se registran; el CSV puede conservar filas con fallo parcial. No usa `isfinite` ni una validación completa de RH, viento o coherencia Tmin≤Tmax. Los scripts de variantes propagan resultados inválidos; diferencias térmicas negativas o RH>100 pueden producir NaN en potencias fraccionarias. El filtro de métricas excluye NaN, pero no infinito.

Una capa nueva debe conservar la observación y asignar motivos de invalidez por método; no heredar la eliminación masiva por ±3σ ni filtrar el objetivo para hacer parecer mejores las estimaciones.

## Metadatos y trazabilidad

`estaciones_baleares.csv` raíz y `legacy/data/siar-derived/metadata/estaciones_baleares.csv` contienen **12 códigos**, incluido IB101. Tres tienen `Fecha_Baja`: IB07, IB09 e IB101. La afirmación de docs_1.2 «11 activas IB01–IB11» es una descripción histórica incorrecta para el catálogo preservado. No se debe construir el inventario con `range(1,12)` porque omite IB101.

PM2 consulta por `Codigo`, usa latitud DMS compacta con hemisferio y altitud; no usa longitud ni UTM para Ra. Para los códigos del catálogo su conversión de latitud representa correctamente cadenas como `390036000N`→39.01°. La función no es un parser general robusto para todos los formatos de coordenadas.

**Discrepancia de metadatos histórica demostrada en IB03 e IB04.** Los CSV calculados de ambas estaciones contienen los mismos N y Ra para una fecha común. Invertir la ecuación N=(24/π)ωₛ da latitud 39.5° exacta para ambas; el catálogo declara IB03=39.545833…°, 79 m e IB04=39.562222…°, 8 m. Recalcular todas sus filas con PM2, latitud 39.5° y **altitud 100 m**, reproduce ET₀ almacenada con error máximo 1.78·10⁻¹⁵ mm/día. Con sus metadatos actuales los errores máximos son respectivamente 0.004681 y 0.016790 mm/día. Es evidencia numérica de uso histórico de esos valores; no prueba por sí sola qué script ejecutó el autor. El archivo alternativo `legacy/et0/PM` contiene justamente generación de metadatos placeholder lat=39.5°, z=100 m, compatible con la explicación.

IB01, IB02, IB05 e IB06 reproducen PM2 con sus metadatos actuales al mismo nivel de redondeo de máquina. La proximidad a EtPMon no detectó por sí sola el problema de IB03/IB04.

`PM2EXCEL.py` es un generador ilustrativo, no un oráculo independiente: escribe resultados en la columna D mientras varias fórmulas intermedias apuntan a la columna B; además referencia DAYOFYEAR. No sirve como validación externa de las salidas sin reparar y recalcular ese libro.

## Evidencia numérica de los outputs preservados

Comprobación independiente mediante lector CSV y aritmética de biblioteca estándar; se extrajeron únicamente las funciones puras de PM2 para la comparación de ese programa. No se importó el script interactivo ni se alteraron sus rutas. Las fórmulas de variantes y métricas se recalcularon directamente.

| Estación | Depuradas / calculadas / ajustadas | Eliminadas preservadas | Periodo de los derivados ajustados |
|---|---:|---:|---|
| IB01 | 6787 / 6787 / 6787 | 564 | 2004-03-30 a 2024-12-30 |
| IB02 | 6952 / 6952 / 6952 | 509 | 2004-03-31 a 2024-12-31 |
| IB03 | 6994 / 6994 / 6994 | 514 | 2004-04-01 a 2024-12-31 |
| IB04 | 2047 / 2047 / 2047 | 138 | 2004-04-01 a 2010-03-31 |
| IB05 | 6713 / 6713 / 6713 | 481 | 2004-04-05 a 2024-03-31 |
| IB06 | 5312 / 5312 / 5312 | 454 | 2004-04-05 a 2020-04-05 |

Los tres modelos alternativos recalculados reproducen los valores de los CSV ajustados de las seis estaciones con discrepancia máxima ≤1.78·10⁻¹⁵ mm/día. Son 34 805 filas históricas derivadas; el recuento no es el tamaño de una nueva cohorte científica. Los eliminados preservados no bastan para demostrar la reproducción del filtro contra el CSV canónico actual, que ha evolucionado desde esas ejecuciones.

Ejemplo IB01, con exactamente la máscara histórica `EtPMon != 0` y pares no nulos, N=6783:

| Modelo | MSE sin redondear | MAE sin redondear | RRMSE sin redondear |
|---|---:|---:|---:|
| PM2 | 0.0044008095 | 0.0171647807 | 0.0211937193 |
| Hargreaves Ra | 0.5668386910 | 0.6187181029 | 0.2405307671 |
| Hargreaves Rs | 0.2297439100 | 0.4100586703 | 0.1531309219 |
| Variante Valiantzas | 0.2236000482 | 0.3548481955 | 0.1510695166 |

Redondeadas a dos decimales, coinciden con `legacy/data/siar-derived/reports/analisis_errores_ajustado.md` y la tabla de IB01 en docs_1.3. El MSE PM publicado como 0.00 no significa coincidencia exacta. La media de `diff` PM2 en IB05 es 0.01511649 mm/día, no el ejemplo ilustrativo 0.12 de docs_1.2.

## Métricas y AHC: qué miden realmente

`calculate_errors2.py` amplía los modelos de `calculate_errors.py` a Hargreaves Ra/Rs. Ambos:

1. Eliminan observaciones cero de **todas** las métricas, aunque MSE/MAE/RMSE sí están definidas con esos ceros. Por estación se excluyen 4, 5, 5, 1, 4 y 3 filas cero, respectivamente.
2. Llaman `RMSE (adimensional)` a √MSE/media(EtPMon): es **RRMSE**, no RMSE en mm/día. docs_1.3 mejora el nombre, docs_1.2 mantiene la ambigüedad.
3. Llaman R² al cuadrado de Pearson: no implementan 1−SSE/SST. Una multiplicación positiva AHC conserva Pearson² aunque cambie el sesgo y el error. Esto explica los R² iguales antes/después del ajuste en las tablas.
4. Calculan AARE con observaciones no cero. Se necesita informar N elegible y N cero excluido de AARE por separado, sin trasladar esa exclusión al resto.
5. Calculan el resultado «General (Media de Todas Estaciones)» **concatenando días y recalculando** las métricas; no es la media simple de métricas por estación. IB04 aporta muchos menos días que IB02/IB03. Una comparación nueva debe distinguir agregación por observación y por estación.
6. Redondean métricas a dos decimales antes de escribirlas y no conservan un manifiesto, configuración, denominadores o particiones junto a cada resultado.

`calculate_ahc2.py` calcula por estación AHC=media(ET0_calc/ET0_empírico) con toda la serie disponible, después multiplica cada estimación por ese AHC y `calculate_errors2.py` evalúa los mismos días contra EtPMon. El objetivo de calibración es **ET0_calc**, no EtPMon; aun así hay **fuga temporal de evaluación** y ajuste dentro de muestra. `nanmean` no protege frente a denominadores cero o infinito. El «General» de AHC sí es media simple de coeficientes de estación, una agregación distinta a las métricas generales.

En IB01, los coeficientes recalculados son 0.8287608127, 0.9163822784 y 0.9161777438, que reproducen la tabla histórica (0.8288, 0.9164, 0.9162). Esto confirma el cálculo histórico; **no valida generalización**. Las versiones de AHC sin sufijo `2` usan otros nombres y hasta un rango de estaciones distinto al comentario: no deben mezclarse con los outputs de la versión ampliada.

Los baselines prioritarios nuevos deben ser sin calibración. Si se incorpora AHC después, se ajustará exclusivamente en entrenamiento cronológico, se congelará y evaluará en fechas posteriores; también cualquier umbral aprendido debe quedar dentro de entrenamiento.

## Lectura crítica de docs_1.3

Las tablas por estación y sus coeficientes se corresponden con los informes históricos, pero la prosa mezcla MSE y MAE. Por ejemplo, da para Hargreaves Rs un rango «MAE 0.19–0.35» cuando esos son valores MSE; su propia tabla da MAE 0.36–0.51. Para Valiantzas el rango «MAE 0.20–0.31» también corresponde a MSE. No se deben reutilizar esos rangos como conclusiones científicas.

La comparación con el TFG no controla estaciones, periodos, máscara, calidad o calibración. La afirmación de que datos más recientes o una limpieza más estricta causan mejoras no está demostrada por esos resultados agregados. IB04 solo llega a 2010 e IB06 a 2020 en los derivados, por lo que «datos hasta 2024» no describe una cohorte homogénea. Las conclusiones causales y el ranking ajustado se conservan como historia, no como evidencia prospectiva.

## Disposición para la nueva capa

- Reproducido: fórmulas alternativas históricas, PM2 de cuatro estaciones con catálogo actual y de dos con metadatos históricos inferidos, métricas ejemplificadas y AHC de IB01.
- Explicado: discrepancia IB03/IB04 por metadatos; RMSE/RRMSE, Pearson²/R², periodos desiguales, ±3σ secuencial, ceros y fuga AHC. No copiar estas decisiones implícitamente.
- Interpretación de unidades reforzada y cerrada para los snapshots auditados: Radiacion MJ m⁻² día⁻¹, VelViento m/s; no convertir de nuevo. Esta conclusión se apoya en el conjunto de evidencia offline, no solo en un ranking de errores.
- No demostrado: altura real del viento por estación/periodo y certificación de configuración efectiva del origen; atribución causal de mejoras respecto al TFG; validez de la interpretación Valiantzas; reproducción de la selección legacy contra una fuente histórica inmutable.
- Decisión vigente 00 (2026-10-19): RESOLVED_FOR_V1 con viento directo según convención SIAR de 2 m. UNITS_PROVENANCE_PARTIAL y station_period_provenance_not_exhaustively_verified son limitaciones no bloqueantes. Solo evidencia positiva de altura distinta de 2 m detiene PM en el ámbito afectado hasta resolución. La construcción definitiva sigue esperando POST y handoff autorizado; el contraste forense no crea ese dataset ni sus resultados finales.
