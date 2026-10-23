# Supuestos científicos v1 y decisión de alcance

Decisión de Dirección 00: 2026-10-19, Europe/Madrid. Estado forense: **RESOLVED_FOR_V1**. Este documento fija convenciones reproducibles; no certifica exhaustivamente la instrumentación histórica ni publica resultados del Dataset Observational v1.

## Unidades y viento

- `Radiacion`: energía solar diaria en MJ m⁻² día⁻¹; lectura directa. No multiplicar por 0.0864.
- `VelViento`: velocidad media en m/s; lectura directa. No dividir por 3.6 ni sustituirla por `VelVientoMax`.
- Para v1 se acepta la **convención SIAR de medición a 2 m**. La altura desconocida en una ficha local no excluye una observación por sí sola.
- Registrar `UNITS_PROVENANCE_PARTIAL` y `station_period_provenance_not_exhaustively_verified` como limitaciones no bloqueantes en dataset y manifiestos. El campo de altura aceptada no significa verificación instrumental exhaustiva.
- Si aparece evidencia positiva de altura distinta de 2 m, detener PM en la estación/periodo afectado hasta resolver identidad, vigencia y tratamiento. No aplicar una conversión ni atribuir altura a partir del error contra EtPMon. HS puede seguir cuando sus propias entradas sean válidas.

Estas decisiones combinan documentación del proyecto, código y contraste offline. La procedencia parcial no se vuelve a tratar como ambigüedad física de escala. El [informe forense](units_offline_forensic.md) conserva sus resultados y su diagnóstico preliminar fechado; la presente decisión posterior establece el estado operativo vigente.

## Penman–Monteith diario FAO-56

Referencia: [FAO-56, meteorología, capítulo 3](https://www.fao.org/4/X0490E/x0490e07.htm) y [ET₀, capítulo 4](https://www.fao.org/4/X0490E/x0490e08.htm). Son referencias metodológicas ya seleccionadas en esta misión.

| Componente | Convención v1 |
|---|---|
| Superficie | Pasto de referencia; albedo fijo 0.23 |
| Temperatura media de cálculo | (Tmax+Tmin)/2; conservar `TempMedia` original separadamente |
| G | 0 MJ m⁻² día⁻¹ para cálculo diario |
| e°(T) | 0.6108 exp[17.27T/(T+237.3)] kPa |
| eₛ | [e°(Tmax)+e°(Tmin)]/2 |
| eₐ | [e°(Tmin)RHmax+e°(Tmax)RHmin]/200 |
| VPD | eₛ−eₐ; presiones finitas y no negativas; exceso eₐ>eₛ solo tolerado hasta 1e−12 kPa absoluto, sin tolerancia relativa |
| Δ | 4098 e°(Tcalc)/(Tcalc+237.3)² |
| P | 101.3[(293−0.0065z)/293]^5.26 kPa |
| γ | 0.000665P kPa °C⁻¹ |
| Ra | Ecuación diaria de geometría solar, Gsc=0.0820, día del año y denominador 365 |
| Rso | (0.75+2·10⁻⁵z)Ra |
| Rns | 0.77Rs |
| Rnl | σ=4.903·10⁻⁹; temperaturas Kelvin=T+273.16; media de cuartas potencias de extremos; emisividad 0.34−0.14√eₐ |
| Nubosidad | 1.35·min(Rs/Rso,1)−0.35; **sin floor inferior por defecto** |
| Cociente PM | [0.408ΔRn+γ·900/(Tcalc+273)·u₂·VPD]/[Δ+γ(1+0.34u₂)] |
| Salida | mm/día; conservar valores negativos con flag, sin truncar a cero |

Rso≤0 es un dominio no calculable para Rnl; no se sustituye por un valor positivo. Rs/Rso>1 se marca aunque el término de nubosidad tenga límite superior 1. Rs/Rso<0.35/1.35 produce factor negativo; conservar el diagnóstico, marcar la condición y no imponer el floor histórico. También marcar Rnl o ET₀ negativos; una flag informativa no equivale automáticamente a exclusión. Toda política alternativa de floor requiere nombre/configuración propios, nunca selección posterior por menor error SIAR.

Corrección numérica autorizada por Dirección 00 el 2026-10-23:
`vapour_pressure_deficit` usa `math.isclose(ea, es, rel_tol=0.0, abs_tol=1e-12)`
únicamente cuando `ea > es`. En ese caso devuelve VPD=0 si el exceso está
dentro de tolerancia; fuera de ella conserva `ValueError`. También rechaza
presiones negativas incluso dentro de esa escala. Un VPD positivo, por pequeño
que sea, se devuelve exactamente como `es-ea`. No redondea ni modifica es, ea,
humedad, temperaturas o EtPMon; no aplica `max(es-ea, 0)` general.

La tolerancia cubre los excesos IEEE-754 de 1–2 ULP (2.22e−16 a 4.44e−16 kPa)
en las tres filas IB10 con RHmin=RHmax=100 %. El margen de 1e−12 kPa está nueve
órdenes por debajo de una diferencia de 0.001 kPa, y la tolerancia relativa cero
evita que crezca con la escala de las presiones. No es un umbral meteorológico
ni una calibración; los excesos superiores siguen siendo errores. El valor y
la política quedan en `PM_FAO56_V1_PARAMETERS.vpd_roundoff` y, por tanto, en
los parámetros y hash de configuración de cada manifest de ejecución.

Las diferencias respecto a PM2 legacy son intencionales: Tcalc de extremos en lugar de `TempMedia`; 0.6108 en lugar de 0.611; γ estándar en lugar de λ variable; Kelvin 273.16 en lugar de 273.15; política de nubosidad sin límite inferior 0.2 del factor; errores explícitos en lugar de sustituciones silenciosas; metadatos del catálogo en lugar de placeholders IB03/IB04. No es objetivo reproducir exactamente EtPMon o el legado. EtPMon es una referencia de cálculo SIAR, no una observación independiente de lisímetro.

## Hargreaves–Samani fijo

ET₀=**0.0023·0.408**·Ra·(Tcalc+17.8)·√(Tmax−Tmin), con Ra en MJ m⁻² día⁻¹ y Tcalc=(Tmax+Tmin)/2. Sin calibración AHC, sustitución de Ra por Rs ni ajuste estacional. Latitud y fecha son necesarias para Ra; radiación medida, humedad y viento no lo son. Valores negativos se conservan y marcan. Diferencias térmicas negativas o entradas no finitas son motivos explícitos de no elegibilidad.

## Selección, métricas y futuro aprendizaje

Conservar cada fila y sus motivos por método; no aplicar ±3σ global ni exigir variables auxiliares ajenas al cálculo. EtPMon ausente no impide una predicción. Su ausencia sí impide formar un par de evaluación. Los ceros de referencia permanecen en MAE, RMSE, MBE y R².

Error=predicción−EtPMon. MAE, RMSE y MBE/bias están en mm/día; R²=1−SSE/SST, no Pearson². n<2 o SST=0 deja R² indefinido (`null`); cohorte vacía informa n=0 y métricas `null`. Las referencias negativas se conservan con flag y se excluyen de la evaluación v1, cuya referencia elegible es EtPMon finita y no negativa, con Quality PASS/WARN y observación dentro de periodo oficial; no se recortan ni se cambian silenciosamente.

Publicar tanto cohorte propia por método como intersección pareada PM/HS, con n y hashes. Las métricas globales reúnen pares, sin promediar RMSE/R² por estación. Desgloses: estación, año, mes y temporada DJF/MAM/JJA/SON; diciembre pertenece al año de enero/febrero en DJF. La convención y los denominadores quedan en el manifiesto.

Los baselines v1 no aprenden parámetros. Cualquier AHC, filtro aprendido, escalado o ML futuro requiere cortes globales cronológicos TRAIN<VALIDATION<TEST, fijados tras POST antes de consultar errores. Ningún día cambia de partición según estación; no se redistribuyen datos de estaciones cerradas para rellenar TEST. TRAIN estima, VALIDATION selecciona y TEST se abre una vez con configuración congelada. Sin partición aleatoria ni ajuste de umbrales sobre toda la serie.
