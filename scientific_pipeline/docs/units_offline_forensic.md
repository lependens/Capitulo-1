# UNITS-OFFLINE-FORENSIC — conclusión consolidada

**Interpretación respaldada: Radiacion en MJ m⁻² día⁻¹ y VelViento en m/s.**
No aplicar ×0.0864 ni /3.6 a estos snapshots. La conclusión combina el contrato
documentado en docs_1.1/docs_1.2, el uso directo legacy, la búsqueda de conversiones
y el contraste numérico amplio; no se selecciona una unidad por mínimo error.
**Cierre posterior de Dirección 00 (2026-10-19): RESOLVED_FOR_V1.**
Se acepta el uso directo de VelViento como u2 según convención SIAR de 2 m.
UNITS_PROVENANCE_PARTIAL y station_period_provenance no verificada de forma
exhaustiva se conservan como limitaciones no bloqueantes. Los diagnósticos STOP
preliminares de esta investigación quedan superados; solo evidencia positiva
contradictoria exige revisión. Este informe conserva el experimento, sin generar
Dataset v1 ni sus resultados finales. El cierre de integración se registra en
mission_status.md, separado de la fecha del experimento forense.

## Alcance y reproducibilidad

Solo lectura offline de los snapshots locales; ninguna consulta SIAR/API, ni
descarga, interpolación, imputación, calibración o cambio de datos. Se calculan
diagnósticos de hipótesis autorizados por 00, no Dataset Observational v1 ni sus
resultados finales. SIAR EtPMon es referencia calculada, no medición independiente.

Herramienta reproducible: `scientific_pipeline/forensics/units_offline.py`,
Python de biblioteca estándar. Desde la raíz del checkout:

```console
python scientific_pipeline/forensics/units_offline.py --repo . --output NUEVO_DIRECTORIO
python scientific_pipeline/forensics/report_units.py NUEVO_DIRECTORIO
python -m unittest discover -s scientific_pipeline/forensics -p "test_*.py" -v
```

El segundo comando escribe este informe en stdout. La herramienta exige un
directorio nuevo y rechaza destinos dentro de datos, legacy y adquisición.
El generador de informe está acotado por hash a los scores revisados de v2:
acepta otra ruta con esos mismos resultados y exige revisar la narrativa si
cambia la evidencia. No es un clasificador automático de unidades.
`results/units-offline-v2/manifest.json` registra SHA del checkout, estado dirty,
hash del script ejecutado, Python, catálogo, funciones PM2, cada CSV y cohortes.
El hash del script, y no solo el HEAD anterior a incorporarlo, identifica la
versión ejecutada. El manifest incluye hashes de las dos salidas numéricas.

**Bytes y finales de línea:** el checkout Windows usado tenía
`core.autocrlf=true`; los CSV y PM2 se leyeron con CRLF aunque Git almacena LF.
Los hashes del manifest describen esos bytes locales realmente leídos, no los
bytes originales del servidor ni una certificación POST. Un checkout LF puede
tener otro hash de archivo aun conservando los mismos valores. Los tests
verifican las fuentes reconstruyendo ambas representaciones en memoria, sin
reescribirlas; no admiten cambios de contenido. La herramienta registra en cada
ejecución su hash real, sin normalizarlo. `.gitattributes` solo dentro de la
carpeta forense preserva los bytes de las salidas y fija LF para sus scripts;
no cambia atributos, contenido o esquema de los CSV de origen.

Los tres corpus se mantienen separados, porque se solapan y no son muestras
independientes. No sumar sus n como si fueran observaciones distintas.

| Corpus | Archivos | Filas | Pares elegibles |
|---|---|---|---|
| canonical_snapshot | 12 | 75495 | 75236 |
| legacy_calculated | 6 | 34805 | 34805 |
| legacy_backup | 4 | 24505 | 24478 |

Canónicos: 12 estaciones, 2004-03-30 a 2025-12-31. Se conservan 50 referencias EtPMon=0 en las métricas. Exclusiones diagnósticas: {'missing_or_unparseable': 255, 'temperature_order': 4}. Los periodos son los presentes en archivos; no se redefine el periodo oficial ni cobertura Quality.
El inventario físico de 75495 filas no sustituye los 75.494 días observados comunicados en PRE: este experimento no aplica el calendario oficial y no compara denominadores de cobertura.

La máscara común exige fecha parseable, inputs finitos, Tmin≤TempMedia≤Tmax,
RH extrema ordenada en [0,100], viento/radiación no negativos y temperaturas
en el dominio (-100,100) °C. No usa ±3σ, precipitación, suelo ni dirección.
No se eliminan ni reescriben filas: se cuentan exclusiones del experimento.
Duplicados o identidad de estación incoherente entre las filas elegibles
detienen la herramienta; las filas excluidas no pasan ese control. Esto no
sustituye los controles de integridad completos del Quality Audit.

## Hipótesis y fórmulas controladas

| Hipótesis | Rs de cálculo | u de cálculo |
|---|---|---|
| Rad1_Wind1 | Radiacion directa | VelViento directo |
| Rad1_Wind2 | Radiacion directa | VelViento / 3.6 |
| Rad2_Wind1 | Radiacion × 0.0864 | VelViento directo |
| Rad2_Wind2 | Radiacion × 0.0864 | VelViento / 3.6 |

0.0864=86400/10⁶ solo convierte W/m² como potencia **media durante 24 h** a
MJ m⁻² día⁻¹. No sería una conversión válida de una lectura instantánea sin
integración temporal. /3.6 convierte km/h a m/s. Son hipótesis explícitas del
experimento, no transformaciones aplicadas a los CSV.

`legacy_pm2` extrae por AST únicamente nueve funciones matemáticas de PM2.py,
sin importar ni ejecutar su I/O. Mantiene gamma variable, e° con 0.611,
273.15 K y su floor de nubosidad. `fao56_sensitivity` cambia a gamma=0.000665P,
e° con 0.6108, 273.16 K y cap superior Rs/Rso=1 sin floor inferior. Mantiene
TempMedia SIAR en ambos para aislar las hipótesis; no define con ello la futura
convención de temperatura del baseline. Se usa G=0 y RH extrema, sin calibración.
Dentro de cada kernel solo cambian los dos factores; todos los demás inputs,
metadatos, selección de días y supuesta altura de 2 m son idénticos.

Se usa el catálogo raíz para las métricas. Los placeholders IB03/IB04
(39.5°,100 m) se usan exclusivamente para comprobar los outputs históricos y
en una columna separada del ejemplo, nunca para mejorar el ajuste del contraste.

## Métricas globales por hipótesis

Bias=predicción−EtPMon; MAE y RMSE en mm/día, calculados sobre pares agrupados.
No son medias de métricas de estaciones. n es idéntico entre hipótesis.

| Corpus | Kernel | Hipótesis | n | bias | MAE | RMSE |
|---|---|---|---|---|---|---|
| canonical_snapshot | fao56_sensitivity | Rad1_Wind1 | 75236 | -0.001300 | 0.003599 | 0.022715 |
| canonical_snapshot | fao56_sensitivity | Rad1_Wind2 | 75236 | -0.437701 | 0.438844 | 0.528303 |
| canonical_snapshot | fao56_sensitivity | Rad2_Wind1 | 75236 | -1.403445 | 1.420902 | 1.863759 |
| canonical_snapshot | fao56_sensitivity | Rad2_Wind2 | 75236 | -1.964387 | 1.965391 | 2.440937 |
| canonical_snapshot | legacy_pm2 | Rad1_Wind1 | 75236 | -0.023222 | 0.025729 | 0.087799 |
| canonical_snapshot | legacy_pm2 | Rad1_Wind2 | 75236 | -0.462255 | 0.463333 | 0.547631 |
| canonical_snapshot | legacy_pm2 | Rad2_Wind1 | 75236 | -2.059799 | 2.060198 | 2.419298 |
| canonical_snapshot | legacy_pm2 | Rad2_Wind2 | 75236 | -2.685245 | 2.685283 | 3.069445 |
| legacy_backup | fao56_sensitivity | Rad1_Wind1 | 24478 | 0.000393 | 0.003065 | 0.022477 |
| legacy_backup | fao56_sensitivity | Rad1_Wind2 | 24478 | -0.543347 | 0.544469 | 0.628084 |
| legacy_backup | fao56_sensitivity | Rad2_Wind1 | 24478 | -1.381167 | 1.399118 | 1.831663 |
| legacy_backup | fao56_sensitivity | Rad2_Wind2 | 24478 | -2.072517 | 2.073344 | 2.554151 |
| legacy_backup | legacy_pm2 | Rad1_Wind1 | 24478 | -0.017664 | 0.021914 | 0.078197 |
| legacy_backup | legacy_pm2 | Rad1_Wind2 | 24478 | -0.564134 | 0.565133 | 0.641808 |
| legacy_backup | legacy_pm2 | Rad2_Wind1 | 24478 | -2.022892 | 2.023359 | 2.380624 |
| legacy_backup | legacy_pm2 | Rad2_Wind2 | 24478 | -2.790756 | 2.790797 | 3.188206 |
| legacy_calculated | fao56_sensitivity | Rad1_Wind1 | 34805 | -0.000589 | 0.003287 | 0.021023 |
| legacy_calculated | fao56_sensitivity | Rad1_Wind2 | 34805 | -0.489971 | 0.490786 | 0.568314 |
| legacy_calculated | fao56_sensitivity | Rad2_Wind1 | 34805 | -1.438785 | 1.456989 | 1.882802 |
| legacy_calculated | fao56_sensitivity | Rad2_Wind2 | 34805 | -2.064623 | 2.065501 | 2.545090 |
| legacy_calculated | legacy_pm2 | Rad1_Wind1 | 34805 | -0.014416 | 0.017778 | 0.068517 |
| legacy_calculated | legacy_pm2 | Rad1_Wind2 | 34805 | -0.505339 | 0.506155 | 0.578679 |
| legacy_calculated | legacy_pm2 | Rad2_Wind1 | 34805 | -2.094678 | 2.095079 | 2.445184 |
| legacy_calculated | legacy_pm2 | Rad2_Wind2 | 34805 | -2.786962 | 2.786991 | 3.181813 |

## Estaciones canónicas: PM2, mismo catálogo y mismos pares

| Estación | Hipótesis | n | bias | MAE | RMSE |
|---|---|---|---|---|---|
| IB01 | Rad1_Wind1 | 7702 | -0.018192 | 0.022687 | 0.081708 |
| IB01 | Rad1_Wind2 | 7702 | -0.504918 | 0.505485 | 0.577329 |
| IB01 | Rad2_Wind1 | 7702 | -2.087311 | 2.087665 | 2.433328 |
| IB01 | Rad2_Wind2 | 7702 | -2.780825 | 2.780851 | 3.162286 |
| IB02 | Rad1_Wind1 | 7693 | -0.015985 | 0.022124 | 0.076865 |
| IB02 | Rad1_Wind2 | 7693 | -0.630258 | 0.630955 | 0.709037 |
| IB02 | Rad2_Wind1 | 7693 | -2.005778 | 2.006329 | 2.388236 |
| IB02 | Rad2_Wind2 | 7693 | -2.850086 | 2.850146 | 3.270391 |
| IB03 | Rad1_Wind1 | 7864 | -0.016898 | 0.019803 | 0.073551 |
| IB03 | Rad1_Wind2 | 7864 | -0.570254 | 0.571992 | 0.644814 |
| IB03 | Rad2_Wind1 | 7864 | -1.977714 | 1.978245 | 2.334016 |
| IB03 | Rad2_Wind2 | 7864 | -2.765040 | 2.765082 | 3.161828 |
| IB04 | Rad1_Wind1 | 7707 | -0.021807 | 0.022900 | 0.081171 |
| IB04 | Rad1_Wind2 | 7707 | -0.446515 | 0.447448 | 0.519402 |
| IB04 | Rad2_Wind1 | 7707 | -2.096359 | 2.096650 | 2.437704 |
| IB04 | Rad2_Wind2 | 7707 | -2.695605 | 2.695626 | 3.086445 |
| IB05 | Rad1_Wind1 | 7826 | -0.019993 | 0.022002 | 0.081036 |
| IB05 | Rad1_Wind2 | 7826 | -0.398114 | 0.398780 | 0.462805 |
| IB05 | Rad2_Wind1 | 7826 | -2.122241 | 2.122482 | 2.482935 |
| IB05 | Rad2_Wind2 | 7826 | -2.644693 | 2.644694 | 3.045558 |
| IB06 | Rad1_Wind1 | 7859 | -0.026962 | 0.027855 | 0.091316 |
| IB06 | Rad1_Wind2 | 7859 | -0.494741 | 0.495568 | 0.571446 |
| IB06 | Rad2_Wind1 | 7859 | -2.033542 | 2.033823 | 2.398146 |
| IB06 | Rad2_Wind2 | 7859 | -2.679411 | 2.679428 | 3.065430 |
| IB07 | Rad1_Wind1 | 2927 | -0.063920 | 0.064488 | 0.147152 |
| IB07 | Rad1_Wind2 | 2927 | -0.276425 | 0.276755 | 0.340709 |
| IB07 | Rad2_Wind1 | 2927 | -2.169807 | 2.169807 | 2.532174 |
| IB07 | Rad2_Wind2 | 2927 | -2.452443 | 2.452443 | 2.814998 |
| IB08 | Rad1_Wind1 | 7050 | -0.032314 | 0.033344 | 0.101762 |
| IB08 | Rad1_Wind2 | 7050 | -0.351748 | 0.352408 | 0.425609 |
| IB08 | Rad2_Wind1 | 7050 | -2.054965 | 2.055168 | 2.420988 |
| IB08 | Rad2_Wind2 | 7050 | -2.514951 | 2.514951 | 2.880590 |
| IB09 | Rad1_Wind1 | 4380 | -0.024487 | 0.027755 | 0.093506 |
| IB09 | Rad1_Wind2 | 4380 | -0.303106 | 0.303704 | 0.359080 |
| IB09 | Rad2_Wind1 | 4380 | -2.099602 | 2.099704 | 2.465924 |
| IB09 | Rad2_Wind2 | 4380 | -2.498293 | 2.498293 | 2.867638 |
| IB10 | Rad1_Wind1 | 6971 | -0.022374 | 0.024117 | 0.087101 |
| IB10 | Rad1_Wind2 | 6971 | -0.369475 | 0.370551 | 0.442945 |
| IB10 | Rad2_Wind1 | 6971 | -2.075840 | 2.076313 | 2.429390 |
| IB10 | Rad2_Wind2 | 6971 | -2.559607 | 2.559628 | 2.926905 |
| IB101 | Rad1_Wind1 | 243 | -0.063333 | 0.070091 | 0.110336 |
| IB101 | Rad1_Wind2 | 243 | -0.498442 | 0.504552 | 0.587908 |
| IB101 | Rad2_Wind1 | 243 | -2.190291 | 2.191814 | 2.586312 |
| IB101 | Rad2_Wind2 | 243 | -2.941593 | 2.941593 | 3.381151 |
| IB11 | Rad1_Wind1 | 7014 | -0.017279 | 0.019526 | 0.077685 |
| IB11 | Rad1_Wind2 | 7014 | -0.541473 | 0.544560 | 0.643428 |
| IB11 | Rad2_Wind1 | 7014 | -2.014081 | 2.015080 | 2.367082 |
| IB11 | Rad2_Wind2 | 7014 | -2.851479 | 2.851679 | 3.178259 |

## Robustez temporal y meteorológica

| Kernel | Agrupación | Grupos con menor RMSE Rad1_Wind1 | Grupos evaluados |
|---|---|---|---|
| legacy_pm2 | station | 12 | 12 |
| legacy_pm2 | year | 22 | 22 |
| legacy_pm2 | station_year | 220 | 220 |
| legacy_pm2 | season | 4 | 4 |
| legacy_pm2 | raw_radiation_bin | 3 | 3 |
| legacy_pm2 | raw_wind_bin | 3 | 3 |
| legacy_pm2 | temperature_bin | 3 | 3 |
| legacy_pm2 | humidity_bin | 2 | 2 |
| fao56_sensitivity | station | 12 | 12 |
| fao56_sensitivity | year | 22 | 22 |
| fao56_sensitivity | station_year | 219 | 220 |
| fao56_sensitivity | season | 4 | 4 |
| fao56_sensitivity | raw_radiation_bin | 3 | 3 |
| fao56_sensitivity | raw_wind_bin | 3 | 3 |
| fao56_sensitivity | temperature_bin | 3 | 3 |
| fao56_sensitivity | humidity_bin | 2 | 2 |

**Excepción local conservada:** IB101:2008, n=26. En fao56_sensitivity, Rad2_Wind1 obtiene RMSE 0.155049 frente a 0.192114 con Rad1_Wind1. En PM2: 0.588405 frente a 0.228986. Bajo la alternativa la radiación media sería 0.551099 MJ/m²/día y Rs/Rso medio 0.053761; Rnl es negativa en 26 días del kernel sin floor. La onda larga negativa compensa parte de la disminución de Rs. La sensibilidad del ranking al kernel y la cohorte corta impiden interpretar este mínimo local como un cambio de unidad. No se oculta la discrepancia ni se calibra contra ella. Es un ejemplo de por qué no elegir unidades solo por menor error.


Ese recuento es un contraste de consistencia entre hipótesis, no una prueba
estadística ni un criterio aislado de selección. Años parciales y periodos de
estaciones cerradas no se ponderan como años completos. El CSV contiene las
métricas completas por año, estación-año, estación climática y condiciones.
DJF/MAM/JJA/SON son estaciones meteorológicas del hemisferio norte, agrupadas
aquí sobre todos los años. Bins físicos previos: temperatura media ≤10,10–25,>25 °C;
RHmin≤40 o >40 %. Bins de escala bruta previos: radiación 0–5,5–20,>20 y viento
0–1,1–4,>4. Estos dos últimos conservan la escala fuente bajo TODAS las hipótesis,
para evitar cambiar la cohorte al convertir unidades; no se presentan como
clases físicas independientes de la hipótesis.

## Escala y diagnósticos físicos, canónicos

| Hipótesis | Rs media MJ/m²/d | u media m/s | Rs/Rso medio | ET0 media mm/d | ET0 mín | ET0 máx | Rs/Rso bajo | Rs>Rso | ET0<0 |
|---|---|---|---|---|---|---|---|---|---|
| Rad1_Wind1 | 15.815368 | 1.288598 | 0.719305 | 2.911215 | -0.151051 | 8.391714 | 3139 | 526 | 42 |
| Rad1_Wind2 | 15.815368 | 0.357944 | 0.719305 | 2.472182 | -0.168274 | 6.177864 | 3139 | 526 | 125 |
| Rad2_Wind1 | 1.366448 | 1.288598 | 0.062148 | 0.874638 | -0.260867 | 5.192841 | 75236 | 0 | 3249 |
| Rad2_Wind2 | 1.366448 | 0.357944 | 0.062148 | 0.249192 | -0.264087 | 2.003834 | 75236 | 0 | 16755 |

Rs/Rso bajo significa <0.35/1.35: la expresión de nubosidad FAO sin floor cambia
de signo. Con Rad2 esta situación se generaliza; el kernel FAO puede producir
Rnl negativa y por ello amortiguar artificialmente parte de la caída de ET0.
El kernel legacy impide esa compensación mediante su floor; ambos rechazan
numéricamente la escala alternativa frente a EtPMon. No interpretar esta
comparación como validación del floor legacy para producción. Rs>Rso y ET0<0
son diagnósticos, no exclusiones adicionales ni correcciones silenciosas.

La reducción del viento /3.6 no crea por sí sola una velocidad imposible. Su
rechazo interpretativo combina la declaración explícita m/s en adquisición,
la ausencia de conversión y la pérdida sistemática de coherencia entre estaciones
y años. La escala de radiación, su relación con Rso y los cálculos diarios
documentados proporcionan evidencia adicional distinta del ranking de errores.

## Ejemplo IB04 y regresión histórica

| Kernel | Hipótesis | EtPMon | PM catálogo 39.562222°,8m | PM placeholder 39.5°,100m |
|---|---|---|---|---|
| legacy_pm2 | Rad1_Wind1 | 3.205964 | 3.205125 | 3.214339 |
| legacy_pm2 | Rad1_Wind2 | 3.205964 | 2.689196 | 2.703091 |
| legacy_pm2 | Rad2_Wind1 | 3.205964 | 1.278449 | 1.272037 |
| legacy_pm2 | Rad2_Wind2 | 3.205964 | 0.416402 | 0.413916 |
| fao56_sensitivity | Rad1_Wind1 | 3.205964 | 3.200715 | 3.209908 |
| fao56_sensitivity | Rad1_Wind2 | 3.205964 | 2.681923 | 2.695805 |
| fao56_sensitivity | Rad2_Wind1 | 3.205964 | 1.857255 | 1.854560 |
| fao56_sensitivity | Rad2_Wind2 | 3.205964 | 1.095983 | 1.097265 |

IB04 2004-04-01: Radiacion=19.999, VelViento=1.908. El resultado 3.214339 corresponde al placeholder histórico, no al catálogo real. La regresión de los seis outputs PM2 preservados reproduce 34805 filas con discrepancia máxima 1.78e-15 mm/día. Eso verifica la extracción de funciones y explica el ejemplo; no certifica instrumentos.


## Interpretación, procedencia y altura: tres conclusiones distintas

1. **Unit interpretation: sólidamente respaldada para los snapshots auditados.**
   Radiacion es energía diaria MJ m⁻² día⁻¹ y VelViento está en m/s. Docs_1.1
   documenta la salida de adquisición, docs_1.2 la metodología y los programas
   preservados consumen esos campos directamente. La búsqueda exhaustiva no
   encuentra una conversión histórica observacional y el experimento amplio
   confirma coherencia de escala en múltiples estaciones, años y condiciones.
   No convertir de nuevo. No se extiende esta conclusión a futuras fuentes o
   lotes cuya escala cambie; seguir validando distribuciones y procedencia.
2. **Provenance certification: pendiente.** El manifest identifica los bytes
   efectivamente auditados y la evidencia de interpretación, pero no inventa
   historial de configuración del proveedor. El manual oficial previamente
   consultado admite configuraciones alternativas y no identifica la efectiva
   del lote. Esa limitación administrativa no obliga a mantener como desconocida
   una interpretación física fuertemente corroborada. Debe conservarse y
   completarse en el handoff POST; no certifica ni invalida Quality desde aquí.
3. **Wind sensor height: convención SIAR 2 m aceptada para v1.** Ninguna de las cuatro
   hipótesis varía altura; todas suponen 2 m al introducir u en la ecuación.
   Coincidir con EtPMon no demuestra altura real ni historia instrumental. Los
   catálogos locales no contienen ese campo y comentarios/plantillas no son
   fichas de sensor. La procedencia instrumental no exhaustiva se registra como
   limitación, sin investigación estación×año ni STOP global. Solo evidencia
   positiva de altura distinta exigiría una revisión versionada y resolución de
   la transformación afectada. No convertir altura por conjetura.

La documentación oficial ya consultada es compatible con MJ/m² y m/s en la
interfaz diaria, pero su manual API no garantiza por sí solo esa configuración
para el archivo histórico. Se combina como contexto, no como una certificación
que no proporciona. No se consultó API ni web adicional durante esta misión.

**Resultado forense, cerrado por 00:** RESOLVED_FOR_V1; no aplicar las dos
conversiones alternativas. Usar VelViento como u2 directo con
wind_height_assumption = 2 m / SIAR convention y station_period_provenance =
not exhaustively verified. UNITS_PROVENANCE_PARTIAL no bloquea la preparación.
La integración, CI y autorización de Dataset v1 se controlan por separado en
mission_status.md. No se solicita consulta SIAR adicional ni más investigación
histórica para discriminar estas unidades.

Evidencia y búsqueda detallada: [units_offline_evidence.md](units_offline_evidence.md).
Contrato actualizado: [contracts.md](contracts.md). Resultados agregados,
manifest y ejemplo: `../forensics/results/units-offline-v2/`.
