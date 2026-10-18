# Contratos científicos preparatorios v1

Estado: **BLOCKED_UNITS**. Especificación preparatoria, no Dataset Observational v1 publicado. No certifica
Quality POST, no cambia sus reglas y no genera resultados sobre observaciones
reales. Las pruebas deben utilizar ejemplos sintéticos.

## Autoridad y handoff de Quality

`docs/siar/quality.md` conserva autoridad sobre los informes y estados de Quality.
Los tres informes esperados son `siar_station_audit.csv`,
`siar_missing_days.csv` y `siar_dataset_manifest.csv`. Actualmente el README de
`data_quality/siar/` indica que no hay artefactos versionados y POST está pendiente.
No sustituirlos por PRE ni fabricar filas para completar un contrato.

La construcción futura requiere autorización explícita en un handoff de Dirección
00 después de Quality closeout. El adaptador científico consume, sin modificar,
los informes entregados. Un handoff debe incluir:

* `authorized: true`, identidad de autorización y fecha UTC verificables;
* `quality_post: {id, sha256}` para identificar el paquete de evidencias;
* rutas y SHA-256 de los tres informes originales y del catálogo contrastado;
* `source_repo_sha` exacto y `source_data_hashes` (ruta relativa → SHA-256);
* `official_periods` por estación, límites inclusivos y cutoff explícito de activas;
* `statuses` por clave estación/fecha, derivados del informe de huecos POST;
* evidencias de unidades y metadatos científicos, con alcance temporal explícito.

`authorized` es una condición de entrada, no una forma de autorizarse a sí mismo.
La implementación del adaptador no confiere autoridad a un JSON arbitrario. Debe
comprobar hashes contra los bytes entregados y rechazar contradicciones entre
informes, catálogo, fechas, estados y archivos. Las discrepancias se devuelven a
00; nunca se corrigen silenciosamente ni se cambia el Quality Audit.

## Dataset Observational v1

Representación normalizada preparada: JSON UTF-8; `schema_version` vale
`observational-v1`. No se fija una extensión de almacenamiento definitiva.

| Campo | Semántica |
|---|---|
| `dataset_id` | Identificador inmutable de una publicación; nunca reutilizar para bytes distintos |
| `source_repo_sha` | Commit exacto de adquisición auditado |
| `source_data_hashes` | Ruta relativa canónica → hash SHA-256 de bytes originales |
| `quality_post` | Identidad y hash del paquete POST; conservar referencias individuales |
| `units` | Unidades normalizadas esperadas; su declaración no demuestra unidades de adquisición |
| `stations` | Código → latitud decimal, altitud en m, altura de viento en m; evidencia y vigencia |
| `rows` | Filas únicas por `station_code,date`, orden estable estación/fecha |

Las 12 estaciones previstas son IB01–IB11 e IB101. El conjunto final y sus
periodos deben verificarse contra POST, sin inferirlos de la cobertura PRE.

Cada fila conserva `station_code`, `date` ISO YYYY-MM-DD, `values`,
`observation_status`, `quality_status`, `quality_flags`, `provenance` y
`source_data_sha`. La procedencia permite localizar archivo y fila originales
sin confundir el SHA del archivo con un hash de registro. Los valores originales
se conservan; el cálculo añade resultados separados y nunca sobrescribe EtPMon.

`values` preserva las 23 columnas y su significado canónico:

```text
Fecha, TempMedia, TempMax, HorMinTempMax, TempMin, HorMinTempMin,
HumedadMedia, HumedadMax, HorMinHumMax, HumedadMin, HorMinHumMin,
VelViento, DirViento, VelVientoMax, HorMinVelMax, DirVientoVelMax,
Radiacion, Precipitacion, TempSuelo1, TempSuelo2, EtPMon, PePMon, Estacion
```

`date` y `station_code` normalizados no reinterpretan `Fecha` y `Estacion`:
se valida su correspondencia mediante catálogo y parsing explícito. Los HHMM
son horas de extremos, no timestamps para geometría solar diaria. No usar
`VelVientoMax` como sustituto de viento medio. `EtPMon` es ET₀ calculada y
proporcionada por SIAR, referencia de comparación, no medición independiente
de lisímetro ni verdad de terreno.

Una fila observada se conserva aunque tenga variables ausentes. Ausencias
numéricas se representan mediante `null`, nunca cero ni NaN/Infinity JSON.
No agregar registros meteorológicos inventados para cubrir el calendario.
Duplicados de estación/fecha son error de integridad; no elegir arbitrariamente
primera fila ni promediarla. El raw y el ledger siguen conservando la evidencia.

### Calendario y estados

El calendario/ledger complementa las filas observadas, sin simular observaciones.
`OBSERVED` identifica una observación existente; `RECOVERED` una observación
recuperada con evidencia POST. `CONFIRMED_NO_DATA` exige consulta válida con
cero registros; `PENDING_RETRY` conserva incertidumbre y nunca se convierte en
ausencia confirmada. `MISSING` significa que no hay fila en el snapshot y no
autoriza inventar un resultado de consulta. `OUTSIDE_OFFICIAL_PERIOD` identifica
fechas fuera del periodo y no aumenta el denominador de huecos.

Si se representa una fecha sin observación en `rows`, `values` debe ser vacío o
nulo, las dos elegibilidades son falsas y no se considera una fila observacional
para cobertura ni métricas. Es preferible un ledger separado. Una observación
raw fuera de periodo se conserva y marca, sin borrarla ni convertirla en hueco.
No permitir una fila presente etiquetada `CONFIRMED_NO_DATA` o `PENDING_RETRY`
sin resolver la contradicción del handoff.

## Unidades y evidencia pendiente

Unidades normalizadas: temperatura `degC`; viento `m/s`; radiación
`MJ/m2/day`; humedad `%`; altitud `m`; latitud `degrees`; presiones `kPa`;
ET₀ `mm/day`. Conversiones deben declarar unidad de origen, fórmula y alcance.
Latitud se convierte a radianes solo dentro de geometría solar.

Evidencia primaria consultada el 18-10-2026:

* El [Manual API SIAR v2.2, mayo 2025, pp. 8 y 14](https://servicio.mapa.gob.es/siarweb/documentos/manuales/ManualTecnico_WEB_API_SIAR.pdf)
  documenta DMS y altitud en metros. Describe radiación diaria con unidad
  dependiente de configuración y no fija inequívocamente la de viento diario.
* La [consulta web diaria SIAR](https://servicio.mapa.gob.es/siarweb/consultaDatos/consultaDatos)
  muestra m/s, radiación en MJ/m² y ET₀ en mm. Eso caracteriza esa interfaz,
  pero no prueba la configuración de la API usada para el snapshot canónico.
* La [descripción oficial de estaciones](https://servicio.mapa.gob.es/siarweb/masInformacion)
  describe estructura de 2 m para estaciones propiedad del MAPA. No demuestra
  por sí sola altura ni historial del sensor de cada estación autonómica.

`docs_1.2` asume m/s y MJ/m² y viento a 2 m. SIAR Sync solicita diarios con
`DatosCalculados=true`, pero no fija ni registra unidad en esos parámetros.
Por tanto **no está demostrada la unidad crítica del snapshot por el mero
nombre de columna o por plausibilidad de valores**. Exigir evidencia del
proveedor/configuración de adquisición vinculada a estaciones, fechas y hashes;
registrar `units_provenance` en el paquete futuro. No usar una correlación con
EtPMon para inferir unidades o decidir una conversión.

El catálogo local proporciona latitud DMS y altitud para las 12 estaciones.
Contrastar identidad, coordenadas, altitud y vigencia antes del handoff; preservar
el snapshot y su hash. La discrepancia histórica `CATALOG-TZ` no se resuelve
desde esta capa: no usar XUTM/YUTM/Huso para derivar latitud. La altura de viento
debe estar documentada; si difiere de 2 m, normalizar explícitamente mediante
FAO-56, sin cambiar el raw.

**STOP científico para uso real:** si el handoff no demuestra radiación, viento
o metadatos requeridos por PM, elevar a 00 y mantener ejecución real bloqueada.
Las funciones matemáticas y sus pruebas sintéticas pueden verificarse mientras
se obtiene evidencia. No declarar READY_FOR_DATASET_V1 incondicional con esa
incertidumbre abierta.

Resolución concreta requerida: entregar evidencia de configuración de la cuenta/
servicio API utilizado por SIAR Sync y adquisiciones históricas, o confirmación
del proveedor aplicable a esos mismos datos y periodos. Debe precisar unidad de
`VelViento` y si `Radiacion` es energía diaria integrada o potencia media;
la unidad W/m² por sí sola no permite decidir una integración temporal.
Vincular la evidencia a los hashes y periodos de los CSV POST, identificando
posibles cambios de configuración históricos. Verificar también altura de
anemómetro por estación/periodo. Una descarga nueva bajo configuración conocida
solo prueba los bytes nuevos: no certifica automáticamente todos los bytes
históricos ni autoriza reemplazar raw. El bloqueo impide integrar/ejecutar
baselines reales y afirmar READY; no requiere alterar Quality ni el esquema raw.

## QC y elegibilidad independientes

Separar cinco capas: integridad de origen; estado de Quality POST; plausibilidad
por variable; elegibilidad por modelo; resultados del modelo. Las flags
científicas son derivadas y no cambian el dictamen Quality original.

Errores de integridad (identidad, fecha, duplicados, esquema, hash) impiden
construcción certificada. Valores inválidos o ausentes producen flags por
variable y motivos de no elegibilidad, conservando la fila.

Reglas físicas mínimas: valores finitos, Tmin ≤ Tmax, humedad en [0,100],
RHmin ≤ RHmax, viento y radiación no negativos; coordenadas válidas y altura
de sensor válida. Inconsistencias de temperatura media respecto a extremos,
Rs/Rso anómalo o ET₀ SIAR negativa se señalan; no aplicar winsorización ni
recortes silenciosos. Umbrales estadísticos exploratorios no eliminan registros.
No se hereda el filtro global ±3σ ni un `dropna` sobre todas las columnas.

| Salida | Requisitos propios |
|---|---|
| PM | Fecha, latitud, altitud, altura de viento; TempMax/Min, HumedadMax/Min, VelViento y Radiacion válidos |
| Hargreaves-Samani Ra | Fecha, latitud, TempMax/Min válidos; rango térmico no negativo |

Para la ejecución diaria FAO-56 estándar, calcular la temperatura media de
cálculo como `(TempMax + TempMin) / 2`, según la recomendación de estandarización
del [capítulo 3 FAO-56](https://www.fao.org/4/X0490E/x0490e07.htm).
Conservar `TempMedia` SIAR y registrar esta elección como parámetro del método;
no sobrescribirla ni confundirla con la media de extremos. La misma convención
se propone para HS v1. Una variante con media horaria/SIAR debe identificarse
por separado y no elegirse para reducir el error frente a EtPMon.
| Evaluación de un modelo | Predicción finita y EtPMon finita y admitida por política de referencia |

EtPMon ausente no impide calcular ninguno de los dos modelos. Radiacion ausente
impide PM pero no Hargreaves. HumedadMedia ausente no impide PM cuando están
los extremos requeridos. Variables auxiliares y precipitaciones ausentes no
excluyen por sí solas baselines. Ningún fallback de humedad, imputación o
calibración entra silenciosamente en v1.

Cada método produce `eligible`, lista de motivos y resultado nullable para cada
clave del dataset. Una excepción numérica se registra como fallo del método;
no se convierte en cero. Comparaciones usan cohortes propias y, separadamente,
intersección pareada PM/HS; registrar ambas selecciones y sus n evita comparar
métricas calculadas sobre periodos distintos sin advertirlo.

## Métricas y agrupaciones

Error firmado `prediction-reference`. `n` es número de pares finitos elegibles;
MAE = media de |error|; RMSE = raíz de media de error²; bias/MBE = media de
error. R² predictivo = 1 − SSE/SST, con SST centrado en la referencia; no es
correlación de Pearson². R² es indefinido si n < 2 o SST=0. Cohorte vacía:
n=0, métricas null y motivo explícito. Nunca devolver un R² perfecto artificial.

MAE, RMSE y MBE en mm/día. Agregación global primaria reúne pares, sin promediar
RMSE ni R² de estaciones. Puede ofrecerse media por estación como salida
secundaria identificada. Agrupar por estación, año, mes y estación climática
DJF/MAM/JJA/SON (hemisferio norte); para temporadas DJF, diciembre pertenece al
año de enero/febrero. Definir esta convención antes de crear agrupaciones.
RRMSE y AARE opcionales deben documentar denominadores y excluir/reportar
referencias cercanas a cero según umbral previo; no se requieren para baseline v1.

## Separación temporal sin leakage

PM y HS v1 tienen constantes publicadas sin parámetros aprendidos. Para futuras
calibraciones, fijar cortes globales de calendario después de POST según
cobertura, antes de observar errores: TRAIN < VALIDATION < TEST. No fijar hoy
años específicos ni ordenar filas aleatoriamente. Ninguna fecha puede ocupar
particiones distintas según estación; estaciones cerradas sin test se reportan
con ausencia de cobertura, no se redistribuyen temporalmente.

Estimaciones estadísticas, escalados, selección de variables, umbrales aprendidos
y AHC se ajustan únicamente con TRAIN. VALIDATION elige configuración; TEST se
abre una sola vez tras congelar configuración. Toda transformación aprendida
conserva hash, parámetros y fechas de ajuste. Ventanas derivadas respetan el
tiempo; futuras variables objetivo necesitan embargo/gap acorde al horizonte.
Los cortes, cobertura y reglas de elegibilidad se versionan. No aplicar al total
histórico el filtro ±3σ o coeficientes ajustados antes de partir.

## Run manifest futuro

Una ejecución escribe en un directorio nuevo por `run_id` y no sobrescribe
resultados previos. El manifest JSON identifica:

* `schema_version`, `run_id`, timestamps UTC de inicio/fin y estado final;
* `dataset_id`, hash del dataset, `source_data_hashes`, catálogo y Quality POST;
* Git SHA del pipeline, indicador dirty, versión de paquete, Python y entorno;
* métodos/versiones, constantes, unidades, parámetros y todas las políticas QC;
* configuración completa y hash, particiones temporales y hash de cohortes;
* hashes/rutas de predicciones, métricas, flags y logs de errores;
* conteos por modelo: entrada, elegibles, calculados, fallidos y pares evaluados;
* referencias metodológicas y limitaciones activas de la ejecución.

Rutas relativas son respecto al directorio de ejecución. Hashes SHA-256 en
hexadecimal minúscula; fechas UTC ISO 8601 terminadas en Z. No inventar commits
ni sustituir metadatos desconocidos por valores plausibles. Una ejecución fallida
conserva manifest con estado FAILED y diagnóstico, nunca resultados parciales
etiquetados como éxito. Los resultados son comparación con SIAR y no validación
contra mediciones independientes de ET₀.
