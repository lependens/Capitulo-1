# Contratos científicos v1

Decisión 00, 2026-10-19: forense **RESOLVED_FOR_V1**. `UNITS_PROVENANCE_PARTIAL` y `station_period_provenance_not_exhaustively_verified` son limitaciones no bloqueantes. Preparación integrada **READY_FOR_DATASET_V1**, con pruebas, revisión y CI verificadas en el estado de misión. Esto no afirma que exista Dataset Observational v1 publicado o Quality POST entregado.

## Autoridad y handoff de Quality

`docs/siar/quality.md` conserva autoridad sobre estados e informes de Quality. La nueva capa no modifica adquisición, Quality, CSV canónicos ni legacy. El handoff autorizado después del cierre Quality es la entrada necesaria para construir el dataset definitivo; no se usa PRE como POST ni se fabrican artefactos.

El JSON de handoff incluye:

| Campo | Contrato |
|---|---|
| `gate` | Literal `SIAR-QUALITY-CLOSEOUT` |
| `authorized` | `true`; representa autorización existente de 00, no autoautorización del adaptador |
| `dataset_id` | Identidad inmutable del dataset a construir |
| `source_repo_sha` | Commit exacto de adquisición |
| `units` | Unidades aceptadas para las variables de entrada |
| `stations` | Metadatos identificados por código, latitud decimal y altitud; convenciones y evidencias disponibles |
| `catalog` | `{path, sha256}` del snapshot de catálogo; hash igual a `estaciones_catalog_sha256` de POST |
| `quality_post` | Objeto `{id, sha256, path}` que identifica el paquete POST JSON |
| `source_data_hashes` | Mapa ruta canónica → SHA-256 de los bytes originales |

Las rutas del handoff son relativas a su propio directorio. `quality_post.path` apunta a un paquete JSON con `phase: POST` y `reports`; sus tres claves son `siar_station_audit.csv`, `siar_missing_days.csv` y `siar_dataset_manifest.csv`, cada una con `{path, sha256}`. Las rutas de esos informes son relativas al directorio del paquete POST. Sus esquemas son los exactos de `docs/siar/quality.md`; no se crea un formato POST paralelo. Los periodos oficiales se obtienen de sus CSV, no de un calendario inventado o de la fecha extrema de un archivo observado. El catálogo verificado debe contener Codigo, Latitud (decimal o DMS compacto) y Altitud; comprobar correspondencia con los metadatos científicos, conservando la referencia y el hash. Altitud vacía puede conservarse como desconocida: impide PM, permite HS si sus otras entradas son válidas.

Verificar hashes, identidad, columnas, unicidad, fechas, periodos, estados y correspondencia entre los tres informes, catálogo y fuentes. Las contradicciones se devuelven a 00; nunca se reparan silenciosamente. Un JSON arbitrario con `authorized: true` no acredita autoridad por sí mismo: la publicación y revisión del handoff forman parte del control de entrada.

## Dataset Observational v1

Representación preparada: JSON UTF-8, `schema_version: observational-v1`. La construcción consume el handoff; no descarga datos ni escribe sobre sus fuentes.

| Campo | Semántica |
|---|---|
| `dataset_id` | Identificador inmutable, nunca reutilizado para bytes distintos |
| `source_repo_sha` | Commit de adquisición auditado |
| `source_data_hashes` | Hashes de fuentes originales |
| `quality_post` | Identidad/hash del paquete POST y referencias conservadas |
| `units` | Unidades normalizadas aceptadas; no certificación instrumental exhaustiva |
| `stations` | Código → metadatos, convención de viento y evidencia disponible |
| `rows` | Filas únicas por `station_code,date`, con orden estable |

Las 12 estaciones previstas son IB01–IB11 e IB101. El conjunto y periodos definitivos se contrastan con POST. No generar inventario mediante `range(1,12)` porque omite IB101. No convertir altitud de estación en altura de sensor ni derivar latitud de UTM para resolver discrepancias ajenas a esta capa.

Cada fila conserva `station_code`, `date` ISO YYYY-MM-DD, `values`, `observation_status`, `quality_status`, `quality_flags`, `provenance` y `source_data_sha`. La procedencia permite identificar archivo/fila originales; el hash de un archivo no se presenta como hash de registro. `values` preserva las 23 columnas canónicas:

```text
Fecha, TempMedia, TempMax, HorMinTempMax, TempMin, HorMinTempMin,
HumedadMedia, HumedadMax, HorMinHumMax, HumedadMin, HorMinHumMin,
VelViento, DirViento, VelVientoMax, HorMinVelMax, DirVientoVelMax,
Radiacion, Precipitacion, TempSuelo1, TempSuelo2, EtPMon, PePMon, Estacion
```

Normalizar claves no reinterpreta silenciosamente Fecha/Estacion. HHMM identifica horas de extremos y no sustituye la fecha civil para geometría solar. Conservar `TempMedia` SIAR aunque el cálculo use media de extremos. EtPMon nunca se sobrescribe con ET₀ nueva.

Valores ausentes se serializan como `null`, no cero ni NaN/Infinity JSON. Conservar filas incompletas. Duplicados estación/fecha son error de integridad, sin promediar ni elegir una fila arbitrariamente. No añadir observaciones meteorológicas inventadas para completar el calendario.

### Calendario y estados

El ledger complementa las observaciones. `OBSERVED` identifica fila existente; `RECOVERED` recuperación acreditada; `CONFIRMED_NO_DATA` consulta válida con cero registros; `PENDING_RETRY` incertidumbre pendiente; `MISSING` ausencia en snapshot sin inventar resultado de consulta; `OUTSIDE_OFFICIAL_PERIOD` fecha fuera de periodo, excluida del denominador de huecos. Estos estados se toman de la evidencia POST.

Las fechas sin observación se representan exclusivamente en `missing_days`, sin crear filas meteorológicas en `rows`. Las filas recuperadas permanecen en `rows` y su evidencia también puede constar en el ledger como RECOVERED. Una observación raw fuera de periodo se conserva y marca. Presencia de observación y estado `CONFIRMED_NO_DATA`/`PENDING_RETRY` contradictorios requieren resolver el handoff. El constructor adjunta QC científico y elegibilidad por método sin sustituir quality_flags o el dictamen POST. Los tokens numéricos no finitos del CSV se conservan semánticamente como null, con flag source_nonfinite por columna y procedencia que identifica los bytes originales; no se imputan valores.

## Unidades, metadatos y limitaciones aceptadas

Unidades: temperatura `degC`, viento `m/s`, radiación `MJ/m2/day`, humedad `%`, altitud `m`, latitud `degrees`, presiones `kPa`, ET₀ `mm/day`. Para los snapshots auditados usar directamente Radiacion y VelViento: **sin ×0.0864 ni /3.6**.

Viento a 2 m es la **convención SIAR aceptada para v1** por decisión 00. Ausencia de historial instrumental exhaustivo no bloquea PM. Conservar `UNITS_PROVENANCE_PARTIAL` y `station_period_provenance_not_exhaustively_verified` en la trazabilidad y manifiestos. No llamar «verificado por estación/periodo» a una altura aceptada por convención.

Una evidencia positiva de altura distinta de 2 m detiene PM para el ámbito afectado hasta resolución explícita. No se fuerza 2 m sobre evidencia contradictoria ni se aplica normalización silenciosa. HS sigue sujeto a su elegibilidad propia. Los nuevos lotes que contradigan unidades/convenciones deben revisarse; el cierre forense se refiere al ámbito auditado, no certifica cualquier fuente futura.

Fuentes y límites ya investigados: [manual API SIAR v2.2](https://servicio.mapa.gob.es/siarweb/documentos/manuales/ManualTecnico_WEB_API_SIAR.pdf), [consulta diaria SIAR](https://servicio.mapa.gob.es/siarweb/consultaDatos/consultaDatos), [descripción de estaciones SIAR](https://servicio.mapa.gob.es/siarweb/masInformacion), documentación histórica del proyecto y [búsqueda offline](units_offline_evidence.md). El [informe numérico forense](units_offline_forensic.md) se conserva sin reescribir sus cifras; su diagnóstico preliminar de bloqueo queda superado por decisión 00 del 2026-10-19. No se requiere continuar búsqueda histórica para avanzar v1.

## Matemáticas y QC por método

Las fórmulas, constantes y decisiones completas están en [assumptions.md](assumptions.md). V1 fija:

- PM diario FAO-56: Tcalc=(Tmax+Tmin)/2; G=0; albedo=0.23; γ=0.000665P; VPD a partir de humedad y temperaturas extremas, sin fallback; Rnl con σ=4.903·10⁻⁹ y Kelvin=T+273.16.
- Nubosidad `1.35*min(Rs/Rso,1)-0.35`, sin floor inferior por defecto. Rso≤0 impide ese cálculo; ratios muy bajos, Rnl negativo y ET₀ negativa se marcan sin recorte silencioso.
- HS fijo: `0.0023*0.408*Ra*(Tcalc+17.8)*sqrt(Tmax-Tmin)`. Sin ajuste, AHC o sustitución por Rs.

Separar integridad de origen, estado Quality, plausibilidad por variable, elegibilidad por método y diagnóstico de resultados. Las flags científicas derivadas no modifican Quality. No heredar ±3σ global ni `dropna` de todas las variables. Errores de identidad, fecha, duplicados, esquema o hash impiden una construcción válida; entradas ausentes o fuera de dominio generan motivos por método conservando su fila.

| Operación | Entradas necesarias |
|---|---|
| PM | Fecha, latitud, altitud, convención/evidencia de viento; TempMax/Min, HumedadMax/Min, VelViento, Radiacion válidos |
| HS | Fecha, latitud y TempMax/Min válidos; rango térmico no negativo |
| Evaluación | Predicción finita y EtPMon finita conforme a política de referencia registrada |

TempMedia SIAR ausente no impide los baselines de media de extremos. Radiacion ausente excluye PM, no HS; HumedadMedia ausente no impide PM con extremos válidos; variables auxiliares o EtPMon ausentes no impiden predicción. Exigir Tmin≤Tmax, 0≤RHmin≤RHmax≤100 y viento/radiación no negativos. Errores de dominio no producen cero; una excepción queda registrada como fallo. Mantener ET₀ negativa y flags diagnósticas, sin convertir una flag informativa en exclusión encubierta.

## Evaluación y particiones

Error firmado=predicción−EtPMon. MAE, RMSE y MBE/bias en mm/día; R² predictivo=1−SSE/SST. R² no definido con n<2 o SST=0; cohortes vacías informan n=0, métricas `null` y motivo. Los ceros de referencia permanecen en esas métricas. La política v1 admite EtPMon finita y no negativa, Quality PASS/WARN y observación dentro de periodo oficial. Las referencias negativas se conservan y marcan, excluidas de métricas sin recortarlas. RRMSE/AARE no son necesarias en v1; si se añaden deben informar denominador y exclusiones cerca de cero.

Publicar cohortes propias por modelo y cohorte pareada PM/HS, n y hashes de selección. Las métricas globales reúnen pares, no promedian RMSE/R² de estaciones. Desgloses por estación, año, mes y temporadas DJF/MAM/JJA/SON; diciembre se asigna al año siguiente en DJF. No presentar cohortes de distinto periodo como comparación pareada.

PM y HS v1 no aprenden parámetros. Para calibración/ML futuros: cortes globales cronológicos TRAIN<VALIDATION<TEST fijados tras POST antes de ver errores, sin partición aleatoria ni fecha en distinta partición según estación. TRAIN ajusta AHC, escalas, filtros y demás transformaciones; VALIDATION selecciona; TEST se abre una vez tras congelar configuración. Estaciones cerradas sin cobertura test se reportan así. Las ventanas y objetivos futuros respetarán embargo acorde al horizonte. No aplicar filtros/coefs estimados en todo el histórico antes de dividirlo.

## Interfaz y manifiestos de ejecución

Interfaz de módulo: `python -m siar_science`. `build-dataset` consume `--handoff` y escribe un archivo `--output` nuevo; requiere cierre autorizado. `validate`, `calculate-pm`, `calculate-hargreaves`, `evaluate` y `run-baselines` consumen `--dataset`, `--output` y opcionalmente `--pipeline-sha`. El SHA se obtiene del checkout cuando es posible; nunca inventarlo. `evaluate` requiere además uno o más `--predictions` repetibles, con artefactos `predictions.json` de cálculo previo que preserven identidad y hash del dataset y métodos sin solapamiento; no recalcula las predicciones. La semántica concreta de cada comando se documenta en el [README](../README.md) y su `--help`.

Las ejecuciones escriben un directorio nuevo sin sobrescribir resultados. Contiene `predictions.json`, `metrics.json` y `manifest.json`; en validación se conservan metadatos y elegibilidad por fila sin ET₀ calculada, con métricas vacías. Un resultado no calculable es `null` con motivo. Las rutas de outputs son relativas al directorio de ejecución y los hashes SHA-256 corresponden a bytes realmente escritos. Las predicciones transportan producer: identidad/hashes del código, parámetros, políticas, estado y referencias upstream; evaluate exige productor exitoso y parámetros coincidentes con el identificador versionado, y conserva esa autoría en el nuevo manifest. La cohorte paired exige siempre PM y HS, incluso al evaluar un solo modelo: si falta otro modelo su n es cero.

El manifiesto identifica dataset/hash, fuentes, POST/catálogo, SHA y estado del pipeline, versión/entorno, modelos/constantes/políticas, configuración, cohortes/particiones, conteos, timestamps UTC, archivos y hashes, flags y limitaciones activas. Registra la convención SIAR de 2 m y procedencia parcial. No presentar un SHA suministrado como prueba de checkout limpio ni un ensayo sintético como ejecución del dataset final. Un fallo debe quedar distinguido de éxito y no publicar resultados parciales etiquetados como completos.
