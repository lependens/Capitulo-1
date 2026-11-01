> **Estado posterior de preservación — 01/11/2026:** WINDOWS-SCIENCE-INGEST-AUTOMATION PASS; PROJECT-ARTIFACT-BACKUP-V1 PASS / MAINTAINED. Las referencias inferiores a transferencia manual/Windows sin protección automática describen el cierre anterior. El límite vigente es finalización/preparación y READY firmado explícitos; después, con clave SFTP desbloqueada en ssh-agent, transferencia, verificación servidor, ingestión y preservación automáticas. [Contrato operativo y evidencia E2E](../../docs/backup/windows-ingestion.md). Esta nota no modifica ciencia ni operación Forecast.

> **Estado posterior — 31/10/2026:** WEB-V0.2-AHC ya está desplegada (PR #46/#47) y el runtime ML rolling está implementado (PR #48); rolling validation y Linux smoke PASS. La campaña científica oficial permanece pendiente de despliegue/ejecución; ANN NOT STARTED. Las referencias inferiores a «web no construida», ML futuro o merge pendiente conservan el estado previo de su entrega. No cambian la semántica contractual ni resultados congelados. Para el contrato rolling vigente, véase [refinamiento temporal](ml_validation_contract_refinement.md); estado rápido en [PROJECT_STATUS](../../docs/PROJECT_STATUS.md).

# Contratos científicos v1

Decisión 00, 2026-10-19: forense **RESOLVED_FOR_V1**. `UNITS_PROVENANCE_PARTIAL` y `station_period_provenance_not_exhaustively_verified` son limitaciones no bloqueantes. Preparación integrada **READY_FOR_DATASET_V1**, con pruebas, revisión y CI verificadas en el estado de misión. Esta frase describía el gate de preparación del 19/10. A 23/10, POST corregido entregado, Dataset Observational v1 construido/validado y primera ejecución oficial PM/HS PASS; [evidencia publicada](../results/pm-vpd-20261023T194553Z/README.md). El contrato sigue aplicando a reconstrucciones y nuevos lotes; no autoautoriza handoffs.

## Autoridad y handoff de Quality

`docs/siar/quality.md` conserva autoridad sobre estados e informes de Quality. La nueva capa no modifica adquisición, Quality, CSV canónicos ni legacy. El handoff autorizado después del cierre Quality es la entrada necesaria para construir el dataset definitivo; no se usa PRE como POST ni se fabrican artefactos.

El JSON de handoff incluye:

| Campo | Contrato |
|---|---|
| `gate` | Literal `SIAR-QUALITY-CLOSEOUT` |
| `authorized` | `true`; representa autorización existente de 00, no autoautorización del adaptador |
| `dataset_id` | Identidad inmutable del dataset a construir |
| `audit_source_repo_sha` | SHA Git obligatorio de 40 caracteres hexadecimales: commit exacto utilizado por Quality Audit POST |
| `source_repo_sha` | SHA Git obligatorio de 40 caracteres hexadecimales: publicación canónica que ya contiene las fuentes, el catálogo y Quality POST utilizados |
| `units` | Unidades aceptadas para las variables de entrada |
| `stations` | Metadatos identificados por código, latitud decimal y altitud; convenciones y evidencias disponibles |
| `catalog` | `{path, sha256}` del snapshot de catálogo; hash igual a `estaciones_catalog_sha256` de POST |
| `quality_post` | Objeto `{id, sha256, path}` que identifica el paquete POST JSON |
| `source_data_hashes` | Mapa ruta canónica → SHA-256 de los bytes originales |

Las rutas del handoff son relativas a su propio directorio. `quality_post.path` apunta a un paquete JSON con `phase: POST` y `reports`; sus tres claves son `siar_station_audit.csv`, `siar_missing_days.csv` y `siar_dataset_manifest.csv`, cada una con `{path, sha256}`. Las rutas de esos informes son relativas al directorio del paquete POST. Sus esquemas son los exactos de `docs/siar/quality.md`; no se crea un formato POST paralelo. Los periodos oficiales se obtienen de sus CSV, no de un calendario inventado o de la fecha extrema de un archivo observado. El catálogo verificado debe contener Codigo, Latitud (decimal o DMS compacto) y Altitud; comprobar correspondencia con los metadatos científicos, conservando la referencia y el hash. Altitud vacía puede conservarse como desconocida: impide PM, permite HS si sus otras entradas son válidas.

Verificar hashes, identidad, columnas, unicidad, fechas, periodos, estados y correspondencia entre los tres informes, catálogo y fuentes. Las contradicciones se devuelven a 00; nunca se reparan silenciosamente. Un JSON arbitrario con `authorized: true` no acredita autoridad por sí mismo: la publicación y revisión del handoff forman parte del control de entrada.

### Identidades auditada y canónica: contrato de dos SHA

Los tres informes POST conservan su `repo_git_sha` original. El constructor exige
`POST.repo_git_sha == handoff.audit_source_repo_sha` en cada registro, además de
la trazabilidad común entre informes. No exige igualdad con `source_repo_sha`.
Para el cierre comunicado por 00, el commit auditado es
`18e6d893d119d7a6bd2959d6fdeb69fbdfa2c865`; no se modifica POST para sustituirlo
por un commit posterior.

`audit_source_repo_sha` y `source_repo_sha` pueden ser distintos o iguales.
La continuidad audit → canonical se demuestra para **todos** los archivos de la
publicación: el conjunto de `source_data_hashes` debe coincidir con los archivos
del manifest POST, cada hash debe ser igual a `canonical_sha256` y los bytes
leídos deben producir ese mismo SHA-256. Para el cierre Baleares son los 12 CSV.
Se conserva además `catalog.sha256 == estaciones_catalog_sha256` del POST y la
verificación de bytes de catálogo, paquete y los tres informes. Separar los SHA
Git no relaja ninguna comprobación de integridad.

**Sin autorreferencia Git:** `source_repo_sha` no identifica el commit del propio
handoff. Debe referenciar una publicación canónica anterior que ya contiene los
12 CSV exactos, el catálogo correspondiente y Quality POST publicado. El handoff
puede generarse/versionarse después y apuntar a ese commit. El constructor local
verifica los bytes entregados contra POST; no inventa ni descubre el SHA de
publicación, ni convierte por sí solo una declaración de commit en certificación
institucional.

Ambas identidades son obligatorias en el dataset normalizado. Un handoff o dataset
sin `audit_source_repo_sha`, o con un SHA inválido, se rechaza sin inferirlo de
`source_repo_sha`. Un artefacto anterior al cambio necesita la procedencia
auditada explícita antes de validarse bajo este contrato; nunca modificar raw
o POST para suplirla.

## Dataset Observational v1

Representación preparada: JSON UTF-8, `schema_version: observational-v1`. La construcción consume el handoff; no descarga datos ni escribe sobre sus fuentes.

| Campo | Semántica |
|---|---|
| `dataset_id` | Identificador inmutable, nunca reutilizado para bytes distintos |
| `audit_source_repo_sha` | Procedencia del audit: commit exacto preservado por Quality POST |
| `source_repo_sha` | Procedencia de la publicación canónica de fuentes, catálogo y Quality POST; no commit del handoff |
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

Los cuatro baselines v1 no aprenden parámetros. Para calibración/ML futuros: cortes globales cronológicos TRAIN<VALIDATION<TEST fijados tras POST antes de ver errores, sin partición aleatoria ni fecha en distinta partición según estación. Para ML, TRAIN ajusta escalas, filtros y demás transformaciones; VALIDATION selecciona; TEST se abre una vez tras congelar configuración. Estaciones cerradas sin cobertura test se reportan así. Las ventanas y objetivos futuros respetarán embargo acorde al horizonte. Para evaluación predictiva de ML, no aplicar filtros/coefs estimados en todo el histórico antes de dividirlo.

## Interfaz y manifiestos de ejecución

Interfaz de módulo: `python -m siar_science`. `build-dataset` consume `--handoff` y escribe un archivo `--output` nuevo; requiere cierre autorizado. `validate`, `calculate-pm`, `calculate-hargreaves`, `evaluate` y `run-baselines` consumen `--dataset`, `--output` y opcionalmente `--pipeline-sha`. El SHA se obtiene del checkout cuando es posible; nunca inventarlo. `evaluate` requiere además uno o más `--predictions` repetibles, con artefactos `predictions.json` de cálculo previo que preserven identidad y hash del dataset y métodos sin solapamiento; no recalcula las predicciones. La semántica concreta de cada comando se documenta en el [README](../README.md) y su `--help`.

Las ejecuciones escriben un directorio nuevo sin sobrescribir resultados. Contiene `predictions.json`, `metrics.json` y `manifest.json`; en validación se conservan metadatos y elegibilidad por fila sin ET₀ calculada, con métricas vacías. Un resultado no calculable es `null` con motivo. Las rutas de outputs son relativas al directorio de ejecución y los hashes SHA-256 corresponden a bytes realmente escritos. Las predicciones transportan producer: identidad/hashes del código, parámetros, políticas, estado y referencias upstream; evaluate exige productor exitoso y parámetros coincidentes con el identificador versionado, y conserva esa autoría en el nuevo manifest. La cohorte paired exige siempre PM y HS, incluso al evaluar un solo modelo: si falta otro modelo su n es cero.

El manifiesto identifica dataset/hash, fuentes, POST/catálogo, SHA y estado del pipeline, versión/entorno, modelos/constantes/políticas, configuración, cohortes/particiones, conteos, timestamps UTC, archivos y hashes, flags y limitaciones activas. Registra la convención SIAR de 2 m y procedencia parcial. No presentar un SHA suministrado como prueba de checkout limpio ni un ensayo sintético como ejecución del dataset final. Un fallo debe quedar distinguido de éxito y no publicar resultados parciales etiquetados como completos.

Dataset, run manifest y artefactos `predictions.json` conservan ambos campos
`audit_source_repo_sha` y `source_repo_sha`. Cada fila de métricas serializada
conserva igualmente las dos identidades, sin modificar los cálculos de métricas;
las colecciones vacías siguen vinculadas mediante el manifest del run. La
evaluación verifica ambos SHA de los artefactos importados frente al dataset,
además de su identidad y hash de bytes. Ningún output sustituye el SHA del audit
por el de publicación ni por el SHA del código del pipeline.

## Extensión clásica 2026-10-24

`calculate-valiantzas` añade `VALIANTZAS_V1` / `ET0_Valiantzas` a los artefactos
de resultados, con HumedadMedia y extremos térmicos, fecha y latitud como
entradas independientes. La elegibilidad VAL se calcula en resultados;
el dataset existente, su esquema y su SHA no se alteran. Parámetros/domino:
[contrato metodológico](valiantzas_v1.md). Se mantienen los motivos PM/HS.

`evaluate-classical` exige predicciones exitosas de los tres métodos, dataset
y ambos SHA coincidentes, hashes del productor y parámetros versionados.
No recalcula ni sustituye PM/HS importados. `run-classical-baselines` calcula
los tres. Ambos publican cohortes own, PM_HS, PM_VAL, HS_VAL, PM_HS_VAL,
con hashes de claves estación-fecha evaluables; PM_HS_VAL es la comparación
principal. `paired_predictions` y `paired_evaluation` siguen significando PM/HS.
No se reutiliza `paired` con un significado distinto.

Se añade `classical-baselines.csv` con todas las filas/columnas originales,
ET0_PM_FAO56_V1, ET0_HS_RA_V1, ET0_VALIANTZAS_V1, eligible/evaluated/reasons
para PM/HS/VAL, fechas, estación, estados, flags y provenance JSON. Las
ausencias permanecen vacías y los ceros son ceros. Incluye dataset_id/SHA,
ambos SHA audit/canonical, POST ID, pipeline SHA y run timestamp. El manifest
conserva hashes POST/fuentes/catálogo, configuración, productores upstream y
hash de cada salida. Su propio hash se publica en el cierre, evitando
autorreferencia. UTF-8 BOM, coma, punto decimal, sin redondeo; ver importación
Excel/Power Query en README. Directorios nuevos, sin sobrescritura.

## Extensión HG_RS_V1 — cuatro métodos

`calculate-hargreaves-rs` añade `HG_RS_V1` / `ET0_Hargreaves_Rs` con extremos
térmicos y Radiacion medida; [contrato científico](hargreaves_rs_v1.md).
No modifica Dataset v1, su esquema ni su SHA. `evaluate-four` exige exactamente
los cuatro métodos exitosos y conserva los productores de los tres anteriores;
`run-four-baselines` calcula los cuatro. Validación de dataset, ambos SHA,
parámetros, hashes y protección contra sobrescritura siguen siendo obligatorios.
Las cohortes propias y todas las intersecciones de dos/tres/cuatro métodos
se identifican separadamente. `PM_HS_VAL_HGRS` es la principal; no se redefine
la cohorte triple anterior ni `paired` PM/HS. Desgloses y métricas no cambian.

El CSV de cuatro añade ET0_HG_RS_V1, eligible_HGRS, evaluated_HGRS y
eligibility_reasons_HGRS. Los comandos de tres métodos conservan su esquema
anterior; ningún output previo se sobreescribe. Se mantienen formato, nulos,
ceros, estados, provenance y ambas identidades audit/canonical; manifest
conserva dataset/source/POST/pipeline/config y hashes de todas las salidas.

## Bytes contractuales y checkout reproducible

El catálogo y los 12 CSV `datos_siar_baleares/IB*_datos_completos.csv`
se materializan con LF mediante `.gitattributes`, incluso con
`core.autocrlf=true`. Los informes POST publicados mantienen su regla `-text`
y sus bytes exactos. La prueba sintética de checkout cubre ambas políticas.
Un hash distinto sigue siendo **FAIL**: el builder no normaliza finales de
línea ni acepta representaciones alternativas.

En `source_repo_sha=63e413d82dec4cae09e562aa0266ca566b9d2863`, los bytes
del blob Git del catálogo tienen SHA-256
`98abd16dcd6ea42ad46a30c331df47150f07dffdca140c14e4445890ce68a357`.
Los 12 blobs CSV también coinciden con los hashes del handoff autorizado y
son LF. La conversión de checkout a CRLF causó el bloqueo previo del catálogo;
la protección se limita a estos archivos contractuales demostrados.
El fix no cambia sus blobs científicos, el POST ni el handoff.

Para ejecutar, usar un checkout nuevo del pipeline corregido y copiar el
handoff autorizado sin cambiar sus bytes junto a los CSV, preservando sus
rutas relativas al catálogo y POST. Comprobar todos los hashes antes de
construir. `source_repo_sha` sigue identificando la publicación canónica
anterior; `pipeline_sha` identifica por separado el código del fix. No es el
SHA de un commit que deba contener su propio handoff.

## AHC y preservación: estado vigente 25/10/2026

**AHC-METHODOLOGY-AND-TEMPORAL-VALIDATION PASS**, PR #38/#39; OLS0 por estación a=Σ(x·PM)/Σ(x²), TRAIN 2004–2015 únicamente, coeficientes congelados. VALIDATION 2016–2020 seleccionó HS annual / HG-Rs seasonal / VAL annual; TEST 2021–2025 retrospective_locked, sin refit ni reselección. [Metodología](ahc_methodology.md) y [evidencia](../results/ahc-20261024T172400Z/README.md).

AHC_CLIMATOLOGY se calcula después de TEST con todo el histórico declarado hasta 2025; identidad separada y sin atribución de métricas TEST. El legacy mean-ratio full-series/in-sample sigue siendo historia. Los cuatro baselines, raw, Quality y Dataset no cambian.

**PROJECT-ARTIFACT-BACKUP-V1 PASS** según cierre de 02 comunicado por Dirección 00; snapshot backup_20261024T215639Z_63d0ef01 REMOTE_VERIFIED, 128 archivos. Los artefactos científicos cubiertos ya tienen preservación externa; no implica que cualquier resultado futuro solo Windows esté automáticamente protegido. Requiere transferencia/ingestión explícita y reconciliación diaria para lo ingerido. [Alcance, pruebas y límites](../../docs/DATA_LOCATIONS.md).

ML contract FROZEN/READY FOR FUTURE EXPERIMENTS y visualization contract FROZEN/AVAILABLE TO 06; no ML/ANN entrenado ni web construida.
