# PROJECT STATUS — SIAR / ET₀

**Última actualización:** 2026-10-12  
**Repositorio principal:** `lependens/Capitulo-1`  
**Rol de este documento:** punto de recuperación rápido del estado real del proyecto.

> Este archivo debe permitir retomar el proyecto aunque los chats hayan crecido demasiado o se pierda contexto operativo.  
> Para conocer hacia dónde va el proyecto, consultar también `docs/ROADMAP.md`.

---

## 1. Objetivo global

Construir una infraestructura reproducible para adquirir, conservar, validar y explotar datos meteorológicos de las estaciones SIAR de Baleares, con especial foco en ET₀.

La evolución prevista es:

```text
SIAR observaciones ──> SIAR Sync ───────────────┐
                                                  │
SIAR forecast ──────> Raw Forecast Capturer ──> Forecast Collector
                                                  │
AEMET forecast ───────────────────────────────────┤
                                                  ↓
                                         Dataset científico
                                                  ↓
                                  ET₀ física / empírica / ML
                                                  ↓
                                  validación histórica/predictiva
                                                  ↓
                                         ET₀ + Kc -> ETc
                                                  ↓
                                      necesidades de riego
                                                  ↓
                                           Web / API
```

Principios vigentes:

- integridad de datos antes que velocidad;
- cambios pequeños y verificables;
- no borrar ni modificar producción sin comprender antes el estado;
- GitHub como fuente de verdad del código;
- adquisición, procesamiento, análisis científico y producto final separados;
- evitar data leakage en cualquier modelo predictivo;
- conservar datos raw cuando puedan ser irrecuperables.

---

## 2. Cómo retomar el proyecto si se pierde el contexto

Orden recomendado de lectura:

1. `docs/PROJECT_STATUS.md` — estado actual y decisiones vigentes.
2. `docs/ROADMAP.md` — fases, dependencias y objetivos.
3. `README.md` — navegación general del repositorio.
4. `siar-sync/README.md` — componente SIAR Sync.
5. `docs_1.1_Obtención de datos.md` — bitácora de adquisición SIAR.
6. Documentación científica específica (`docs_1.2`, `docs_1.3`, `docs_1.4`, etc.) cuando se trabaje en ET₀/ML.

Los chats especializados siguen siendo útiles para el detalle operativo, pero este documento debe conservar el estado consolidado.

---

## 3. Arquitectura y responsabilidades

### 3.1 Monorepo

Se mantiene un único repositorio principal:

```text
lependens/Capitulo-1
```

La separación se realiza por componentes y carpetas, no mediante repositorios independientes.

Estructura real del monorepo en `main`:

```text
Capitulo-1/
├── siar-sync/
├── forecast_collector/
├── datos_siar_baleares/       # 12 CSV canónicos
├── data_quality/siar/         # reservado para informes de auditoría
├── legacy/                    # código y datos históricos/no productivos
├── docs/
│   └── siar/
├── estaciones_baleares.csv
└── README.md
```

`datos_siar_baleares/` contiene los 12 CSV canónicos: `IB01`–`IB11` e `IB101`. El backfill temporal de adquisición ha alcanzado las 12 estaciones. La certificación científica de calidad y cobertura sigue pendiente; no se presupone que todos los días esperados tengan observaciones válidas.

### 3.2 Producción y GitHub

Producción SIAR Sync se ejecuta actualmente desde:

```text
/home/josep/siar-sync
```

El repositorio contiene desde la PR #2 una copia canónica en:

```text
siar-sync/
```

La incorporación al repositorio no implicó despliegue ni sustitución del directorio productivo.

---

## 4. Estado maestro por componente

| Componente | Estado | Situación actual | Próximo hito |
|---|---|---|---|
| SIAR Sync | 🟢 | v2.3.3 desplegada y validada; ejecución actual terminada por ahora | Quality Recovery / POST autorizado y en curso |
| TLS SIAR API | 🟢 | operativo con `SIAR_CA_BUNDLE`, manteniendo verificación TLS | retirar workaround cuando MAPA sirva una cadena compatible |
| Datos históricos SIAR | 🟠 | Quality Audit v1 PRE completado: 12 estaciones, cobertura ponderada 96,99 %; recuperación/POST y certificación pendientes | Quality Recovery / POST y certificación observacional |
| Docker / servidor | 🟢 | `init: true` canónico; INC-INFRA-001 resuelta | mantener `init: true` en `siar-sync/docker-compose.yml` |
| GitHub / monorepo | 🟢 | PR #1–#15 fusionadas; PR #9/#10 reorganizaron código y datos históricos bajo `legacy/` | mantener estado consolidado |
| Forecast SIAR | 🟢 | E2E y LIVE-IB04 #2 completos; permanent runtime preflight PASS | permanent smoke run autorizado, resultado pendiente |
| Raw Forecast Capturer | 🟠 | v0.1 fusionado; runtime permanente preparado, sin ejecución periódica | smoke run; después backup off-site y scheduler experimental |
| AEMET forecast | 🟠 | Track A aceptado: forecast municipal diario más reciente; Track B en investigación | cerrar contrato raw Track A y estudiar feed AEMET usado por SIAR |
| Dataset científico | ⚪ | aún no consolidado | después de asegurar adquisición |
| ET₀ / modelos físicos | 🟡 | existe trabajo previo legacy | revisión científica posterior |
| Machine Learning | ⚪ | no priorizar todavía | después de datasets reproducibles |
| Web / API final | ⚪ | futura | después de modelos y validación |
| Kc / ETc / riego | ⚪ | futura | etapa de producto |
| Backup externo | 🟠 | rclone/off-site pendiente | configurar y validar backup antes del scheduler |

---

## 5. SIAR Sync

### 5.1 Versión productiva

La versión productiva actual es `v2.3.3`, desplegada y validada desde:

```text
/home/josep/siar-sync
```

v2.3.2 fue desplegada y validada como corrección funcional intermedia. El worker productivo v2.3.3 tiene SHA-256:

```text
7d48be629cd6cf584bed491f9e30da57f62a025f67654be1c33706b02886a81f
```

Otros hashes productivos cotejados:

```text
Dockerfile
39998beee1043cc1a12388038ea3b9faf6aa831adeeabc2cf90c88f0ae96f9f9

certs/siar_extra_chain.pem
3b212076d42035c737e112287fef31db3fbcea2e907e17c9a95cd13bb31b97cf

app.py
c2569944a4ef27fe42ff9168022065f58db72a78bb9acc6417eaacb1ef8e2d02

templates/index.html
1de14ab615bf1b63a86495e1238988db03f3be27e0f7cba161d8fd7354d49f73

requirements.txt
3af465eba4e9d46d574053b001bdbf5c3799279b819cf8a84adf6d6a0c99df3d
```

La plantilla y el worker v2.3.3 están cotejados con el contenedor. La autenticación real y `Info/ACCESOS` fueron validados. Tras los despliegues v2.3.2 y v2.3.3, los nueve CSV IB01–IB09 permanecieron byte a byte idénticos. El TLS está operativo mediante `SIAR_CA_BUNDLE` y la verificación TLS sigue activa.

La versión v2.3.3 queda canonizada en GitHub mediante la PR #4, sin desplegar ni sustituir la instancia activa.

### 5.2 Funciones actualmente consolidadas

- autenticación SIAR mediante NIF y contraseña;
- obtención dinámica de token;
- consulta de `Info/ACCESOS`;
- bloques de descarga de máximo 7 días;
- detección de cobertura y huecos;
- creación automática de CSV para estaciones nuevas;
- esquema canónico fijo de 23 columnas;
- deduplicación por `Fecha`;
- orden cronológico;
- escritura atómica;
- refresco de días recientes;
- reintentos ante errores HTTP transitorios;
- persistencia local aunque Git falle;
- checkpoints Git y sincronización con GitHub;
- interfaz FastAPI/Uvicorn en puerto 8000;
- interfaz web para análisis, sincronización, estado y cuota SIAR.

### 5.3 v2.3.2 — corrección funcional intermedia desplegada y validada

El artefacto final v2.3.2 fue desplegado y validado antes de v2.3.3. Su SHA-256 exacto fue:

```text
87e583dd6576d74cc4b51e1abb7ccbdd34ec94ec1fc15619e1999786367d1916
```

Incluye el tratamiento de fechas civiles de instalación/baja en `Europe/Madrid`, aplicar la baja al reanudar estaciones, intentar checkpoint Git al alcanzar la cuota diaria y normalizar permisos finales de CSV a `0644` manteniendo escritura atómica.

### 5.4 v2.3.3 — compatibilidad TLS desplegada y validada

v2.3.3 incorpora `SIAR_CA_BUNDLE` opcional para utilizar una cadena CA compatible con validación TLS activa. Sin la variable se conserva la verificación estándar. No se utiliza `verify=False`.

El certificado adicional versionado es material CA público. El bundle `siar_bundle.pem` se genera durante el build combinando el trust store de `certifi` con la cadena adicional; el bundle generado no se versiona. El workaround puede retirarse cuando MAPA sirva una cadena compatible.

## 6. Catálogo y datos SIAR de Baleares

Catálogo oficial confirmado:

```text
IB01 IB02 IB03 IB04 IB05 IB06 IB07 IB08 IB09 IB10 IB101 IB11
```

Total:

```text
12 estaciones
```

Cerradas:

```text
IB07
IB09
IB101
```

Activas:

```text
9 estaciones
```

### Quality Audit v1 — PRE completado

La auditoría PRE de los 12 CSV canónicos está completada. El cálculo usa los periodos oficiales disponibles y un cutoff explícito para estaciones activas; la discrepancia de huso sigue pendiente:

| Métrica PRE | Resultado |
|---|---:|
| Estaciones auditadas | 12 |
| ACTIVE / CLOSED | 9 / 3 |
| WARN / FAIL / INCOMPLETE | 11 / 1 / 0 |
| Días esperados | 77.840 |
| Días con dato | 75.494 |
| Huecos PRE | 2.346 |
| Cobertura ponderada | 96,99 % |

Los 2.346 huecos se mantienen como `PENDING_RETRY`, `NOT_ATTEMPTED`, con `attempt_count=0`. No se ha declarado ningún `CONFIRMED_NO_DATA`. Quality Recovery / POST está autorizado y en curso; aún no consta como completado.

**IB09 permanece FAIL:** la fecha oficial de instalación es 2005-10-24 y la fecha oficial de baja 2018-10-10; el CSV raw contiene una fila de 2005-10-23. La fila se conserva íntegra y queda fuera del periodo oficial. No se ha corregido ni reescrito el CSV.

Hallazgo CATALOG-TZ pendiente de investigación semántica: el huso del snapshot local es 31 y el de Info/ESTACIONES es 30. No se corregirá automáticamente `estaciones_baleares.csv` ni se usará ese campo científicamente hasta conocer su significado. Por sí sola, esta discrepancia no degrada el resultado PRE ni bloquea la certificación observacional, salvo que se demuestre que afecta al periodo o a la semántica de los datos.

Las políticas y formatos están en [Calidad SIAR](siar/quality.md); el resumen PRE y el estado de recuperación están en [sync-status](siar/sync-status.md). Esta actualización documental no genera ni modifica artefactos de auditoría.

### Saneamiento operativo posterior a v2.3.2 — completado

El modo canónico esperado de los CSV es `0644`. La lógica correctora ya forma parte de SIAR Sync v2.3.2; el saneamiento puntual de archivos existentes fue una operación de filesystem posterior a v2.3.2, no una nueva versión funcional.

Resultado: 12/12 CSV están en modo `0644`; SHA-256 PRE y POST coincide en 12/12; el contenido raw no cambió. No se reescribieron filas, esquema, fechas ni observaciones.

---

## 7. Esquema canónico SIAR

Los CSV de observaciones deben conservar exactamente 23 columnas:

```text
Fecha
TempMedia
TempMax
HorMinTempMax
TempMin
HorMinTempMin
HumedadMedia
HumedadMax
HorMinHumMax
HumedadMin
HorMinHumMin
VelViento
DirViento
VelVientoMax
HorMinVelMax
DirVientoVelMax
Radiacion
Precipitacion
TempSuelo1
TempSuelo2
EtPMon
PePMon
Estacion
```

Regla:

> No modificar el esquema canónico sin una decisión explícita de arquitectura y migración.

---

## 8. Incidencia de procesos zombie — servidor/Docker

Incidencia:

```text
INC-INFRA-001
```

La incidencia queda **RESUELTA** tras validar `init: true` en producción: `docker-init` como PID 1, 0 procesos zombie, 265 checkpoints Git, 254 subidas correctas, 0 errores Git, `pids.current=12` y `RestartCount=0`.

`init: true` queda como requisito canónico de `siar_sync`.

---

## 9. GitHub y trazabilidad

### PR #1 — completada

Objetivo:

```text
higiene inicial del repositorio
```

Resultado:

- `.gitignore` añadido;
- `venv/` eliminado del seguimiento;
- sin cambios funcionales.

### PR #2 — completada

Objetivo:

```text
incorporar SIAR Sync canónico de producción al monorepo
```

Merge commit:

```text
7902c2ac1c2c8c8d3d0dd06fe3cbd1bb417f1264
```

Se añadieron únicamente los nueve archivos aprobados bajo:

```text
siar-sync/
```

No se desplegó esta versión desde GitHub al servidor.

### PR #3 — documentación maestra fusionada

Se incorporaron `docs/PROJECT_STATUS.md` y `docs/ROADMAP.md`, junto con las actualizaciones documentales aprobadas.

### PR #4 — canonización de SIAR Sync v2.3.3

Fusionada. Incorpora a GitHub el estado productivo v2.3.3 con la trazabilidad de v2.3.2 y v2.3.3. No desplegó ni sustituyó producción.

### PR #8 — Raw Forecast Capturer v0.1

Fusionó en `main` la implementación del capturador bajo `forecast_collector/`; el gate técnico previo al merge pasó 29 tests sintéticos.

### PR #14 — contrato E2E y live test

Fusionada. El contrato E2E SIAR y `LIVE-IB04 #2` están completos. La infraestructura permanente superó el preflight; el smoke run está autorizado y pendiente de resultado. Rclone/off-site backup está pendiente, no hay scheduler configurado y el Collector no opera de forma periódica.

### PR #9 y PR #10 — reorganización legacy

Fusionadas. PR #9 archivó código histórico bajo `legacy/`; PR #10 separó los datos derivados históricos y dejó entonces `datos_siar_baleares/` con los diez CSV canónicos IB01–IB10.

Cambios de producción y de infraestructura deben reflejarse mediante PR revisables y separadas.

---

## 10. Forecast SIAR

La primera fase de ingeniería inversa está completada.

Flujo HTTP reproducido:

```text
GET /siarweb/necesidadesHidricas/inicio
    ↓
sesión web + CSRF
    ↓
llamadas auxiliares estación/cultivo
    ↓
validación formulario
    ↓
POST /siarweb/necesidadesHidricas/calculo
    ↓
HTML de forecast
    ↓
GET /siarweb/necesidadesHidricas/exportCSV
    ↓
ZIP con CSV oficial
```

Características confirmadas:

- no requiere login de usuario;
- requiere mantener la sesión web;
- usa `JSESSIONID`, `XSRF-TOKEN` y `_csrf`;
- el CSV oficial utiliza `;`;
- decimal con coma;
- columnas observadas:
  - Fecha
  - Kc
  - ET0
  - ETc
  - Pe
  - ETc - Pe
- el HTML contiene mayor precisión numérica que el CSV;
- todavía no está resuelta la semántica exacta del horizonte;
- no existe evidencia de recuperación de forecasts históricos.

Regla provisional:

> Una emisión no capturada puede ser irrecuperable.

---

## 11. Raw Forecast Capturer

v0.1 está implementado en `forecast_collector/` y fusionado en `main` mediante PR #8. El gate técnico previo al merge pasó **29 tests**. La validación sintética cubre JSON Schema, guard rail de secretos, lectura segura de ZIP, SHA-256, estados `complete` / `partial` / `failed`, append-only y publicación atómica. `source.source_issue_at` es `null`.

El contrato E2E SIAR está demostrado, PR #14 está fusionada y `LIVE-IB04 #2` está completo. El release preparado es `e850827e24cff11ae46f51c9bc3e8a2f36b0d2da`, con Python 3.14.4. El permanent runtime preflight pasó. El permanent smoke run está autorizado y pendiente de resultado; rclone/off-site backup está pendiente y no hay scheduler configurado. El Collector no está desplegado periódicamente.

La infraestructura física validada incluye el usuario de servicio `siar-forecast`, `/srv/siar-forecast/raw/{runs,tmp}`, permisos base/SGID, patrón de releases en `/opt/siar-forecast`, configuración en `/etc/siar-forecast`, TLS bundle permanente y runtime independiente de `/home`.

Responsabilidad:

```text
SIAR web
   ↓
captura HTTP
   ↓
RAW append-only
```

No normaliza ni interpreta científicamente.

### Unidad de captura

```text
1 estación
+
1 instante de recogida
+
HTML original
+
ZIP oficial
+
CSV extraído
+
metadata
+
hashes
```

### Principios

- `run_id` por ejecución;
- `capture_id` por estación;
- append-only;
- no deduplicación destructiva;
- no sobrescritura;
- estados `complete`, `partial`, `failed`;
- fallo de una estación no aborta el resto;
- cookies y CSRF solo en memoria;
- escritura temporal y publicación mediante rename atómico;
- SHA-256 de artefactos;
- datos legibles posteriormente sin Collector ni base de datos.

Raíz propuesta, todavía no aprovisionada y pendiente de coordinación con 02:

```text
/srv/siar-forecast/raw
```

No hay almacenamiento creado bajo `/srv`; su provisión y el backup off-site siguen pendientes.

Frecuencia experimental aprobada:

```text
08:00
14:00
20:00
Europe/Madrid
```

Debe seguir siendo configurable.

HTML raw:

- conservar byte-for-byte;
- antes del despliegue verificar que no contiene secretos, cookies o CSRF reutilizables;
- si contiene material sensible, no sanitizar silenciosamente: volver a Dirección 00.

Volumen estimado inicial:

- ~1,4 MB de HTML por estación/captura;
- ~5–6 GB/año con 1 captura/día para 9 estaciones;
- ~15–18 GB/año con 3 capturas/día, antes de compresión/backup.

---

## 12. AEMET

La línea AEMET se divide en dos tracks:

- **Track A — ACCEPTED:** AEMET municipal daily latest forecast → captura append-only prospectiva → futuro benchmark Hargreaves.
- **Track B — RESEARCH:** no se ha demostrado cuál es el feed AEMET exacto utilizado por SIAR.

Límites del producto municipal aceptado para Track A:

- no es un forecast puntual de una estación SIAR;
- por sí solo no proporciona todas las variables para FAO-56 Penman–Monteith completo;
- no se ha demostrado histórico de emisiones;
- todavía no tiene asignación D+n.

La captura prospectiva no implica que exista un archivo histórico de emisiones ni autoriza interpretar el producto municipal como observación o forecast puntual SIAR. No entrenar modelos predictivos hasta asegurar que los inputs estaban disponibles en el instante de emisión.

---

## 13. Almacenamiento y backups

Decisiones vigentes:

### Código

```text
GitHub
```

Fuente de verdad de código, documentación, esquemas y configuración reproducible no secreta.

### Observaciones SIAR

Los CSV actuales continúan versionados en GitHub por ahora.

No cambiar este mecanismo estable sin necesidad concreta.

### Forecast raw

Debe permanecer fuera de Git.

Arquitectura prevista:

```text
Servidor = copia operativa
+
backup externo = copia off-site
+
GitHub = código/documentación
```

Proveedor externo todavía pendiente:

- OneDrive; o
- Google Drive.

Para una futura base DuckDB:

> No sincronizar directamente una base activa mientras está siendo escrita.  
> Crear snapshot consistente y respaldar ese snapshot.

---

## 14. Decisiones arquitectónicas vigentes

1. `Capitulo-1` sigue siendo el monorepo principal.
2. SIAR Sync y Forecast Collector son componentes independientes.
3. Forecast Collector no debe importar ni depender de `siar_worker.py`.
4. `datos_siar_baleares/` no debe moverse mientras producción dependa de esa ruta.
5. `estaciones_baleares.csv` no debe moverse mientras producción dependa de esa ruta.
6. Forecast raw vive fuera de Git.
7. Emisiones forecast se conservan append-only.
8. HTML y ZIP/CSV oficiales deben preservarse en raw cuando sea seguro.
9. `source_issue_at` no debe confundirse con `collected_at`.
10. ET0 vacío no equivale a cero.
11. No asignar D+0/D+1/D+N hasta demostrar la semántica temporal.
12. Futuras NN solo pueden usar información disponible en el instante de emisión.
13. No usar `verify=False` para resolver el TLS de SIAR.
14. Los cambios funcionales deben entrar mediante PR pequeñas.
15. Producción y GitHub pueden estar desacoplados durante una transición controlada, pero la divergencia debe ser temporal y documentada.

---

## 15. Bloqueadores e incidencias abiertas

### Alta prioridad

- completar Quality Recovery / POST y certificar científicamente cobertura, validez y huecos; el PRE mantiene 2.346 fechas pendientes de consulta. Investigar CATALOG-TZ como hallazgo semántico no bloqueante por sí solo;
- completar el permanent smoke run ya autorizado y configurar rclone/off-site backup antes del scheduler experimental.

### No bloqueantes inmediatos

- validar la cadencia real y la semántica D+n del forecast;
- completar el smoke run autorizado, validar backup off-site y configurar scheduler experimental antes de cualquier operación periódica;
- mantener Track B de AEMET en investigación y avanzar después con dataset científico reproducible, ET₀, ML y aplicación final.

## 16. Próximos pasos recomendados

```text
1. Completar Quality Recovery / POST; no interpolar, imputar, borrar ni editar manualmente observaciones. Los registros reales devueltos por SIAR para fechas ausentes pueden incorporarse mediante el mecanismo canónico de SIAR Sync.
2. Revisar los resultados POST y certificar el periodo observacional; investigar CATALOG-TZ sin corregir automáticamente `estaciones_baleares.csv` ni usar el campo científicamente hasta conocer su significado.
3. Ejecutar el permanent smoke run autorizado y revisar su resultado
4. Configurar rclone/off-site backup y validarlo
5. Configurar scheduler experimental solo tras superar backup
6. Observar la operación y medir cadencia/semántica D+n
7. Cerrar el contrato raw prospectivo de AEMET Track A; continuar Track B en investigación
8. ET₀, ML y producto web/API/riego
```

Quality Audit v1 PRE está completado; Recovery / POST y certificación siguen abiertos. El permanent runtime preflight de Forecast pasó, mientras que el smoke run autorizado aún espera resultado, rclone/off-site backup y scheduler siguen pendientes, y no hay operación periódica.

## 17. Protocolo de actualización de este documento

Actualizar `PROJECT_STATUS.md` cuando ocurra un hito material:

- nueva versión desplegada;
- PR relevante fusionada;
- estación histórica completada;
- incidencia importante abierta/cerrada;
- Raw Forecast Capturer desplegado;
- decisión arquitectónica nueva;
- cambio de fase del proyecto;
- modificación del almacenamiento/backups.

No actualizar por cada conversación menor.

Cada actualización debe cambiar también:

```text
Última actualización: YYYY-MM-DD
```

---

## 18. Propiedad documental

Conceptualmente:

```text
00 · Dirección
```

mantiene el estado y las decisiones globales.

Operativamente:

```text
03 · GitHub
```

materializa las actualizaciones documentales en GitHub mediante PR revisables.
