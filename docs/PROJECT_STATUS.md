# PROJECT STATUS — SIAR / ET₀

**Última actualización:** 2026-10-18\
**Repositorio principal:** `lependens/Capitulo-1`  
**Rol de este documento:** punto de recuperación rápido del estado real del proyecto.

> Este archivo permite retomar el proyecto ante cambios de equipo o pérdida de contexto operativo.
> Para conocer hacia dónde va el proyecto, consultar también `docs/ROADMAP.md`.

---

## 1. Objetivo global

Construir una infraestructura reproducible para adquirir, conservar, validar y explotar datos meteorológicos de las estaciones SIAR de Baleares, con especial foco en ET₀.

Las líneas de adquisición trabajan en paralelo:

| Fuente | Componente | Salida conservada |
|---|---|---|
| Observaciones SIAR | `siar-sync/` | CSV canónicos; auditoría Quality por separado |
| Forecast SIAR | `forecast_collector/` | Capturas raw append-only fuera de Git |
| Forecast municipal AEMET | `aemet_forecast_collector/` | Wrapper y payload raw separados, append-only |

El cierre Quality habilita Dataset Observacional v1. La captura prospectiva de forecasts avanza en paralelo y no espera a que terminen los modelos. Después, datasets reproducibles → FAO-56 PM → Hargreaves → baselines comparables → ML y validación predictiva → Web/API, ETc y riego.

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

1. [PROJECT_STATUS](PROJECT_STATUS.md) — estado actual y decisiones vigentes.
2. [ROADMAP](ROADMAP.md) — líneas paralelas, fases y dependencias.
3. [MILESTONES](MILESTONES.md) — hitos materiales y trazabilidad append-only.
4. [README raíz](../README.md) — navegación general.
5. [SIAR Sync](../siar-sync/README.md), [Forecast SIAR](../forecast_collector/README.md) y [AEMET](../aemet_forecast_collector/README.md) — contratos y límites de cada componente.
6. [Quality](siar/quality.md) y [sync-status](siar/sync-status.md) — reglas de auditoría y estado observacional.
7. [Bitácora de adquisición](../docs_1.1_Obtenci%C3%B3n%20de%20datos.md) y documentación científica `docs_1.*` / `docs_2.*` cuando se trabaje en ET₀/ML.

El estado consolidado, las decisiones y sus referencias deben permitir recuperar el trabajo sin depender del historial de conversaciones.

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
├── aemet_forecast_collector/
├── datos_siar_baleares/       # 12 CSV canónicos
├── data_quality/siar/         # reservado para informes de auditoría
├── legacy/                    # código y datos históricos/no productivos
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── ROADMAP.md
│   ├── MILESTONES.md
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
| SIAR Sync | 🟢 | v2.3.3 productiva; backfill temporal de 12 estaciones alcanzado | mantener adquisición e integridad raw mientras Quality R2 avanza |
| TLS SIAR API | 🟢 | operativo con `SIAR_CA_BUNDLE`, manteniendo verificación TLS | retirar workaround cuando MAPA sirva una cadena compatible |
| Datos históricos SIAR | 🟠 | PRE completo; R1 superado como etapa histórica; R2 en ejecución autónoma; POST pendiente | Quality closeout y Dataset Observacional v1 |
| Docker / servidor | 🟢 | `init: true` canónico; INC-INFRA-001 resuelta | mantener `init: true` en `siar-sync/docker-compose.yml` |
| GitHub / monorepo | 🟢 | PR #1–#20 fusionadas; collectors SIAR/AEMET y legacy separados | mantener estado y [hitos](MILESTONES.md) recuperables |
| Forecast SIAR | 🟢 | `SMOKE-FC-001 = PASS`; `BACKUP-FC-001 = PASS`; `/srv` operativo | scheduler experimental y cierre backup/DR |
| Raw Forecast Capturer | 🟠 | runtime permanente validado; Google Drive + rclone crypt + restore demostrados; scheduler periódico = NO | cerrar gates operativos y comenzar captura prospectiva periódica |
| AEMET forecast | 🟠 | Collector Track A fusionado (#18), canonicalización (#19), permisos POSIX (#20); A1/A2 PASS, A3 COMPLETE | confirmación E2E final pendiente por timeout operacional; después despliegue prospectivo + backup; Track B OPEN |
| Dataset científico | 🟡 | Dataset Observacional v1 aún no generado; depende de Quality closeout | Dataset v1 → FAO-56 PM → Hargreaves → baselines → ML |
| ET₀ / modelos físicos | 🟡 | existe trabajo previo legacy | revisión científica posterior |
| Machine Learning | ⚪ | no priorizar todavía | después de datasets reproducibles |
| Web / API final | ⚪ | futura | después de modelos y validación |
| Kc / ETc / riego | ⚪ | futura | etapa de producto |
| Backup externo | 🟠 | `BACKUP-FC-001 = PASS`; copia off-site y restore validados; recuperación ante pérdida total del servidor aún pendiente | confirmar recuperabilidad del material crypt fuera del servidor; repetir tras multi-smoke antes del scheduler |

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

En el snapshot PRE, los 2.346 huecos quedaron registrados como `PENDING_RETRY` / `NOT_ATTEMPTED` / `attempt_count=0`. El PRE no declaró ningún `CONFIRMED_NO_DATA`. Son estados de un snapshot inmutable: no describen necesariamente la ejecución actual de recuperación.

### Quality Recovery — R2 en ejecución

R1 queda superado como etapa histórica. R2 está en ejecución autónoma y POST permanece pendiente. Un `INVALID_RESPONSE` aislado se registra y trata dentro del ámbito afectado; por sí solo no bloquea globalmente R2. No se declara el cierre Quality ni se inventan resultados POST mientras esa etapa no concluya y se revise su evidencia.

El checkpoint R1 206/431 y el hallazgo IB07 pertenecen al historial de R1, no al estado operativo actual ni a una orden de pausar R2. No se infiere aquí una clasificación científica nueva de la fila IB07.

**IB09 en el PRE: FAIL.** La instalación oficial es 2005-10-24 y la baja 2018-10-10; la auditoría PRE detectó una fila raw de 2005-10-23, fuera del periodo oficial. El hallazgo y el snapshot se preservan; el resultado de POST aún no está disponible.

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

### PR #12–#14 — almacenamiento, contrato SIAR y HTML sensible

Fusionadas: PR #12 añadió permisos de grupo compartido; PR #13 adaptó el cliente al contrato E2E demostrado; PR #14 permitió omitir HTML sensible sin abortar la captura. El contrato SIAR y el runtime permanente tienen evidencia de validación; `SMOKE-FC-001` y `BACKUP-FC-001` están PASS. El scheduler periódico aún no está activo.

### PR #9 y PR #10 — reorganización legacy

Fusionadas. PR #9 archivó código histórico bajo `legacy/`; PR #10 separó los datos derivados históricos y dejó entonces `datos_siar_baleares/` con los diez CSV canónicos IB01–IB10.

### PR #18–#20 — Collector AEMET Track A

Fusionadas: PR #18 incorporó el Collector; PR #19 añadió la comparación canónica de identificadores municipales; PR #20 completó los permisos POSIX de los directorios ancestros. El detalle material y los SHA están en [MILESTONES](MILESTONES.md).

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

El contrato E2E SIAR está demostrado, PR #14 está fusionada y `LIVE-IB04 #2` está completo. `SMOKE-FC-001 = PASS` para IB04 + Aguacate: run y capture completos, TLS verificado, hashes/schemas PASS, permisos `2770/0660` + SGID PASS, `html_persistence = omitted_sensitive` y `raw/tmp` vacío.

`BACKUP-FC-001 = PASS`: rclone 1.75.2 a Google Drive con rclone crypt (contenido, nombres y directorios cifrados), `rclone copy` sin `sync`, restore independiente PASS, SHA256SUMS, schemas, ZIP/CSV y `COMPLETE` remoto validados.

El runtime permanente está validado como `siar-forecast`, con release funcional `e850827e24cff11ae46f51c9bc3e8a2f36b0d2da` y Python 3.14.4. `/opt/siar-forecast` y `/etc/siar-forecast` están operativos; `/srv/siar-forecast/raw` está aprovisionado y validado. El scheduler periódico aún NO está activo. El cierre de backup/DR, incluida la recuperabilidad del material crypt fuera del servidor, sigue siendo un gate separado del PASS de `BACKUP-FC-001`.

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

Raíz operativa aprovisionada y validada:

```text
/srv/siar-forecast/raw
```

`/srv/siar-forecast/raw/{runs,tmp}` está aprovisionado y validado; `BACKUP-FC-001` pasó. Queda pendiente confirmar material crypt recuperable fuera del servidor.

Frecuencia propuesta para un futuro scheduler experimental (scheduler = NO; aún no hay captura periódica):

```text
08:00
14:00
20:00
Europe/Madrid
```

Debe seguir siendo configurable.

HTML raw:

- conservar byte-for-byte cuando el guard rail lo permita;
- si contiene material sensible reutilizable, no persistirlo ni sanitizarlo silenciosamente: aplicar la política `omitted_sensitive` y conservar el ZIP/CSV seguro;
- escalar hallazgos que impliquen seguridad o integridad raw.

Volumen estimado inicial:

- ~1,4 MB de HTML por estación/captura;
- ~5–6 GB/año con 1 captura/día para 9 estaciones;
- ~15–18 GB/año con 3 capturas/día, antes de compresión/backup.

---

## 12. AEMET

### Track A — Collector municipal fusionado

El producto aceptado es AEMET municipal daily latest forecast, para captura prospectiva append-only y un futuro benchmark Hargreaves. El código está en [aemet_forecast_collector/](../aemet_forecast_collector/README.md).

| Hito | Estado consolidado |
|---|---|
| PR #18 | Collector Track A fusionado |
| PR #19 | Canonicalización del identificador municipal fusionada |
| PR #20 | Permisos POSIX de ancestros fusionados |
| AEMET-A1 | PASS |
| AEMET-A2 | PASS |
| AEMET-A3 | COMPLETE |
| Confirmación E2E final | Pendiente por timeout operacional |
| Track B | OPEN |

El flujo wrapper → `datos` → payload y el contrato municipal ya tienen evidencia. Se observó `id = 7040` como entero JSON frente al identificador configurado `"07040"`; la canonicalización solo compara y no altera raw. En POSIX, `data_root/runs/YYYY/MM/DD` y el árbol publicado usan `2770`, ficheros `0660`, SGID en directorios y permisos `other = 0`.

La última confirmación E2E sigue pendiente por un timeout operacional; ese timeout no demuestra un defecto del Collector ni revoca los hitos A1/A2 PASS y A3 COMPLETE. El siguiente gate es confirmar el E2E en el runtime objetivo y preparar despliegue prospectivo con backup. No se declara operación periódica AEMET.

### Track B — OPEN

La investigación del feed AEMET exacto utilizado por SIAR sigue abierta y es independiente de Track A.

Límites científicos del producto municipal:

- representa la capital municipal, no un forecast puntual de estación SIAR;
- por sí solo no aporta todas las variables para FAO-56 Penman–Monteith completo;
- no demuestra recuperación de emisiones históricas;
- no tiene todavía asignación científica D+n ni timezone de emisión resuelta.

No usar observaciones futuras como inputs ni entrenar modelos predictivos antes de fijar disponibilidad temporal y semántica científica.

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

Para Forecast SIAR se ha demostrado Google Drive mediante rclone crypt, copia off-site y restore independiente (`BACKUP-FC-001 = PASS`). El backup de AEMET y la automatización periódica se cierran en sus workstreams. El PASS de una prueba de backup no equivale a dar por cerrada la recuperación ante pérdida total del servidor: el material crypt debe ser recuperable fuera de él.

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

## 15. Gates abiertos y prioridades

- **Observaciones / 01:** completar R2, generar y revisar POST, cerrar Quality y preparar Dataset Observacional v1. Un `INVALID_RESPONSE` aislado no detiene globalmente R2.
- **Infraestructura y Forecast SIAR / 02:** cerrar gates de scheduler y backup/DR, con `/srv` y la prueba de backup ya operativos; no declarar captura periódica hasta validarla.
- **AEMET / 04:** confirmar E2E final tras el timeout operacional y continuar con despliegue prospectivo y backup. Track B sigue OPEN.
- **Ciencia:** mantener CATALOG-TZ como investigación semántica no bloqueante por sí sola; definir Dataset v1 y baselines antes de ML.

## 16. Trabajo paralelo y secuencia científica

| Workstream | Trabajo autónomo actual | Salida siguiente |
|---|---|---|
| 01 — Observaciones y Quality | R2 → POST → Quality closeout | Dataset Observacional v1 |
| 02 — Plataforma y Forecast SIAR | scheduler + backup/DR | captura prospectiva periódica validada |
| 04 — AEMET | confirmación E2E final | despliegue prospectivo + backup |

La adquisición prospectiva de forecasts no espera a que terminen Dataset v1, los baselines ni ML. Cada fuente comienza a conservar emisiones cuando supera sus propios gates de adquisición, seguridad y persistencia. El cierre observacional tampoco requiere que ambas fuentes forecast hayan terminado su despliegue.

La secuencia científica es Dataset Observacional v1 → FAO-56 PM → Hargreaves → baselines comparables → ML. La semántica D+n, cadencia y correspondencia estación/municipio se validarán antes de construir comparativas predictivas.

Quality PRE está completo, R1 es histórico, R2 está en ejecución y POST pendiente. Forecast SIAR conserva `SMOKE-FC-001` y `BACKUP-FC-001` PASS con scheduler periódico aún NO. AEMET Track A está fusionado, A1/A2 PASS y A3 COMPLETE, con confirmación E2E final pendiente; no se declaran cerradas las fases observacional ni de captura prospectiva.

## 17. Protocolo de actualización de este documento

Actualizar `PROJECT_STATUS.md` cuando ocurra un hito material:

- nueva versión desplegada;
- PR relevante fusionada;
- hito material registrado al final de [MILESTONES](MILESTONES.md);
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

## 18. Modelo operativo y gobierno

Dirección de proyecto (00) fija objetivos, invariantes y criterios STOP. Los workstreams 01, 02 y 04 trabajan autónomamente dentro de esos límites y resuelven los errores operativos en su propio ámbito. Solo se escalan decisiones de arquitectura, seguridad, integridad raw o semántica científica; un fallo operativo aislado no se convierte automáticamente en un bloqueo global.

La responsabilidad documental (03) consolida estado, enlaces e hitos mediante PR revisables. `PROJECT_STATUS` resume el presente, `ROADMAP` ordena el trabajo y `MILESTONES` conserva decisiones e hitos materiales de forma append-only. Ninguno sustituye las evidencias de ejecución ni autoriza inventar resultados.
