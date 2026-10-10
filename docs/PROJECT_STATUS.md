# PROJECT STATUS — SIAR / ET₀

**Última actualización:** 2026-10-10  
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
├── datos_siar_baleares/       # CSV canónicos IB01–IB10
├── legacy/                    # código y datos históricos/no productivos
├── docs/
├── estaciones_baleares.csv
└── README.md
```

`datos_siar_baleares/` contiene exclusivamente los diez CSV canónicos `IB01`–`IB10`. `IB101` e `IB11` aún no tienen CSV canónico. La validación científica y la completitud histórica de las estaciones siguen pendientes.

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
| SIAR Sync | 🟢 | v2.3.3 desplegada y validada; ejecución actual terminada por ahora | reanudar backfill cuando se autorice |
| TLS SIAR API | 🟢 | operativo con `SIAR_CA_BUNDLE`, manteniendo verificación TLS | retirar workaround cuando MAPA sirva una cadena compatible |
| Datos históricos SIAR | 🟠 | diez CSV canónicos IB01–IB10; completitud científica pendiente | validar/completar histórico de estaciones |
| Docker / servidor | 🟢 | `init: true` canónico; INC-INFRA-001 resuelta | mantener `init: true` en `siar-sync/docker-compose.yml` |
| GitHub / monorepo | 🟢 | PR #1–#10 fusionadas; PR #9/#10 reorganizaron código y datos históricos bajo `legacy/` | mantener estado consolidado |
| Forecast SIAR | 🟠 | flujo HTTP reproducido; contrato exacto del formulario/payload actual pendiente de cierre con 04W | cerrar contrato actual y después runbook + live test IB04 |
| Raw Forecast Capturer | 🟢 | v0.1 fusionado mediante PR #8; gate técnico sintético superado (29 passed), sin live test ni despliegue | runbook + prueba live manual IB04 |
| AEMET forecast | 🟠 | pendiente de investigación específica | identificar producto e inputs |
| Dataset científico | ⚪ | aún no consolidado | después de asegurar adquisición |
| ET₀ / modelos físicos | 🟡 | existe trabajo previo legacy | revisión científica posterior |
| Machine Learning | ⚪ | no priorizar todavía | después de datasets reproducibles |
| Web / API final | ⚪ | futura | después de modelos y validación |
| Kc / ETc / riego | ⚪ | futura | etapa de producto |
| Backup externo | 🟠 | diseño pendiente | coordinar con raw forecast |

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

### Último estado de backfill reportado a Dirección 00

- IB01–IB08: CSV válidos según las auditorías realizadas.
- IB01–IB06 e IB08: alcanzaban 31/12/2025 en el último reporte consolidado.
- IB07: cerrada; cobertura histórica hasta el entorno de su fecha de baja, pendiente de validar semántica exacta de fecha civil.
- IB09: backfill histórico parcialmente completado; progreso protegido y checkpoint realizado.
- IB10: pendiente.
- IB101: pendiente.
- IB11: pendiente.

> Este bloque debe actualizarse cuando 01 complete nuevas estaciones.  
> No asumir que este snapshot sigue vigente sin revisar la fecha de actualización del documento.

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

Fusionó en `main` la implementación del capturador bajo `forecast_collector/`. El gate técnico previo al merge pasó 29 tests sintéticos; aún no hay prueba live ni despliegue.

### PR #9 y PR #10 — reorganización legacy

Fusionadas. PR #9 archivó código histórico bajo `legacy/`; PR #10 separó los datos derivados históricos, dejando `datos_siar_baleares/` únicamente con los diez CSV canónicos IB01–IB10.

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

Todavía no hay prueba live contra SIAR, almacenamiento bajo `/srv`, scheduler ni despliegue. El siguiente gate es preparar un runbook y realizar una prueba live manual con IB04. El almacenamiento definitivo y el backup off-site siguen pendientes de coordinación con 02.

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

Línea todavía abierta.

Objetivo:

Identificar un producto automatizable que proporcione para ubicaciones equivalentes a las estaciones SIAR:

- temperatura;
- humedad relativa;
- viento;
- radiación global;
- precipitación;
- horizonte y cadencia conocidos.

Debe investigarse:

- API/producto;
- resolución temporal;
- resolución espacial;
- horizonte;
- frecuencia de actualización;
- autenticación/límites;
- posibilidad de conservar histórico de predicciones;
- correspondencia reproducible con estaciones SIAR.

No entrenar modelos predictivos hasta asegurar que todos los inputs utilizados estaban disponibles en el instante de emisión.

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

- completar el backfill y validar científicamente la cobertura histórica de las estaciones; que haya diez CSV canónicos no significa que el histórico esté completo;
- completar el siguiente gate de Forecast: runbook y prueba live manual con IB04;
- coordinar con 02 el almacenamiento definitivo fuera de Git y el backup off-site.

### No bloqueantes inmediatos

- validar la cadencia real y la semántica D+n del forecast;
- desplegar y observar el capturador durante 7–14 días después de aprobar la prueba live y la infraestructura;
- investigación AEMET, dataset científico reproducible, ET₀, ML y aplicación final.

## 16. Próximos pasos recomendados

```text
1. Cerrar con 04W el contrato actual del formulario/payload SIAR
2. Ajustar configuración/código solo si la evidencia demuestra que faltan campos requeridos
3. Finalizar runbook + prueba live manual IB04
4. Validar la respuesta contra SIAR real
5. Acordar almacenamiento definitivo y backup off-site con 02
6. Despliegue experimental y observación 7–14 días
7. Medir cadencia real y resolver semántica D+n
8. Completar/validar el backfill SIAR y consolidar dataset científico
9. ET₀, ML y producto web/API/riego
```

La ejecución actual de SIAR Sync ha terminado por ahora. Raw Forecast Capturer v0.1 ya está fusionado en `main`, pendiente del gate live y de infraestructura.

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
