# PROJECT STATUS — SIAR / ET₀

**Última actualización:** 2026-10-09  
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

Estructura consolidada actualmente:

```text
Capitulo-1/
├── siar-sync/
├── datos_siar_baleares/
├── estaciones_baleares.csv
├── docs_*.md
└── ...
```

Estructura futura prevista, todavía no completada:

```text
Capitulo-1/
├── siar-sync/
├── forecast-collector/       # futuro
├── et0/                      # futuro
├── analysis/                 # futuro
├── ml/                       # futuro
├── docs/
├── tests/                    # futuro
├── config/                   # futuro
├── datos_siar_baleares/      # no mover mientras producción dependa de esta ruta
└── estaciones_baleares.csv   # no mover mientras producción dependa de esta ruta
```

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
| SIAR Sync | 🟠 | v2.3.1 estable en producción | preparar/probar v2.3.2 |
| TLS SIAR API | 🟠 | fallo externo reproducible con cadena CA normal | v2.3.3 con `SIAR_CA_BUNDLE` |
| Datos históricos SIAR | 🟠 | backfill avanzado, aún no finalizado | completar estaciones restantes |
| Docker / servidor | 🟠 | incidencia de zombies diagnosticada | aplicar/validar `init: true` |
| GitHub / monorepo | 🟢 | PR #1 y PR #2 fusionadas | documentación y siguientes PR pequeñas |
| Forecast SIAR | 🟢 | contrato HTTP reproducido | Raw Forecast Capturer |
| Raw Forecast Capturer | 🟠 | diseño aprobado, no desplegado | implementación aislada |
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

Versión estable actual:

```text
v2.3.1
```

Código productivo validado por hashes entre host y contenedor:

```text
app.py
siar_worker.py
templates/index.html
Dockerfile
requirements.txt
docker-compose.yml
```

Los tres archivos principales validados host ↔ contenedor fueron:

```text
app.py
SHA-256 c2569944a4ef27fe42ff9168022065f58db72a78bb9acc6417eaacb1ef8e2d02

siar_worker.py
SHA-256 0bd2c2a541b31238067cd969397110f262eaa994a1540adfa73a58893089999c

templates/index.html
SHA-256 1de14ab615bf1b63a86495e1238988db03f3be27e0f7cba161d8fd7354d49f73
```

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

### 5.3 Próxima versión v2.3.2

Alcance aprobado y cerrado:

1. interpretar correctamente `Fecha_Instalacion` y `Fecha_Baja` en la zona `Europe/Madrid`;
2. aplicar `Fecha_Baja` también a estaciones existentes/reanudadas;
3. intentar checkpoint Git antes de salir por `SiarDailyLimitError`;
4. normalizar permisos finales de CSV a `0644` manteniendo escritura atómica.

No debe incluir refactors, cambios de UI ni modificaciones ajenas a estos cuatro puntos.

### 5.4 Próxima versión v2.3.3

Objetivo aprobado:

```text
SIAR_CA_BUNDLE=/ruta/siar_bundle.pem
```

Debe ser opcional y mantener validación TLS activa.

No se autoriza utilizar:

```text
verify=False
```

La incidencia TLS actual se reproduce fuera del programa y también en el host. SIAR responde correctamente cuando se utiliza explícitamente una cadena CA válida.

---

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

Diagnóstico:

- el contenedor `siar_sync` acumuló más de mil procesos zombie `git`;
- los zombies terminaron adoptados por Uvicorn;
- Uvicorn es actualmente PID 1;
- el contenedor no tenía init/reaper;
- los procesos Git provienen de checkpoints ejecutados por `siar_worker.py`;
- la hipótesis principal es adopción de procesos auxiliares/huérfanos no recolectados;
- el problema no ha demostrado corrupción de CSV.

Corrección autorizada:

```yaml
init: true
```

en el servicio `siar_sync`, recreando únicamente ese contenedor, sin rebuild ni pull.

Estado de este documento:

```text
AUTORIZADA / pendiente de informe final de ejecución y validación prolongada
```

La incidencia no debe cerrarse hasta observar varios checkpoints Git reales sin nueva acumulación.

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

### Regla futura

Cambios de producción deben reflejarse mediante PR pequeñas y trazables.

Ejemplos próximos:

- `init: true`, una vez validado;
- SIAR Sync v2.3.2;
- SIAR Sync v2.3.3.

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

Diseño aprobado.

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

Raíz física decidida:

```text
/srv/siar-forecast/raw
```

Todavía no debe crearse/desplegarse sin coordinación con infraestructura.

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

- TLS de `servicio.mapa.gob.es` impide continuar adquisición normal con validación CA estándar.
- completar v2.3.2 y posteriormente v2.3.3.
- validar corrección `init: true` para zombies.
- desplegar cuanto antes el Raw Forecast Capturer para no perder emisiones potencialmente irrecuperables.

### No bloqueantes

- limpieza legacy adicional del repositorio;
- reorganización completa de carpetas;
- normalización científica;
- ML;
- aplicación web final.

---

## 16. Próximos pasos recomendados

Orden orientativo actual:

```text
1. SIAR Sync v2.3.2
2. validar init: true / cerrar INC-INFRA-001
3. SIAR Sync v2.3.3 y recuperar acceso TLS seguro
4. completar backfill IB10 / IB101 / IB11 y pendientes
5. implementar/desplegar Raw Forecast Capturer
6. observar cadencia/horizonte de forecast
7. investigar AEMET
8. diseñar backup off-site
9. dataset científico reproducible
10. ET₀ y ML
11. producto web/API/riego
```

Algunas tareas pueden ejecutarse en paralelo si no comparten riesgos.

---

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
