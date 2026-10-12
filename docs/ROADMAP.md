# ROADMAP — SIAR / ET₀

**Última actualización:** 2026-10-12  
**Objetivo:** definir las fases del proyecto, sus dependencias y los criterios para considerarlas terminadas.

> Este documento describe hacia dónde va el proyecto.  
> Para el estado operativo actual consultar `docs/PROJECT_STATUS.md`.

---

## Leyenda

```text
✅ completado
🟢 operativo / consolidado
🟠 en curso
🟡 planificado próximo
⚪ futuro
⛔ bloqueado
```

---

## Visión de conjunto

```text
FASE 0
Repositorio + infraestructura base
        ↓
FASE 1
Observaciones SIAR / SIAR Sync
        ├─────────────────────────────┐
        ↓                             ↓
FASE 2                        FASE 3
Forecast SIAR                 Forecast meteorológico
Raw Capture                   AEMET / otros inputs
        └──────────────┬──────────────┘
                       ↓
                    FASE 4
               Dataset científico
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
      FASE 5                      FASE 6
   ET₀ física/empírica              ML
         └─────────────┬─────────────┘
                       ↓
                    FASE 7
             Validación predictiva
                       ↓
                    FASE 8
              Web / API / ETc / riego
```

---

# FASE 0 — Base del proyecto

## Objetivo

Disponer de una base segura, versionada y reproducible desde la que evolucionar.

### Estado

- ✅ repositorio principal `lependens/Capitulo-1`;
- ✅ estrategia de monorepo;
- ✅ `.gitignore`;
- ✅ eliminación de `venv/` del versionado;
- ✅ SIAR Sync canónico incorporado a `siar-sync/`;
- ✅ separación conceptual producción ↔ GitHub;
- ✅ documentación maestra `PROJECT_STATUS` / `ROADMAP`;
- ✅ reorganización legacy segura de código y datos históricos completada mediante PR #9 y PR #10; `datos_siar_baleares/` conserva solo los CSV canónicos.

### Criterio de salida

La fase se considera suficientemente consolidada cuando:

- el código productivo relevante está versionado;
- no existen secretos en Git;
- la navegación documental permite recuperar contexto;
- el repositorio puede evolucionar sin depender de archivos históricos accidentales.

Estado:

```text
🟢 prácticamente completada
```

---

# FASE 1 — Adquisición observacional SIAR

## Objetivo

Mantener una copia histórica fiable y actualizable de las observaciones SIAR de Baleares.

### Ya conseguido

- ✅ SIAR Sync operativo;
- ✅ FastAPI/Uvicorn;
- ✅ Docker;
- ✅ autenticación NIF/password;
- ✅ token dinámico;
- ✅ control de cuota;
- ✅ bloques ≤ 7 días;
- ✅ esquema canónico 23 columnas;
- ✅ detección de huecos;
- ✅ escritura atómica;
- ✅ deduplicación;
- ✅ refresco reciente;
- ✅ persistencia local;
- ✅ checkpoints Git;
- ✅ resiliencia ante errores HTTP transitorios;
- ✅ catálogo oficial de 12 estaciones;
- ✅ detección de estaciones activas/cerradas.

### Trabajo actual

- ✅ backfill temporal de adquisición alcanzado para 12 estaciones;
- ✅ Quality Audit v1 PRE completado: 12 estaciones (9 ACTIVE, 3 CLOSED), 11 WARN, 1 FAIL (IB09), 0 INCOMPLETE;
- ✅ cobertura PRE ponderada 96,99 %: 75.494 días con dato de 77.840 esperados; 2.346 huecos;
- 🟠 Quality Recovery / POST autorizado, aún no completado; los 2.346 huecos siguen PENDING_RETRY / NOT_ATTEMPTED, intentos 0, sin CONFIRMED_NO_DATA;
- 🟠 resolver discrepancia de huso del catálogo: snapshot local 31 frente a Info/ESTACIONES 30, sin corrección automática;
- ✅ saneamiento operativo posterior a v2.3.2: 12/12 CSV con modo 0644, SHA-256 PRE = POST 12/12, contenido raw intacto;
- ✅ v2.3.2 desplegada y validada; lógica correctora de permisos incluida desde esa versión;
- ✅ v2.3.3 desplegada y validada: SIAR_CA_BUNDLE opcional con verificación TLS activa;
- ✅ v2.3.3 canónica en GitHub tras PR #4;
- ✅ INC-INFRA-001 resuelta y init: true canónico.

IB09 permanece FAIL porque el CSV raw conserva la observación 2005-10-23, anterior a la Fecha_Instalacion oficial 2005-10-24; su Fecha_Baja oficial es 2018-10-10. La fila no se elimina y queda fuera del periodo oficial.

La ejecución actual de SIAR Sync terminó por ahora. datos_siar_baleares/ contiene 12 CSV canónicos. El resultado PRE y los formatos se documentan en docs/siar/quality.md y docs/siar/sync-status.md.

### Criterio de salida

La fase observacional se considera consolidada cuando:

1. las 12 estaciones oficiales tienen el tratamiento histórico correcto;
2. activas/cerradas respetan sus periodos reales;
3. no quedan huecos atribuibles a fallos del Collector sin documentar;
4. el TLS funciona con validación segura;
5. Git y persistencia local son robustos;
6. el contenedor no acumula zombies;
7. la versión productiva coincide con una versión trazable en GitHub;
8. existe procedimiento de backup/restauración probado.

Ampliación geográfica futura: SIAR Sync España no forma parte de la implementación actual.

Estado:

```text
🟠 Quality PRE completado; Recovery / POST y certificación pendientes
```

---

# FASE 2 — Forecast SIAR y preservación raw

## Objetivo

Conservar cada emisión publicada por SIAR antes de que pueda desaparecer.

### Ingeniería inversa

- ✅ flujo HTTP identificado;
- ✅ sesión pública y CSRF reproducibles;
- ✅ validación y POST de cálculo reproducidos;
- ✅ exportación ZIP/CSV reproducida;
- ✅ estructura CSV conocida;
- ✅ HTML con mayor precisión identificado;
- ✅ ausencia de login confirmada;
- ✅ contrato E2E SIAR demostrado y `LIVE-IB04 #2` completado; PR #14 fusionada;
- ✅ permanent runtime preflight PASS: service user, almacenamiento base, release pattern, configuración y TLS bundle permanente validados;
- 🟠 permanent smoke run autorizado, resultado pendiente;
- 🟠 semántica exacta del horizonte pendiente;
- 🟠 cadencia de publicación pendiente.

### Raw Forecast Capturer — v0.1 implementado

- ✅ implementación fusionada en `main` mediante PR #8;
- ✅ estados `complete` / `partial` / `failed`;
- ✅ append-only;
- ✅ SHA-256;
- ✅ publicación atómica;
- ✅ guard rail de secretos;
- ✅ manejo seguro de ZIP;
- ✅ JSON Schema;
- ✅ tests sintéticos (29 passed en el gate previo al merge);
- ✅ integración en `main`;
- ✅ `source.source_issue_at = null`.

El código actual está en `forecast_collector/`. `LIVE-IB04 #2` está completo; el capturador aún no está desplegado.

Arquitectura aprobada:

```text
SIAR web
   ↓
captura HTTP
   ↓
raw append-only
```

Decisiones:

- `run_id` por ejecución;
- `capture_id` por estación;
- HTML raw;
- ZIP oficial;
- CSV extraído;
- metadata;
- SHA-256;
- `complete / partial / failed`;
- sin deduplicación destructiva;
- sin normalización;
- sin base de datos obligatoria;
- cookies/CSRF solo en memoria;
- datos fuera de Git;
- raíz prevista `/srv/siar-forecast/raw`;
- frecuencia experimental 08:00 / 14:00 / 20:00 Europe/Madrid.

### Infraestructura permanente y siguiente gate

El release preparado es e850827e24cff11ae46f51c9bc3e8a2f36b0d2da, con Python 3.14.4. El permanent runtime preflight está en PASS. La infraestructura física validada incluye:

- usuario de servicio siar-forecast;
- /srv/siar-forecast/raw/{runs,tmp} creado con permisos base y SGID validados;
- patrón de releases en /opt/siar-forecast;
- configuración en /etc/siar-forecast;
- TLS bundle permanente validado;
- runtime independiente del home validado.

Pendiente:

- 🟠 ejecutar y revisar el permanent smoke run, ya autorizado;
- 🟠 configurar y validar rclone/off-site backup;
- 🟠 configurar scheduler experimental solo después del backup;
- ⚪ determinar cadencia real y semántica D+N;
- ⚪ decidir frecuencia definitiva y parser/normalización posterior.

El Collector no está desplegado de forma periódica. La FASE 2 permanece abierta.

### Criterio de salida

La fase se considera completada cuando:

1. todas las estaciones activas se capturan automáticamente;
2. las capturas son append-only e inmutables;
3. no se pierden capturas parciales/fallidas;
4. existe verificación SHA-256;
5. existe backup externo;
6. se conoce la cadencia real de SIAR;
7. se conoce la semántica temporal del horizonte;
8. el raw puede reprocesarse sin depender del Collector original.

Estado:

```text
🟠 en curso
```

---

# FASE 3 — Fuente meteorológica futura / AEMET

## Objetivo

Obtener inputs meteorológicos futuros reproducibles para modelos propios de ET₀.

Variables prioritarias:

- temperatura;
- humedad relativa;
- viento;
- radiación global;
- precipitación.

### Tracks AEMET

- ✅ **Track A — ACCEPTED:** AEMET municipal daily latest forecast → captura prospectiva append-only → futuro benchmark Hargreaves.
- 🟠 **Track B — RESEARCH:** el feed exacto de AEMET utilizado por SIAR no está demostrado.

Límites de Track A:

- el producto municipal no es un forecast puntual de una estación SIAR;
- por sí solo no permite calcular FAO-56 Penman–Monteith completo;
- no se ha demostrado histórico de emisiones;
- todavía no tiene horizonte D+n asignado.

Contrato empírico raw y diseño del Collector para captura prospectiva siguen pendientes. Aceptar el producto para Track A no cierra la investigación de Track B ni completa la FASE 3.

### Regla científica

Para predecir ET₀ de D+n:

> solo pueden utilizarse variables que estuvieran disponibles en el instante en que se habría emitido la predicción.

Nunca utilizar observaciones futuras como inputs.

### Criterio de salida

- producto automatizable identificado;
- variables/unidades documentadas;
- horizonte y cadencia documentados;
- asociación estación ↔ forecast reproducible;
- emisiones conservables;
- ausencia de data leakage demostrable.

Estado:

```text
🟠 investigación
```

---

# FASE 4 — Dataset científico

## Objetivo

Separar adquisición raw de datos preparados para investigación.

Arquitectura prevista:

```text
RAW
 ↓
validación
 ↓
normalización
 ↓
Parquet / tablas derivadas
 ↓
DuckDB
 ↓
datasets versionados/reproducibles
```

### Trabajo previsto

- ⚪ esquema científico;
- ⚪ parser de observaciones;
- ⚪ parser de forecast;
- ⚪ reconciliación forecast ↔ observado;
- ⚪ control de calidad;
- ⚪ manifiestos/versiones;
- ⚪ particiones train/validation/test;
- ⚪ snapshots reproducibles.

### Regla

Los raw originales no se modifican para “arreglar” problemas.

Las correcciones y transformaciones deben producir datasets derivados.

### Criterio de salida

Un experimento debe poder relacionarse inequívocamente con:

- raw de origen;
- versión de parser;
- versión de dataset;
- parámetros;
- código;
- fecha de ejecución.

Estado:

```text
⚪ futuro próximo
```

---

# FASE 5 — ET₀ física y empírica

## Objetivo

Establecer baselines científicos robustos y reproducibles.

Métodos previstos:

- FAO-56 Penman-Monteith;
- Hargreaves-Samani;
- otros métodos empíricos relevantes;
- ET₀ proporcionada por SIAR como referencia/comparación.

Existe trabajo previo en el repositorio, pero debe revisarse antes de considerarlo pipeline científico definitivo.

### Trabajo previsto

- ⚪ auditar scripts legacy;
- ⚪ verificar unidades;
- ⚪ verificar fórmulas;
- ⚪ reproducir ET₀ SIAR;
- ⚪ métricas por estación;
- ⚪ métricas estacionales;
- ⚪ baseline formal.

### Criterio de salida

- métodos reproducibles;
- resultados validados;
- unidades y supuestos documentados;
- métricas consistentes;
- tests con casos conocidos.

Estado:

```text
⚪ pendiente de revisión científica
```

---

# FASE 6 — Machine Learning

## Objetivo

Evaluar si modelos propios pueden estimar/predicir ET₀ con menos variables o mejor comportamiento predictivo.

### Estrategia

- modelos por estación;
- posible modelo global de Baleares;
- 3–4 inputs cuando científicamente tenga sentido;
- baselines simples antes de redes complejas;
- entrenamiento en nube si el servidor doméstico no es suficiente.

### Modelos candidatos

- regresión/baselines;
- árboles;
- boosting;
- redes neuronales;
- modelos híbridos.

### Trabajo previsto

- ⚪ definición formal de features;
- ⚪ splits temporales;
- ⚪ comparación con métodos físicos;
- ⚪ validación espacial;
- ⚪ experiment tracking;
- ⚪ modelos forecast por horizonte.

### Criterio de salida

- ausencia de leakage;
- baseline superado de forma reproducible;
- métricas fuera de muestra;
- generalización temporal estudiada;
- resultados por estación/horizonte.

Estado:

```text
⚪ futuro
```

---

# FASE 7 — Validación predictiva

## Objetivo

Comparar lo que se predijo con lo que finalmente ocurrió.

Dataset objetivo conceptual:

```text
station
issue/capture time
target date
lead time
SIAR/AEMET forecast ET₀
modelo propio forecast ET₀
SIAR observed ET₀
meteorología forecast
meteorología observada
```

Métricas:

- MAE;
- RMSE;
- bias;
- R²;
- métricas por horizonte;
- métricas por estación;
- métricas estacionales.

### Criterio de salida

Poder responder con evidencia:

- qué error tiene SIAR por D+n;
- qué error tiene nuestro modelo por D+n;
- cuándo un modelo supera a otro;
- cómo cambia el error con estación/época/horizonte.

Estado:

```text
⚪ futuro
```

---

# FASE 8 — Producto: Web / API / ETc / riego

## Objetivo

Convertir la infraestructura científica en una herramienta útil.

Arquitectura funcional prevista:

```text
ubicación/estación
      ↓
meteorología
      ↓
ET₀
      ↓
Kc
      ↓
ETc
      ↓
precipitación efectiva
      ↓
necesidad de riego
```

Posibles funcionalidades:

- mapa;
- selección de estación/ubicación;
- históricos;
- forecast;
- comparación SIAR/modelo;
- gráficos;
- calidad de datos;
- cultivo/Kc;
- necesidades netas/brutas;
- API.

Estado:

```text
⚪ futuro
```

---

# Línea transversal — Infraestructura y backups

No constituye una fase final; acompaña a todas las demás.

### Objetivos

- seguridad de secretos;
- backups verificables;
- restore probado;
- monitoring;
- almacenamiento fuera de Git para raw;
- snapshots consistentes;
- Docker reproducible.

### Próximos puntos

- ✅ `INC-INFRA-001` resuelta; `init: true` validado durante 265 checkpoints Git.
- 🟠 backup raw forecast;
- 🟠 proveedor off-site OneDrive o Google Drive;
- ⚪ checksums/manifest;
- ⚪ prueba periódica de restauración.

---

# Ruta crítica actual

A fecha 2026-10-12:

```text
FASE 1
Quality Audit PRE ✅
   ↓
Quality Recovery / POST 🟠
   ↓
certificación observacional 🟠

FASE 2
E2E / LIVE-IB04 #2 ✅
   ↓
infra preflight permanente ✅
   ↓
permanent smoke run 🟠
   ↓
backup off-site
   ↓
scheduler experimental

FASE 3
AEMET Track A: producto municipal aceptado ✅
   ↓
contrato raw empírico / diseño Collector 🟠

Track B: investigar feed AEMET exacto usado por SIAR 🟠
```

Las fases 1, 2 y 3 permanecen abiertas. AEMET Track A no equivale a un forecast puntual de estación SIAR ni demuestra histórico de emisiones.

# Qué NO priorizar todavía

Mientras las capas de datos no estén consolidadas:

- no entrenar modelos predictivos definitivos;
- no rediseñar toda la aplicación web;
- no migrar masivamente los CSV;
- no mover `datos_siar_baleares/`;
- no mover `estaciones_baleares.csv`;
- no hacer una limpieza legacy destructiva;
- no introducir una base de datos como dependencia del raw capturer.

---

# Protocolo de actualización

Actualizar este roadmap únicamente cuando:

- una fase cambie de estado;
- cambie el orden de dependencias;
- aparezca/elimine una línea de trabajo relevante;
- se modifique un criterio de salida.

Los detalles operativos diarios pertenecen a:

```text
docs/PROJECT_STATUS.md
```
