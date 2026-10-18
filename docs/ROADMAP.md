# ROADMAP — SIAR / ET₀

**Última actualización:** 2026-10-18\
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

## Visión de conjunto — trabajo paralelo

Las fases 1, 2 y 3 están abiertas y avanzan en paralelo; no forman una cadena en la que cada adquisición deba esperar a la anterior.

| Workstream | Trabajo actual | Siguiente salida |
|---|---|---|
| 01 — Observaciones / Quality | R2 autónoma → POST → Quality closeout | Dataset Observacional v1 |
| 02 — Forecast SIAR / infraestructura | scheduler + backup/DR | captura prospectiva periódica validada |
| 04 — AEMET | confirmación E2E final | despliegue prospectivo + backup |

**No esperamos a terminar modelos para empezar a capturar forecasts.** Las emisiones prospectivas se conservan cuando cada capturador supera sus gates operativos, independientemente del calendario de los modelos. Dataset Observacional v1 depende del cierre Quality, no de terminar los despliegues forecast.

Después, la secuencia científica es **Dataset v1 → FAO-56 PM → Hargreaves → baselines → ML**, seguida de validación predictiva y producto Web/API/ETc/riego. Las fases 4–8 desarrollan esa evolución.

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
- 🟠 Quality Recovery: R1 superado como etapa histórica; R2 en ejecución autónoma; POST pendiente. Los 2.346 huecos y sus estados pertenecen al snapshot PRE inmutable, no al estado actual de R2;
- 🟠 CATALOG-TZ: discrepancia sistemática snapshot local 31 frente a Info/ESTACIONES 30, pendiente de investigación semántica; no corregir automáticamente `estaciones_baleares.csv` ni usar el campo científicamente hasta conocer su significado. Por sí sola no degrada PRE ni bloquea certificación salvo evidencia de efecto en periodo/semántica;
- ✅ saneamiento operativo posterior a v2.3.2: 12/12 CSV con modo 0644, SHA-256 PRE = POST 12/12, contenido raw intacto;
- ✅ v2.3.2 desplegada y validada; lógica correctora de permisos incluida desde esa versión;
- ✅ v2.3.3 desplegada y validada: SIAR_CA_BUNDLE opcional con verificación TLS activa;
- ✅ v2.3.3 canónica en GitHub tras PR #4;
- ✅ INC-INFRA-001 resuelta y init: true canónico.

IB09 tuvo resultado FAIL en PRE por la observación 2005-10-23 anterior a la Fecha_Instalacion oficial 2005-10-24 (Fecha_Baja 2018-10-10). Se preserva la evidencia histórica; el resultado POST aún no existe.

El antecedente R1 de IB07 ya no define una pausa operativa vigente: R2 avanza autónomamente. Un `INVALID_RESPONSE` aislado se trata en el bloque o estación afectados y no bloquea globalmente R2. Esta actualización no infiere dictámenes nuevos sobre filas raw ni genera POST.

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
🟠 PRE completo; R1 histórico; R2 en ejecución; POST y Quality closeout pendientes
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
- ✅ `SMOKE-FC-001 = PASS` (IB04 + Aguacate; run/capture completos, TLS, hashes/schemas, permisos y SGID validados; HTML sensible omitido; raw/tmp vacío);
- ✅ `BACKUP-FC-001 = PASS` (rclone 1.75.2, Google Drive + crypt, copy, restore independiente, SHA256SUMS, schemas, ZIP/CSV y COMPLETE remoto validados);
- 🟠 confirmar recuperabilidad del material crypt fuera del servidor; completar `FC-STATIONS-001`, `FC-MULTI-SMOKE-001` y backup del multi-smoke; scheduler experimental después;

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

El código actual está en `forecast_collector/`. Runtime permanente, `SMOKE-FC-001` y `BACKUP-FC-001` están validados; el scheduler periódico aún NO está activo. La cobertura y el backup de ampliaciones deben verificarse antes de automatizarlas.

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
- raíz operativa `/srv/siar-forecast/raw`, aprovisionada y validada;
- frecuencia propuesta para scheduler experimental futuro (scheduler = NO): 08:00 / 14:00 / 20:00 Europe/Madrid.

### Infraestructura permanente y siguiente gate

`SMOKE-FC-001 = PASS` y `BACKUP-FC-001 = PASS`. Runtime permanente como `siar-forecast`; release funcional `e850827e24cff11ae46f51c9bc3e8a2f36b0d2da`, Python 3.14.4. Infraestructura física validada:

- usuario de servicio siar-forecast;
- /srv/siar-forecast/raw/{runs,tmp} creado con permisos base y SGID validados;
- patrón de releases en /opt/siar-forecast;
- configuración en /etc/siar-forecast;
- TLS bundle permanente validado;
- runtime independiente del home validado.

Resultados y gates siguientes:

- `/opt/siar-forecast` y `/etc/siar-forecast` operativos; `/srv/siar-forecast/raw` aprovisionado y validado;
- smoke: TLS verificado; hashes/schemas PASS; permisos `2770/0660` + SGID PASS; `html_persistence = omitted_sensitive`; raw/tmp vacío;
- backup: contenido, nombres y directorios cifrados con rclone crypt; `rclone copy` (no `sync`); restore independiente, SHA256SUMS, schemas, ZIP/CSV y COMPLETE remoto PASS;
- 🟠 confirmar material crypt recuperable fuera del servidor; completar `FC-STATIONS-001`, `FC-MULTI-SMOKE-001` y backup del multi-smoke; scheduler experimental después;
- ⚪ determinar cadencia real, semántica D+N y parser/normalización posterior.

Scheduler periódico = NO. El workstream 02 continúa con scheduler, cobertura operativa y backup/DR; el PASS de la prueba de backup no cierra por sí solo la recuperación ante pérdida total del servidor. La FASE 2 permanece abierta.

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

- ✅ **Track A — Collector fusionado:** PR #18 incorporó `aemet_forecast_collector/`; PR #19 añadió canonicalización municipal y PR #20 permisos POSIX de ancestros.
- ✅ `AEMET-A1 = PASS`; `AEMET-A2 = PASS`; `AEMET-A3 = COMPLETE`.
- 🟠 Confirmación E2E final pendiente por timeout operacional. No demuestra un defecto del Collector ni revoca los gates anteriores; siguiente trabajo: confirmar E2E → despliegue prospectivo + backup.
- 🟠 **Track B = OPEN:** no se ha demostrado el feed AEMET exacto utilizado por SIAR; su investigación es independiente de Track A.

El payload observado contiene `id = 7040` (entero JSON) y la configuración usa `"07040"`; solo se canonicaliza para comparar, manteniendo raw intacto. La política POSIX es `2770` para `data_root/runs/YYYY/MM/DD` y árbol publicado, `0660` para ficheros, SGID en directorios y `other = 0`.

Límites científicos de Track A:

- el producto municipal representa la capital municipal, no un forecast puntual SIAR;
- por sí solo no permite FAO-56 Penman–Monteith completo;
- no se ha demostrado recuperación de emisiones históricas;
- no tiene todavía horizonte D+n ni timezone de emisión científicamente asignados.

La implementación ya no es un trabajo pendiente. Los gates de E2E final, despliegue, backup y operación prospectiva mantienen la FASE 3 abierta.

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
🟠 Track A fusionado; E2E final y operación prospectiva pendientes; Track B OPEN
```

---

# FASE 4 — Dataset Observacional v1 y datasets científicos

## Objetivo

Separar adquisición raw de datos preparados para investigación. La primera entrega es Dataset Observacional v1, después de R2, POST y Quality closeout. No requiere esperar a que los forecasts operen periódicamente; sus emisiones se acumulan en paralelo para comparativas posteriores.

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

- 🟡 definir Dataset Observacional v1 con trazabilidad raw/Quality y criterios de inclusión explícitos;
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

Orden de baselines después de Dataset v1:

1. FAO-56 Penman–Monteith.
2. Hargreaves-Samani.
3. Comparación y validación formal de baselines antes de ML.

Métodos y referencias adicionales:

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
- ✅ `BACKUP-FC-001 = PASS`: rclone 1.75.2, Google Drive crypt, `copy`, restore independiente y checksums validados;
- 🟠 confirmar recuperabilidad del material crypt fuera del servidor;
- 🟠 backup del multi-smoke antes de scheduler;
- ⚪ backup periódico y su verificación;

---

# Rutas actuales en paralelo

A fecha 2026-10-18:

| Línea | Gates vigentes | Resultado buscado |
|---|---|---|
| 01 | PRE completo; R1 histórico; R2 autónoma → POST → Quality closeout | Dataset Observacional v1 |
| 02 | smoke y backup PASS; scheduler + cobertura y backup/DR pendientes de cierre | captura prospectiva SIAR periódica |
| 04 | Collector, canonicalización y permisos fusionados; A1/A2 PASS, A3 COMPLETE → confirmación E2E final | despliegue prospectivo AEMET + backup |

Un `INVALID_RESPONSE` aislado no bloquea globalmente R2. La auditoría no interpola ni reescribe CSV raw; Recovery solo incorpora registros reales de SIAR mediante el mecanismo canónico, sin imputar, borrar ni editar manualmente observaciones. CATALOG-TZ sigue como investigación semántica separada y no bloqueante por sí sola.

Los workstreams avanzan autónomamente con objetivos, invariantes y STOP fijados por dirección. Resuelven los errores operativos en su propio ámbito y escalan arquitectura, seguridad, integridad raw o semántica científica.

La captura de forecasts **no espera a los modelos**. El camino científico posterior es Dataset v1 → FAO-56 PM → Hargreaves → baselines → ML; después se abordarán validación predictiva y producto. Las fases 1, 2 y 3 siguen abiertas; ningún timeout operacional se registra como revocación automática de hitos ya demostrados.

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
