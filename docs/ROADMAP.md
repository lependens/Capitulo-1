# ROADMAP — SIAR / ET₀

**Última actualización:** 2026-10-10  
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

- 🟠 completar el backfill restante;
- ✅ v2.3.2 desplegada y validada: fechas civiles de instalación/baja, tratamiento de estaciones reanudadas, checkpoint al alcanzar cuota y permisos finales `0644`;
- ✅ v2.3.3 desplegada y validada: `SIAR_CA_BUNDLE` opcional con verificación TLS activa;
- ✅ v2.3.3 canónica en GitHub tras PR #4;
- ✅ `INC-INFRA-001` resuelta y `init: true` canónico.

La ejecución actual de SIAR Sync ha terminado por ahora. `datos_siar_baleares/` contiene exclusivamente los diez CSV canónicos IB01–IB10; IB101 e IB11 todavía no tienen CSV canónico. La validación/completitud científica de las estaciones sigue pendiente. El TLS no es un bloqueador inmediato.

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

Estado:

```text
🟠 en curso avanzado
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

El código actual está en `forecast_collector/`. Aún no hay prueba live contra SIAR ni despliegue.

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

### Trabajo pendiente

- 🟠 cerrar con 04W el contrato actual de `#necesidadesNetasForm` y clasificar sus campos como `required`, `optional` o `UI-only` antes del live test;
- 🟠 preparar runbook y realizar prueba live manual con IB04;
- 🟠 validar el flujo contra SIAR real;
- 🟠 acordar con 02 el almacenamiento definitivo `/srv` y los permisos;
- 🟠 definir y probar backup off-site;
- 🟠 desplegar después de superar el gate live y de infraestructura;
- 🟠 observar 7–14 días tras el despliegue;
- ⚪ determinar cadencia real;
- ⚪ determinar semántica D+N;
- ⚪ decidir frecuencia definitiva;
- ⚪ parser/normalización posterior.

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

### Investigación necesaria

- 🟠 identificar producto AEMET actual;
- 🟠 API/automatización;
- ⚪ resolución espacial;
- ⚪ resolución temporal;
- ⚪ horizonte;
- ⚪ frecuencia de actualización;
- ⚪ autenticación/límites;
- ⚪ posibilidad de histórico de forecast;
- ⚪ correspondencia espacial con estaciones SIAR;
- ⚪ conservación de cada emisión.

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

A fecha 2026-10-10:

```text
v2.3.3 canónica en GitHub, desplegada y validada en producción
   ↓
reanudar/completar backfill SIAR
```

En paralelo:

```text
runbook + prueba live manual IB04
   ↓
validación contra SIAR real
   ↓
almacenamiento `/srv` + backup off-site
   ↓
despliegue y observación
```

También en paralelo:

```text
AEMET
   ↓
inputs futuros
```

Después convergen en:

```text
Dataset científico
   ↓
ET₀ + ML
```

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
