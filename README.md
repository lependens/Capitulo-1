# Capítulo 1 — Proyectos y ampliaciones del TFG

Este repositorio contiene la evolución y ampliación del trabajo realizado originalmente en el TFG, incorporando nuevas herramientas, datos actualizados, automatización, análisis de evapotranspiración de referencia (ET₀), Machine Learning y futuras aplicaciones orientadas a la predicción y al riego.

El proyecto comenzó como una continuación experimental del TFG y ha evolucionado progresivamente hacia una infraestructura de datos y modelado para las estaciones meteorológicas SIAR de las Islas Baleares.

---

# 1. Origen del proyecto

El punto de partida es el TFG original, basado en la estimación de la evapotranspiración de referencia mediante diferentes métodos y modelos de redes neuronales.

A partir de ese trabajo se planteó la posibilidad de:

* migrar los cálculos a Python;
* obtener datos meteorológicos actualizados;
* automatizar el tratamiento de datos;
* comparar métodos de estimación de ET₀;
* desarrollar nuevos modelos de Machine Learning;
* crear herramientas de visualización;
* estudiar la predicción futura de ET₀;
* aplicar posteriormente los resultados a necesidades de riego.

La documentación histórica de esta evolución se conserva en los diferentes archivos `docs_*.md`.

---

# 2. Mapa de documentación

## 2.1 Objetivos

### [1.0 — Objetivos alcanzables](docs_1.0_Objetivos%20alcanzables.md)

Documento que recoge los objetivos iniciales del proyecto y su evolución.

Incluye:

* obtención de datos;
* exploración;
* migración de cálculos;
* modelos de ET₀;
* automatización;
* visualización;
* integración de IA;
* evolución de los objetivos durante 2026.

---

## 2.2 Obtención de datos

### [1.1 — Obtención de datos](docs_1.1_Obtenci%C3%B3n%20de%20datos.md)

Documento dedicado a la evolución del sistema de adquisición de datos SIAR.

Incluye:

* primeras consultas a SIAR;
* identificación de estaciones;
* estaciones de Baleares;
* descargas históricas;
* consultas por períodos;
* formato de los CSV;
* API oficial;
* detección de cobertura;
* detección de huecos;
* actualización automática;
* validación;
* Docker;
* Git;
* SIAR Sync;
* futura recopilación de predicciones.

---

## 2.3 Tratamiento de datos y cálculo de ET₀

### [1.2 — Tratamiento de datos y cálculo de estimaciones](docs_1.2_Tratamiento%20de%20datos%20y%20c%C3%A1lculo%20de%20estimaciones.md)

Documentación del tratamiento de los datos meteorológicos y de la implementación de los cálculos de ET₀.

Incluye el trabajo realizado con:

* depuración;
* preparación de datos;
* Penman-Monteith;
* radiación;
* variables meteorológicas;
* cálculo de Ra;
* cálculo de Rso;
* comparación con ET₀ de SIAR.

---

## 2.4 Análisis de errores

### [1.3 — Análisis errores](docs_1.3_An%C3%A1lisis%20errores.md)

Documentación de la comparación entre diferentes métodos de estimación de ET₀.

Se estudian diferentes métricas y métodos, incluyendo:

* MSE;
* RMSE;
* MAE;
* R²;
* AARE;
* Penman-Monteith;
* Hargreaves;
* variantes de Hargreaves;
* Valiantzas;
* métodos ajustados.

Este documento sirve como continuación del análisis realizado en el TFG y como base para futuras comparaciones con Machine Learning.

---

## 2.5 Redes neuronales

### [1.4 — Modelos redes neuronales](docs_1.4_Modelos%20redes%20neuronales.md)

Documentación de los primeros modelos de redes neuronales desarrollados a partir de los datos meteorológicos.

Esta fase permite conectar el análisis clásico de ET₀ con los posteriores experimentos de Machine Learning.

---

## 2.6 Estimación de ET₀ mediante ANN

### [2.1 — Estimación ET₀ ANN (TensorFlow)](docs_2.1_Estimaci%C3%B3n%20Et0%20ANN%20%28tensorflow%29.md)

Documentación de la implementación de redes neuronales artificiales para estimar ET₀ utilizando Python, TensorFlow y Keras.

Esta etapa representa la transición hacia una infraestructura de Machine Learning más completa.

---

# 3. Estructura conceptual del proyecto

El proyecto actual se organiza en varias capas:

```text
                    PROYECTO SIAR BALEARES
                             │
             ┌───────────────┴───────────────┐
             │                               │
       DATOS HISTÓRICOS                 PREDICCIONES
             │                               │
         SIAR Sync                    Forecast Collector
             │                               │
             └───────────────┬───────────────┘
                             │
                             ↓
                     DATASET CIENTÍFICO
                             │
                ┌────────────┴────────────┐
                │                         │
              ET₀                   MACHINE LEARNING
                │                         │
        Métodos clásicos          ANN / otros modelos
                │                         │
                └────────────┬────────────┘
                             │
                             ↓
                       VALIDACIÓN
                             │
                    ┌────────┴────────┐
                    │                 │
                HISTÓRICA         PREDICTIVA
                    │                 │
                    └────────┬────────┘
                             │
                             ↓
                         WEB / API
                             │
                             ↓
                       ET₀ → Kc → ETc
                             │
                             ↓
                           RIEGO
```

---

# 4. Etapa actual — 2026

Durante 2026 el proyecto ha dejado de centrarse únicamente en scripts independientes y ha comenzado a desarrollar una infraestructura reproducible.

La primera parte consolidada es la adquisición de datos históricos mediante **SIAR Sync**.

Sus funciones principales son:

* conexión con la API oficial;
* autenticación;
* consulta controlada;
* detección de cobertura;
* detección de huecos;
* descarga incremental;
* validación;
* deduplicación;
* ordenación;
* actualización;
* escritura segura;
* automatización;
* control mediante Git.

La adquisición histórica debe mantenerse estable mientras se desarrollan las siguientes etapas.

---

# 5. Próximas etapas

## 5.1 Dataset científico

Convertir los datos históricos en un dataset preparado para análisis y Machine Learning.

Posible arquitectura:

```text
CSV
 ↓
Validación
 ↓
Normalización
 ↓
Parquet
 ↓
DuckDB / Python
```

---

## 5.2 ET₀

Establecer una referencia sólida para ET₀ y reproducir los cálculos necesarios.

Se estudiarán:

* Penman-Monteith;
* Hargreaves;
* variantes;
* otros métodos empíricos;
* ET₀ proporcionada por SIAR.

---

## 5.3 Machine Learning

Desarrollar y comparar modelos capaces de estimar ET₀ utilizando diferentes combinaciones de variables meteorológicas.

Se estudiarán, entre otros:

* redes neuronales;
* modelos de árboles;
* modelos de boosting;
* modelos híbridos;
* modelos locales por estación;
* modelos globales para Baleares.

El objetivo científico será determinar hasta qué punto es posible reproducir la referencia de ET₀ utilizando menos variables meteorológicas.

---

## 5.4 Validación

La evaluación deberá realizarse mediante divisiones temporales y espaciales.

Se estudiarán:

* entrenamiento;
* validación;
* prueba;
* generalización temporal;
* generalización entre estaciones.

Las métricas podrán incluir:

* MAE;
* RMSE;
* R²;
* MBE;
* error relativo;
* comportamiento estacional;
* comportamiento por rango de ET₀.

---

## 5.5 Predicción

Se desarrollará una segunda línea de trabajo dedicada a la predicción de ET₀.

Antes de entrenar modelos propios se estudiará la información disponible en el sistema de predicción de SIAR.

El objetivo será conservar las predicciones emitidas en cada momento para poder compararlas posteriormente con las observaciones reales.

La arquitectura prevista será independiente de SIAR Sync:

```text
SIAR
 ├── SIAR Sync
 │      └── Datos históricos
 │
 └── Forecast Collector
        └── Predicciones
```

---

## 5.6 Aplicación web

Una futura aplicación permitirá consultar los resultados de forma sencilla.

Posibles funcionalidades:

* mapa de Baleares;
* selección de estación o ubicación;
* datos meteorológicos;
* ET₀ histórica;
* ET₀ calculada;
* comparación de modelos;
* predicciones;
* gráficos;
* información de calidad de datos.

---

## 5.7 Necesidades de riego

Una fase posterior permitirá utilizar ET₀ para estimar las necesidades de agua de los cultivos.

La base será:

```text
ETc = ET₀ × Kc
```

Posteriormente podrán incorporarse:

* precipitación efectiva;
* eficiencia del riego;
* superficie;
* volumen de agua;
* necesidades netas;
* necesidades brutas.

---

# 6. Principios del proyecto

## Datos

Los datos originales no deben modificarse artificialmente para eliminar problemas.

Cuando SIAR no proporciona información para una fecha, debe distinguirse entre:

* ausencia real de datos;
* dato pendiente;
* error de consulta;
* error de conexión.

---

## Reproducibilidad

Cada resultado importante debe poder relacionarse con:

* datos utilizados;
* código;
* versión;
* parámetros;
* fecha de ejecución.

---

## Separación de responsabilidades

Los sistemas de adquisición, análisis, Machine Learning y aplicación deben mantenerse suficientemente separados.

Un experimento nuevo no debe poner en riesgo la infraestructura de adquisición histórica.

---

## Validación

Un modelo no debe evaluarse únicamente con los mismos datos utilizados para entrenarlo.

La validación temporal y espacial será una parte fundamental de la metodología.

---

# 7. Estado general

| Área                      | Estado                           |
| ------------------------- | -------------------------------- |
| Datos históricos SIAR     | 🟢 En desarrollo avanzado        |
| SIAR Sync                 | 🟢 Implementado                  |
| Control de cobertura      | 🟢 Implementado                  |
| Validación de datos       | 🟢 Implementado                  |
| Docker                    | 🟢 Implementado                  |
| Git / trazabilidad        | 🟢 Implementado                  |
| ET₀ Penman-Monteith       | 🟢 Implementado inicialmente     |
| Métodos empíricos         | 🟢 Implementado inicialmente     |
| Redes neuronales          | 🟢 Primera etapa implementada    |
| Dataset científico        | 🟡 Próxima etapa                 |
| Nuevos modelos ML         | 🟡 Próxima etapa                 |
| Forecast Collector        | 🟡 Investigación / próxima etapa |
| Predicción propia de ET₀  | ⚪ Futura                         |
| Aplicación web            | ⚪ Futura                         |
| Kc / necesidades de riego | ⚪ Futura                         |

---

# 8. Regla de evolución de la documentación

La documentación de este repositorio debe conservar la evolución histórica del proyecto.

Los documentos antiguos **no se eliminan simplemente porque el proyecto haya evolucionado**.

Cuando una nueva etapa modifica sustancialmente un objetivo o sistema, se añadirá una sección de actualización al documento correspondiente.

Por ejemplo:

```text
docs_1.0
 ├── Objetivos originales
 └── ACTUALIZACIÓN 2026

docs_1.1
 ├── Desarrollo inicial
 └── ACTUALIZACIÓN 2026

docs_1.2
 └── Tratamiento y ET₀

docs_1.3
 └── Análisis de errores

docs_1.4
 └── Redes neuronales

docs_2.1
 └── ANN con TensorFlow
```

De esta forma, GitHub funciona simultáneamente como:

1. repositorio de código;
2. historial del proyecto;
3. documentación técnica;
4. bitácora científica;
5. mapa de navegación hacia las diferentes líneas de trabajo.

---

# 9. Punto de partida para la siguiente etapa

La infraestructura de datos históricos constituye actualmente la base sobre la que se desarrollarán las siguientes fases.

El flujo previsto es:

```text
          SIAR
           ↓
       SIAR Sync
           ↓
    Datos históricos
           ↓
   Dataset científico
           ↓
     ┌─────┴─────┐
     ↓           ↓
    ET₀          ML
     │           │
     └─────┬─────┘
           ↓
      Predicción
           ↓
       Aplicación
           ↓
         Riego
```

El siguiente objetivo práctico es consolidar el dataset y comenzar la nueva fase de modelado y predicción sin comprometer la infraestructura de adquisición ya construida.

---

## Navegación rápida

* **Objetivos:** [`docs_1.0_Objetivos alcanzables.md`](docs_1.0_Objetivos%20alcanzables.md)
* **Obtención de datos:** [`docs_1.1_Obtención de datos.md`](docs_1.1_Obtenci%C3%B3n%20de%20datos.md)
* **Tratamiento y ET₀:** [`docs_1.2_Tratamiento de datos y cálculo de estimaciones.md`](docs_1.2_Tratamiento%20de%20datos%20y%20c%C3%A1lculo%20de%20estimaciones.md)
* **Análisis de errores:** [`docs_1.3_Análisis errores.md`](docs_1.3_An%C3%A1lisis%20errores.md)
* **Redes neuronales:** [`docs_1.4_Modelos redes neuronales.md`](docs_1.4_Modelos%20redes%20neuronales.md)
* **ANN / TensorFlow:** [`docs_2.1_Estimación Et0 ANN (tensorflow).md`](docs_2.1_Estimaci%C3%B3n%20Et0%20ANN%20%28tensorflow%29.md)

---

# 10. Idea central

El proyecto puede resumirse actualmente en una única línea:

> **Construir una infraestructura reproducible de datos meteorológicos de las Islas Baleares que permita estudiar, estimar y predecir ET₀ mediante métodos físicos, empíricos y de Machine Learning, con una futura aplicación práctica orientada al riego.**
