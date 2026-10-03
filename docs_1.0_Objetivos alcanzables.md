# Objetivos alcanzables

Este documento recoge pequeños objetivos que sean **realistas y alcanzables**, con resultados próximos y fáciles de ejecutar. La idea es crear una base sólida de trabajo en Python, siguiendo la metodología y el hilo del TFG, pero con herramientas modernas y flexibles.

---

## 1. Obtención de datos históricos y actuales de las estaciones meteorológicas de Baleares

**Objetivo:** Conseguir todos los datos históricos y actualizados hasta 2024 de las estaciones meteorológicas de las Islas Baleares de manera automatizada, para almacenarlos en una **base de datos local** en formato CSV.

**Pasos principales:**
1. Identificar fuentes de datos: API del SIAR, archivos históricos CSV/Excel.
2. Descargar datos automáticamente con Python (`requests` o `httpx`).
3. Depurar datos con `pandas` (eliminar nulos, estandarizar columnas y fechas).
4. Guardar los datos en CSV organizados por estación y año.
5. Verificar consistencia con un análisis estadístico preliminar.

**Beneficio:** Base de datos confiable y actualizada lista para análisis y entrenamientos de modelos.

09/09/2025: Con siarID he podido conseguir una lista de todas las estaciones de España, con los siguientes datos:
1. Nombre estación
2. Código estación
3. Nombre provincia
4. Latitud y altitud
5. Fecha inicio datos

---

## 2. Exploración y análisis preliminar de los datos

**Objetivo:** Familiarizarse con los datos y obtener estadísticas básicas de cada variable.

**Pasos principales:**
1. Cargar los CSV en Python con `pandas`.
2. Explorar los datos:
   - Resumen estadístico (`describe()`).
   - Distribución de valores y detección de valores atípicos.
3. Visualizar tendencias temporales simples:
   - Temperatura media anual.
   - Precipitación acumulada.
   - Evapotranspiración estimada.
4. Guardar gráficos preliminares como PNG para documentación.

**Beneficio:** Comprender los datos y detectar problemas antes de entrenar modelos.

---

## 3. Migración de cálculos del TFG a Python

**Objetivo:** Reproducir los cálculos clave realizados en MATLAB del TFG, pero en Python.

**Pasos principales:**
1. Identificar las fórmulas y métodos usados (ET₀, medias, desviaciones, etc.).
2. Implementar scripts en Python usando `numpy` y `pandas`.
3. Comparar resultados con los del TFG original para validar la implementación.
4. Documentar el código y los resultados obtenidos.

**Beneficio:** Tener un código base en Python que replica el TFG original y sirve como punto de partida para mejoras.

---

## 4. Desarrollo de pequeños modelos de ET₀ en Python

**Objetivo:** Crear versiones básicas de redes neuronales para estimar ET₀ usando Python (`scikit-learn`, `TensorFlow` o `PyTorch`).

**Pasos principales:**
1. Preparar datos de entrenamiento y prueba desde la base de datos creada.
2. Implementar un modelo simple (MLP o regresión lineal).
3. Evaluar el modelo con métricas básicas (RMSE, MAE).
4. Guardar el modelo entrenado y los resultados de evaluación.

**Beneficio:** Primera versión funcional de predicción de ET₀ en Python.

---

## 5. Automatización de análisis y generación de reportes

**Objetivo:** Crear scripts que generen análisis automáticos y reportes con gráficos y estadísticas.

**Pasos principales:**
1. Crear funciones que calculen estadísticas y generen gráficos automáticamente.
2. Guardar reportes en PDF o HTML usando `matplotlib`, `seaborn` y `pandas`.
3. Automatizar la ejecución para nuevos datos descargados automáticamente.

**Beneficio:** Flujo de trabajo reproducible y escalable, con resultados visuales listos para documentación.

---

## 6. Visualización interactiva de resultados

**Objetivo:** Explorar los datos y resultados mediante dashboards interactivos.

**Pasos principales:**
1. Crear dashboards simples con `Streamlit` o `Plotly Dash`.
2. Visualizar variables meteorológicas y ET₀ por estación y por año.
3. Incluir filtros dinámicos (por estación, por año, por variable).
4. Añadir comparaciones entre métodos empíricos y modelos de machine learning.

**Beneficio:** Herramienta interactiva para análisis exploratorio y divulgación.

---

## 7. Preparación para integración futura de IA

**Objetivo:** Preparar la infraestructura y scripts para que la IA pueda sugerir mejoras o generar código automáticamente.

**Pasos principales:**
1. Estructurar los scripts y módulos en Python de forma clara y modular.
2. Documentar funciones y procesos.
3. Guardar logs y resultados de cada ejecución.
4. Mantener el repositorio listo para integración con asistentes de IA o scripts automáticos de optimización.---

# ACTUALIZACIÓN 2026 — Evolución de los objetivos

Durante 2026 el proyecto ha evolucionado considerablemente respecto a los objetivos iniciales planteados en 2025.

El objetivo original era disponer de datos meteorológicos de las estaciones SIAR de las Islas Baleares y utilizarlos posteriormente para reproducir los cálculos del TFG y desarrollar modelos de estimación de ET₀.

A partir de ese punto, el proyecto ha pasado a plantear una infraestructura más completa, en la que la obtención de datos, su validación, los modelos de ET₀, la predicción y una futura aplicación web forman parte de un mismo flujo de trabajo.

La idea principal de esta nueva etapa es que cada fase sea reproducible, verificable y pueda servir como base para la siguiente.

---

## 8. Sistema automatizado de obtención y actualización de datos SIAR

### Objetivo

Disponer de un sistema capaz no solamente de descargar datos históricos, sino de mantenerlos actualizados y detectar automáticamente qué información falta.

La obtención de datos deja de ser un proceso manual basado en ejecutar scripts para convertirse en un proceso de sincronización.

### Estado

**En gran parte conseguido.**

Se ha desarrollado un sistema independiente denominado **SIAR Sync**, encargado de trabajar con la API oficial de SIAR y mantener los archivos históricos de las estaciones.

El sistema actualmente contempla:

1. Autenticación con la API de SIAR.
2. Consulta de información de las estaciones.
3. Lectura de los datos meteorológicos diarios.
4. Comprobación de los datos existentes localmente.
5. Detección de la cobertura temporal real.
6. Detección de huecos internos.
7. Identificación de fechas pendientes.
8. Descarga mediante bloques pequeños para evitar superar los límites de la API.
9. Comprobación de los datos recibidos antes de incorporarlos.
10. Eliminación de duplicados.
11. Ordenación cronológica.
12. Validación de la estructura del CSV.
13. Escritura atómica de los archivos.
14. Actualización incremental.
15. Refresco de datos modificados por SIAR.
16. Registro y control de errores.
17. Integración con Git para conservar un historial de cambios.

Este sistema se mantiene separado del resto de experimentos para evitar que las pruebas de nuevos modelos puedan afectar al proceso de adquisición de datos.

---

## 9. Mejora de la calidad e integridad de los datos

### Objetivo

No considerar que un archivo completo en apariencia implica necesariamente que los datos sean completos.

El sistema debe conocer:

* primera fecha disponible;
* última fecha disponible;
* número de registros;
* fechas duplicadas;
* huecos internos;
* fechas pendientes;
* fechas sin datos disponibles en SIAR;
* estructura y orden de las columnas.

### Estado

**Implementado en SIAR Sync.**

Una de las mejoras importantes respecto al sistema inicial ha sido pasar de comprobar únicamente si existe un archivo a analizar su **cobertura temporal real**.

Esto permite distinguir entre:

* un período correctamente cubierto;
* un período con huecos;
* un período que todavía no ha sido descargado;
* una fecha para la que SIAR no devuelve datos.

Esta distinción es especialmente importante para evitar descargar repetidamente períodos que ya están completos y, al mismo tiempo, evitar asumir que una fecha sin datos significa necesariamente que el sistema de descarga ha fallado.

---

## 10. Control de límites de la API de SIAR

### Objetivo

Adaptar las consultas a los límites establecidos por SIAR y evitar que el proceso dependa de valores fijados manualmente.

### Estado

**Implementado.**

Se ha trabajado con la información proporcionada por la propia API para conocer sus límites de acceso y de registros.

El sistema utiliza esta información para diseñar las consultas de forma conservadora.

Además, las descargas se realizan en bloques reducidos, actualmente con un máximo de siete días por bloque.

Esto permite:

* reducir el riesgo de superar los límites;
* repetir únicamente el período que haya fallado;
* controlar mejor los errores;
* comprobar los datos recibidos antes de modificar el archivo local.

---

## 11. Automatización y ejecución continua

### Objetivo

Conseguir que la actualización de datos pueda ejecutarse automáticamente sin intervención manual.

### Estado

**Implementado.**

SIAR Sync se ha integrado en un entorno Linux mediante Docker.

La arquitectura actual permite mantener separado:

* el código;
* la configuración;
* las credenciales;
* el proceso de sincronización;
* los datos;
* el repositorio Git.

Esto supone un cambio importante respecto a los primeros scripts desarrollados durante 2025, que estaban pensados principalmente para ejecutar consultas concretas desde un ordenador personal.

---

## 12. Control de versiones y trazabilidad

### Objetivo

Conservar un historial de las modificaciones realizadas en los datos y en el código.

### Estado

**Implementado.**

Los datos y el código se mantienen bajo control de versiones mediante Git.

Cada actualización automática puede generar un commit con los cambios producidos.

De esta forma se puede conocer:

* cuándo se incorporaron nuevos datos;
* qué archivos cambiaron;
* qué versión del código realizó la actualización;
* qué estado tenía el repositorio en cada momento.

También se ha incorporado control ante errores temporales de conexión con GitHub, de forma que un fallo de red o TLS no provoque la pérdida de los datos descargados localmente.

---

## 13. Separación entre adquisición de datos y análisis

Una decisión importante tomada durante 2026 ha sido separar claramente la adquisición de datos del resto del proyecto.

La arquitectura conceptual pasa a ser:

```text
SIAR
  │
  ├── SIAR Sync
  │       │
  │       └── Datos históricos
  │
  └── Futuro Forecast Collector
          │
          └── Predicciones
```

SIAR Sync debe mantenerse estable y dedicado principalmente a los datos históricos.

Las futuras pruebas relacionadas con las predicciones de SIAR se desarrollarán mediante un proceso independiente.

Esto permite experimentar con nuevas funcionalidades sin poner en riesgo el sistema de adquisición histórica.

---

## 14. Preparación de un dataset científico

### Objetivo

Pasar de disponer de una colección de archivos CSV a disponer de un dataset preparado específicamente para análisis científico y aprendizaje automático.

La arquitectura prevista será:

```text
Datos SIAR
    ↓
CSV histórico
    ↓
Validación y normalización
    ↓
Dataset científico
    ↓
Parquet / DuckDB
    ↓
Análisis y Machine Learning
```

El CSV se conserva como formato sencillo, auditable y directamente legible.

Para fases posteriores se plantea utilizar formatos columnares como Parquet y herramientas analíticas ligeras como DuckDB para trabajar con grandes cantidades de datos de forma más eficiente.

---

## 15. Nueva etapa de Machine Learning

El objetivo inicial de utilizar redes neuronales para ET₀ se amplía.

Ya no se plantea únicamente reproducir los modelos del TFG, sino estudiar de forma sistemática diferentes métodos:

* Penman-Monteith como referencia;
* Hargreaves;
* variantes de Hargreaves;
* otros métodos empíricos;
* redes neuronales;
* otros modelos de Machine Learning.

Una de las líneas principales será estudiar si es posible estimar ET₀ utilizando un número reducido de variables meteorológicas manteniendo un error suficientemente bajo respecto al método de referencia.

---

## 16. Validación temporal y espacial

La validación de los modelos deberá realizarse evitando que los datos de entrenamiento y prueba representen exactamente las mismas condiciones temporales.

Se plantea utilizar:

* división temporal de los datos;
* períodos de entrenamiento;
* períodos de validación;
* períodos de prueba;
* validación por estaciones;
* validación dejando una estación fuera del entrenamiento.

El objetivo es distinguir entre:

**capacidad para reproducir datos conocidos**

y

**capacidad para generalizar a períodos o estaciones no utilizados durante el entrenamiento.**

---

## 17. Predicción de ET₀

Una nueva línea de trabajo será estudiar la predicción de ET₀ para días futuros.

Se plantea comparar:

1. ET₀ observado posteriormente.
2. Predicción oficial disponible mediante SIAR.
3. Predicción meteorológica utilizada como entrada.
4. Modelo propio de Machine Learning.

La comparación deberá realizarse para diferentes horizontes de predicción.

Antes de desarrollar este modelo será necesario estudiar cómo se obtienen y almacenan las predicciones de SIAR y conservar las predicciones históricas sin sobrescribirlas posteriormente.

---

## 18. Aplicación web

Una vez que los datos y modelos sean suficientemente estables, se plantea desarrollar una aplicación web para las Islas Baleares.

La aplicación podría permitir:

* seleccionar una estación;
* consultar datos meteorológicos;
* consultar ET₀;
* comparar diferentes métodos;
* visualizar históricos;
* consultar predicciones;
* representar información geográfica;
* incorporar posteriormente información de cultivos.

La aplicación web se considera una etapa posterior al desarrollo y validación científica de los modelos.

---

## 19. Integración con necesidades de riego

Una posible etapa posterior consiste en utilizar ET₀ para estimar las necesidades hídricas de cultivos.

La relación fundamental será:

```text
ETc = ET₀ × Kc
```

donde:

* ET₀ = evapotranspiración de referencia;
* Kc = coeficiente de cultivo;
* ETc = evapotranspiración del cultivo.

Posteriormente podrían incorporarse:

* precipitación efectiva;
* eficiencia del sistema de riego;
* superficie cultivada;
* volumen de agua;
* necesidades netas y brutas.

Esta parte se desarrollará separadamente de los datos meteorológicos para mantener una arquitectura modular.

---

## 20. Objetivo general de la nueva etapa

El proyecto deja de considerarse únicamente como una colección de scripts para analizar datos meteorológicos.

El objetivo pasa a ser construir un flujo completo:

```text
Obtención de datos
       ↓
Validación
       ↓
Dataset científico
       ↓
ET₀ de referencia
       ↓
Métodos empíricos
       ↓
Machine Learning
       ↓
Validación
       ↓
Predicción
       ↓
Aplicación web
       ↓
Necesidades hídricas
       ↓
Aplicación práctica al riego
```

Cada etapa deberá poder validarse independientemente antes de utilizarse como entrada de la siguiente.

---

## Estado general del proyecto en 2026

| Área                                   | Estado                    |
| -------------------------------------- | ------------------------- |
| Obtención inicial de estaciones SIAR   | Completado                |
| Descarga histórica de datos            | Completado / en evolución |
| Actualización automática               | Implementado              |
| Detección de huecos                    | Implementado              |
| Validación de archivos                 | Implementado              |
| Automatización con Docker              | Implementado              |
| Control mediante Git                   | Implementado              |
| Cálculo de ET₀ en Python               | Implementado inicialmente |
| Comparación de métodos empíricos       | Implementado inicialmente |
| Redes neuronales para ET₀              | Implementado inicialmente |
| Dataset científico unificado           | Próxima etapa             |
| Predicción de ET₀                      | Próxima etapa             |
| Recopilación histórica de predicciones | Próxima etapa             |
| Aplicación web definitiva              | Futura etapa              |
| Integración Kc / riego                 | Futura etapa              |

---

> **Conclusión de la actualización 2026:**
> El proyecto ha pasado de una primera fase centrada en la exploración y obtención de datos a una arquitectura más completa basada en adquisición automatizada, control de calidad, trazabilidad, análisis de ET₀, Machine Learning y futuras aplicaciones de predicción y riego.


**Beneficio:** Facilita la ampliación del proyecto y la integración de nuevas técnicas sin rehacer la base de datos ni los scripts existentes.

---

> Siguiendo estos objetivos paso a paso, se logra un flujo de trabajo completo: desde la obtención de datos hasta la creación de modelos y dashboards interactivos, todo en Python, con resultados concretos en cada etapa y posibilidad de expansión futura.
