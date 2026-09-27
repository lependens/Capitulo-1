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
4. Mantener el repositorio listo para integración con asistentes de IA o scripts automáticos de optimización.

**Beneficio:** Facilita la ampliación del proyecto y la integración de nuevas técnicas sin rehacer la base de datos ni los scripts existentes.

---

> Siguiendo estos objetivos paso a paso, se logra un flujo de trabajo completo: desde la obtención de datos hasta la creación de modelos y dashboards interactivos, todo en Python, con resultados concretos en cada etapa y posibilidad de expansión futura.
