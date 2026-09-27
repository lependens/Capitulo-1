# Capítulo 1: Proyectos y Ampliaciones del TFG

Tras subir a GitHub el TFG y revisar con detalle la metodología utilizada, en este capítulo se exploran posibles proyectos que podrían realizarse a partir de los conocimientos adquiridos. Además, se considera cómo la guía de la Inteligencia Artificial puede facilitar el aprendizaje de nuevos programas y técnicas, potenciando la aplicabilidad del trabajo original.

## 1.1 Objetivo del capítulo
El objetivo de este capítulo es identificar oportunidades de expansión y mejora del TFG mediante:

- La implementación de herramientas en nuevos lenguajes (por ejemplo, Python).  
- La conexión con APIs externas para obtener datos actualizados.  
- La automatización del preprocesamiento y análisis de datos.  
- El uso de IA para sugerir optimizaciones y nuevas técnicas de modelado.  

## 1.2 Ideas de proyectos derivados

1. **Herramienta en Python para conexión con la API de SIAR**
   - **Objetivo:** Obtener datos meteorológicos y agrícolas actualizados de las Islas Baleares.
   - **Funcionalidades:**
     - Descarga de datos históricos y en tiempo real.
     - Limpieza y depuración de los datos.
     - Cálculos preliminares como medias, desviaciones y tendencias.
     - Exportación de los datos en formatos compatibles con análisis posteriores.
   - **Ventaja:** Permite reproducir y ampliar los análisis del TFG con datos más recientes y bajo un lenguaje de programación más flexible que MATLAB.

2. **Reentrenamiento de modelos de ET₀ en Python**
   - Migración de las redes neuronales implementadas en MATLAB a Python (`TensorFlow` o `PyTorch`).
   - Comparativa de resultados entre MATLAB y Python.
   - Posibilidad de experimentar con arquitecturas más modernas (redes profundas, LSTM, etc.).

3. **Visualización interactiva de resultados**
   - Desarrollo de dashboards con `Plotly Dash` o `Streamlit` para:
     - Explorar datos meteorológicos y resultados de modelos.
     - Comparar distintos métodos de estimación de ET₀.
     - Generar informes automáticos y personalizables.

4. **Predicción de ET₀ con aprendizaje incremental**
   - Implementación de un sistema que actualice los modelos a medida que se obtienen nuevos datos, permitiendo que el modelo “aprenda” en tiempo real.

5. **Ampliación hacia predicción de otros indicadores agrícolas**
   - Estimación de necesidades hídricas de cultivos específicos.
   - Integración con modelos de riego inteligente o de planificación agrícola.

6. **Automatización de procesos con IA**
   - Uso de IA como asistente de programación para:
     - Generar código automáticamente.
     - Optimizar scripts de análisis de datos.
     - Aprender nuevas técnicas estadísticas y de modelado.

7. **Comparativa de modelos clásicos vs. machine learning**
   - Implementación de modelos empíricos tradicionales (Hargreaves, Penman-Monteith) en Python.
   - Comparación de resultados con redes neuronales y otros algoritmos de machine learning.

8. **Análisis de sensibilidad de variables**
   - Determinar cuáles factores climáticos tienen mayor influencia sobre ET₀.
   - Implementar visualizaciones y métricas de importancia de variables.

9. **Creación de un repositorio educativo**
   - Documentar paso a paso la migración de MATLAB a Python.
   - Generar notebooks interactivos para enseñar la metodología a otros estudiantes o investigadores.

10. **Exploración de datasets globales**
    - Aplicar el mismo enfoque a otras regiones fuera de las Islas Baleares.
    - Comparar resultados y adaptar los modelos a nuevas condiciones climáticas.

11. **Simulación y pronóstico a futuro**
    - Integración de modelos de predicción climática para estimar ET₀ bajo distintos escenarios de cambio climático.
    - Visualización de proyecciones en mapas y gráficos interactivos.

12. **Aplicación móvil o web para ET₀**
    - Desarrollo de una aplicación que muestre ET₀ diario, alertas de sequía o recomendaciones de riego.
    - Integración con APIs externas y modelos entrenados en Python.

## 1.3 Consideraciones finales
Estos proyectos permiten no solo actualizar y mejorar el TFG original, sino también explorar nuevas competencias técnicas y metodológicas. La combinación de análisis meteorológico, machine learning y automatización de procesos abre un campo amplio de aplicaciones prácticas y académicas.

---
