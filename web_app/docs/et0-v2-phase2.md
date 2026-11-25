# ET₀ V2 · Explorador retrospectivo, fase 2

El menú lateral del explorador abre estaciones, comparaciones, AHC, ML, ANN, diagnóstico temporal, metodología, guía y auditoría. La vista anterior está disponible con `?science=et0-v2&phase1`. La guía incluye siete ejemplos navegables. El parámetro `seccion` permite enlazar cada apartado.

El comparador filtra las filas firmadas por familia, calendario, método y grupo, con un máximo de 12 claves y tres calendarios por consulta. Calcula diferencias descriptivas solo para la misma clave de grupo, la misma huella de cohorte y el mismo n positivo. RMSE y MAE admiten porcentaje respecto a una referencia positiva. Consultar varias estaciones conserva sus poblaciones independientes. La selección vacía muestra el agregado original de las nueve estaciones. Los residuos utilizan exclusivamente los pares diarios entregados por la API, limitada a 367 días; las bandas de la distribución tienen ancho 0,5 mm/día.

## Procedencia y alcance

- `et0-v2-memo.json` reproduce tablas descriptivas de los capítulos 7–9 de la [memoria](../../scientific_pipeline/docs/ET0_V2_MEMORIA_CIENTIFICA.md). Regenerar con `python web_app/ops/extract-v2-memo.py`. Incluye coeficientes de IB01/HG-Rs, selecciones ML y candidatos ANN de IB01/HG-Rs anual, y frecuencias ANN. Conserva el redondeo de la memoria.
- El detalle DEVELOPMENT de otras celdas requiere extracción verificable del árbol congelado. Solicitar a 05 vistas de solo lectura por estación, familia, calendario y grupo AHC, y por algoritmo o arquitectura para ML y ANN, con hash del freeze y de cada ficha de origen. La interfaz identifica esta limitación y permite consultar TEST por esas dimensiones.
- Las predicciones, resultados firmados y catálogo VERIFIED se utilizan en solo lectura. El estado científico es NOT_PUBLISHED.

## Despliegue reversible

Probar un proyecto Compose independiente con puerto libre y la misma configuración de montajes VERIFIED. Ejecutar la comprobación del catálogo, de la API de comparación, de series y de la navegación. Antes de reutilizar 8090, conservar IDs de contenedores, etiquetar las dos imágenes de fase 1 y copiar la configuración Compose. El proyecto previo puede detenerse con `stop`, conservando los contenedores. Para volver a fase 1, detener el proyecto de fase 2, arrancar los dos contenedores previos mediante `docker compose start` con su configuración original y comprobar salud y catálogo en 8090. V0.3 usa un proyecto separado en 8088.
