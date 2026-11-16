# Web 2.0 · integración provisional de ET₀ V2

La vista previa implementa la primera comparación de AHC, ML clásico, ANN A y ANN E junto con la fórmula original. Está en español y usa ejemplos ficticios. La interfaz tiene filtros de fórmula y sus entradas, calendario del modelo, método y grupo. Permite ordenar métricas comparables, consultar dos grupos en paralelo y ver ET₀ diaria frente a PM_FAO56_V1, una referencia calculada. El AHC anual de histórico completo tiene una vista propia que identifica el ajuste y la evaluación dentro de muestra.

## Abrir la vista previa

Desde la raíz del repositorio, configurar `WEB_DATA_DIR` con un derivado WEB existente y verificado. Activar `WEB_ET0_V2_PREVIEW=1` solo en el entorno local. Arrancar `python -m uvicorn web_app.backend.app:app --factory --port 8001` y, desde `web_app/frontend`, ejecutar `npm run dev -- --host 127.0.0.1`. Abrir `http://127.0.0.1:5173/?preview=et0-v2`. Los ejemplos de la API viven en el código y su activación es explícita. Los resultados reales requieren un adaptador posterior y el catálogo final de 05.

## API provisional de solo lectura

| GET bajo `/api/et0-v2` | Filtros | Contenido |
|---|---|---|
| `/catalog` | — | Versión, referencia, periodo, fórmulas, entradas, métodos, estaciones y claves de agrupación |
| `/metrics` | `formula`, `calendar`, `group`, `key`, `method` repetible | Una población por respuesta, con n, MAE, RMSE, bias, R², cobertura de catálogo e identidad de cohorte |
| `/series` | `formula`, `calendar`, `station`, `start`, `end` inclusivos | Días de la estación con referencia y estimaciones. Un valor ausente conserva el hueco visible |
| `/in-sample` | `formula`, `station` | AHC anual dentro de muestra con coeficiente, métricas y pares por partición |

Cada respuesta identifica la versión y el alcance. Los errores de contrato devuelven 422 y las selecciones válidas sin información devuelven listas vacías. La serie de prueba contiene ocho días ficticios de Hargreaves Rs anual en DEMO-01. La petición provisional permite hasta 367 días inclusivos. Dimensionar el límite de producción con el Parquet final y pruebas de navegación en móvil. El separador de las curvas y la tabla accesible usan el mismo conjunto de puntos. Las métricas de ejemplo se definen de forma independiente de los puntos.

## Correspondencia con el productor y puerta de publicación

El código productor leído en la revisión `3357f4f` define grupos global, estación, año, estación-año, mes y estación climática. Anual, estacional y mensual describen el calendario del ajuste y son un eje distinto. El error usa claves comunes dentro de cada fórmula y calendario. Los campos `common_keys_sha256`, `n` y el contexto de la fila determinan si se permite ordenar varios métodos juntos. La cobertura usa observaciones propias del método y se muestra sobre días de catálogo. El productor también emite cobertura respecto a días con PM disponible. Incluir su denominador, nombre y valor en la ficha de detalle una vez confirmado el catálogo. Leer las columnas exactas de la publicación antes de mapearlas al contrato web. El código original incluye algoritmos ML individuales además de ML principal, que es el método incluido en esta prueba. [Productor retrospectivo](https://github.com/lependens/Capitulo-1/blob/3357f4f/scientific_pipeline/src/siar_science/et0_v2_retrospective.py) · [Ejecución documentada](https://github.com/lependens/Capitulo-1/blob/3357f4f/scientific_pipeline/docs/et0_v2_execution.md).

Dirección informa de 2.081.376 registros de predicción verificados y 9.612 filas de métricas. La unidad física de almacenamiento y los esquemas reales se cotejarán con el catálogo de 05. Bajo `/srv/siar-project/scientific-artifacts/et0-v2/` se han comunicado `original/retrospective-20261116/`, `derived/v1/` y `catalog/`. Tratarlos como ubicaciones previstas hasta que 05 certifique publicación, integridad y restauración. La prueba no abre esas rutas.

Para integrar la publicación, validar el catálogo y sus hashes, fijar versión, referencia, identidad del dataset, manifest original y SHA del freeze. Montar el derivado publicado de solo lectura. Resolver archivos exclusivamente desde un inventario permitido y usar consultas parametrizadas. Cotejar una muestra de la API con los archivos de 05, incluidos cobertura, n, nulos, sesgo, R² y huellas de clave. El cambio de versión debe dejar registrada la configuración anterior y permitir restaurarla con sus archivos e imágenes. El freeze se trata como evidencia de procedencia. La inferencia sobre meteorología nueva tendrá su propio módulo.

## Decisiones y necesidades para Dirección

- La etapa base puede diseñarse sobre las tablas documentadas. Confirmar con 05 el inventario final con esquema, versiones, alcance de cada tabla y campos de fechas, grupo y huella de cohorte. La vista de años comparará tablas en paralelo con sus respectivas poblaciones.
- Para comparar algoritmos con inputs de fórmulas distintas hace falta una cohorte común publicada entre esas fórmulas. Mientras no exista, el ranking se limita a la misma fórmula, el mismo calendario y la misma cohorte. El filtro por inputs selecciona la fórmula equivalente.
- Solo si se priorizan métricas estación × mes, estación × estación climática o de intervalos arbitrarios, pedir a 05 que confirme si se publican esos grupos. Si faltan, acordar claves, denominadores, soporte y método para publicar los agregados. La serie diaria sí admite recortes de fecha.
- Reservar el mapa de estaciones, fichas imprimibles completas, resúmenes meteorológicos, correlaciones y gráficos de residuales para una etapa posterior. Reutilizar el catálogo de estaciones para mapa y fichas, sujeto a disponibilidad de coordenadas y procedencia. Al priorizar los gráficos adicionales, pedir solo los valores que falten con método y trazabilidad.
