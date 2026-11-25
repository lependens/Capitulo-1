# ET₀ V2 · Explorador retrospectivo

La aplicación reúne estaciones, comparador, AHC, ML, ANN, diagnóstico temporal, metodología, guía y auditoría. La interfaz de fase 1 está disponible con `?science=et0-v2&phase1`. El parámetro `seccion` identifica un apartado. Los ejemplos abren filtros preparados.

## Procedencia y contratos

- TEST 2020–2025 procede de `retrospective-metrics.csv` y las particiones Parquet del catálogo retrospectivo VERIFIED. SHA-256 del catálogo `fef85f0a6ae981792d867a9de2b10b9f7787bc680461820d2284cd20453df4ab`.
- [Contrato del handoff #82](../../scientific_pipeline/docs/et0_v2_experimental_metadata.md). DEVELOPMENT 2004–2019 procede del catálogo complementario con SHA-256 `3207fe042abb004b7cd69122339845adc8d55cd65b5cfd4d676d9e5512e6eea9`. La copia autorizada para lectura reside en `/home/josep/siar-et0-v2-development-prep/data` y procede de `/srv/siar-project/scientific-artifacts/et0-v2-experimental-metadata-v1`. El original tiene permisos restrictivos. Su copia usa un directorio padre privado, CSV y catálogo sin escritura y montaje de solo lectura. Snapshot `backup_20261125T123805Z_0944177f`.
- Al arrancar, el backend coteja SHA-256, tamaño, esquema, claves únicas y recuento de cada CSV. Enlaza ambos catálogos por identidad Dataset/freeze y cada ficha con TEST por estación, familia, calendario y método congelado. Las consultas a las tablas requieren estación y familia, se paginan y los folds ML/ANN requieren candidato. Los tres casos mensuales de IB101 carecen de soporte para ML y ANN. AHC conserva sus grupos publicados.
- Las tablas de candidatos y fold representan DEVELOPMENT. La tabla `ahc-history-in-sample.csv` se consulta en un panel independiente. ANN agrupa el ensemble una vez por candidato y fold y permite consultar cada semilla por separado. Los avisos ML conservan el significado del artefacto de origen.
- La comparación de TEST conserva filas del catálogo y muestra diferencias únicamente para el mismo grupo, huella de cohorte y n positivo. Comparar años distintos es un análisis descriptivo. El porcentaje de mejora se reserva a RMSE y MAE con referencia positiva.
- Las medias meteorológicas son descriptivas, calculadas sobre valores disponibles de cada variable en el `observations.parquet` ya verificado. La vista mensual agrupa por mes del calendario a lo largo del intervalo. La línea de referencia utiliza la media de todo el histórico de la estación. Para series largas se limita el gráfico a 1.000 bloques de observaciones, manteniendo el intervalo entero. La dispersión usa pares diarios y muestreo visual acotado. Las métricas congeladas permanecen intactas.
- La costa local es un recorte simplificado de la capa [Land 1:10 millones de Natural Earth 5.1.1](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/), dominio público, en WGS84. Límites longitud 0,95–4,45° E, latitud 38,55–40,22° N. La costa y las coordenadas SIAR usan la misma transformación lineal. La vista funciona sin teselas externas.

## Seguimiento del Word de Josep

| Observación | Modificación realizada | Sección | Estado | Evidencia de revisión |
| --- | --- | --- | --- | --- |
| Fondo de islas que coincida con estaciones | Costa georreferenciada y marcadores SIAR | Estaciones | Implementada | Seleccionar IB01, IB08 o IB11 en el mapa |
| Medias de variables por estación | Medias históricas con número de días válidos por variable | Estaciones | Implementada | Tabla bajo el mapa |
| Temperatura y humedad a la vez y segunda variable | Selección múltiple simultánea y dispersión de dos variables por pares válidos | Estaciones | Implementada | Marcar temperatura y humedad, alternar Segunda variable |
| Serie histórica 2009–2020 | Consulta de todo el intervalo, con agrupación visual documentada para series largas | Estaciones | Implementada | Introducir 01/01/2009 a 31/12/2020 y pulsar Aplicar fechas |
| Medias por año y referencia histórica | Gráfico de medias anuales y línea histórica activable | Estaciones | Implementada | Medias por año |
| Medias por mes | Gráfico climatológico mensual y tabla de días válidos | Estaciones | Implementada | Medias por mes del calendario |
| Comparar Rs, Ra y Valiantzas | Filtro múltiple y puntos por grupo con símbolos por familia | Comparador | Implementada | Ejemplo Enfrentar 2020 y 2025 con tres familias |
| Identificación de estaciones | Código SIAR junto al nombre | Comparador | Implementada | Abrir selector Estaciones |
| Enfrentar años TEST 2020–2025 | Selección simultánea de anualidades publicadas | Comparador | Implementada | Seleccionar 2020 y 2025 |
| Guía detallada por sección | Instrucciones y ejemplos navegables | Guía práctica | Implementada | Abrir Qué puedes hacer en cada sección |
| Ecuaciones de errores, tres fórmulas y AHC | Fórmulas, unidades y sentido de los grupos AHC | Metodología | Implementada | Abrir Métricas de evaluación y Fórmulas empíricas |

## Pruebas y despliegue

Ejecutar `python -m pytest web_app/tests -q` con los artefactos de aceptación de V0.2/V0.3, TEST y `TEST_ET0_V2_DEVELOPMENT_DIR` para comprobar el handoff real. Ejecutar lint, las pruebas y la compilación de frontend.

Añadir `compose.et0-v2-development.yaml` a los Compose ya usados para el catálogo retrospectivo VERIFIED. Definir `WEB_ET0_V2_DEVELOPMENT_DIR` con la ruta del paquete de lectura. Probar en proyecto y puerto independientes. Comparar salud, SHA-256, montajes de solo lectura y resultados representativos antes de mover la versión probada a 8090. V0.3 usa un proyecto separado en 8088. Los dos catálogos mantienen `NOT_PUBLISHED`.

En iaserver el proyecto final es `siar-et0-v2-final`. Para regresar a la revisión anterior:

```sh
docker stop siar-et0-v2-final-frontend-1 siar-et0-v2-final-backend-1
docker start siar-et0-v2-phase2-test-backend-1 siar-et0-v2-phase2-test-frontend-1
```

Comprobar salud, HTTP 200 y catálogo firmado en 8090. Para volver a fase 1, detener también la revisión previa, arrancar `siar-et0-v2-review-backend-1` y `siar-et0-v2-review-frontend-1`, y repetir las comprobaciones. El inventario del rollback está en `/home/josep/siar-et0-v2-phase2-rollback/`.
