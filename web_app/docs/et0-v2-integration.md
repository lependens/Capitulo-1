# Integración de ET₀ V2 para revisión aislada

## Alcance y datos comprobados

La entrada `/?science=et0-v2` sigue la guía acordada. Inicio, Estaciones y datos, Modelos y metodología, Resultados por familia, Comparar modelos y Fuentes y auditoría. Incluye fichas imprimibles, catálogo de estaciones y mapa interactivo esquemático. El mapa usa las coordenadas del catálogo SIAR identificado por el protocolo de ciencia. Las fichas consultan observaciones de Dataset V1 en páginas de 200, con filtros de fechas, serie y dispersión de dos variables.

Se inspeccionó la copia local de la entrega DRAFT, contrastada con el SHA de la [PR #76](https://github.com/lependens/Capitulo-1/pull/76) en `a5cc056`. La ruta de servidor comunicada es `/srv/siar-project/scientific-artifacts/et0-v2/derived/v1/`. El catálogo de la copia, SHA `f497074a39d25a38d637329a87a652c470875502a2c9f796cf2eca6706148d81`, lista 54 archivos Parquet de predicción por estación y año. Se comprobó el conteo total de 2.081.376 registros y se leyeron las 9.612 filas de métricas. La aplicación verifica tamaños y SHA de los archivos que consume. Los CSV de predicciones duplicados quedan fuera de la lectura. El paquete científico y los modelos congelados se conservan en sus ubicaciones de origen.

## API de solo lectura

| GET bajo `/api/et0-v2` | Consulta |
|---|---|
| `/catalog` | Identidad, referencia, estado, estaciones, métodos, entradas y grupos |
| `/metrics` | `formula, calendar, group, key` y `method` repetible. Filtra filas del CSV publicado |
| `/series` | `formula, calendar, station, start, end, cohort` y `method` repetible. Límite de 367 días inclusivos. Solo abre las particiones de la estación y los años del intervalo |
| `/evolution` | `formula, calendar`, `station` opcional y `method` repetible. Usa filas año o estación-año |
| `/in-sample` | `formula, station`. Coeficiente AHC anual de histórico completo y métricas dentro de muestra |

La API compara cada versión del catálogo con un SHA fijado por configuración y conserva los identificadores del dataset, freeze, ejecución y manifiesto original. Los valores de métricas, cobertura y predicciones vienen de tablas precalculadas. La tabla permite ordenar conjuntamente cuando coinciden huella de cohorte, n y cobertura de catálogo. Al mezclar cohortes o coberturas la interfaz usa lectura paralela, con denominadores detallados. Las variantes ML individuales conservan sus filas y huellas. Los filtros de fecha libre afectan la serie diaria. Los grupos de métricas conservan su población publicada.

## Revisión local

Configurar las siguientes variables en el proceso backend. Las rutas guardan los artefactos fuera del repositorio.

```text
WEB_DATA_DIR=<derivado WEB V0.3 verificado, para observaciones SIAR y compatibilidad>
WEB_ET0_V2_DIR=<directorio DRAFT que contiene catalog.json>
WEB_ET0_V2_CATALOG_SHA256=f497074a39d25a38d637329a87a652c470875502a2c9f796cf2eca6706148d81
WEB_ET0_V2_STATION_CSV=<ruta a estaciones_baleares.csv verificado>
WEB_ET0_V2_MODE=review
```

Desde la raíz ejecutar `python -m uvicorn web_app.backend.app:app --factory --host 127.0.0.1 --port 8012`. Desde `web_app/frontend` ejecutar `npm ci`, configurar `WEB_VITE_API=http://127.0.0.1:8012` y ejecutar `npm exec vite -- --host 127.0.0.1 --port 5182 --strictPort`. Abrir `http://127.0.0.1:5182/?science=et0-v2`.

## Despliegue aislado y reversión

Preparar un archivo de entorno privado con las rutas anteriores, `WEB_BIND_IP=127.0.0.1` y un `WEB_PORT` libre distinto de 8088. Desde `web_app`, crear el proyecto de revisión separado:

```bash
docker compose -p siar-et0-v2-review --env-file /ruta/review.env \
  -f compose.yaml -f compose.et0-v2-review.yaml config --quiet
docker compose -p siar-et0-v2-review --env-file /ruta/review.env \
  -f compose.yaml -f compose.et0-v2-review.yaml up -d --build
```

Los montajes científicos son de solo lectura. Verificar salud, puerto ligado a loopback, versión DRAFT, conteos y consultas antes de compartir la revisión. Para revertir la instancia de revisión ejecutar `down` con esos mismos argumentos y el nombre de proyecto `siar-et0-v2-review`. Al autorizarse un futuro despliegue, verificar el catálogo aprobado y usar `WEB_ET0_V2_MODE=approved`, que rechaza DRAFT. Mantener la imagen y configuración previas de WEB V0.3 y sus datos verificados disponibles según el [procedimiento de rollback](release-v03.md).

## Pendientes para 05 y Dirección

- Entregar y aprobar el catálogo final con SHA. Si cambia el esquema, ajustar el adaptador y repetir la paridad antes de habilitar el modo aprobado.
- Para un ranking cruzado entre fórmulas con entradas distintas, aportar métricas recalculadas por ciencia sobre una cohorte común definida, n, cobertura propia y huella de las claves. El paquete actual permite comparaciones dentro de fórmula, calendario y grupo.
- Para barras de medias meteorológicas anuales y línea de media del periodo de referencia, solicitar si se priorizan una tabla por estación, variable y periodo con unidad, media, n, cobertura, denominador, periodo de referencia y regla para faltantes. Para correlaciones o residuales publicados, acordar método y población con 05. Las observaciones diarias ya permiten visualizar relaciones y huecos.
- Los grupos estación×mes, estación×temporada y fechas arbitrarias precisan publicación propia si se necesitan métricas oficiales. Los coeficientes AHC por grupo y las fichas de ganador por celda requieren un extracto de metadatos del freeze autorizado para la web. La tabla disponible de histórico completo aporta su coeficiente anual y sus pares.

La futura inferencia sobre nueva meteorología se mantiene como módulo independiente.
