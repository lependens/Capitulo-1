# Integración de ET₀ V2 para revisión aislada

La configuración vigente del corte del 25/11 figura en [Catálogo VERIFIED](#catálogo-verified-en-revisión-lan-24-11-2026) y [Recuperación operativa de V0.3](#recuperación-operativa-de-v03--25-11-2026). Los apartados DRAFT preservan el procedimiento de revisión y la alternativa de reversión.

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

## Instancia LAN de revisión (24-11-2026)

En iaserver se ha preparado una instancia independiente en `192.168.1.43:8090`, con entrada `/?science=et0-v2` y proyecto Compose `siar-et0-v2-review`. La web anterior continúa en 8088. El archivo de configuración está en `/home/josep/siar-et0-v2-review.env`, y la copia de trabajo en `/home/josep/siar-et0-v2-review-work`. El backend monta el derivado V2, el catálogo SIAR y el derivado de observaciones V1 con `:ro`. La exposición es en la red local: los usuarios de esa red pueden abrir esta instancia sin autenticación adicional.

Para iniciar o reconstruir después de actualizar el código, desde `~/siar-et0-v2-review-work/web_app`:

```bash
docker compose -p siar-et0-v2-review --env-file /home/josep/siar-et0-v2-review.env -f compose.yaml -f compose.et0-v2-review.yaml up -d --build
```

Para comprobar o retirar exclusivamente esta instancia, respectivamente:

```bash
docker compose -p siar-et0-v2-review --env-file /home/josep/siar-et0-v2-review.env -f compose.yaml -f compose.et0-v2-review.yaml ps
docker compose -p siar-et0-v2-review --env-file /home/josep/siar-et0-v2-review.env -f compose.yaml -f compose.et0-v2-review.yaml down
```

No aplicar estos comandos al proyecto de 8088. Tras una actualización, verificar el SHA y estado del catálogo, los montajes y la respuesta real de la API antes de utilizarla; no sustituir el SHA configurado sin verificar la nueva publicación de ciencia.

La interfaz permite filtrar por entradas exactas o por subconjunto de variables disponibles. El filtro selecciona una familia de fórmula cada vez; no habilita ranking cruzado entre fórmulas sin población común publicada. Las fichas aún no incluyen matriz de disponibilidad variable × año ni promedios anuales ni correlaciones. Residuales, distribuciones, matriz de resultados por estación y fichas completas de parámetros/coeficientes quedan pendientes de completar o de concretar con las tablas publicadas. Las pruebas visuales y automáticas de esta instancia no constituyen certificación científica.

## Catálogo VERIFIED en revisión LAN (24-11-2026)

La instancia `192.168.1.43:8090/?science=et0-v2` utiliza `et0-v2-retrospective-20261116` con catálogo técnicamente `VERIFIED` y publicación científica `NOT_PUBLISHED`. La fuente canónica es `/srv/siar-project/scientific-artifacts/et0-v2-verified-catalog-20261124/catalog.json`; su copia de lectura es `/home/josep/siar-et0-v2-verified-prep/catalog.json`, protegida por un directorio 700. Ambas tienen SHA256 `fef85f0a6ae981792d867a9de2b10b9f7787bc680461820d2284cd20453df4ab`. Los archivos de datos y el catálogo se montan de solo lectura. No se ha modificado el catálogo canónico ni los datos científicos.

La configuración adicional `/home/josep/siar-et0-v2-verified-prep/compose.verified.yaml` monta la copia sobre `/et0-v2/catalog.json` y fija su SHA. El entorno continúa en modo `review`, ya que VERIFIED no implica publicación científica. Para arrancar o actualizar desde `~/siar-et0-v2-review-work/web_app`:

```bash
docker compose -p siar-et0-v2-review --env-file /home/josep/siar-et0-v2-review.env -f compose.yaml -f compose.et0-v2-review.yaml -f /home/josep/siar-et0-v2-verified-prep/compose.verified.yaml up -d --build
```

Para volver a la configuración DRAFT, sin tocar datos científicos, ejecutar desde el mismo directorio:

```bash
docker compose -p siar-et0-v2-review --env-file /home/josep/siar-et0-v2-verified-prep/review-draft.env -f compose.yaml -f compose.et0-v2-review.yaml up -d --force-recreate
```

Verificar a continuación `/api/et0-v2/catalog` y el estado de los contenedores. Se conservaron también las imágenes `siar-et0-v2-review-backend:before-verified-20261124` y `siar-et0-v2-review-frontend:before-verified-20261124` para una restauración adicional si fuera necesaria.

Antes del cambio se comprobó el SHA del catálogo, los 111 archivos de su inventario y la paridad de consultas representativas entre DRAFT y VERIFIED: métricas globales y por estación, predicciones diarias y AHC dentro de muestra. El cambio no supone publicación ni autorización de acceso por Internet.

## Registro de V0.3 y resultados ET₀ V1 — 24-11-2026

Josep comunicó que la retirada de la aplicación V0.3 desde CasaOS fue accidental. En la inspección anterior, el puerto 8088 no respondía, los contenedores V0.3 no figuraban y la copia `/home/josep/siar-et0-web` mostraba sus archivos `web_app` como eliminados. No se ha atribuido una causa técnica adicional ni se ha ejecutado una recuperación de la aplicación.

Los derivados utilizados por V0.3 se localizaron en `/home/josep/siar-et0-web-data/releases/web-v0.3-explorer-20261101/data-verified`; las pruebas de aceptación de V0.3 se ejecutaron con esos datos y el paquete ML V1 de `/home/josep/project-artifact-inbox/science-ml-v1-test-recovery-20261101T090828Z/scientific-results/ml-v1-test-recovery-20261101T090828Z/`, montados en solo lectura. Esto verifica su disponibilidad en aquella inspección; no acredita una restauración completa de la web.

El código de la versión originalmente desplegada está identificado por el commit `c44d1187760e19ed67269c0dbc05d2c5b0255ad1`. El snapshot de preservación comunicado es `backup_20261101T123421Z_8022d3fd`. Identidades, hashes y detalles de preservación figuran en [release-v03.md](release-v03.md). Cualquier recuperación futura debe utilizar una copia independiente y verificar antes las rutas, el snapshot y los hashes; no regenerar resultados científicos ni sobrescribir la copia antigua.

## Recuperación operativa de V0.3 — 25-11-2026

La aplicación V0.3 se reconstruyó en una copia independiente `/home/josep/siar-et0-v03-recovery` desde el commit `c44d1187760e19ed67269c0dbc05d2c5b0255ad1`, sin modificar la antigua copia de trabajo. Su estado y diff previos se conservaron en `/home/josep/siar-et0-v03-recovery-evidence/`. El proyecto Compose independiente `siar-et0-v03-recovery` usa el derivado histórico en `/home/josep/siar-et0-web-data/releases/web-v0.3-explorer-20261101/data-verified`, montado de solo lectura.

La web recuperada arrancó en `192.168.1.43:8088`, con backend y frontend saludables, 75.496 observaciones y navegación comprobada. Se verificó el SHA del `info.json` (`9893055e97d75684a311f7270ee57d90dd9b7e934d01efb0679e0c90c4da3b94`) y los 18 hashes de artefactos que declara. Una consulta AHC SELECTED, common_all, devolvió n=15.958 y RMSE=0,40139207829124707. La instancia V2 siguió activa en 8090, VERIFIED y NOT_PUBLISHED. La sección anterior refleja la situación previa a esta recuperación.

Para iniciar o reconstruir V0.3, después de verificar que 8088 está libre:

```bash
cd /home/josep/siar-et0-v03-recovery/web_app
docker compose -p siar-et0-v03-recovery --env-file /home/josep/siar-et0-v03-recovery-evidence/recovery.env -f compose.yaml up -d --build
```

Para dejarla apagada sin borrar los datos ni las imágenes:

```bash
docker compose -p siar-et0-v03-recovery --env-file /home/josep/siar-et0-v03-recovery-evidence/recovery.env -f compose.yaml stop
```

Verificar `/api/health` y los servicios después de iniciar. Este proyecto es independiente de `siar-et0-v2-review`. El registro local de la recuperación se encuentra en `/home/josep/siar-et0-v03-recovery-evidence/RECOVERY.md`.
