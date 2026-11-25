# ET₀ V2 · operación estable

Aprobada para uso privado en LAN. Acceso: http://192.168.1.43:8090/?science=et0-v2. V0.3 continúa en http://192.168.1.43:8088/. El explorador ofrece estaciones, comparación, AHC, ML, ANN, series, metodología, guía y auditoría. La referencia PM FAO-56 es calculada. DEVELOPMENT, TEST retrospectivo e histórico completo in-sample se interpretan según sus respectivos alcances. Publicación científica NOT_PUBLISHED.

Código desplegado: c7f8ebbde5e69ec32012739d669d65880c13614a. Merge PR #81: 7c672da4c2787b5f0d22695afeb5ca105a2b448b. Proyecto Compose: siar-et0-v2-final. Código en /home/josep/siar-et0-v2-final/web_app. Las imágenes locales son siar-et0-v2-final-backend y siar-et0-v2-final-frontend. Contenedores con restart unless-stopped y healthcheck.

Configuración operativa en /home/josep/siar-et0-v2-review.env y /home/josep/siar-et0-v2-verified-prep/compose.verified.yaml. DEVELOPMENT se monta desde /home/josep/siar-et0-v2-development-prep/data. Retrospectiva y DEVELOPMENT se montan en solo lectura. Sus catálogos SHA-256 son respectivamente fef85f0a6ae981792d867a9de2b10b9f7787bc680461820d2284cd20453df4ab y 3207fe042abb004b7cd69122339845adc8d55cd65b5cfd4d676d9e5512e6eea9. La fuente canónica de DEVELOPMENT y sus respaldos se describen en et0-v2-phase2.md.

Para reconstruir o arrancar en el servidor, desde /home/josep/siar-et0-v2-final/web_app:

```sh
WEB_PORT=8090 WEB_ET0_V2_DEVELOPMENT_DIR=/home/josep/siar-et0-v2-development-prep/data docker compose -p siar-et0-v2-final --env-file /home/josep/siar-et0-v2-review.env -f compose.yaml -f compose.et0-v2-review.yaml -f /home/josep/siar-et0-v2-verified-prep/compose.verified.yaml -f compose.et0-v2-development.yaml up -d --build --wait
```

Para detener y arrancar conservando contenedores: `docker stop siar-et0-v2-final-frontend-1 siar-et0-v2-final-backend-1` y `docker start siar-et0-v2-final-backend-1 siar-et0-v2-final-frontend-1`. El procedimiento de rollback a la revisión anterior y fase 1, las fuentes, limitaciones y seguimiento histórico se encuentran en [et0-v2-phase2.md](et0-v2-phase2.md). El inventario de contenedores anteriores reside en /home/josep/siar-et0-v2-phase2-rollback/.

## Preservación operativa verificada

Paquete `et0-v2-stable-20261125`, en `/home/josep/siar-et0-v2-operational-preservation-20261125` y en el destino cifrado `forecast-offsite-crypt:SIAR-project/operational/et0-v2-stable-20261125`. SHA-256 de `SHA256SUMS`: `69af43f822a8b2ab32dd7df2cbbb89ef21cfc381f929fcfd201371c8c91df2fc`. La verificación remota con descarga cotejó ocho archivos y no encontró diferencias. Es una copia operativa independiente del sistema de snapshots cumulativos `backup_*`.

Restauración independiente en `/home/josep/siar-et0-v2-operational-restore-20261125`. Los siete archivos de contenido superaron `sha256sum -c` y la configuración Compose restaurada superó `docker compose config --quiet`. Se reconstruyó y arrancó el proyecto aislado `siar-et0-v2-restore-check` en `127.0.0.1:8091`. Ambos contenedores alcanzaron estado saludable, la web respondió HTTP 200 y el catálogo DEVELOPMENT devolvió el hash autorizado, NOT_PUBLISHED y 108 celdas. Los contenedores aislados se detuvieron después.

El paquete preserva el código web, la configuración y los inventarios de rollback. La prueba utiliza las ubicaciones científicas originales montadas en solo lectura. Los artefactos científicos y las imágenes históricas se recuperan mediante sus procedimientos y copias independientes, documentados en `et0-v2-phase2.md` y `release-v03.md`.
