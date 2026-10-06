# SIAR Sync

Componente FastAPI para consultar SIAR y sincronizar los CSV diarios de estaciones de Baleares con el repositorio de datos. La interfaz se sirve desde `templates/index.html`; la lógica de descarga y sincronización está en `siar_worker.py`.

## Estado de producción

La instancia de producción continúa desplegada desde `/home/josep/siar-sync` en el servidor. Esta carpeta del monorepo documenta y versiona la copia canónica de producción; incorporarla a GitHub no despliega ni sustituye la instancia activa.

## Configuración

Compose carga las variables desde `.env`, que no debe versionarse. `.env.example` enumera las variables leídas por la aplicación y sus valores predeterminados. `SIAR_NIF`, `SIAR_PASSWORD` y `GITHUB_TOKEN` se necesitan para las consultas autenticadas y para publicar cambios en GitHub. Las variables de Telegram son opcionales.

`SIAR_API_KEY` se conserva como variable legacy/deprecada: `app.py` aún la lee por compatibilidad, mientras la autenticación actual usa NIF y contraseña.

## Compose

`docker-compose.yml` conserva el servicio, el puerto, el usuario, el archivo de entorno y la configuración de build de producción. El montaje del monorepo es relativo (`../:/app/Capitulo-1`) para que el componente pueda ejecutarse desde esta estructura de carpetas.

No ejecutes esta configuración en el servidor de producción como parte de esta incorporación. La producción sigue usando `/home/josep/siar-sync` hasta que una decisión y un procedimiento de despliegue independientes indiquen lo contrario.
