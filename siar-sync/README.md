# SIAR Sync

Componente FastAPI para consultar SIAR y sincronizar los CSV diarios de estaciones de Baleares. La interfaz se sirve desde `templates/index.html`; la lógica de descarga y sincronización está en `siar_worker.py`.

## Estado de producción

Producción continúa desplegada desde `/home/josep/siar-sync`. La versión v2.3.2 se desplegó y validó como corrección funcional intermedia; v2.3.3 es la versión productiva actual, desplegada y validada. La sincronización de esta copia canónica en GitHub no despliega ni sustituye la instancia activa. GitHub queda pendiente de esta PR.

El worker productivo v2.3.3 tiene SHA-256:

```text
7d48be629cd6cf584bed491f9e30da57f62a025f67654be1c33706b02886a81f
```

## Configuración

Compose carga variables desde `.env`, que no debe versionarse. Usa `.env.example` como referencia sin secretos. `SIAR_NIF` y `SIAR_PASSWORD` autentican con SIAR; `GITHUB_TOKEN` permite al worker publicar cambios en GitHub; Telegram es opcional. `SIAR_API_KEY` se conserva como variable legacy/deprecada porque `app.py` todavía la lee por compatibilidad.

`SIAR_CA_BUNDLE` es opcional. Si no se define, Requests aplica TLS estándar con `verify=True`. Si se define, se utiliza la ruta del bundle indicado y se mantiene la verificación TLS. Nunca se utiliza `verify=False`.

## Certificados y workaround TLS

`certs/siar_extra_chain.pem` contiene únicamente certificados CA públicos, sin claves privadas ni credenciales. Durante el build, el Dockerfile combina el trust store de `certifi` con esa cadena adicional y genera `/app/certs/siar_bundle.pem`. El bundle generado es un artefacto de build y no se versiona.

Este workaround puede retirarse cuando MAPA sirva una cadena TLS compatible con el trust store estándar.

## Compose

`docker-compose.yml` conserva el montaje relativo `../:/app/Capitulo-1` para esta estructura del monorepo. No ejecutes este Compose como despliegue de producción. La instancia activa continúa usando `/home/josep/siar-sync` hasta que exista una decisión y un procedimiento de despliegue independientes.
