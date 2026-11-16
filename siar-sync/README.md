# SIAR Sync

Componente FastAPI para consultar SIAR y sincronizar los CSV diarios de estaciones de Baleares. La interfaz se sirve desde `templates/index.html`; la lógica de descarga y sincronización está en `siar_worker.py`.

## Estado de producción

**Actualizado: 2026-10-23. SIAR-SYNC-RUNTIME = HOLD.** v2.3.2 y v2.3.3 fueron desplegadas y validadas; v2.3.3 es la versión canónica en GitHub, no una PR pendiente. El despliegue conocido está en `/home/josep/siar-sync`; esta copia no sustituye ni actualiza producción.

El diagnóstico comunicado por 02 conserva un checkout residual de datos `quality/recovery-20261012T085131Z`, HEAD `18e6d893...`. El snapshot científico 2025 ya está cerrado y main contiene su recuperación. El worker tiene target default 2025-12-31; todavía no hay modo incremental 2026 demostrado desde último dato útil hasta actualidad sin reabrir huecos históricos. /start podría intentar aproximadamente 1.958 huecos históricos de activas.

El panel mezcla historical gaps y current/prospective lag; tampoco aplica correctamente Fecha_Baja a cerradas aunque el worker sí. Reactivación condicionada a **SIAR-SYNC-PRODUCTION-ALIGNMENT + SIAR-SYNC-INCREMENTAL-2026**. Forecast SIAR/AEMET no dependen de levantar este HOLD. Ver [estado maestro](../docs/PROJECT_STATUS.md).

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

`docker-compose.yml` conserva el montaje relativo `../:/app/Capitulo-1` para esta estructura del monorepo. No ejecutes este Compose como despliegue de producción. El despliegue conocido, actualmente HOLD, utiliza `/home/josep/siar-sync` hasta que exista una decisión y un procedimiento de despliegue independientes.

`init: true` es requisito canónico del servicio; INC-INFRA-001 quedó resuelta. El worker conserva autenticación dinámica, bloques ≤7 días, esquema 23 columnas, escritura atómica, reintentos y persistencia local aunque falle Git.
