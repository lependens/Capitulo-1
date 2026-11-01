# Operación y semántica

**Actualizado: 2026-11-01**, incluido cierre operativo Windows PASS de 02B comunicado por Dirección 00. Este documento describe mecanismos inventariados, no ordena ejecutarlos ni instala automatización. Los paths y las identidades están en [arquitectura](architecture.md).

## Ingestión event-driven existente

Flujo real: inbox del servidor → `ingest_incremental.py` → verificación SHA256/inventario → `scientific-artifacts` → receipt de ingestión → solicitud de `siar-project-preservation.service`.

Idempotencia: un artefacto idéntico se acepta como **unchanged**. La misma identidad con contenido diferente se rechaza como **collision**; no sobrescribirla ni reutilizarla. No se deducen formatos de receipt ni flags del importador que no figuren en el inventario.

La entrada Windows ya está automatizada **después de finalización explícita**: resultado completo → preparación/inventario/hash y firma READY local → watcher cada 5 min → SFTP restringido → staging/chroot → verificación SHA servidor → inbox/importer existentes. Debe estar desbloqueada la clave SFTP dedicada en ssh-agent. READY firmado es independiente de la presencia del paquete; no lo firma autónomamente el watcher. [Contrato y precondiciones](windows-ingestion.md).

Tras aceptación: scientific-artifacts → receipt → solicitud del servicio de preservación → snapshot → Drive crypt → verificación remota/receipt. Originales y OneDrive no cambian. Fallos conservan paquete/material para retry y no publican staging incompleto ni borran originales. E2E, idempotencia (new=0, unchanged=1) y failure/retry PASS, con artefacto sintético 47 bytes y snapshot `backup_20261101T065239Z_71bbe72e` REMOTE_VERIFIED. El manejo de fallo se validó tras corregir el contrato de retorno del importador, sin cambiar su política de colisiones.

## Preservación general

El servicio ejecuta `preserve-v3.py snapshot` y después `preserve-v3.py reconcile`. El snapshot incluye solo bytes nuevos/modificados. Esto no autoriza sobrescribir una identidad existente ni afirmar que cada snapshot incremental sea una imagen completa independiente.

| Estado | Significado |
|---|---|
| NO_NEW_BYTES | No hay bytes nuevos/modificados: no crear snapshot nuevo |
| REMOTE_VERIFIED | Copia remota + verificación de datos + verificación de marcador + receipt local coincidente |
| PREVIOUSLY_REMOTE_VERIFIED | Un receipt existente coincidente permite omitir una auditoría repetida |
| COMPLETE | Se publica solo después de verificar payload |

Operación remota: `rclone copy --immutable`. Verificación: `rclone check --download --one-way`. No es sync destructivo. No rebajar un fallo de copia/verificación a REMOTE_VERIFIED ni deducirlo únicamente de la existencia de un directorio remoto.

El timer diario es `siar-project-preservation.timer`: 04:15 UTC, Persistent=true, retraso aleatorio hasta 15 minutos. Servicio static/inactive entre ejecuciones y timer enabled/active son estados compatibles; inactive no es por sí solo fallo de un job que termina.

## SIAR Forecast específico

Fuente: runs publicados en `/srv/siar-forecast/raw/runs`, estados **complete / partial / failed**. `raw/tmp` queda fuera. Wrapper y reconciliador usan `copy --ignore-existing`, verifican payload remoto y publican/leen de vuelta `COMPLETE`.

Su timer enabled/active se ejecuta cada 15 minutos tras boot / ejecución previa. El servicio `siar-forecast-backup.service` es disabled/inactive entre runs; el wrapper utiliza root:root. La captura SIAR solicita backup desde ExecStopPost y mantiene su calendario separado 08:00/14:00/20:00 Europe/Madrid.

La disponibilidad de timer/hook no sustituye la evidencia pendiente de un nuevo run automático real → backup automático → COMPLETE remoto. Esta misión no declara CAPTURE/PILOT RUNNING.

## AEMET específico

`aemet-forecast-backup.service` ejecuta `ops.py reconcile` con identidad aemet-forecast:aemet-forecast, sin timer independiente. Usa `rclone copy --immutable`, `rclone check --download --one-way` y después COMPLETE en `forecast-offsite-crypt:AEMET-forecast/raw/runs`.

La captura permanece disabled/inactive. La inspección no observó hook de backup en su servicio: queda retirada la afirmación genérica de backup automático tras cada captura (success/failure). No sustituirla por otra dependencia no demostrada. La inclusión de AEMET en preservación general es un mecanismo distinto y no prueba nuevas capturas ni backup específico automático activo.

## Revisión operativa sin cambiar infraestructura

Contrastar la identidad del artefacto y la receipt de ingestión, el manifest del snapshot, la verificación del payload/marcador y la receipt remota local coincidente. Ante colisión o discrepancia de hashes, conservar originales y evidencia y escalar integridad; no borrar ni forzar una identidad. Para recuperación, seguir [restore](restore.md), sin publicar configuración secreta ni volcar entornos.
