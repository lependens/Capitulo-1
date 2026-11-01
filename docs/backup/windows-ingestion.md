# Ingestión Windows — automatización operativa

**Actualizado: 2026-11-01. WINDOWS-SCIENCE-INGEST-AUTOMATION = PASS. PROJECT-ARTIFACT-BACKUP-V1 = PASS / MAINTAINED.** Fuente: cierre operativo de 02B comunicado por Dirección 00. Esta documentación no inspecciona nuevamente el servidor ni instala o modifica infraestructura.

El límite vigente es **EXPLICIT FINALIZATION REQUIRED + AUTOMATED TRANSFER/PRESERVATION**. El productor declara completo el resultado y prepara/firma READY localmente. La automatización no decide que un resultado científico está completo ni firma autónomamente en su nombre.

## Flujo vigente y precondiciones

Resultado científico completo en Windows → preparación explícita e inventario/hash → **READY firmado** → watcher Windows cada 5 minutos → SFTP restringido → staging/chroot limitado → verificación SHA servidor → inbox existente → importer existente → scientific-artifacts → receipt → preservation service → snapshot → Drive crypt → verificación remota y receipt.

| Elemento | Estado operativo comunicado |
|---|---|
| Raíz de paquetes | `%LOCALAPPDATA%\SIAR-Science-Ingest\ready-for-backup` |
| Watcher | Cada 5 minutos |
| Elegibilidad | Resultado declarado completo, paquete preparado y READY firmado |
| Autenticación transporte | Clave Ed25519 dedicada, desbloqueada/cargada mediante `ssh-agent` |
| Clave privada | Solo Windows |
| Contraseñas almacenadas | NO |
| Acceso servidor | SFTP restringido, staging/chroot limitado |
| Inbox existente | `/home/josep/project-artifact-inbox` |
| Importador existente | `ingest_incremental.py`; verificación SHA256/inventario |
| Publicación científica | `/srv/siar-project/scientific-artifacts` |
| Receipts ingestión | `/srv/siar-project/ingestion-receipts` |
| Trigger preservación | Solicitud de `siar-project-preservation.service` tras ingestión aceptada |

**Precondiciones:** resultado completo, paquete preparado, READY firmado y clave SFTP dedicada desbloqueada en `ssh-agent`. READY es una declaración firmada independiente de la mera existencia del paquete. La passphrase de firma se introduce únicamente en Windows, no se almacena ni se publica. No se presupone que firma y autenticación SFTP sean el mismo mecanismo ni se documentan detalles no reportados de claves/firmas.

## Doble verificación e idempotencia

Windows calcula inventario/hash durante la preparación. El servidor **revalida SHA antes de publicar** mediante el flujo receptor/importador. Haber transferido staging no equivale a publicación científica ni a REMOTE_VERIFIED.

Publicación idempotente: mismo artefacto y mismos bytes se aceptan unchanged, sin publicación duplicada. Misma identidad con bytes distintos sigue siendo **collision/error**, nunca overwrite silencioso. Retry demostrado: **new=0, unchanged=1, PASS**.

## Fallos y retry

Originales intactos; OneDrive intacto. Un fallo no destruye el paquete preparado ni borra originales Windows por haberse transferido. El staging incompleto no se publica como resultado completo; se conserva material/estado suficiente para retry. No se utiliza destructive sync. **Failure/retry handling E2E = PASS**, tras corregir el contrato de retorno del importador; no es necesario reproducir internals del fallo para operar el flujo.

## Cierre E2E — 01/11/2026

| Evidencia | Resultado |
|---|---|
| Cadena Windows → staging → inbox → importer → scientific-artifacts → receipt → preservación → Drive crypt → verificación | PASS |
| Artefacto | Sintético, **47 bytes**, no resultado científico |
| Snapshot | `backup_20261101T065239Z_71bbe72e` |
| Estado remoto | **REMOTE_VERIFIED** |
| Idempotencia retry | **new=0 / unchanged=1, PASS** |
| Fallos/reintento | **PASS** |

El PASS demuestra la cadena con ese artefacto; no afirma que todo resultado Windows ya haya sido finalizado o verificado remotamente. El estado anterior DESIGN_READY / STILL_PENDING del 31/10 queda supersedido operacionalmente, conservado en [MILESTONES](../MILESTONES.md).

## Mecanismos separados

Ingestión event-driven tras READY no sustituye la reconciliación diaria de preservación **enabled/active**. Tampoco sustituye el backup SIAR Forecast de 15 minutos ni crea un timer independiente AEMET. [Arquitectura](architecture.md), [operación](operations.md), [cobertura](coverage.md) y [secretos](secrets-and-recovery.md).
