# Ingestión Windows — diseño pendiente

**WINDOWS-SCIENCE-INGEST-AUTOMATION = DESIGN_READY / STILL_PENDING.** Esta sección es diseño propuesto, **NOT DEPLOYED / NOT INSTALLED**. Provisionar SFTP **NO ESTÁ AUTORIZADO** por esta misión. La existencia del importador del servidor no demuestra un watcher Windows instalado.

## Diseño propuesto

| Elemento | Propuesta, no despliegue actual |
|---|---|
| Raíz ready-for-backup | `%LOCALAPPDATA%\SIAR-Science-Ingest\ready-for-backup` |
| Inbox servidor | `/home/josep/project-artifact-inbox` |
| Transporte | Staging SFTP restringido |
| Autenticación | Clave Ed25519 dedicada, con passphrase; privada solo Windows |
| Cuenta SFTP | Locked, chroot, solo SFTP, sin shell ni forwarding |
| Watcher | Cada 5 min, solo mientras el usuario Windows esté conectado |
| Importador | Reutilizar el existente, sí; verificación SHA256/inventario |
| Receipts | `/srv/siar-project/ingestion-receipts` |
| Trigger de preservación | Importador solicita `siar-project-preservation.service` |

Los controles concretos para autenticar una cuenta restringida y configurar el receptor deben resolverse en su futura misión; esta tabla no es una receta de aprovisionamiento.

## Integridad e idempotencia

SHA256 local + SHA256 del importador del servidor. Identidad idempotente: batch ID + inventario SHA256. Un paquete parcial no pasa a READY; READY solo después de estar completo y verificado. Un artefacto idéntico se acepta unchanged; misma identidad con bytes distintos se rechaza.

Ante fallo: conservar paquete local y staging, registrar pending/failure y reintentar. **Nunca borrar originales.** No confundir staging transferido con receipt de ingestión ni con REMOTE_VERIFIED.

## Trabajo aún manual / pendiente

Desarrollar y provisionar receptor + watcher en una misión autorizada y realizar E2E sintético. Hasta entonces, transferencia/ingestión explícita sigue siendo necesaria: resultados solo Windows **NOT COVERED**. OneDrive no es un backup auditado. La reconciliación diaria únicamente opera sobre ámbitos ya presentes en el servidor; no recoge un workspace Windows arbitrario.

[Cobertura vigente](coverage.md) y [operación del importador existente](operations.md).
