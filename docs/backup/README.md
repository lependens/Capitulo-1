# Preservación y recuperación del proyecto

**Canonización documental: 2026-10-31.** Fuente operativa: inventario `BACKUP-IMPLEMENTATION-INVENTORY-V1 = PASS` recibido de Dirección 00. Esta documentación transcribe ese inventario; no es una nueva inspección del servidor, no ejecuta backups ni modifica infraestructura. Los estados de unidades indicados son los del inventario o su estado esperado entre ejecuciones, no una garantía de disponibilidad permanente.

GitHub conserva código, documentación y evidencia versionable. Los artefactos científicos completos y los raw Forecast se preservan fuera de Git, dentro de ámbitos explícitos. Un archivo en Windows no queda protegido por existir un timer en el servidor.

| Documento | Contenido canónico |
|---|---|
| [Arquitectura](architecture.md) | Servicios, identidades, rutas y destinos lógicos |
| [Cobertura](coverage.md) | Incluidos, exclusiones y snapshot verificado |
| [Operación](operations.md) | Ingestión, estados, verificación y tres mecanismos separados |
| [Restore](restore.md) | Cuarentena, comprobaciones y límites de la evidencia |
| [Ingestión Windows](windows-ingestion.md) | Diseño pendiente, no instalado ni autorizado para provisionar SFTP |
| [Secretos y recuperación](secrets-and-recovery.md) | Dependencias de recuperación y política de exclusión |

## Tres mecanismos distintos

| Mecanismo | Activación | Ámbito |
|---|---|---|
| Preservación general | Ingestión event-driven y timer diario 04:15 UTC, con retraso aleatorio hasta 15 min | Ciencia ya ingerida y fuentes explícitamente incluidas en el snapshot |
| Backup SIAR Forecast | Timer cada 15 min y solicitud desde ExecStopPost de captura | Runs publicados complete, partial y failed; nunca raw/tmp |
| Backup AEMET específico | Servicio disponible, sin timer independiente | Runs AEMET; no se observó hook de backup en el servicio de captura |

`PROJECT-ARTIFACT-BACKUP-V1 = PASS` conserva su significado histórico limitado: snapshot `backup_20261024T215639Z_63d0ef01`, 128 archivos verificados. No equivale a backup completo del servidor ni de todo futuro resultado. [Matriz y evidencia](coverage.md).

El estado general y los gates de captura permanecen en [PROJECT_STATUS](../PROJECT_STATUS.md); los próximos trabajos, en [ROADMAP](../ROADMAP.md); las ubicaciones científicas, en [DATA_LOCATIONS](../DATA_LOCATIONS.md). Esta misión no cambia el estado de Forecast ni de AEMET.
