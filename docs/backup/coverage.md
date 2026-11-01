# Cobertura y límites

**Actualizado: 2026-11-01; PROJECT-ARTIFACT-BACKUP-V1 = PASS / MAINTAINED.** La matriz refleja `BACKUP-IMPLEMENTATION-INVENTORY-V1 = PASS` y el cierre `WINDOWS-SCIENCE-INGEST-AUTOMATION = PASS`, no una auditoría nueva de cada archivo. Cobertura condicionada a ingestión no equivale a verificación individual de un output recién creado.

| Categoría | Cobertura exacta |
|---|---|
| Dataset Observational v1 | Cubierto una vez bajo scientific-artifacts; reconciliación diaria |
| scientific-results | Cubierto después de ingestión |
| four-classical | Cubierto |
| AHC | Cubierto |
| WEB | Cubierto después de ingestión |
| Futuros ML/AHC | Cubiertos después de ingestión a scientific-artifacts |
| SIAR Forecast published Raw | Backup programado cada 15 min y, adicionalmente, representado en el ámbito de preservación general |
| AEMET Raw runs | Ámbito de preservación general; servicio específico disponible sin timer independiente; captura deshabilitada |
| Working copies del repositorio | Sin backup masivo; GitHub es autoridad |
| Configuración no secreta / metadatos de releases | Solo allowlist explícita |
| Resultado Windows no finalizado explícitamente | NOT YET ELIGIBLE FOR AUTOMATED TRANSFER |
| Resultado Windows finalizado + paquete preparado + READY firmado | Watcher transfiere automáticamente con clave SFTP desbloqueada en ssh-agent; servidor verifica SHA, importer publica y preservación/verificación remota automáticas |
| OneDrive | NO es un backup auditado |
| Archivos arbitrarios del servidor | NO cubiertos |

## Snapshot histórico verificado

`PROJECT-ARTIFACT-BACKUP-V1 = PASS`

| Evidencia | Resultado |
|---|---|
| Snapshot | `backup_20261024T215639Z_63d0ef01` |
| Estado | REMOTE_VERIFIED |
| scientific-results | 82 archivos; 1,421,919,603 bytes científicos |
| SIAR published Raw | 46 archivos |
| Total verificado | 128 archivos |
| Dataset v1 SHA256 | PASS |
| CSV cuatro métodos SHA256 | PASS |
| cryptcheck | PASS |
| Restore de ciencia | 2 restores PASS |
| Restore de referencias operativas | 2 restores PASS |
| Descarga completa redundante | INTERRUPTED; **NO PASS** |

La aceptación histórica se apoya en cryptcheck + restores independientes + SHA256. No se transforma el intento interrumpido en una prueba superada. Los 221 entries del inventario del servidor no equivalen a 221 archivos protegidos. La evidencia general de restore tampoco demuestra por sí sola un restore específico del namespace SIAR Raw.

## Exclusiones explícitas

- Sistema operativo completo; `/proc`, `/sys`, `/dev`, `/tmp`.
- Capas Docker y cachés.
- `/etc` completo y configuraciones secretas.
- `raw/tmp` de captura; no confundir con cuarentena temporal de restore.
- Directorios arbitrarios de repositorios o workspaces.
- Workspaces Windows no ingeridos y OneDrive.
- Archivos fuera de raíces allowlisted del servidor.
- CSV SIAR arbitrarios fuera de las raíces de runs declaradas: el backup Forecast no protege automáticamente los CSV de SIAR Sync.
- Credenciales, tokens, claves privadas y material secreto de configuración.

La **finalización científica sigue siendo explícita/manual**, no la transferencia/ingestión posterior. [Watcher operativo y READY firmado](windows-ingestion.md): E2E PASS con artefacto sintético 47 bytes y snapshot `backup_20261101T065239Z_71bbe72e` REMOTE_VERIFIED; retry new=0/unchanged=1 y failure/retry PASS. Un paquete elegible sigue necesitando completar la cadena y su verificación: no se certifica por anticipado todo archivo futuro. La política de [secretos y recuperación](secrets-and-recovery.md) evita depender únicamente del backup cifrado para poder abrirlo.
