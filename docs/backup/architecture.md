# Arquitectura de preservación

Fuentes: [inventario operativo y cierre Windows de 01/11/2026](README.md). Se documenta la implementación existente, sin desplegar ni cambiar unidades.

## Preservación general

| Elemento | Implementación inventariada |
|---|---|
| Servicio | `siar-project-preservation.service`; static/inactive entre ejecuciones |
| Timer | `siar-project-preservation.timer`; enabled/active |
| Calendario | Diario 04:15 UTC; `Persistent=true`; `RandomizedDelaySec` hasta 15 min |
| User/Group | `root:root` |
| Exec, en orden | `preserve-v3.py snapshot`, después `preserve-v3.py reconcile` |
| Script | `/opt/siar-project-preservation/preserve-v3.py`; root:root, modo 0600 |
| ReadWritePaths | `/srv/siar-project/preservation` |

Scripts de ingestión/recuperación existentes, todos **root:root, 0600**:

- `/opt/siar-project-preservation/ingest_incremental.py`
- `/opt/siar-project-preservation/ingest.py`
- `/opt/siar-project-preservation/restore_science_reference.py`

Los modos anteriores describen archivos inventariados; no implican permiso ejecutable directo. No se inventa un intérprete o una línea ExecStart completa que el inventario no haya proporcionado.

## Raíces locales

| Función | Ruta |
|---|---|
| Ciencia preservada | `/srv/siar-project/scientific-artifacts` |
| Inbox Windows/servidor | `/home/josep/project-artifact-inbox` |
| Receipts de ingestión | `/srv/siar-project/ingestion-receipts` |
| Preservación | `/srv/siar-project/preservation` |
| Manifest de snapshot | `snapshots/<backup_id>/manifest.json`, relativo a preservación |
| COMPLETE local | `markers/<backup_id>/COMPLETE` |
| Receipt de verificación remota | `verified/<backup_id>.json` |
| Cuarentena de restore | `restores/<backup_id>_<suffix>/` |

La raíz de preservación contiene `snapshots/`, `staging/`, `markers/`, `verified/` y `restores/`. Las rutas son referencias operativas, no directorios creados por esta PR.

## Entrada Windows operativa — PASS

`%LOCALAPPDATA%\SIAR-Science-Ingest\ready-for-backup`: raíz operativa. El productor finaliza explícitamente, prepara inventario/hashes y firma READY localmente; la simple presencia de un paquete no inicia su publicación como resultado completo. Watcher cada 5 min, con clave SFTP Ed25519 dedicada cargada/desbloqueada en `ssh-agent` → SFTP restringido/staging-chroot limitado → revalidación SHA servidor → inbox/importer existentes → scientific-artifacts → receipt → preservación/snapshot → Drive crypt → verificación remota. [Precondiciones y evidencia](windows-ingestion.md).

Clave privada Windows-only, contraseñas almacenadas NO, passphrase de firma solo introducida en Windows y no almacenada. Originales y OneDrive intactos; no cambios operativos desde esta documentación.

## Destino remoto lógico

`forecast-offsite-crypt:SIAR-project/snapshots/<backup_id>/`

Cada snapshot puede representar `data/science/`, `data/forecast/siar/`, `data/forecast/aemet/`, manifest(s) y `COMPLETE`, según su inventario. No implica que cada snapshot contenga datos nuevos de las tres fuentes. El destino es lógico; no se publica configuración de remotes ni material de cifrado.

## SIAR Forecast específico

| Elemento | Implementación inventariada |
|---|---|
| Servicio | `siar-forecast-backup.service`; disabled/inactive entre ejecuciones |
| Timer asociado | enabled/active; cada 15 min tras boot / ejecución previa; el inventario no aporta su nombre exacto |
| User/Group del wrapper | root:root |
| Wrapper | `/usr/local/sbin/siar-forecast-reconcile-complete-pending`; root:root, 0750 |
| Reconciliador | `/opt/siar-forecast/bin/siar-forecast-reconcile`; root:siar-forecast, 0750 |
| Lectura | `/srv/siar-forecast/raw`, `/opt/siar-forecast` |
| Escritura | `/run/siar-forecast-backup`, `/etc/siar-forecast/rclone` |

Contexto de captura, sin alterar su estado: timer enabled/active, 08:00, 14:00 y 20:00 **Europe/Madrid**; servicio disabled/inactive entre ejecuciones; identidad `siar-forecast:siar-forecast`; escritura en `/srv/siar-forecast/raw`. `ExecStopPost` solicita el servicio de backup. No confundir horario local de captura con 04:15 UTC de preservación general ni con la cadencia de 15 minutos del backup SIAR.

## AEMET específico

| Elemento | Implementación inventariada |
|---|---|
| Servicio backup | `aemet-forecast-backup.service`; static/inactive |
| Timer de backup | No existe timer independiente |
| Exec backup | `ops.py reconcile` |
| Identidad | `aemet-forecast:aemet-forecast` |
| ReadWritePaths backup | `/srv/aemet-forecast/raw/tmp` |
| Script | `/opt/aemet-forecast/current/ops.py`; current apunta a release `c3602a09…` (prefijo recibido, no SHA completo) |
| Captura | Timer disabled/inactive; OnCalendar placeholder 2099; servicio static/inactive |
| Exec captura / escritura | `ops.py capture`; `/srv/aemet-forecast/raw` |
| Remoto específico | `forecast-offsite-crypt:AEMET-forecast/raw/runs` |

**No se observó un hook de backup en el servicio de captura AEMET.** No afirmar backup automático después de cada intento sin evidencia de otro mecanismo explícito. Un servicio disponible no demuestra programación automática.

Semántica de cada mecanismo: [operación](operations.md). Inclusiones y exclusiones: [cobertura](coverage.md).
