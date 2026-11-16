# Recuperación y comprobaciones

Los procedimientos describen interfaces existentes en el inventario. No se han ejecutado durante esta misión. Recuperar primero a cuarentena; no reemplazar raw, datasets o producción para probar un restore.

## Preservación general

Interfaz del script `/opt/siar-project-preservation/preserve-v3.py`:

```text
preserve-v3.py restore --id <backup_id>
```

El script es root:root 0600; la línea muestra la interfaz, no presupone que sea directamente ejecutable. Utilizar el contexto operativo autorizado y el intérprete de la instalación existente, sin cambiar permisos ni adivinar configuración.

Cuarentena: `/srv/siar-project/preservation/restores/<backup_id>_<suffix>/`.

Antes de aceptar recuperación, comprobar:

1. Marcador `COMPLETE` correspondiente.
2. Manifest e identidad del snapshot solicitado.
3. SHA256 del contenido recuperado frente al manifest y referencias de origen.

Conservar manifests/receipts para interpretar el conjunto de snapshots de bytes nuevos/modificados. No asumir una copia completa monolítica por el nombre del backup. No promover cuarentena a ubicación canónica sin validación y autorización operativa.

## Referencias científicas

Helper: `/opt/siar-project-preservation/restore_science_reference.py`, root:root 0600. Validado para Dataset v1 y CSV clásicos/artefactos de referencia. El inventario no proporciona sus argumentos completos; no se inventa una CLI adicional.

El snapshot histórico registra Dataset v1 SHA256 PASS, CSV cuatro métodos SHA256 PASS y restores independientes de 2 artefactos científicos + 2 referencias operativas PASS. [Alcance y límites](coverage.md).

## AEMET específico

Interfaz existente de `/opt/aemet-forecast/current/ops.py`:

```text
ops.py restore <run-relative>
```

Destino: `raw/tmp/restore-*`, bajo la raíz AEMET. Validar `COMPLETE` frente a `SHA256SUMS` antes de considerar íntegro el run restaurado. La cuarentena no es una captura nueva ni parte del raw publicado. Esta descripción no acredita una ejecución nueva de restore ni reactiva AEMET.

## SIAR Raw específico: límite de evidencia

El inventario no registra un test independiente de restore específico del namespace SIAR Raw. **No se declara demostrado.** La evidencia de recuperación de preservación general se documenta separadamente y no debe reclasificarse como ese test específico.

## Dependencias fuera del backup

Antes de restaurar, comprobar acceso al remoto, material crypt y recuperación OAuth/rclone disponibles fuera del servidor, GitHub y manifests/receipts preservados. Nunca imprimir secretos para demostrar su disponibilidad. [Política de secretos y recuperación](secrets-and-recovery.md).
