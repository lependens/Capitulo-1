# Secretos y recuperación independiente

Se documentan dependencias y políticas; **no valores ni contenido secreto**.

No publicar ni incluir en snapshots: `rclone.conf`, OAuth access/refresh tokens, crypt password, crypt password2, credenciales de remotes, claves SSH privadas, API keys ni otros secretos. Los paths no autorizan leerlos o copiar su contenido a GitHub, logs, issues o PRs.

## Dependencias de recuperación

- Credenciales de acceso al remoto.
- Material crypt para descifrar.
- Material de recuperación OAuth/rclone.
- Repositorio GitHub para código/documentación versionable.
- Manifests y receipts preservados para identificar y comprobar artefactos.

El material de recuperación crypt/OAuth debe existir **off-server**. El cierre histórico de PROJECT-ARTIFACT-BACKUP-V1 lo confirmó OFF-SERVER; este documento no hace una inspección nueva ni publica su ubicación sensible. Evitar dependencia circular: las credenciales necesarias para recuperar no pueden existir únicamente dentro del backup cifrado que permiten abrir.

La aplicación OAuth pasó a Production y se reautorizó: elimina el límite específico de siete días de Testing, pero no garantiza que nunca se invalide un token. Revalidar acceso y recuperación sin imprimir valores ni versionar configuraciones secretas.

## Configuración y alcance

Solo configuración **no secreta** y metadatos de releases explícitamente allowlisted pueden entrar en preservación. No afirmar que se respalda `/etc` completo o todo el servidor. Los working copies no tienen backup masivo; GitHub es su autoridad versionable.

**Transporte Windows operativo — 01/11/2026:** autenticación mediante clave Ed25519 dedicada, privada solo Windows y desbloqueada/cargada mediante `ssh-agent`; acceso SFTP y staging/chroot restringidos. Contraseñas almacenadas: **NO**. READY firmado es requisito independiente de la presencia del paquete. La passphrase de firma se introduce solo en Windows, no se almacena ni publica. La automatización no firma autónomamente resultados científicos. [Flujo operativo PASS](windows-ingestion.md).

No publicar contenidos de claves públicas/privadas innecesarios, fingerprints sensibles, passphrases, secretos rclone ni tokens OAuth. Esta documentación no proporciona credenciales ni cambia su provisión.

La recuperación se acepta por integridad verificable, no por mostrar secretos: [restore y validaciones](restore.md), [evidencia histórica y exclusiones](coverage.md).
