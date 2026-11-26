# Integración AEMET, 26 de noviembre de 2026

Rama de trabajo para consolidar #72, #73 y #74 con el runtime AEMET observado en el servidor. Los documentos A2 históricos describen el estado de su fecha, no el estado actual.

El calendario usa horas UTC 00:20, 06:20, 12:20 y 18:20; backup horario e independiente. El guard local comprueba integridad y espacio antes de captura. Las credenciales, configuración rclone y RAW no se incluyen en Git. El estado de backup del lector representa la última evidencia del journal; no es una consulta actual al remoto.

Captura programada 2026-11-26 00:20 UTC: completa; emisión literal 2026-11-25T21:03:08, horizonte 25 noviembre–1 diciembre, manifiesto local y backup verificados. Captura 06:20 UTC: fallo HTTP 429, sin emisión, run fallido y backup conservados. No inferir zona horaria de emisión. Piloto pendiente.

Rollback: conservar el backup de configuración en /var/backups/aemet-a3-20261125T201155Z. Antes de restaurar, comprobar unidades activas, comparar archivos y no reemplazar credenciales OAuth renovadas; no borrar RAW o copia remota. Los controles de local_guard.py no deben retirarse sin verificar almacenamiento e integridad. La web y el tratamiento específico de rate limit están pendientes.
