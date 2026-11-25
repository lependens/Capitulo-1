# SIAR Forecast: administracion y preservacion

El colector captura respuestas RAW sin sobrescribirlas. El producto puede devolver CSV con solo cabecera; eso no es una prevision util. El indice complementario clasifica cada captura por contenido de ET0: useful, partial, empty o failed, sin modificar RAW ni sus manifiestos. La calidad no demuestra identidad ni hora de emision del proveedor.

La cola verifica como maximo cuatro runs por ejecucion, priorizando los mas recientes. Un recibo se escribe solo tras verificar el run local y la copia remota; los fallos se reintentan y los recibos se revisan cada siete dias. Un recibo describe una verificacion pasada. Los runs elegibles para revision no equivalen necesariamente a copias ausentes. La continuidad del historico depende de que las ejecuciones sucesivas terminen y debe supervisarse.

## Configuracion

La configuracion privada de rclone y la direccion LAN se mantienen fuera de Git. Para la unidad de backup se suministran SIAR_FORECAST_RCLONE_CONFIG y SIAR_FORECAST_REMOTE_ROOT mediante un EnvironmentFile protegido. Para la web se suministran SIAR_WEB_BIND, SIAR_WEB_PORT y SIAR_WEB_ORIGIN mediante /etc/siar-forecast-web/environment. No almacenar contrasenas, claves privadas ni configuraciones reales en el repositorio.

La interfaz usa HTTPS y credenciales independientes. El servicio escucha solo en la direccion LAN configurada; no se debe exponer a Internet. El usuario web no tiene shell ni socket Docker. Un helper propiedad de root acepta exclusivamente operaciones fijas para SIAR Forecast y registros de sus servicios. El certificado, clave y contrasena tienen permisos restringidos.

## Verificacion

Ejecutar desde forecast_collector: python3 tests/test_reconcile_local.py; python3 tests/test_queue_local.py; PYTHONPATH=src python3 -m pytest tests/test_quality.py; python3 tests/test_web_admin.py. Las pruebas que interactuan con el servidor o alteran temporalmente el timer son locales y no deben ejecutarse en CI.

Comprobar systemctl status de siar-forecast-capture.timer, siar-forecast-backup.timer, siar-forecast-quality.timer y siar-forecast-web.service; consultar journalctl -u siar-forecast-backup.service. El indice es /var/lib/siar-forecast-quality/status.json. Comprobar periodicamente espacio, errores y recibos pendientes. La hora de captura no constituye hora de emision.

## Reversion

Antes de desplegar, conservar copia de las unidades y ejecutables existentes. Para retirar la web, deshabilitar su servicio y retirar su regla sudoers; conservar su configuracion de forma segura y no tocar captura, backup ni RAW. Para revertir la cola, detener temporalmente el timer de backup, restaurar el ejecutable y drop-in previamente respaldados, ejecutar systemctl daemon-reload y reactivar el timer. La version antigua puede rechazar respuestas vacias y agotar el tiempo: una reversion no resuelve esas incidencias. Para retirar solo el indice, detener y deshabilitar siar-forecast-quality.timer y conservar sus datos. Nunca borrar RAW, manifiestos ni recibos como parte de la reversion.

## Evidencia y limites

En una consulta manual del producto de pronostico diario con el cultivo aguacate, cinco estaciones baleares mostraron cero registros en la web oficial. Sus CSV automaticos recientes tambien carecian de filas. No se pudo verificar independientemente la descarga manual y la causa no esta determinada. Una copia remota de una captura reciente fue verificada de nuevo sin duplicacion y restaurada de forma independiente, con manifiesto SHA256 valido. Esa muestra no certifica toda la historia.
