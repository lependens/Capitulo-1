# Interfaz conjunta SIAR y AEMET

La web SIAR escucha solo en la direccion LAN configurada y mantiene TLS, controles CSRF y acceso limitado a la LAN 192.168.1.0/24, sin usuario ni contrasena. Los dos Collectors y sus timers son independientes. La pagina consulta los siete dias de captura hasta hoy por defecto; el calendario filtra fechas de captura, no fechas objetivo. AEMET muestra la prevision municipal asociada a cada estacion mediante /var/lib/aemet-forecast/export/status.json. Las capturas y predicciones se vinculan por capture_id, station_code y municipality_id. La fuente se marca desactualizada a partir de 15 minutos o si su hora queda mas de 5 minutos en el futuro. Los estados de backup son evidencias previas, no verificacion en tiempo real.

La web lee el export de AEMET directamente con su cuenta de servicio y sin acceso a sus RAW. El helper privilegiado controla dos acciones fijas del timer de AEMET. No concede shell, Docker ni argumentos arbitrarios. No cambia el Collector ni su configuracion.

Despliegue: instalar los cuatro archivos de aemet_read en /opt/siar-forecast-aemet-read como se indica, control.py root:root 0755 en /usr/local/libexec/siar-forecast-web-control y server.py, index.html, combined.js root:siar-forecast-web 0640 en /opt/siar-forecast/web; reiniciar solo siar-forecast-web.service. Los permisos y sudoers existentes se mantienen. Revisar tambien las rutas y dependencia de runuser.

Rollback: restaurar server.py, index.html y el control de /var/backups/siar-forecast-20261125-corrective/web-joint-20261126/ a sus destinos originales, y reiniciar exclusivamente siar-forecast-web.service. No modificar los timers ni los archivos RAW. La copia del lector de AEMET puede permanecer sin utilizarse.

El 26/11/2026 se verificaron siete pruebas de web, HTTPS/acceso LAN/API/CSRF, lectura AEMET con cuatro capturas y 14 dias de previsiones en total, filtros IB04/IB02, fechas historicas y pausa/reanudacion independiente de ambos timers. Ningun backup remoto se comprueba al cargar el panel.

Sin usuario ni contrasena, cualquier dispositivo de la LAN permitida puede consultar y controlar las programaciones. El servidor rechaza IP fuera de esa subred y nombres de host distintos del origen configurado.
