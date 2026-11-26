# Interfaz conjunta SIAR y AEMET

La web SIAR escucha solo en la direccion LAN configurada y mantiene TLS, autenticacion y controles CSRF. Los dos Collectors y sus timers son independientes. La pagina consulta los siete dias de captura hasta hoy por defecto; el calendario filtra fechas de captura, no fechas objetivo. AEMET solo se presenta asociado a IB04 y Palma 07040. Los estados de backup son evidencias previas, no verificacion en tiempo real.

El puente de AEMET usa copias del lector preparado por AEMET en un directorio root:aemet-forecast 0750, con archivos root:aemet-forecast 0640. La lectura RAW se ejecuta como aemet-forecast; el helper privilegiado consulta exclusivamente el journal de backup y controla dos acciones fijas del timer de AEMET. No concede shell, Docker ni argumentos arbitrarios. No cambia el Collector ni su configuracion.

Despliegue: instalar los cuatro archivos de aemet_read en /opt/siar-forecast-aemet-read como se indica, control.py root:root 0755 en /usr/local/libexec/siar-forecast-web-control y server.py, index.html, combined.js root:siar-forecast-web 0640 en /opt/siar-forecast/web; reiniciar solo siar-forecast-web.service. Los permisos y sudoers existentes se mantienen. Revisar tambien las rutas y dependencia de runuser.

Rollback: restaurar server.py, index.html y el control de /var/backups/siar-forecast-20261125-corrective/web-joint-20261126/ a sus destinos originales, y reiniciar exclusivamente siar-forecast-web.service. No modificar los timers ni los archivos RAW. La copia del lector de AEMET puede permanecer sin utilizarse.

El 26/11/2026 se verificaron siete pruebas de web, HTTPS/autenticacion/API/CSRF, lectura AEMET con cuatro capturas y 14 dias de previsiones en total, filtros IB04/IB02, fechas historicas y pausa/reanudacion independiente de ambos timers. Ningun backup remoto se comprueba al cargar el panel.
