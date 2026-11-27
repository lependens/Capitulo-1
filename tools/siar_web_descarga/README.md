# Prototipo de descarga de datos diarios desde la web de SIAR

Estado: experimental; independiente de SIAR Sync V3. Usa un navegador Chromium automatizado sobre el formulario público, sin llamar a la API de SIAR ni usar su clave. Conserva los valores y encabezados tal como aparecen en la web, con su precisión visible; no son equivalentes demostrados de la respuesta de la API ni de sus 23 columnas canónicas.

## Ejecución en el servidor

```bash
cd /home/josep/siar-auditoria-web-code/tools/siar_web_descarga
PLAYWRIGHT_BROWSERS_PATH=/home/josep/siar-auditoria-web-browsers \
/home/josep/siar-auditoria-web-venv/bin/python siar_web_descarga.py \
  --station AL10 --start 2012-03-01 --end 2012-03-31 \
  --output /home/josep/siar-descarga-web-pruebas
```

Se puede cambiar estación y fechas. Divide el intervalo en bloques consecutivos de hasta 31 días, con pausa mínima de 10 segundos. Por defecto procesa como máximo un bloque por ejecución; `--max-blocks 12` permite hasta 12. Repetir el mismo comando, con igual fecha inicial y tamaño de bloque, evita volver a consultar bloques guardados. No ejecutar dos procesos simultáneamente sobre el mismo directorio de salida.

Los CSV se generan por estación con todas las columnas de la tabla visible; la base `progreso.sqlite` conserva fecha de captura, bloques, valores, hash SHA-256 y lista de fechas no devueltas. Una fecha no devuelta queda pendiente, no como ausencia confirmada. Los bloques con fechas pendientes se guardan y no se reintentan automáticamente. Una respuesta vacía, error o cambio de estructura interrumpe el programa sin registrar la consulta como resuelta. No hay reintentos automáticos ni programación de nuevas ejecuciones. Es una captura de la tabla web: si el portal pagina, trunca, retrasa filas o modifica el formato, deben revisarse los pendientes y la evidencia. El programa regenera el CSV a partir de la base guardada al reanudar.

## Prueba real 27/11/2026

AL10, 14–20/03/2012: siete filas, 21 columnas web, sin fechas pendientes. AL10, 01–31/03/2012: 31 filas, sin fechas pendientes. Se verificaron cabeceras y primera fila del CSV. Repetir el comando mensual mantuvo un único bloque guardado. Cuatro pruebas automatizadas cubren intervalos, fechas ausentes, rechazos y preservación de valores.

La prueba no valida todas las estaciones, periodos, esquemas ni escenarios de fallos de la web. No escribe en los CSV ni en SQLite de V3, ni importa sus resultados. Queda por diseñar y probar de forma separada la reconciliación, normalización de variables y precisión, tratamiento de huecos, bloqueo de ejecuciones concurrentes y su integración en V3.
