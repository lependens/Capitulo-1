# AEMET Forecast: estaciones SIAR de Baleares

Configuración derivada de estaciones_baleares.csv, columna Termino, y códigos municipales de AEMET. La predicción es para la capital municipal y no un pronóstico puntual en la estación.

| SIAR | Nombre | Municipio | Código AEMET |
|---|---|---|---|
| IB01 | Santa Eulalia | Santa Eulària des Riu | 07054 |
| IB02 | Inca | Inca | 07027 |
| IB03 | Manacor | Manacor | 07033 |
| IB04 | Son Ferriol | Palma | 07040 |
| IB05 | Felanitx | Felanitx | 07022 |
| IB06 | Sa Pobla | Sa Pobla | 07044 |
| IB10 | Calvià | Calvià | 07011 |
| IB11 | S’Estany des Peix | Formentera | 07024 |

La captura consulta una vez por municipio y reutiliza el resultado entre estaciones asociadas; separa municipios tres segundos. La programación permanece cada seis horas. El límite general publicado por AEMET es 40 solicitudes por minuto por usuario, con posibles límites adicionales y otras aplicaciones que compartan cuota. El servicio espera hasta 4 horas para completar el lote y los reintentos; las ejecuciones simultáneas de la unidad no se lanzan.

El snapshot conserva los campos existentes y añade station_code, station_name, municipality_id y municipality_name. Los valores disponibles son temperatura y humedad mínimas y máximas; la identidad de emisión permanece literal. La web SIAR debe eliminar su filtro exclusivo IB04 y relacionar cada fila por station_code, indicando el municipio. La evidencia de backup es histórica, no una comprobación remota actual.

Rollback: comprobar que capture.service no está activo y restaurar desde /var/backups/aemet-baleares-20261126/ config.toml, capture.py, dashboard_data.py y emissions.py a sus ubicaciones respectivas, respetando propietarios y modos; retirar solo el drop-in 40-baleares-timeout.conf, hacer daemon-reload y verificar timer, exportador y hashes. No modificar RAW ni respaldos. La muestra de recuperación histórica previa no sustituye una restauración de un lote ampliado.

## Resultado de la captura controlada

El 26/11/2026 14:48 UTC se inició el lote controlado run_20261126T144810.746682Z_24c571c1dab8. Resultado parcial: IB01, IB02, IB03, IB05, IB06, IB10 e IB11 completos; IB04 falló con HTTP 429 tras reintento. Se verificó el manifiesto local; no se han modificado capturas históricas. Las primeras ejecuciones naturales de la configuración ampliada y la restauración independiente de este lote siguen pendientes. El snapshot expone siete fechas para cada uno de los siete destinos completos.

La instalación actual de la web conjunta ejecuta /opt/siar-forecast/aemet-read/report.py con una copia anterior del lector y combined.js filtra exclusivamente IB04. El equipo SIAR debe consumir /var/lib/aemet-forecast/export/status.json con su identidad autorizada, y filtrar captures y forecasts por station_code. Hasta ese cambio la web no refleja automáticamente la ampliación.

Ante un fallo en la publicación atómica, el directorio oculto de staging se conserva para recuperación. La prueba verifica bytes RAW y ambos manifiestos, y simula su publicación tras resolver el fallo. En operación debe revisarse la causa y verificarse la integridad antes de recuperar; no se debe sobrescribir ningún run existente.
