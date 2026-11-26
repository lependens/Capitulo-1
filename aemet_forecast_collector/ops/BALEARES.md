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
