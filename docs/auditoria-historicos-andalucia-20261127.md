# Auditoría inicial de históricos manuales de Andalucía

Fecha de revisión: 2026-11-27. Estado: EN CURSO. El propietario autoriza la importación progresiva tras contrastar las variables y preparar pruebas y copias de seguridad.

## Procedencia
Archivos aportados por el propietario, recopilados manualmente desde SIAR para el TFM. Origen local: carpeta estacions andalucia, estructurada por provincias. Se observa diversidad de formatos; no debe inferirse el código de estación exclusivamente del nombre de archivo. Ejemplo: Almería/a1.txt tiene cabeceras AL10. La sección inicial de Almería/GENERAL ALMERIA.csv identifica AL02.

Fuente de contraste operativa: /home/josep/siar-sync-v3-data/csv/AL02_datos_completos.csv, leída el 27/11/2026. Es una fuente derivada SIAR V3, no una comprobación independiente de la web SIAR.

## Muestra preliminar AL02
Se inspeccionaron las cinco fechas 01–05/06/2020, en la sección inicial del CSV histórico y en AL02 de V3. Las temperaturas, humedad, viento, dirección, radiación, precipitación y horas observadas son compatibles en estas filas con el redondeo visible del archivo histórico; esto no equivale a una auditoría automatizada completa de cada campo.

| Fecha | ET0 histórica | EtPMon V3 | Diferencia histórica menos V3 |
|---|---:|---:|---:|
| 01/06/2020 | 5.18 | 5.195760 | -0.015760 |
| 02/06/2020 | 4.61 | 4.607870 | +0.002130 |
| 03/06/2020 | 5.26 | 5.240662 | +0.019338 |
| 04/06/2020 | 5.48 | 5.445802 | +0.034198 |
| 05/06/2020 | 5.42 | 5.406935 | +0.013065 |

Las discrepancias ET0 no se explican todas por redondear directamente EtPMon a dos decimales. Causa no determinada: no corregir ni sustituir valores sin investigación; comprobar definiciones, fuente original y posibles cambios posteriores en SIAR; no atribuir el valor histórico al cálculo posterior del propietario.

## Contraste con portal oficial SIAR
Consulta efectuada el 27/11/2026: AL02 - Almería, datos diarios del 01/06/2020 al 05/06/2020, todas las variables, en https://servicio.mapa.gob.es/siarweb/consultaDatos/consultaDatos. Resultado visible de ET0, respectivamente: 5,20; 4,61; 5,24; 5,45; 5,41 mm. Coincide con V3 a la precisión mostrada por la web. El archivo histórico coincide en una de estas cinco fechas a dos decimales. El propietario aclara que la ET0 de estos archivos se descargó directamente de SIAR y que sus cálculos propios fueron posteriores. La causa de las diferencias permanece sin determinar. La consulta es evidencia puntual; no valida otras estaciones ni el histórico entero.

## Ampliación: AL01 y AL04
AL01: archivo original Almería/a6.txt, identificado por sus cabeceras; contraste con el CSV V3 para el 01–05/06/2020. ET0 histórica / EtPMon V3: 5.27 / 5.256639; 4.48 / 4.470225; 5.62 / 5.570457; 5.26 / 5.231101; 6.02 / 5.940239. Los cinco pares difieren al comparar con dos decimales. Las variables comunes observadas son compatibles con el redondeo de las cifras históricas; no se ha hecho un cotejo automático exhaustivo ni una validación de las otras fechas.

AL04: archivo Almería/a9.txt, identificado por cabeceras. Contraste directo en web oficial SIAR para 01–05/06/2020: temperaturas máxima, mínima y media, sus horas de máxima y mínima, y humedades máxima, mínima y media son compatibles con la precisión visible del original. La respuesta consultada solo presentó temperatura y humedad: no se verificaron en AL04 viento, radiación, precipitación ni ET0. No se encontraron filas para estas fechas en el CSV V3 AL04 durante la revisión.

En V3 el CSV conserva, en estas muestras, decimales adicionales, por ejemplo AL01 05/06/2020 EtPMon 5.940239; el redondeo a dos decimales se hizo para la presentación comparativa, no modificando el CSV. No se ha auditado todo el proceso de escritura o su fidelidad al payload original.

## Pendiente
- Inventario y hashes de cada fuente; identificar cada archivo y las secciones con código de estación y evidencia, especialmente archivos sin cabecera.
- Comparación reproducible por fecha y variable, con reglas explícitas de redondeo y tratamiento de valores ausentes.
- Contrastar cinco fechas por estación con V3 cuando exista solapamiento y, si no existe, con el portal oficial SIAR. Registrar respuesta y evidencia.
- Resolver discrepancias y determinar compatibilidad con las 23 columnas; las variables ausentes no deben inventarse.
- Evaluar duplicados, periodos, huecos, conflictos y cómo planificador y SQLite reconocen datos importados.
- Preparar prueba aislada y copias de seguridad antes de cualquier incorporación. No se han modificado los CSV, SQLite ni trabajos de producción en esta revisión.

## Decisión del propietario
Conservar e importar los valores de ET0 históricos originales junto a las variables comprobadas, sin sobrescribir valores de V3 en conflicto. Registrar procedencia y discrepancias. Campos faltantes quedan vacíos y documentados para posible recuperación futura en V3.A; no se deben reconsultar todas las fechas únicamente por dichos campos. Comprobar previamente la interacción con la cola, los huecos y publicación.
