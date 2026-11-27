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

## Revisión estructural 27/11/2026

- AL05 / a4.txt: 7410 líneas con fecha y contenido, 23 líneas adicionales no vacías fuera de ese patrón, una fecha duplicada; SHA256 1153c1ca89b31200ffeba3f0d56bf00cfe0eaea59892dbe29d71232aa84e49b0.
- AL01 / a6.txt: 7417 líneas con fecha y contenido, 16 líneas adicionales no vacías fuera de ese patrón, una fecha duplicada; SHA256 f18272c7c54d0cdfbf186a290435077fe0e11843c07e65ef6c5cc212298ef96d.
- AL04 / a9.txt: 7411 líneas con fecha y contenido, 22 líneas adicionales no vacías fuera de ese patrón, una fecha duplicada; SHA256 7150c9a2daa6d199aade034efaf1d7a9b013cb4609a98dfba64caea7c03781d9.

La fecha duplicada es 29-12-15 en los tres archivos; una fila tiene día anual 364 y otra 363. 364 corresponde al 30/12/2015, de modo que existe una posible discordancia fecha/día anual; la causa y la corrección requieren comprobación frente a SIAR antes de asignar un día. Ejemplo AL01 filas 1623–1624. También hay líneas que contienen solo una fecha, incluida 31-12-15. No tratarlas automáticamente como observaciones ni como ausencias confirmadas.

Revisión del código: worker_v3.py toma fechas de filas CSV y planner_s1a.py da prioridad a fechas observadas con acción SKIP, incluso si faltan variables. Esto debe verificarse con pruebas aisladas y con los estados del ledger antes de importar. La presencia de filas totalmente vacías para algunas estaciones es un caso especial que no debe imitarse. La publicación toma snapshots y está vinculada a entradas pendientes; aún debe probarse cómo registrar la publicación de una importación.

Contraste adicional AL01: la fila histórica marcada 29/12/2015 y día anual 363 es compatible en las variables visibles con el registro V3 del 29/12/2015. La otra, marcada 29/12/2015 y día anual 364, es compatible con V3 del 30/12/2015 (TMax 17.5 frente a 17.47; TMin 11.0 frente a 10.96; radiación 12.3 frente a 12.28). Es evidencia puntual para AL01, no autoriza corregir automáticamente los demás archivos. Los originales se conservarán sin modificar.

## Ampliación AL05 — 27/11/2026

Consulta oficial SIAR de AL05 Fiñana, datos diarios del 01 al 05/06/2020, comparada con a4.txt (identidad indicada en cabecera). Las temperaturas, sus horas, humedad media/máxima/mínima, velocidad y dirección de viento, radiación y precipitación visibles son compatibles con la precisión aproximada del histórico. ET0 en archivo histórico: 6.00, 5.67, 8.51, 6.16, 7.11 mm. ET0 en web oficial: 5.89, 5.57, 8.19, 6.10, 6.93 mm, respectivamente. Las diferencias están pendientes de explicación y no justifican alterar el original.

El análisis estructural inicial de a4.txt encuentra 7410 filas con fecha y campos, 23 líneas que solo contienen fecha, dos discrepancias entre fecha y día anual, y una fecha duplicada. Concretamente, la línea 1622 dice 30-12-15 con día anual 365 (31 de diciembre) y la línea 1623 dice 29-12-15 con día 364 (30 de diciembre); la línea 1624 dice 29-12-15 y día 363. No se debe corregir automáticamente sin contrastar días y valores con SIAR o V3. Las filas de fecha sin valores no son datos recuperados ni ausencia confirmada.

En el momento de la revisión la cola V3 contiene múltiples estaciones andaluzas; AL05 estaba WAITING_QUOTA hasta 2026-11-28 01:00 UTC. No se cambió ni interrumpió la cola. Una futura importación requiere exclusión mutua con el worker, backup consistente de SQLite y CSV, manejo del diario y registro de publicación, además de pruebas aisladas.

## Criterio de fechas acordado y prueba aislada

El propietario solicita que las fechas incoherentes con el día anual y las líneas sin datos queden pendientes de comprobación por SIAR Sync, no como ausencias confirmadas ni observaciones. Una incoherencia puede afectar tanto a la fecha escrita como a la fecha derivada del día anual: ambas se aíslan. Si existe otra fila histórica con una de esas fechas, también se reserva para evitar introducir un falso positivo. Datos ya verificados en V3 deben mantenerse y resolverse mediante reconciliación, sin sobrescribirlos.

Se ha preparado tools/historicos_andalucia/andalucia_stage.py, herramienta aislada sin escritura a V3. Solo admite formato con estación en cabecera y siglo explícito. La prueba de AL05/a4.txt obtiene 7402 filas candidatas y 31 excepciones: 23 fechas sin datos, dos incoherencias de día anual, cinco valores u horas no interpretados y una fila con fecha afectada por evidencia ambigua. Cuatro pruebas unitarias verifican exclusión de fechas ambiguas, fechas sin datos, duplicados y horas no interpretadas. Estos resultados son preliminares y no validan todas las variables ni la importación.

Pendiente: guardar evidencia y resultados de la conversión, contrastar fechas y cobertura por estación, revisar estados previos CONFIRMED_NO_DATA y PENDING_RETRY, probar el planificador real y la publicación, y hacer backup consistente bajo exclusión mutua. No se ha creado ningún estado pendiente en SQLite ni se han escrito datos de producción. Las candidatas de fecha o día anual no deben añadirse ciegamente a PENDING_RETRY si ya existen observaciones verificadas.

## Inventario ampliado de todos los TXT — 27/11/2026

Se examinaron 121 archivos TXT, excluidos dos auxiliares identificados. De ellos 84 contienen cabecera reconocible y 37 utilizan otros formatos o carecen de cabecera; estas cifras son archivos, no estaciones validadas. Los alias de provincia HU→H y JA→J son propuestas para comparar con el catálogo, no identificaciones confirmadas. El inventario por archivo, con hash cuando se analizó la cabecera, se conserva en inventario-historicos-andalucia-20261127.json.

En la exploración preliminar de los 84 archivos con cabecera se obtuvieron 473968 filas candidatas, no importadas ni verificadas por estación, y 9361 excepciones de línea: 8134 líneas con solo fecha, 180 discrepancias fecha/día anual, 77 filas adicionales en fechas con evidencia ambigua, 893 valores u horas no interpretados, 75 fechas fuera de la ventana 2000–05/06/2020 y 2 filas duplicadas. Estas últimas se mantienen en cuarentena. Los años de dos dígitos fuera de esa ventana no se asignan automáticamente: podrían representar 1999, entre otras posibilidades, y necesitan interpretación documentada.

Tres archivos Huelva (h2, h10, h13) incluyen radiación pero no precipitación ni ET0. Se conserva el esquema explícito y se dejan ambas variables vacías; no se construyen valores. Se han ejecutado siete pruebas unitarias del preparador aislado, incluida la protección de fechas, duplicados, siglos y variables ausentes. Estos controles no suplen las cinco comparaciones por estación, ni el cotejo con el catálogo, ni las pruebas de integración con V3 y Git. Los 37 archivos sin cabecera se consideran pendientes y no producen candidatos de estación.

## Muestreo solicitado — 27/11/2026

El propietario solicita 10 fechas aleatorias por estación, verificando las variables meteorológicas frente a SIAR y sin descalificar por ET0. Para los 84 archivos con cabecera reconocida se han seleccionado diez fechas de filas candidatas por archivo. La selección es reproducible: se usa como semilla los primeros 16 dígitos hexadecimales de SHA-256 de cada fuente, y se comprueba que el archivo no haya cambiado. Se registra en muestreo-historicos-andalucia-20261127.json; todas las entradas se marcan PENDING_SIAR. Los alias y muestras no son validaciones.

Se ha hecho una comparación exploratoria de diez fechas adicionales en AL01 entre el archivo histórico y el CSV V3, pero no se ha comparado aún cada fecha con la web oficial ni se ha completado un cotejo automatizado campo por campo. No contabilizarla como cumplimiento del muestreo oficial. Fecha y variables, tolerancias y discrepancias deben registrarse por separado, distinguiendo SIAR web, API y CSV V3.

DADES EST.xlsx contiene hojas por provincia, incluyendo CADIZ, CORDOBA y GRANADA, con número, nombre y otros metadatos de estación. Esta información apoya propuestas de correspondencia pero no confirma que c1, co1 o g1 correspondan a esos números: se requiere cotejo de las observaciones. Los 37 archivos de otros formatos siguen sin muestreo ni identidad validada. No se ha importado ningún dato, ni activado pendientes en SQLite.

## AL05: diez fechas aleatorias frente a SIAR

Comprobación 27/11/2026 para AL05 Fiñana, fuente a4.txt SHA256 1153c1ca89b31200ffeba3f0d56bf00cfe0eaea59892dbe29d71232aa84e49b0. Se consultaron las diez fechas predeterminadas en el muestreo: 11/03/2001, 25/03/2006, 22/10/2008, 25/12/2008, 17/01/2009, 08/12/2011, 28/01/2015, 01/02/2016, 02/02/2019 y 19/03/2019. Los 12 campos meteorológicos comunes, incluidas horas de temperatura máxima y mínima, fueron compatibles en las diez fechas. Para los valores numéricos se usa tolerancia igual a medio último decimal mostrado en el histórico más 0,005 por la precisión de la pantalla web; horas exactas. La ET0 se conserva separada y no penaliza, de acuerdo con la instrucción; difiere en las diez fechas. Consultas en la web SIAR, una fecha cada vez; se preservan los valores y los resultados en docs/evidencia-muestreo/AL05-siar-web-20261127.json y AL05-comparacion-20261127.json. Datos capturados de la interfaz, no respuestas RAW de la API; el contraste es una muestra y no garantiza la totalidad de la serie.

Estado de la muestra AL05: PASS para identidad y variables observadas, pendiente de revisar anomalías, solapamientos, seguridad de la conversión y ejecución de importación. Las otras estaciones no se consideran validadas por este resultado. La preparación del 84 archivos se mantiene como PENDING_SIAR salvo esta evidencia puntual; la lista original de selección es inmutable, no un indicador vivo de estado.

## AL04: diez fechas aleatorias frente a SIAR

Comprobación 27/11/2026 de AL04 Tabernas, fuente a9.txt SHA256 7150c9a2daa6d199aade034efaf1d7a9b013cb4609a98dfba64caea7c03781d9. Se consultaron las diez fechas predeterminadas de 2000-06-29 a 2018-02-10. Los 12 campos meteorológicos comunes, incluidas ambas horas de temperatura extrema, resultan compatibles aplicando la tolerancia indicada para AL05 y horas exactas en las diez fechas. La ET0 se registra separada y difiere en diez fechas; no penaliza la muestra. Evidencia en docs/evidencia-muestreo/AL04-siar-web-20261127.json y AL04-comparacion-20261127.json. Datos capturados de la web, no RAW API. Muestra PASS, no es validación total del conjunto ni autorización por sí sola para modificar datos sin resolver seguridad de la importación.

## AL02: diez fechas aleatorias frente a SIAR

Comprobación 27/11/2026 de AL02 Almería, fuente a2.txt SHA256 06eda863db3b5418e5dd66e3267f56f8038f299a070770282afca2cb190d1fcd. Las diez fechas seleccionadas entre 2003 y 2020 tienen 12 campos meteorológicos comunes compatibles según la tolerancia de redondeo de AL05 y horas exactas. La ET0 se registra separada sin penalizar. Evidencias docs/evidencia-muestreo/AL02-siar-web-20261127.json y AL02-comparacion-20261127.json. Muestra PASS, no valida todas las filas ni importa datos.

## AL01: diez fechas aleatorias frente a SIAR

Comprobación 27/11/2026 de AL01 La Mojonera, fuente a6.txt SHA256 f18272c7c54d0cdfbf186a290435077fe0e11843c07e65ef6c5cc212298ef96d. Las diez fechas predeterminadas entre 2000 y 2019 tienen 12 campos meteorológicos comunes compatibles según la tolerancia de redondeo de AL05 y horas exactas. La ET0 se registra separada sin penalizar. Evidencias docs/evidencia-muestreo/AL01-siar-web-20261127.json y AL01-comparacion-20261127.json. Muestra PASS, no valida todas las filas ni importa datos.

## AL03: diez fechas aleatorias frente a SIAR

Comprobación 27/11/2026 de AL03 Níjar (inactiva), fuente a8.txt SHA256 f050ae8651c4d2fb668dad7ff4f3f6a927dfb159834e7290b13a15895119f5ac. Las diez fechas predeterminadas entre febrero y junio de 2000 tienen 12 campos meteorológicos comunes compatibles según la tolerancia de redondeo de AL05 y horas exactas. Se consultaron intervalos pequeños para agrupar fechas próximas. La ET0 se registra separada sin penalizar. Evidencias docs/evidencia-muestreo/AL03-siar-web-20261127.json y AL03-comparacion-20261127.json. Muestra PASS, no valida todas las filas ni importa datos.

## Muestreo acordado de dos bloques y AL06

El 27/11/2026 el propietario autorizó cambiar la muestra de diez días separados a dos bloques de cinco días consecutivos por estación, con periodos separados cuando sea posible. Los 79 archivos restantes con cabecera reconocida tienen un plan reproducible derivado de su hash en muestreo-bloques-andalucia-20261127.json. Las 37 fuentes sin cabecera requieren identificación y planificación adicional. Un plan no constituye una validación ni una importación.

AL06 Virgen de Fátima-Cuevas de Almanzora, a12.txt SHA256 1585042346240bc2a7b4cab5fc837855b8d7c8d5c92cf160be796eba20b64996: consultas oficiales SIAR de 16–20/08/2002 y 25–29/10/2013. En 10/10 días coinciden los 12 campos meteorológicos comunes según precisión del histórico, con horas exactas. ET0 se registra pero no penaliza, y difiere en las diez fechas. Evidencia en docs/evidencia-muestreo/AL06-siar-web-20261127.json y AL06-comparacion-20261127.json. Muestra PASS; no valida la integridad de cada fila ni supone que se hayan importado datos. Siete pruebas del conversor y dos pruebas de selección de bloques pasan localmente.

## AL07: contraste de dos bloques

AL07 Huércal-Overa, a5.txt SHA256 fab27a8702c5d44fe08b6fd6ad1a7bedd3771a50931308bd73b48fc32a0f7f01: consulta web oficial SIAR el 27/11/2026 de 20–24/05/2001 y 28/08–01/09/2014. Los 12 campos meteorológicos comparables son compatibles en las diez filas con la precisión del histórico y horas exactas. ET0 se registra sin penalizar y difiere en los diez días. Evidencia en docs/evidencia-muestreo/AL07-siar-web-20261127.json y AL07-comparacion-20261127.json. Muestra PASS, sin importación ni garantía de integridad de todas las filas.

## AL08: contraste de dos bloques en web pública

AL08 Cuevas de Almanzora, a3.txt SHA256 160148b0c9203c66e2da4f445ea13572181ac7fd031179d968ed357f98157be9: consulta web pública SIAR el 27/11/2026 de 26–30/09/2000 y 12–16/09/2011. Los 12 campos meteorológicos comparables son compatibles en las diez filas con la precisión del histórico y horas exactas. ET0 se registra sin penalizar. Evidencia en docs/evidencia-muestreo/AL08-siar-web-20261127.json y AL08-comparacion-20261127.json. Muestra PASS, sin importación ni garantía de integridad de todas las filas. No se empleó la API autenticada para esta consulta web.

## Comparador sin API

A petición del propietario no se usará la cuota agotada de la API SIAR para estas comprobaciones. Las referencias se obtienen mediante la web pública. Se añadió tools/historicos_andalucia/andalucia_compare_web.py, que compara sin red las diez filas capturadas con la fuente local, verifica las fechas y el hash del plan, excluye ET0 de la calificación y devuelve PASS, REVIEW o INCOMPLETE. Incluye pruebas con discrepancias y datos ausentes; 13 pruebas de las tres piezas pasan. Se comprobó el CLI con los datos reales de AL08, resultado 10/10 PASS.

Limitación: este comparador no navega ni descarga por sí mismo de la web y no existe un proceso autónomo ejecutándose tras finalizar la sesión. Los datos capturados deben alimentarse explícitamente; faltan validación y comprobaciones por las otras estaciones y, si se desea una automatización autónoma, implementar y probar la navegación y el almacenamiento de progreso por separado, respetando los límites de la web. No se consultó la API para el contraste de AL08.

## Automatización mediante la web pública: avance posterior

Se ha preparado `tools/historicos_andalucia/andalucia_web_runner.py` para consultar mediante navegador automatizado los dos bloques de cinco días, comparar las variables meteorológicas sin penalizar ET0 y conservar el progreso y las respuestas en SQLite separada de V3. El proceso se detiene si falla una consulta y puede reanudar los bloques ya guardados. La selección y el archivo fuente se vinculan mediante sus hashes. No utiliza la API SIAR.

En el servidor se ha preparado un entorno aislado en `/home/josep/siar-auditoria-web-venv`, el navegador en `/home/josep/siar-auditoria-web-browsers` y el código en `/home/josep/siar-auditoria-web-code`. Han pasado 16 pruebas automatizadas y se han recuperado desde la web los cinco días 26–30/09/2000 de AL08 mediante navegador sin interfaz. Esta prueba de navegación no equivale a ejecutar el flujo completo sobre los históricos del propietario.

La selección disponible contiene 79 fuentes con cabecera, incluyendo AL06, AL07 y AL08 ya contrastadas; estas deben excluirse para no repetir consultas. Otros 37 archivos sin cabecera todavía necesitan identificación y tratamiento específicos. Los archivos originales están en el ordenador Windows y aún no están disponibles en el servidor. No hay un proceso de auditoría ejecutándose ni se ha importado dato alguno a V3. Falta verificar el flujo completo con las fuentes reales, automatizar y revisar el registro exportable de resultados y completar el tratamiento de los archivos sin cabecera.
