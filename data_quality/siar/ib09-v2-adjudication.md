# IB09: evidencia Quality para V2

Estado: IB09-OFFICIAL-PERIOD-VERIFICATION = UNRESOLVED; IB09-V2-SCIENTIFIC-ADMISSION = PENDING. Esta publicación no activa el prototipo ni modifica el FAIL de Quality V1. El contrato y la integración corresponden a 05.

## Fuentes y límites

El catálogo auditado `estaciones_baleares.csv` y el POST SIAR-QA-POST-20261020T193140Z coinciden en IB09 Artá, CLOSED, 2005-10-24 a 2018-10-10, límites inclusivos. El POST registra FAIL por una fila exterior. Su revisión independiente publicada indica que comprobó periodos, pero no incluye aquí la respuesta original de Info/ESTACIONES, por lo que no constituye una verificación independiente reproducible de las fechas.

Consulta de fuentes el 11/11/2026: el portal oficial [SiAR - consulta de datos](https://servicio.mapa.gob.es/siarweb/consultaDatos/inicio) enumera IB09 - Artá como inactiva, sin certificar las fechas históricas en la información consultada. El endpoint de Info/ESTACIONES del worker requiere autenticación SIAR y no se ha recuperado su respuesta preservada en este checkout. Se revisaron catálogo, POST vigente, informe consolidado y certificación independiente versionados. No se afirma una nueva consulta autenticada ni que las fechas hayan sido verificadas independientemente. El cambio de día por huso es hipótesis no demostrada. CATALOG-TZ-001 permanece OPEN / NON_BLOCKING.

## Adjudicación y política propuesta

El raw SHA256 `2cbd9bb289404264fc78dad21322562d9d22b64ce4bbc87e51512ca763ea7a8a` contiene 4423 filas, una de ellas 2005-10-23 fuera del periodo del catálogo. Las 4422 interiores tienen fechas únicas, ordenadas y estación correcta. Los 313 días restantes del periodo coinciden exactamente con el ledger como CONFIRMED_NO_DATA. La presencia de un día exterior explica el FAIL de estación, sin convertir las filas interiores en PASS V1. Controles descriptivos de numericidad, finitud, humedad [0,100], orden Tmin/Tmax, y no negatividad de viento, radiación y precipitación no detectaron incidencias en filas interiores. No se certifica plausibilidad meteorológica ni se inventan valores.

Propuesta: excluir la fila exterior de la futura vista V2, manteniéndola en raw. El prototipo `scientific_pipeline/src/siar_science/quality_ib09_v2.py` es explícitamente opt-in, está limitado a estación IB09, FAIL, fechas del catálogo y SHA del snapshot, y llama a los controles por método existentes. La función recibe un SHA que el llamador debe verificar sobre bytes reales; comparar cadenas no prueba por sí mismo la procedencia de los valores de fila. 05 debe verificar el raw y vincular las filas a esos bytes, registrar la versión de política en linaje y aplicar el gate de verificación oficial antes de cualquier uso. No se modifica ni se sustituye el estado histórico de entrada. El pipeline V1 continúa rechazando FAIL.

## Inventario y soporte

`ib09_v2_inventory.json` puede reproducirse ejecutando `python data_quality/siar/tools/ib09_v2_inventory.py`. Los conteos son preliminares para las 4422 observaciones oficiales del catálogo: PM 4390, HS-Ra 4406, HG-Rs 4406, Valiantzas 4400; intersección de los cuatro 4390. Las razones por variable se solapan y no deben sumarse. Los recuentos expresan disponibilidad y dominio, no resultados calculados ni capacidad de entrenamiento. La elegibilidad para comparar contra PM requiere la intersección con PM; 05 fijará soporte, particiones y validación temporal.

El inventario incluye días oficiales, observados, ausencias y disponibilidad por año 2005–2018. 2005 y 2018 son años oficiales parciales; 2017 y 2018 presentan 149 y 102 ausencias, respectivamente. No se fija TEST ni se consultan métricas científicas para elegirlo. La regla proporcional de Dirección está pendiente del contrato de 05.

## Validación

En Python 3.12.14 con scikit-learn 1.9.1 instalado aisladamente: 106 tests científicos PASS. El prototipo tiene cinco pruebas específicas, incluida separación de viento para PM/HS, rechazo de otro SHA/estación y exclusión de la fecha exterior. No se ejecutaron campañas ni se alteraron raw, POST o Dataset V1. Los tests no sustituyen la validación científica ni la evidencia oficial pendiente.
