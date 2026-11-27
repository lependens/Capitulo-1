> **Lectura posterior, 27/11/2026:** la [auditoría científica independiente TFG–TFM–ET₀ V2](et0_v2_auditoria_cientifica_tfg_tfm.md) complementa esta certificación técnica con una comparación histórica y bibliográfica focalizada y sus cuestiones pendientes. Se conserva el alcance de las comprobaciones y el estado científico NOT_PUBLISHED.

# Evidencia de certificación técnica ET₀ V2

El ZIP retrospectivo Kaggle `370251881` conserva 108 celdas AHC, 105 ML, 105 ANN A y 105 ANN E; en el árbol congelado se inventariaron 3.885 checkpoints de candidatos ML y 525 para cada ANN, además de 735 modelos ML y 105 modelos por variante ANN. La mera cuenta no reemplaza el cotejo de hashes y relaciones; el expediente previo registra los cuatro índices y su verificación.

El freeze SHA256 `97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd` declara `FROZEN_BEFORE_RETROSPECTIVE_ACCESS`. La revisión de 9.612 filas reprodujo n, MAE, RMSE y sesgo; R² se verificó salvo grupos indefinidos o casi constantes. Se contrastaron las 2.081.376 claves y valores del CSV derivado y se realizó una comparación de ida y vuelta CSV/Parquet. La prueba aislada de tres modelos figura en `et0_v2_contrato_inferencia.md`.

La copia cifrada y restauración independiente del ZIP retrospectivo están registradas en snapshot `backup_20261116T132334Z_7a6c2b24`, SHA256 `2fddd95c2374a95b4bd1885a6d13f4455ed89d45b8e0d9e854e4d846d40cd420`. El ZIP derivado se restauró desde `backup_20261116T163815Z_35960ca0`, SHA256 `c502530fa70321246c26ed0c9c59aaf766d4ac619251bc1ab99f854695701690`. La diferencia de SHA del notebook original sigue documentada, sin causa demostrada.

## Comparación exploratoria entre fórmulas

Al intersectar estación y fecha sobre la salida anual, las tres fórmulas y PM calculado tienen 19.251 pares comunes. RMSE en mm/día: HS-Ra 0,9491; HG-Rs 0,5670; Valiantzas 0,6236. El denominador de 19.272 es el número de claves observadas en el archivo, no días de catálogo. El contraste no iguala sensores, ni usa observaciones instrumentales independientes. La tabla derivada y sus hashes permanecen pendientes; no utilizar estas cifras como clasificación operativa sin evaluar cobertura y disponibilidad de entradas.

El catálogo anterior se conserva DRAFT; el catálogo técnico definitivo está VERIFIED y la publicación sigue NOT_PUBLISHED. Las salidas originales no se han alterado.

## Verificación adicional del árbol instalado (24-11-2026)

En `/srv/siar-project/scientific-artifacts/et0-v2/derived/v1/` se ha leído el catálogo DRAFT y comprobado el tamaño y SHA256 de todos sus 111 archivos declarados: 111 coinciden y no hay discrepancias. SHA256 del catálogo: `f497074a39d25a38d637329a87a652c470875502a2c9f796cf2eca6706148d81`. Esta verificación no constituye por sí misma firma ni aprobación de publicación.

Se contrastó la cadena SHA256 desde `freeze.json` hasta los cuatro índices, las celdas congeladas, los modelos y los checkpoints incluidos en el ZIP retrospectivo. Índices: AHC `d475ec833ace324d8fb83ba038566c1e6a795707a12c8aa5272ebd485ef9c86e`; ML `8a78db402354a9da42dc8be18237f02c5d12a813fdb448aa5da2688534fd4bd7`; ANN A `0a10d6416c47f1745c4f717edbea429f8985b4211504ae82a96bc59a433f6f58`; ANN E `81524c1cecb6fff1a97acba5f18168e943c2d4f6367d37fbb4f4485883f5f256`. La declaración de congelación y estos enlaces acreditan integridad; no son por sí solos un registro exhaustivo de todos los accesos históricos a datos.

La lectura del archivo de dataset para calcular su hash desde Windows tuvo lugar tras la autorización de Dirección y antes del journal de Kaggle; no debe confundirse con entrenamiento o selección. La incidencia sobre el hash del notebook permanece abierta como procedencia, sin atribuirle una causa no demostrada.

La cohorte `common_core` incluye las familias principales FORMULA, AHC, ML, ANN A y ANN E. Para variantes individuales de ML, una comparación pareada requiere su propia intersección y denominador; no presentar esos subconjuntos como si fueran automáticamente idénticos. `eligibility_status` representa presencia de salida y no sustituye la elegibilidad científica completa.

## Repetición de comprobaciones

Los comprobadores añadidos a esta PR permiten verificar, respectivamente, la cadena de artefactos dentro del ZIP retrospectivo y las 9.612 filas de métricas a partir de las predicciones expandidas. Uso: `python scientific_pipeline/tools/verify_et0_v2_freeze.py RUTA_DEL_ZIP` y `python scientific_pipeline/tools/verify_et0_v2_metrics.py DIRECTORIO_CON_CSV`. El directorio debe contener los archivos originales de la inspección `predictions.csv.gz` y `retrospective-metrics.csv`. Los resultados son pruebas de integridad y consistencia aritmética, no sustituyen la revisión científica independiente ni prueban ausencia absoluta de accesos no registrados.

La auditoría preservada del lote 01 registra 51 candidatos con avisos de convergencia durante los folds y siete reajustes Elastic Net sin alcanzar la tolerancia, ninguno seleccionado como método ML principal. Los resultados de variantes individuales afectadas conservan esa limitación; no se reajustaron para eliminar el aviso.

## Trazabilidad de checkpoints por lote conservado

Se cotejaron por ruta y SHA256 los 4.935 checkpoints incluidos en el árbol congelado con los lotes conservados. Primer lote disponible con bytes idénticos: ML 3.885 en `368475953`; ANN A 304 en `368475953`, 85 en `369095500` y 136 en `369223074`; ANN E 101 en `369632717`, 101 en `369691836`, 206 en `369788169` y 117 en `369953511`. La primera aparición comprobada no demuestra por sí sola el instante de creación. La relación detallada de ruta, tamaño, hash y lote se encuentra en el archivo de inspección `checkpoint-trace.json`, SHA256 `4165faa41500314caa424f75318dfff5aa3a2e63e0ea05f34b4d7e82ae12f1eb`; su preservación remota y restauración independiente se verificaron.

Los comprobadores de esta PR se ejecutaron de nuevo: cadena del freeze PASS y 9.612 filas de métricas PASS, con la salvedad ya indicada de R² indefinido o casi constante. El catálogo candidato histórico se conserva; el catálogo técnico definitivo está VERIFIED y el DRAFT instalado no se ha modificado.

## Valores fuera de rango físico

Una comprobación de todos los valores presentes no encontró valores infinitos ni NaN. Sí hay estimaciones negativas (el conteo es por fila de modelo, familia y calendario; no por fechas únicas): ANN A 2, ANN E 1, regresión lineal 377, Ridge 373 y Elastic Net 367. Los tres valores ANN corresponden a IB02 2023-01-27 Valiantzas mensual (ANN A −0,03365 y ANN E −0,03292 mm/día) e IB10 2023-02-27 Valiantzas mensual (ANN A −0,02169 mm/día). Son salidas del experimento sin restricción no negativa y deben identificarse como tales; no se han truncado ni modificado en las métricas o predicciones. La interfaz debe mostrar esta limitación y conservar las cifras originales. Una comparación que solo considere ET₀ físicamente no negativa requeriría una política explícita y una evaluación versionada posterior, no una corrección silenciosa.

Se verificaron además las dos razones de cobertura en las 9.612 filas: `coverage_calendar = predicted_days / calendar_days` y `coverage_pm_own = pm_pairs_own / pm_days` cuando el denominador es positivo. Son denominadores distintos; un pronóstico puede existir en un día sin referencia PM. La comprobación aritmética de la razón no sustituye la comprobación independiente del calendario de cada estación.

El paquete adicional de seis archivos con el catálogo candidato y la trazabilidad fue firmado por el propietario, recibido con `published-verified` (lote `science-20261124T124326Z-d8398c59`) y figura en el snapshot `backup_20261124T134355Z_2e5ca9e9`, cuyo servicio registra `REMOTE_VERIFIED`. La restauración independiente de los seis archivos del paquete pasó: se cotejó cada SHA256 con el manifiesto remoto y el original del servidor. Recibo `/srv/siar-project/ingestion-receipts/et0-v2-certification-evidence-restore-20261124.json`, SHA256 `8f578fb13b9fe5617f27b709f414bae9905cde4a8e03d798a3d82f8aa072e931`; copia restaurada `/tmp/et0-cert-restore-wwql1l4t`. El manifiesto del snapshot tiene SHA256 `ce4fcad6e58fe5ab3281a49743ed0e0e46d66399b7c284403325c97d491ae6e5`. El catálogo candidato histórico se conserva por trazabilidad; el catálogo técnico definitivo consta por separado.

Una comprobación independiente reconstruyó la lista ordenada de claves estación-fecha, con la serialización del pipeline congelado, para cada una de las 9.612 filas de métricas. Todos los SHA256 de cohorte coinciden y las claves de las filas de métricas son únicas. El procedimiento reproducible está en `scientific_pipeline/tools/verify_et0_v2_cohorts.py`; requiere el directorio con las predicciones y métricas originales de inspección.

Para los denominadores de calendario se reconstruyeron todas las fechas de 2020–2025, incluidos bisiestos, para las nueve estaciones retrospectivas y para cada agrupación declarada; coinciden las 9.612 filas. Esta prueba verifica el calendario explícito de nueve estaciones, no identifica por sí sola indisponibilidades de sensores.

Se ha preparado un catálogo separado con estado técnico `VERIFIED` y publicación `NOT_PUBLISHED`. Su SHA256 es `fef85f0a6ae981792d867a9de2b10b9f7787bc680461820d2284cd20453df4ab`. El inventario que lo vincula fue firmado por el propietario y recibió `published-verified` en el servidor (lote `science-20261124T155413Z-9e2d549c`). El archivo `/srv/siar-project/scientific-artifacts/et0-v2-verified-catalog-20261124/catalog.json` reproduce ese SHA256. El snapshot `backup_20261124T160201Z_5a69c7b9` contiene el catálogo y su recibo de ingreso y consta como `REMOTE_VERIFIED`. Su restauración independiente desde la copia cifrada pasó con el SHA256 anterior; recibo `/srv/siar-project/ingestion-receipts/et0-v2-verified-catalog-restore-20261124.json`, SHA256 `39a71985d727ebd135d8fc6c74de77156e0edc554698e17cfcce4129752f04c8`. El SHA256 del manifiesto del snapshot es `1c223f87523a62d5d25edba71db3f5907ccdf768e6e929fffbb04190be62f307`. El catálogo DRAFT anterior permanece intacto; la aprobación y publicación siguen reservadas a Dirección.
