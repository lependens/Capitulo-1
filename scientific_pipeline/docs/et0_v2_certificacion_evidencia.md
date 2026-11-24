# Evidencia de certificación ET₀ V2 — revisión pendiente

El ZIP retrospectivo Kaggle `370251881` conserva 108 celdas AHC, 105 ML, 105 ANN A y 105 ANN E; en el árbol congelado se inventariaron 3.885 checkpoints de candidatos ML y 525 para cada ANN, además de 735 modelos ML y 105 modelos por variante ANN. La mera cuenta no reemplaza el cotejo de hashes y relaciones; el expediente previo registra los cuatro índices y su verificación.

El freeze SHA256 `97acad7ce7cc39c65d82f6efdc36290ed0325caaef2dedb69d9ffbefe1addfbd` declara `FROZEN_BEFORE_RETROSPECTIVE_ACCESS`. La revisión de 9.612 filas reprodujo n, MAE, RMSE y sesgo; R² se verificó salvo grupos indefinidos o casi constantes. Se contrastaron las 2.081.376 claves y valores del CSV derivado y se realizó una comparación de ida y vuelta CSV/Parquet. La prueba aislada de tres modelos figura en `et0_v2_contrato_inferencia.md`.

La copia cifrada y restauración independiente del ZIP retrospectivo están registradas en snapshot `backup_20261116T132334Z_7a6c2b24`, SHA256 `2fddd95c2374a95b4bd1885a6d13f4455ed89d45b8e0d9e854e4d846d40cd420`. El ZIP derivado se restauró desde `backup_20261116T163815Z_35960ca0`, SHA256 `c502530fa70321246c26ed0c9c59aaf766d4ac619251bc1ab99f854695701690`. La diferencia de SHA del notebook original sigue documentada, sin causa demostrada.

## Comparación exploratoria entre fórmulas

Al intersectar estación y fecha sobre la salida anual, las tres fórmulas y PM calculado tienen 19.251 pares comunes. RMSE en mm/día: HS-Ra 0,9491; HG-Rs 0,5670; Valiantzas 0,6236. El denominador de 19.272 es el número de claves observadas en el archivo, no días de catálogo. El contraste no iguala sensores, ni usa observaciones instrumentales independientes. La tabla derivada y sus hashes permanecen pendientes; no utilizar estas cifras como clasificación operativa sin evaluar cobertura y disponibilidad de entradas.

El catálogo continúa DRAFT hasta verificar integralmente el árbol instalado y la firma del catálogo definitivo. Las salidas originales no se han alterado.

## Verificación adicional del árbol instalado (24-11-2026)

En `/srv/siar-project/scientific-artifacts/et0-v2/derived/v1/` se ha leído el catálogo DRAFT y comprobado el tamaño y SHA256 de todos sus 111 archivos declarados: 111 coinciden y no hay discrepancias. SHA256 del catálogo: `f497074a39d25a38d637329a87a652c470875502a2c9f796cf2eca6706148d81`. Esta verificación no constituye por sí misma firma ni aprobación de publicación.

Se contrastó la cadena SHA256 desde `freeze.json` hasta los cuatro índices, las celdas congeladas, los modelos y los checkpoints incluidos en el ZIP retrospectivo. Índices: AHC `d475ec833ace324d8fb83ba038566c1e6a795707a12c8aa5272ebd485ef9c86e`; ML `8a78db402354a9da42dc8be18237f02c5d12a813fdb448aa5da2688534fd4bd7`; ANN A `0a10d6416c47f1745c4f717edbea429f8985b4211504ae82a96bc59a433f6f58`; ANN E `81524c1cecb6fff1a97acba5f18168e943c2d4f6367d37fbb4f4485883f5f256`. La declaración de congelación y estos enlaces acreditan integridad; no son por sí solos un registro exhaustivo de todos los accesos históricos a datos.

La lectura del archivo de dataset para calcular su hash desde Windows tuvo lugar tras la autorización de Dirección y antes del journal de Kaggle; no debe confundirse con entrenamiento o selección. La incidencia sobre el hash del notebook permanece abierta como procedencia, sin atribuirle una causa no demostrada.

La cohorte `common_core` incluye las familias principales FORMULA, AHC, ML, ANN A y ANN E. Para variantes individuales de ML, una comparación pareada requiere su propia intersección y denominador; no presentar esos subconjuntos como si fueran automáticamente idénticos. `eligibility_status` representa presencia de salida y no sustituye la elegibilidad científica completa.
