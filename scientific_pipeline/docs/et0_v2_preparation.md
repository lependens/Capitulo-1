# Preparación de ET₀ V2

El [protocolo versionado](../experiments/et0-v2-preparation.json) y su consumidor preparan la vista DEVELOPMENT hasta el 31/12/2019. La salida objetivo es `PM_FAO56_V1` calculada. La evaluación 2020–2025 será retrospectiva. El inventario de ese bloque solo contabiliza disponibilidad de entradas y controles QC. Los resultados finales, los ajustes AHC y los entrenamientos requieren un lanzamiento separado.

## Excepción IB09 de Dirección 00

La [decisión del 11/11/2026](../../data_quality/siar/ib09-handoff-v2-development/decision.md) autoriza únicamente IB09 en DEVELOPMENT V2, con los límites inclusivos del catálogo **2005-10-24 a 2018-10-10**. El dictamen Quality V1 sigue siendo **FAIL**. La verificación oficial independiente sigue **UNRESOLVED**. La observación de 2005-10-23 se conserva con su procedencia y queda excluida de todos los métodos. IB09 figura como serie histórica separada y no aporta observaciones a 2020–2025.

El [handoff efectivo de #68](../../data_quality/siar/ib09-handoff-v2-development/handoff.json) tiene SHA256 `9c14247a67ea6fd870d63f985e96c25b02860ccf41bf7beea9b27ccc20b9ca92`. El [handoff PENDING de #67](../../data_quality/siar/ib09-handoff-v2/handoff.json) permanece íntegro como evidencia. El consumidor lo rechaza para admisión. La excepción exige que la revisión de la evidencia oficial disponible no haya encontrado contradicción. Cualquier contradicción nueva o fallo de reconciliación detiene la admisión y se traslada a Dirección 00.

El consumidor verifica los bytes del CSV IB09 fijado por #66, catálogo, paquete Quality POST y sus tres informes. Coteja fecha, estación, ordinal de registro, procedencia y los 23 campos de cada fila frente al Dataset V1 fijado por SHA256. Comprueba que el calendario del catálogo suma 4.735 fechas, repartidas en 4.422 observadas interiores y 313 ausencias confirmadas. La fila exterior completa las 4.423 del CSV. Para las candidatas interiores aplica los mismos controles físicos y cálculos por método de V1 con el gate científico de la excepción. La vista derivada conserva el FAIL original y el linaje de cada fila.

## Ejecución reproducible de la preparación

Desde la raíz del checkout, con Python 3.11 o posterior, después de revisar la evidencia oficial disponible en las PR #66 y #68 y comprobar que sigue sin contradicción conocida:

```bash
PYTHONPATH=scientific_pipeline/src python -m siar_science.et0_v2 \
  --repo . --dataset /ruta/dataset-v1/dataset.json \
  --ib09-handoff data_quality/siar/ib09-handoff-v2-development/handoff.json \
  --official-evidence-reviewed-no-contradiction \
  --output /ruta/existente/et0-v2-preparation
```

La salida es un directorio nuevo con `manifest.json`, `development.json.gz` y `support-inventory.json`. Cada estimación diaria conserva estado y motivos de elegibilidad. El inventario publica resultados propios y pares con PM válido, observaciones y ausencias por año, grupos AHC identificables, tamaños de los folds, huellas de las claves de cada partición y presupuesto de ajustes. Los datos diarios del bloque retrospectivo quedan fuera de la cápsula. Si se omite el handoff, el comando retiene IB09 y etiqueta el artefacto como parcial.

## Soporte y selección

- Las tres familias tienen las mismas entradas para fórmula, AHC, ML y ANN en cada contraste. HS-Ra usa Tmin, Tmax y Ra. HG-Rs usa la media de los extremos y Rs medida. Valiantzas usa Tmin, Tmax, HR media y Ra. La comparación anual añade cero indicadores. Estacional y mensual añaden los indicadores de la categoría que usa AHC.
- Para disponibilidad se exige cálculo propio finito. Para ajustar o evaluar error se exige además una referencia PM finita y no negativa. Las comparaciones entre métodos se hacen dentro de cada familia y calendario con las mismas claves. La intersección de las cuatro fórmulas se informa por separado.
- Cada fold exterior aparta un año natural completo dentro de DEVELOPMENT. ANN A elige el último 15 % de TRAIN por fecha. E toma el 15 % sin reemplazo mediante orden SHA256 de semilla, estación, familia, año omitido y fecha. Se redondea hacia arriba y se preserva el orden de fecha al ajustar. Las mismas claves internas se reutilizan entre arquitecturas, semillas y calendarios. El código fija la semilla de E aparte de las tres semillas de red.
- Se puntúan categorías con soporte en ambos ajustes internos de ANN y con AHC identificable en TRAIN. Se exigen al menos dos años exteriores con alguna clave común para seleccionar. Los grupos sin soporte se publican como tales. El mínimo técnico no certifica robustez. IB101 se conserva separado por su serie corta.
- ML conserva los 37 candidatos por celda de las rejillas V1. AHC exige pares válidos y denominador positivo. ML y ANN seleccionan por media simple de RMSE anuales con la dispersión y desempates del protocolo. A y E quedan definidos antes de ajustar y se conservarán ambos. El orden de campaña es **ANN A → ANN E**.

## Prevalidación privada en Kaggle CPU

Adjuntar la cápsula DEVELOPMENT como fuente privada y usar el checkout de esta PR. Para un entorno sin red, adjuntar también el código y las dependencias aprobadas como fuente privada. La prevalidación verifica inventario, dependencias, CPU, memoria y swap con el siguiente comando. Guardar su JSON junto al manifiesto.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=/ruta/codigo/scientific_pipeline/src python -m siar_science.et0_v2_kaggle \
  /kaggle/input/ruta-capsula --manifest-sha256 SHA256_PUBLICADO_DEL_MANIFIESTO \
  > /kaggle/working/et0-v2-preflight.json
```

La prevalidación solo lee archivos y metadatos del entorno. Al autorizarse el lanzamiento, medir tiempo, pico RSS, memoria del contenedor, swap y disco con una celda estación × familia × calendario y un candidato activo antes de escalar. Mantener un proceso y un hilo por ajuste al inicio y guardar resultados por celda con sus hashes para recuperar una interrupción. El presupuesto de ajustes del inventario permite estimar tiempo y cuotas a partir de esa medición. La evaluación retrospectiva requiere configuraciones y ajustes congelados con hashes antes de abrir los resultados del periodo.
