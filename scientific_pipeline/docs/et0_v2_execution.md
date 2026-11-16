# Ejecución de ET₀ V2

El [contrato de preparación](et0_v2_preparation.md) fija los datos, los folds, las entradas, los candidatos, las reglas de selección y el orden **AHC → ML clásico → ANN A → ANN E**. La cápsula DEVELOPMENT tiene manifiesto SHA256 `0c43f1020a04b0dedeeff84ccb1cbae4285ff00ac20c48473ff6c81b57ecd3e6`. Su vista contiene las 4.390 fechas comunes de IB09 admitidas por la excepción de la PR #68. IB09 se mantiene como serie histórica separada, con cero observaciones retrospectivas.

## Prevalidación y piloto en Kaggle CPU

El [cuaderno privado, versión 2](https://www.kaggle.com/code/josepiglesiasfranco/et0-v2-kaggle-preflight?scriptVersionId=368467345), verificó los hashes de la cápsula. Kaggle expandió la vista gzip al importarla. El cuaderno comprobó el SHA del JSON expandido y reconstruyó un gzip con el SHA original exacto. El entorno medido fue Python 3.13.18, scikit-learn 1.9.1, NumPy 2.1.3, SciPy 1.16.3 y joblib 1.6.0. El wheel Linux CPython 3.13 de scikit-learn se instaló aislado desde PyPI y quedó identificado por SHA256 `55e79d6e9b0923f1a978179822bd43d7f5543f45e970a00fe861f43486380aba`.

Con el checkout limpio del código y la cápsula fijada, desde la raíz del repositorio:

```bash
export PYTHONPATH="$PWD/scientific_pipeline/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m siar_science.et0_v2_campaign --repo . --capsule "$CAPSULE" --run "$PILOT" pilot
```

El piloto usa IB02 × Valiantzas × mensual y sus dos años extremos aptos. Ejecuta los candidatos de las rejillas completas, interrumpe ML tras el primer candidato y comprueba que el segundo lote recupera el mismo checkpoint. El `pilot.json` distingue tiempo de candidato y refit, pico RSS, espacio de disco, huella recuperada y proyección de cada campaña. La extrapolación de una celda orienta el presupuesto y tiene incertidumbre por estación, familia, calendario y hardware. Los resultados del piloto quedan fuera de la selección oficial.

## Lotes oficiales y congelación

```bash
python -m siar_science.et0_v2_campaign --repo . --capsule "$CAPSULE" --run "$RUN" \
  drive --workers 4 --wall-seconds 28800
```

El límite suave se consulta entre candidatos. Cada trabajador procesa una lista disjunta de celdas y usa un hilo. Se guarda el checkpoint de cada candidato antes de iniciar otro. Cada celda conserva puntuaciones por fold, métricas por semilla cuando corresponden, selección, modelos y escaladores, huellas y tiempos. Una fase se sella cuando el inventario completo coincide con el contrato y cada hash está verificado. `drive` verifica los sellos precedentes y guarda telemetría por lote.

En Kaggle, usar una versión privada con **Save & Run All** y conservar el directorio de ejecución como salida. Para continuar, montar la salida de la última versión válida, copiar el directorio a un área de escritura y volver a invocar `drive` con el mismo commit, cápsula y dependencias. Los checkpoints válidos se reutilizan. El ejecutor comprueba identidad y hashes antes de avanzar. Kaggle documenta un límite de 12 h por sesión CPU y 20 GB de salida persistente. [Especificaciones de Kaggle](https://www.kaggle.com/docs/notebooks)

Cuando las cuatro fases estén selladas, `drive` genera `freeze.json` y su SHA256. Los coeficientes, las configuraciones, los escaladores y los modelos ya quedan vinculados a ese archivo.

## Evaluación retrospectiva

Solo después de verificar `freeze.json`, proporcionar el Dataset V1 fijado en el contrato y ejecutar:

```bash
python -m siar_science.et0_v2_retrospective --repo . --capsule "$CAPSULE" --run "$RUN" \
  --dataset "$DATASET_V1" --freeze-sha256 "$FREEZE_SHA" --output "$OUTPUT"
```

Se registra la intención antes de leer Dataset V1 y la apertura después de cargar las filas retrospectivas. El resultado publica `predictions.csv.gz`, `retrospective-metrics.csv`, `ahc-full-history-in-sample.csv`, `report.md` y `manifest.json`. El error se calcula con claves comunes a la fórmula, AHC, ML principal y ANN A y E dentro de cada familia y calendario. La cobertura propia usa días de catálogo y días con PM disponible. Las estaciones históricas figuran con cero observaciones 2020–2025 y cobertura indefinida cuando el calendario operativo tiene cero días. El AHC de histórico completo se calcula en una tabla descriptiva dentro de muestra.
