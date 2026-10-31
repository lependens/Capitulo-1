# ML v1 — refinamiento de validación temporal

**Estado: contrato congelado antes de selección científica; TEST 2021–2025 no
se modifica ni se usa en el smoke.** El protocolo ejecutable es
[`ml-v1-classical.json`](../experiments/ml-v1-classical.json).

## Qué se conserva del TFG y qué no

El legado `legacy/ml/train_nn_et0.py` separaba sucesivamente un año como test,
entrenaba con todos los demás años y reservaba al azar un 15 % de ese conjunto
para validación/early stopping. La repetición por años es una idea útil para
medir sensibilidad temporal. No se conserva la parte no causal: cuando el año
evaluado no era el último, el entrenamiento contenía años futuros. El mismo
script calculaba además un «mejor modelo por test», lo que usa el conjunto
evaluado para seleccionar y no es una estimación fuera de muestra válida.

La variante histórica `train_nn_et0_ESTYRANG.py` ya distinguía LOYO no causal de
forward chaining, pero seleccionaba el mejor modelo global con el MSE de los
años evaluados y no mantenía el TEST final 2021–2025 sellado. Sirve como
trazabilidad, no como protocolo v1. No se reutilizan la partición aleatoria del
15 %, escaladores por año evaluado, early stopping ANN ni selección por TEST.

## Rolling-origin adoptado

Todo el desarrollo ocurre en 2004–2020, con fechas inclusivas y comunes a las
nueve estaciones primarias:

| Fold | TRAIN expansivo | VALIDATION futura |
|---|---|---|
| F1 | 2004–2009 | 2010–2012 |
| F2 | 2004–2012 | 2013–2015 |
| F3 | 2004–2015 | 2016–2018 |
| F4 | 2004–2018 | 2019–2020 |

La cobertura del Dataset v1 permite mantener estos cortes predeclarados; no se
han desplazado en función de errores ni de disponibilidad de un candidato. Cada
transformación se ajusta exclusivamente con el TRAIN del fold. Las ventanas de
validación son disjuntas y siempre posteriores a su entrenamiento.

TEST continúa exactamente en 2021-01-01–2025-12-31 y conserva la etiqueta
`retrospective_locked`. No interviene en arquitectura, regularización,
profundidad, familias de inputs, hiperparámetros ni importancia de variables.

## Regla de agregación y selección

Para cada candidato se calcula RMSE en cada ventana de validación. La regla
primaria, congelada antes de la campaña, es la media de los cuatro RMSE ponderada
por el número de observaciones elegibles del fold. También se publican RMSE
pooled, desviación ponderada entre folds, mínimo y máximo. Dentro de la tolerancia
relativa del 0,1 %, se prefiere menor dispersión; después, menor complejidad,
menos inputs e ID léxico. Así la estabilidad informa los empates, pero no se
convierte después de ver resultados en una penalización ajustable.

Tras seleccionar exclusivamente con los cuatro folds, la configuración se
congela y se vuelve a ajustar una sola vez con todo DEVELOPMENT 2004–2020. Sus
transformaciones se aprenden en ese mismo periodo. Solo entonces se permite una
lectura de TEST. El mejor valor observado en TEST será descriptivo y no podrá
reemplazar la selección de desarrollo.

AHC v1 no se recalcula ni se invalida. Sus métricas siguen siendo comparables
como referencia publicada, declarando que su protocolo TRAIN/VALIDATION no es
idéntico al rolling-origin de ML.

## Ejecución Linux y smoke

Los nuevos runs canónicos se ejecutan nativamente en el servidor Linux. El
comando `ml-linux-smoke` recorre los cuatro folds con Linear Regression y Ridge
(`alpha=1`) sobre `FEATURE_SET_T_RH_RS`. Es una comprobación de entorno,
determinismo, tiempos, memoria, hashes y outputs. El manifest fija
`scientific_selection_allowed=false` y `test_inspected=false`; sus errores no
se usan para elegir modelo. La campaña Windows TRAIN iniciada con el contrato
anterior terminó antes del cambio, pero nunca abrió VALIDATION ni TEST y queda
supersedida, sin valor de selección científica.

Los artefactos Linux siguen el flujo servidor `scientific-results` → preservación
→ Drive crypt. No se mueven ni eliminan resultados históricos de Windows.

