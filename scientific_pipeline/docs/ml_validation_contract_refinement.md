> **Estado posterior — 31/10/2026:** WEB-V0.2-AHC ya está desplegada (PR #46/#47) y el runtime ML rolling está implementado (PR #48); rolling validation y Linux smoke PASS. La campaña científica oficial permanece pendiente de despliegue/ejecución; ANN NOT STARTED. Las referencias inferiores a «web no construida», ML futuro o merge pendiente conservan el estado previo de su entrega. No cambian la semántica contractual ni resultados congelados. Este documento es la referencia técnica rolling; estado rápido en [PROJECT_STATUS](../../docs/PROJECT_STATUS.md).

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

## Runtime de campaña rolling

La orquestación oficial está expuesta como `ml-rolling-campaign`. Los comandos
`ml-train`, `ml-validate` y `ml-test` siguen siendo las etapas legacy y conservan
su rechazo explícito del protocolo rolling v2. La campaña valida los SHA del
Dataset v1, protocolo congelado y manifiesto AHC TEST antes del ajuste.

`ml-rolling-campaign` ajusta los 185 candidatos en F1–F4 (185×4 evaluaciones),
selecciona una configuración por cada una de las 35 celdas, agrega la cohorte
común por fold, congela selección/configuraciones y reajusta las 35
configuraciones en DEVELOPMENT 2004–2020. El cargador de TEST no se invoca hasta
que selección y bundle de refit están materializados. TEST se lee una vez; el
mejor TEST se conserva como descripción, nunca como selector.

La salida es append-only. Cada run contiene manifiesto con hashes, protocolo,
métricas de folds/candidatos, selección congelada, bundle refitted, predicciones
y métricas TEST, cohortes comunes, desgloses, cobertura, importancia de
permutación F4 y vistas CSV para consumo web. Las vistas no recalculan ciencia.
Un fallo retira el staging de esa ejecución y conserva solo un manifiesto
`FAILED` sin outputs científicos parciales.

```bash
siar-science ml-rolling-campaign --dataset ./derived/dataset.json \
  --ahc-manifest ./runs/ahc/test/manifest.json \
  --protocol scientific_pipeline/experiments/ml-v1-classical.json \
  --output ./scientific-results/ml-v1-classical-UTC \
  --pipeline-sha <SHA-del-checkout-fusionado>
```

La primera campaña oficial se interrumpió después de persistir selección y
bundle de refit; su acceso a TEST es desconocido. La evaluación autorizada de
recuperación, descrita abajo, es una ejecución separada y no afirma ser el
primer acceso a TEST. ANN continúa fuera de alcance hasta analizar estabilidad
y cobertura de la campaña clásica.

## Recuperación de evaluación interrumpida

Si una campaña se interrumpe después de persistir selección congelada y bundle
refitted, `ml-test-recovery` permite validar esos artefactos y evaluar el mismo
bundle en TEST sin volver a seleccionar ni ajustar modelos. Requiere las 185
resúmenes de candidatos con F1–F4, las 35 configuraciones idénticas en selección,
configuración congelada y bundle, y metadatos de refit DEVELOPMENT 2004–2020.
Los hashes del Dataset v1, protocolo y manifiesto AHC también deben coincidir.

La recuperación escribe en una carpeta nueva. Su journal durable registra
`PRE_TEST_VERIFIED` y `TEST_ACCESS_INTENT_RECORDED` antes de invocar el cargador
TEST; las etapas posteriores registran apertura, evaluación y resultado. Si la
campaña previa no permite saber si llegó a abrir TEST, el nuevo run se etiqueta
`retrospective_recovery_evaluation` y conserva `prior_attempt_status` como
`TEST_ACCESS_UNKNOWN`. No se afirma que TEST se abra por primera vez ni que sea
un acceso único a lo largo de todas las ejecuciones.

```bash
siar-science ml-test-recovery --source-run ./scientific-results/<campaña-interrumpida> \
  --dataset ./derived/dataset.json \
  --ahc-manifest ./runs/ahc/test/manifest.json \
  --protocol scientific_pipeline/experiments/ml-v1-classical.json \
  --output ./scientific-results/ml-test-recovery-UTC \
  --pipeline-sha <SHA-del-checkout-fusionado>
```

La recuperación es solo evaluación: no ejecuta selección, entrenamiento ni
refit. Un fallo conserva su manifest `FAILED` y el journal de acceso para que
quede registrada la frontera alcanzada.

