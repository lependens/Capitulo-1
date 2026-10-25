# ML v1 — campaña clásica bloqueada antes del entrenamiento

La configuración ejecutable está en
[`ml-v1-classical.json`](../experiments/ml-v1-classical.json). Se cerró el
25/10/2026 antes de ajustar cualquier modelo ML o inspeccionar sus errores.
Mantiene Dataset v1, target, familias y cortes del contrato ML aprobado.

El objetivo es emular `PM_FAO56_V1`, una referencia calculada en mm/día. No es
ET medida ni un target independiente de sus inputs físicos. La población primaria
son las nueve estaciones con TRAIN, VALIDATION y TEST. `station_code` no entra
como número ni categoría; latitud y altitud sí pertenecen al contexto explícito.

Cada familia se evalúa con casos completos propios. Todos los modelos incluyen
latitud, altitud, seno/coseno anual y Ra versionada. Regresión lineal, Ridge y
Elastic Net usan `StandardScaler` aprendido solo en TRAIN dentro de un pipeline.
Árboles no escalan. No hay imputación, filtro por residuo, clipping ni refit con
TRAIN+VALIDATION.

La rejilla contiene 37 candidatos por familia: 1 lineal, 6 Ridge, 12 Elastic Net,
6 árboles limitados, 4 Random Forest, 4 Extra Trees y 4 HistGradientBoosting;
**185 ajustes** en total. `scikit-learn==1.9.1` es la única dependencia ML.
La elección sigue la [instalación estable oficial](https://scikit-learn.org/stable/install.html)
y usa estimadores mantenidos en el mismo paquete. HistGradientBoosting evita
añadir otro framework. Semillas, paralelismo, iteraciones y rejillas están
fijados en el JSON.

VALIDATION elige un candidato por celda algoritmo×familia. Después elige un
champion de campaña sobre la intersección común de las 35 celdas congeladas.
Empates relativos dentro del 0,1 % favorecen menor complejidad, menos variables
y finalmente ID léxico. TEST se abre una vez y evalúa las 35 celdas sin cambiar
nada. El mejor valor observado en TEST es descriptivo; no reemplaza al champion
seleccionado en VALIDATION. TEST continúa etiquetado `retrospective_locked`.

La importancia por permutación se calcula en VALIDATION sobre una muestra de
claves seleccionada por hash, independiente del target. La caída de rendimiento
describe dependencia predictiva del modelo y las variables correlacionadas;
no mide causalidad ni dirige selección. Los árboles conservan además su
importancia intrínseca cuando el estimador la ofrece, con la misma advertencia.

Se publicarán cohortes propias y una intersección común, desgloses por estación,
año, mes y temporada, matrices de las 35 celdas, predicciones diarias, modelos,
transformaciones TRAIN, configuración, manifests y hashes. Los baselines AHC
se importan de su ejecución oficial; `AHC_CLIMATOLOGY` no participa.
