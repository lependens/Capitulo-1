# Revisión de la metodología de ET₀ y protocolo de V2

V2 compara fórmulas empíricas calibradas, ML clásico y redes neuronales (ANN) para estimar la ET₀ diaria cuando faltan entradas de Penman–Monteith. La comparación se hace **por estación meteorológica**, con la **misma información de entrada** y sobre las **mismas fechas evaluables** dentro de cada escenario. La referencia de error es `PM_FAO56_V1` calculada por el proyecto. Este documento describe el experimento acordado para V2. Las cifras de rendimiento que se citan proceden de V1.

[Presentación editable del protocolo V2](assets/et0_v2_experiment_protocol.pptx) · [Registro de decisiones](et0_v2_decisions.md) · [Contrato ejecutable de preparación](et0_v2_preparation.md) · [Guía de ANN V1](ann_v1_modern_explained.md)

## 1. Decisiones que fijan V2

| Aspecto | Decisión |
|---|---|
| Unidad de ajuste | Un ajuste local por estación meteorológica, método y escenario de entradas. |
| DEVELOPMENT | Desde el inicio elegible de cada estación hasta el **31/12/2019 inclusive**. Incluye la selección y el ajuste final. |
| Evaluación final | Del **01/01/2020 al 31/12/2025**, con configuraciones, escaladores, pesos y coeficientes ya fijados. |
| Fórmulas comparadas | HS-Ra, HG-Rs y Valiantzas. Cada una define un conjunto de entradas para su comparación con AHC, ML y ANN. |
| Calendario | Comparación anual sin calendario adicional, estacional con la temporada y mensual con el mes. |
| AHC | Ajuste directo anual, estacional y mensual con DEVELOPMENT. Cuarto cálculo anual de todo el histórico hasta 2025, con error dentro de muestra. |
| Selección ML y ANN | Dejar un año fuera cada vez dentro de DEVELOPMENT. Seleccionar por media de RMSE anual y reajustar con todo DEVELOPMENT. |
| Parada ANN | Ejecutar A y E. A usa el último 15 % por fecha. E usa el 15 % de filas al azar. Ambas mantienen los parámetros y el escalador de V1. |

La evaluación 2020–2025 es **retrospectiva**. V2 reserva esas fechas para su evaluación final, aunque 2020 formó parte de DEVELOPMENT de ML y ANN V1 y 2021–2025 ya se ha examinado en el proyecto. Si se quiere confirmar en el futuro una elección basada en estos resultados, se reservará otro periodo antes de usarlo para decidir. La duración dependerá de las estaciones y temporadas que se necesite representar.

## 2. Referencia y preparación de datos

### Una fila por estación y fecha

Se parte de las observaciones SIAR y su trazabilidad de Quality. Cada fila identifica **estación + fecha** y conserva los datos meteorológicos válidos, los motivos de exclusión y, si se puede calcular, el valor diario de `PM_FAO56_V1`. La referencia PM se usa como salida objetivo para calibrar y entrenar, y para medir el error. Con todas las entradas válidas disponibles, la vía directa para obtener esa referencia es calcular PM. Evaluar la ET₀ frente a un lisímetro requeriría datos instrumentales independientes. [FAO-56, cálculo de ET₀](https://www.fao.org/4/x0490e/x0490e08.htm)

Para **ajustar y medir error** se exige PM válido y las entradas requeridas por la fórmula correspondiente. Para **medir disponibilidad**, se cuentan aparte los días en que el método puede producir una estimación con sus propias entradas, aunque ese día falte una variable necesaria para calcular PM. Los métodos de cada contraste usan las mismas claves elegibles en el ajuste comparable y se puntúan sobre la intersección de claves evaluables. Las ausencias se registran por método y motivo.

### Entradas meteorológicas por familia

| Familia | Entradas que llegan al método | Origen |
|---|---|---|
| `HS_RA` | `Tmin`, `Tmax`, `Ra` | `Ra` es radiación extraterrestre calculada con latitud y día del año. |
| `HG_RS` | `Tmedia_extremos`, `Rs` | `Tmedia_extremos = (Tmin + Tmax) / 2`. `Rs` es radiación solar medida. |
| `VALIANTZAS` | `Tmin`, `Tmax`, `RHmedia`, `Ra` | `RHmedia` es la humedad relativa media. |

Estas son las entradas para AHC, ML y ANN en la comparación anual de cada familia. La media y la amplitud de temperatura juntas permiten reconstruir los extremos. La media sola pierde la amplitud, por lo que ML y ANN de HG-Rs reciben únicamente la media de los extremos y Rs. La fecha y la latitud intervienen en el cálculo de Ra. ML y ANN reciben la Ra resultante en esa familia. El identificador de estación selecciona el modelo local. El PM, los residuos y las estimaciones de otras fórmulas quedan fuera del vector de entrada. [Entradas AHC V1](../results/ahc-20261024T172400Z/test/input_complexity.csv) · [HG-Rs](hargreaves_rs_v1.md) · [Valiantzas](valiantzas_v1.md)

## 3. Comparación estacional y mensual con la misma información

Se propone codificar **exactamente la categoría de calendario que usa AHC**, con indicadores binarios `one-hot` para ML y ANN. Por ejemplo, el **15 de julio** activa únicamente «verano» en la comparación estacional y únicamente «julio» en la mensual. Esto evita que numerar los meses del 1 al 12 imponga un orden y una distancia artificial al modelo. El día del año, incluso codificado como seno y coseno, aporta una posición más detallada que la que emplea el AHC estacional o mensual. Su efecto queda para un experimento posterior. [Codificación categórica](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) · [Codificaciones temporales](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)

| Comparación | AHC | ML y ANN |
|---|---|---|
| **Anual** | Un coeficiente por estación y fórmula. | Un modelo local por familia y método, con las entradas base. |
| **Estacional** | Hasta cuatro coeficientes. La fecha selecciona DJF, MAM, JJA o SON. | Un modelo local por familia y método con las entradas base más cuatro indicadores de temporada. |
| **Mensual** | Hasta doce coeficientes. La fecha selecciona el mes. | Un modelo local por familia y método con las entradas base más doce indicadores de mes. |

DJF corresponde a diciembre, enero y febrero. MAM corresponde a marzo, abril y mayo. JJA corresponde a junio, julio y agosto. SON corresponde a septiembre, octubre y noviembre. La **granularidad** indica cuándo cambia el coeficiente. El ajuste reúne los días de la misma categoría de todos los años admisibles. Los años que se apartan para validar son años naturales, de enero a diciembre. ML y ANN comparten parámetros entre categorías dentro de un modelo local y pueden aprender interacciones con las entradas, según el algoritmo. AHC aplica multiplicadores independientes por grupo. Se iguala la información disponible para medir la diferencia entre métodos.

El vocabulario de categorías se fija antes de ajustar. Si una categoría carece de días elegibles en el entrenamiento de un método, se marca como **sin soporte** para ese ajuste y se informa su efecto en la cobertura. La calibración local y el análisis estacional tienen precedentes en estudios de ET₀. El rendimiento de esta codificación para Baleares se medirá en V2. [Moratiel et al. (2020), calibración y evaluación estacional de ET₀](https://doi.org/10.5194/nhess-20-859-2020)

## 4. Calibración directa de AHC

AHC toma una fórmula empírica y estima un multiplicador por mínimos cuadrados sin intercepto. Para cada estación, fórmula y grupo de calendario, si `x` es la ET₀ diaria de la fórmula y `y` es el PM del mismo día, se obtiene `a = Σ(x × y) / Σ(x²)`. La estimación diaria ajustada es `a × x`. Así se minimiza la suma de errores diarios al cuadrado. Es el estimador de AHC V1 y difiere de la media de cocientes diarios `y/x`. [Método AHC V1](ahc_methodology.md)

1. **AHC anual.** Ajustar un coeficiente con todas las filas elegibles de DEVELOPMENT de esa estación y fórmula. Aplicarlo a cada día evaluable de 2020–2025.
2. **AHC estacional.** Repetir el cálculo dentro de cada una de las cuatro temporadas usando DEVELOPMENT. Aplicar en 2020–2025 el coeficiente correspondiente a la fecha.
3. **AHC mensual.** Repetir el cálculo para cada mes usando DEVELOPMENT. Aplicar en 2020–2025 el coeficiente del mes correspondiente.
4. **AHC anual de histórico completo.** Ajustar un coeficiente anual con el histórico elegible hasta el 31/12/2025. Medir el error sobre esas mismas filas y etiquetarlo **dentro de muestra**. Tiene el mismo carácter descriptivo que el cálculo de climatología de V1 y se presenta fuera de la clasificación final 2020–2025.

Se ajustan directamente las tres granularidades y se publican sus resultados. Cada coeficiente usa todos los pares disponibles de su grupo. La regla acordada prescinde de un mínimo uniforme de días o años. El cálculo requiere pares válidos y `Σ(x²) > 0`. Si no se cumple, se registra la causa y el grupo queda sin coeficiente. Se guardan los pares, los años representados y las fechas extremas de cada grupo para distinguir una estimación con mucho histórico de una sustentada en pocas observaciones.

Se publican los errores anuales de los coeficientes congelados y su variabilidad en 2020–2025. Un RMSE calculado por año resume el comportamiento del mismo ajuste frente a condiciones distintas. Si más adelante se declara una granularidad ganadora usando esos errores, la confirmación de esa elección corresponderá a otro periodo reservado.

## 5. ML clásico por estación

Por cada **estación × familia × calendario** se prueban las rejillas de V1 para regresión lineal, Ridge, Elastic Net, árbol de decisión, Random Forest, Extra Trees e HistGradientBoosting. Se conservan sus parámetros fijos y las transformaciones de V1. Las tres regresiones usan `StandardScaler` ajustado dentro de cada entrenamiento. Los métodos de árboles reciben las columnas sin ese escalado. [Rejillas y preprocesamiento de ML V1](../experiments/ml-v1-classical.json) · [Implementación de estimadores](../src/siar_science/ml_classical.py)

Para puntuar un candidato, se reserva un año completo disponible dentro de DEVELOPMENT y se ajusta con los demás. Por ejemplo, al puntuar **2014** pueden entrar observaciones de 2015–2019 en TRAIN. El procedimiento evalúa años omitidos dentro del histórico de desarrollo. El corte posterior de 2020–2025 proporciona la evaluación cronológica del modelo congelado. Se rota el año, se calcula el RMSE de cada bloque y se obtiene la media simple de los RMSE sobre los mismos años y las mismas filas comparables. Cada año apto pesa lo mismo.

Se conserva el mejor candidato de cada algoritmo y se identifica el ML principal por **estación, familia y calendario** usando DEVELOPMENT. Para un empate se prioriza menor dispersión de RMSE anual y luego el orden de complejidad y desempate de V1. Se reconstruye el candidato seleccionado con todos los datos elegibles de DEVELOPMENT. Se guarda el resultado de selección y, a continuación, se evalúa el modelo final en 2020–2025.

## 6. ANN por estación, desde los datos hasta el resultado

### Configuraciones comparadas

Se conservan las arquitecturas de V1 y se añaden las dos anchuras propuestas. Se prueban **`[16]`, `[32]`, `[64]`, `[64, 32]` y `[64, 32, 16]`** con las tres semillas de V1, `20261101`, `20261102` y `20261103`. Las redes tienen salida escalar. Conservan ReLU, Adam, la penalización L2 `alpha=0,0001`, tasa inicial `0,001`, lote configurado de 256 y el orden de ajuste fijado por el código de V1. El escalador es `StandardScaler`. Sus estadísticas se calculan de nuevo usando solo las entradas de las filas que ajustan los pesos en cada fase. [Contrato ANN V1](../experiments/ann-v1-modern-et0.json) · [Código de entrenamiento](../src/siar_science/ann_modern.py)

El centrado y el escalado reducen diferencias de magnitud entre entradas durante la optimización. Con una media de entrenamiento de 20 °C y una desviación de 5 °C, 25 °C se convierte en 1 tanto en ajuste como en evaluación. Las mismas estadísticas se aplican a ambos conjuntos hasta el siguiente reajuste. [LeCun et al., *Efficient BackProp*](https://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) · [Escalado de MLP en scikit-learn](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use)

### Dos validaciones con funciones distintas

1. **Dejar fuera el año externo.** Separa, por ejemplo, todas las filas de 2014 de esa estación y familia. El resto de DEVELOPMENT forma TRAIN del fold.
2. **Dividir TRAIN del fold en ajuste y validación interna.** A usa el último 15 % de fechas elegibles ordenadas. E elige al azar aproximadamente el 15 % de filas elegibles, sin reemplazo. Cada estación, familia y año externo conserva un único conjunto de claves para E, compartido por las arquitecturas, semillas de red y versiones de calendario que usan esas mismas filas. La semilla de partición se fija y registra aparte de las semillas de inicialización de la red.
3. **Elegir épocas por semilla.** Se ajusta el escalador y se actualizan los pesos con la parte de ajuste. Se mide el RMSE en el 15 % interno después de cada época. Se conserva la mejor época registrada cuando la mejora supera `0,0001 mm/día`. Se aplica paciencia de **15 épocas** y un máximo de **200**, igual que en V1.
4. **Puntuar en el año externo.** Para cada semilla se inicia otra red, se recalcula el escalador con **todo el TRAIN del fold**, incluido el 15 % interno, y se entrena durante las épocas elegidas. En el año externo, se promedian primero las tres predicciones de cada día y luego se calcula el RMSE anual de ese promedio.
5. **Elegir la arquitectura.** Se repite el ciclo para cada año apto. Se compara la media de los RMSE anuales. Se selecciona una arquitectura por **estación, familia, calendario y regla A o E**. En caso de empate se prefiere menor dispersión anual y luego menor número de parámetros. Se conserva también la comparación A frente a E a arquitectura fija, sobre las mismas filas, para aislar el efecto de cómo se eligió el 15 %.
6. **Ajustar y congelar.** Para cada configuración seleccionada se ajusta un escalador con todo DEVELOPMENT. Se crean tres redes nuevas. Para cada semilla se usa la mediana de sus mejores épocas de los folds, redondeando hacia arriba cuando termina en media época. Se congelan escalador, pesos, épocas y selección. Tanto A como E pasan a la evaluación 2020–2025. La salida diaria de cada modelo final es la media de las tres redes.

| Cómo se divide TRAIN | A · Por fecha | E · Al azar |
|---|---|---|
| Ajuste de pesos | Primer 85 % de fechas. | Aproximadamente 85 % de filas, las que quedan fuera de la muestra aleatoria. |
| Elección de épocas | Último 15 % de fechas. | Aproximadamente 15 % de filas elegidas sin reemplazo. |
| Evaluación externa | Año omitido completo. | El mismo año omitido completo. |

El año externo se usa para comparar arquitecturas. El 15 % interno se usa para elegir épocas. El bloque 2020–2025 mide los modelos finales. La mediana transfiere un presupuesto de épocas desde muestras más pequeñas, con distinta composición y distinto número de actualizaciones por época. Se publicará la frecuencia con que se alcanza el tope de 200 para valorar este efecto. Las opciones B, C y D de la revisión anterior quedan descritas como alternativas para otra fase: bloques internos completos, rejilla de épocas y reserva tardía para fijar el presupuesto final. [Prechelt, sensibilidad a la regla de parada](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)

### Relación con el TFG

El TFG separaba un año entero por estación y describía una validación interna aleatoria del 15 % de las filas restantes. La copia de `validacio4.m` usa `randperm` para elegir esas filas. El anexo impreso de la memoria calcula la permutación, pero extrae las filas en el orden de entrada. La versión usada para generar las tablas históricas queda sin confirmar. El TFG usaba Levenberg–Marquardt, `tansig`, MSE, máximo de 100 épocas y `max_fail=5` en la rutina revisada. E incorpora la idea de la muestra interna aleatoria y conserva el resto del entrenamiento moderno de V1. [MathWorks, `trainlm`](https://www.mathworks.com/help/deeplearning/ref/trainlm.html)

El TFG describía dos criterios de selección de arquitectura, por error interno o por error del año apartado. En V2 se puntúa la **misma arquitectura** a lo largo de los años de DEVELOPMENT y se fija antes de evaluar 2020–2025. Esto separa la selección del resultado final. [Cawley y Talbot, evaluación tras selección](https://www.jmlr.org/papers/v11/cawley10a.html) · [Cerqueira et al., estimación de rendimiento temporal](https://doi.org/10.1007/s10994-020-05910-7)

## 7. Series cortas y cobertura por estación

| Estación | Evidencia disponible | Tratamiento en V2 |
|---|---|---|
| **IB07 · Sóller** | 2.946 filas brutas hasta el 13/11/2012. Hubo PM preservado en 2.929. | Admisión histórica con las filas aptas. Puede aportar ajustes y evaluación anual de DEVELOPMENT según soporte. En 2020–2025 figura sin observaciones. |
| **IB09 · Artà** | 4.423 filas brutas, incluida la del 23/10/2005 exterior al catálogo. Quality V1 sigue FAIL y la verificación oficial independiente UNRESOLVED. | Excepción de Dirección 00 para DEVELOPMENT V2, 24/10/2005–10/10/2018 inclusive. El recálculo confirma 4.390 pares comunes. Serie histórica separada sin observaciones en 2020–2025. |
| **IB101 · Ciutadella** | 249 filas brutas entre 2008 y 2009. Hubo PM preservado en 246. | Calcular los AHC que resulten identificables e indicar su soporte. Comprobar para ML y ANN si hay años, particiones y filas suficientes para calcular la selección definida. Publicar las causas de falta de soporte. |

Las filas disponibles y el soporte técnico determinan qué se puede ajustar. Para ML se necesitan años con TRAIN y validación externa utilizables. Para ANN, además, ambas partes internas deben tener filas válidas. Para seleccionar y comparar candidatos se requieren años aptos comunes. Cuando no pueda completarse el procedimiento, se indica la causa en lugar de producir una clasificación artificial. Las estaciones cerradas se muestran en el informe histórico con el tipo de evaluación correspondiente. Sus errores de ajuste AHC y los errores de años apartados de ML o ANN se rotulan de acuerdo con el conjunto que produjo cada cifra. [Manifiesto Quality](../../data_quality/siar/post/SIAR-QA-POST-20261020T193140Z/siar_dataset_manifest.csv) · [Auditoría por estación](../../data_quality/siar/post/SIAR-QA-POST-20261020T193140Z/siar_station_audit.csv)

## 8. Cómo se mide y publica el error

- **Selección en DEVELOPMENT.** Una fila de resultados por candidato y año apartado. RMSE en mm/día, número de filas y trazabilidad de las claves. Publicar la media simple de los RMSE anuales y, de forma separada, el RMSE diario agrupado. Estas dos magnitudes ponderan de forma distinta los años con diferente número de días.
- **Evaluación final 2020–2025.** Fórmula original, AHC de la granularidad correspondiente, ML seleccionado y ANN A y E. Puntuarlos contra PM en las mismas fechas evaluables de cada contraste. Publicar RMSE, MAE, sesgo medio `estimación − PM`, número de pares y cobertura, por estación y año. Añadir el desglose estacional y mensual cuando haya filas aptas. Una diferencia pareada compara errores de los métodos en las mismas claves.
- **Disponibilidad.** Publicar dos denominadores cuando corresponda: días previstos del calendario operativo y días con PM evaluable. Informar qué método puede emitir una estimación y por qué se pierde un día. Medir aparte el error de una ausencia simulada ocultando la entrada al método y conservándola para calcular PM.
- **AHC anual de histórico completo.** Tabla descriptiva propia con error dentro de muestra, años y pares usados. Su resultado no interviene en el orden de los modelos en el bloque final.
- **Trazabilidad.** Guardar versión de datos y Quality, definición de entradas, claves de cada partición, coeficientes, configuraciones, escaladores, semillas, épocas, resultados por año y predicciones congeladas. Registrar tiempo y memoria por fase, con una comprobación de recursos antes de la campaña.

Antes de la primera ejecución se puede inventariar cuántos días y años aptos aporta cada estación y fórmula sin calcular errores de candidatos en el bloque final. El inventario permite dimensionar el trabajo y marcar las combinaciones que no pueden evaluarse.

## 9. Qué sabemos de V1 y por qué cambia el protocolo

**Sí, ML y ANN V1 eran agrupados.** Sus funciones de ajuste tomaban las filas elegibles de las estaciones primarias y creaban un modelo por combinación de algoritmo y familia en ML, o una ANN por familia y semilla en el ajuste final. Los códigos de estación servían para filtrar y emparejar resultados, y el ajuste no se repetía por código. AHC V1 ya estimaba un coeficiente por estación y fórmula. V2 alinea los tres métodos en el ajuste local. [Ajuste final de ML V1](../src/siar_science/ml_campaign.py#L209) · [Ajuste final de ANN V1](../src/siar_science/ann_campaign.py#L60) · [Protocolo AHC V1](ahc_methodology.md)

AHC V1 ajustó los coeficientes validados en 2004–2015, comparó granularidades en 2016–2020 y evaluó coeficientes congelados en 2021–2025. La variante seleccionada fue HS-Ra anual, HG-Rs estacional y Valiantzas anual. AHC Climatology calculó aparte coeficientes anuales con el histórico elegible completo hasta 2025. ML y ANN V1 usaron ventanas crecientes, selección en DEVELOPMENT 2004–2020 y TEST 2021–2025. [Resultados AHC V1](../results/ahc-20261024T172400Z/README.md) · [Protocolo ML V1](ml_v1_classical_methodology.md) · [Protocolo ANN V1](ann_v1_modern_methodology.md)

En la cohorte pareada de **15.958 filas** de V1, ANN FULL obtuvo **0,051312 mm/día** de RMSE frente a **0,064104** de HGB FULL. Sus sesgos fueron **−0,022449** y **−0,000966 mm/día**, respectivamente. Las comparaciones aproximadas con fórmulas de entradas reducidas dieron los resultados siguientes. Las familias ML y ANN de V1 podían recibir contexto o variables adicionales, y AHC se ajustó con otro corte. V2 controla esas diferencias.

| Fórmula y familia más próxima en V1 | Fórmula original | AHC V1 | HGB | ANN |
|---|---:|---:|---:|---:|
| HS-Ra / T | 0,949 | 0,639 anual | 0,547 | 0,521 |
| HG-Rs / T + Rs | 0,565 | 0,414 anual · 0,401 estacional | 0,302 | 0,342 |
| Valiantzas / T + RH | 0,629 | 0,524 anual | 0,430 | 0,414 |

RMSE en mm/día frente a PM. [Resultados y linaje ANN V1](ann_v1_modern_explained.md) · [Resultados ML V1](../../docs_1.2.6_Modelos%20ML%20cl%C3%A1sicos%20para%20estimaci%C3%B3n%20de%20ET0.md)

La recuperación de ML V1 en Kaggle evaluó 35 modelos ajustados previamente en el servidor Linux. El intento de recuperación local se interrumpió preventivamente con `SIGTERM` durante la carga del bundle por presión de memoria observada, según el registro operativo revisado por 05. El preflight aislado en Kaggle alcanzó 14,62 GiB de RSS al cargar el bundle y la recuperación completa registró 25,23 GiB de RSS máximo, sin desglose del pico por fase. ANN V1 entrenó en el servidor con CPU. Su DEVELOPMENT duró cerca de 25 minutos y registró aproximadamente 693 MiB de RSS máximo según el log consultado por 05. Son fases y cargas de trabajo distintas. [Incidente de recuperación de TEST](ml-v1-test-recovery-access-incident.md) · [Corrección de serialización](https://github.com/lependens/Capitulo-1/pull/54) · [Preservación de artefactos](../../docs/DATA_LOCATIONS.md)

## 10. Puntos abiertos antes de usar los resultados

El contrato ejecutable fija calendario, RMSE medio anual, desempates y orden ANN A seguido de ANN E. El [inventario de soporte](et0_v2_support_20261111.md) registra la excepción de IB09 y la cobertura de las series cortas. Para recomendar un método para uso operativo quedan por fijar **error, sesgo y cobertura aceptables** para cada escenario de sensores. La elección de umbrales operativos y de un periodo adicional de confirmación puede hacerse después de cerrar el protocolo retrospectivo, antes de usar esos datos para decidir.
