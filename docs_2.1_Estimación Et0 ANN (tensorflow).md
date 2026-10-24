# 2.1 Estimación de ET₀ mediante Redes Neuronales Artificiales (ANN)

> **Lectura vigente — 24/10/2026.** Los apartados anteriores a la última actualización conservan snapshots históricos, incluidas tareas después superadas. La consolidación final y [PROJECT_STATUS](docs/PROJECT_STATUS.md) distinguen cuatro baselines oficiales, preservación pendiente y gates operativos actuales; no se reescriben resultados históricos.

> **Estado vigente — 23/10/2026.** Las secciones y actualizaciones anteriores son snapshots históricos; sus planes, inventarios y afirmaciones de estado pueden estar superados. La actualización final de este documento recoge el estado contrastado, y [PROJECT_STATUS](docs/PROJECT_STATUS.md) es la autoridad de estado rápido. La metodología histórica se conserva sin convertir sus resultados en resultados v1.


Este documento explica paso a paso cómo se implementaron las **Redes Neuronales Artificiales (ANN)** para estimar la evapotranspiración de referencia **ET₀**, replicando la metodología del TFG original, pero adaptada a **Python/Keras con TensorFlow**. Se integra como continuación natural del documento:

**`docs_1.2_Tratamiento de datos y cálculo de estimaciones.md`**

---

## 🎯 Objetivo

Desarrollar y evaluar **tres modelos de redes neuronales artificiales** que estimen ET₀ (mm/día), usando como referencia los valores calculados con **Penman-Monteith (PM)** en Python.

Estos modelos se comparan con sus equivalentes empíricos:

| Modelo ANN | Inputs utilizados                                  | Modelo empírico equivalente |
|------------|----------------------------------------------------|-----------------------------|
| **ANN_Rs** | Radiación solar medida (Rs), Temperatura media    | HGRₛ                        |
| **ANN_Ra** | TempMax, TempMin, TempMedia, Radiación extraterrestre (Ra) | HGRₐ          |
| **ANN_HR** | TempMax, TempMin, TempMedia, Ra, Humedad media    | HGHR                        |

El script `train_nn_et0_fast.py` usa **TensorFlow 2.17.0** (biblioteca de ML de Google) con **Keras** (su API de alto nivel) para construir y entrenar los modelos. TensorFlow maneja el cómputo en GPU/CPU, optimizando el entrenamiento para datasets grandes (~32,500 filas).

---

## 📁 Estructura del proyecto

```
📂 Capitulo-1/
  📂 datos_siar_baleares/
     ├─ IB01_et0_variants.csv  # Datos por estación
     ├─ IB02_et0_variants.csv
     ├─ ...
     └─ IB05_et0_variants.csv
  📂 outputs/  # Generados por el script
     ├─ nn_errors_fast.csv         # Errores por año
     └─ nn_errors_summary.csv      # Errores medios resumen
  ├─ train_nn_et0_fast.py     # Script principal ANN
  └─ requirements.txt         # Dependencias
```

- **Ruta base**: `datos_siar_baleares/` (subida manual en Colab o montada desde Drive).
- **Nombre de archivos**: `IBXX_et0_variants.csv` (CSV, UTF-8, separador coma).
- **Formato general**: CSV (pandas.read_csv con encoding='utf-8-sig' para compatibilidad).

---

## 🧠 Metodología aplicada

### ✅ 1. Target (salida de la red)

- Se utiliza **ET0_calc**, calculado con la ecuación FAO-56 Penman-Monteith en Python.
- No se usa directamente EtPMon proporcionado por SIAR.
- **Tipo de dato**: Float (mm/día, normalizado 0-1 con MinMaxScaler durante entrenamiento, denormalizado para métricas).

### ✅ 2. Inputs según Tabla 4 del TFG

| Modelo | Inputs utilizados |
|--------|--------------------|
| ANN_Rs | Radiacion, TempMedia |
| ANN_Ra | TempMax, TempMin, TempMedia, Ra |
| ANN_HR | TempMax, TempMin, TempMedia, Ra, HumedadMedia |

- **Cómo coge los inputs**: El script carga cada CSV con `pd.read_csv()`, verifica columnas con `if col in df.columns`, y accede con `train_df[inputs].values` (array NumPy float).
- **Formato y tipo de datos**: Float (valores numéricos continuos, normalizados 0-1 con MinMaxScaler). El script verifica disponibilidad y avisa si faltan (ej. "Advertencia: Faltan columnas {'Ra'}").

---

### ✅ 3. Arquitectura de la red neuronal

| Parámetro        | Valor aplicado                  |
|------------------|----------------------------------|
| Capas ocultas    | 1                                |
| Neuronas         | 1 a 10                          |
| Activación       | `tanh` (tansig en MATLAB)       |
| Capa de salida   | 1 neurona, activación `linear`  |
| Optimización     | Adam (lr=0.001)                |
| Pérdida          | MSE (Error cuadrático medio)    |
| Early stopping   | Sí, patience = 1                |
| Épocas máximas   | 30 (optimizado)                 |
| Batch size       | 128 (optimizado para GPU)       |

- **TensorFlow/Keras**: Usa `Sequential` para capas secuenciales, `Input(shape=(len(inputs),))` para entrada, `Dense` para capas ocultas/salida. Compila con `compile(optimizer='adam', loss='mse')`.
- **Selección de modelo**: Por cada combinación, selecciona el mejor por MSE en validación (generalización) y test (ajuste).
- **Formato de salida del modelo**: Array NumPy float (predicciones denormalizadas, mm/día).

---

### ✅ 4. Validación cruzada por años (K-Fold temporal)

- Cada año del dataset se usa una vez como test.
- El resto de años se dividen en:
  - 85% entrenamiento
  - 15% validación
- Se repite para todas las estaciones y todos los modelos ANN.
- **Formato y tipo de datos**: Años como int (derivados de 'Fecha' datetime), test/train/val como DataFrames pandas con filas filtradas por año.

---

### ✅ 5. Métricas evaluadas

| Métrica | Descripción |
|---------|-------------|
| MSE     | Error cuadrático medio |
| RMSE    | Raíz de MSE |
| MAE     | Error absoluto medio |
| R²      | Coeficiente de determinación |
| AARE    | Error relativo absoluto medio |

- **Formato y tipo de datos**: Float (precisión decimal, redondeado a 3 dígitos en resumen).

---

### ⚙️ Ejecución del script

#### 📌 1. Instalar dependencias

```bash
pip install pandas numpy scikit-learn tensorflow
```

#### 📌 2. Ejecutar el script

```bash
python train_nn_et0_fast.py
```

#### 📌 3. Archivos generados

| Archivo              | Descripción                           | Formato | Tipo de datos |
|----------------------|---------------------------------------|---------|---------------|
| `nn_errors_fast.csv` | Métricas por estación, año, modelo    | CSV     | Estacion (str), Modelo (str), Seleccion (str), Test_Year (int), MSE (float), RMSE (float), MAE (float), R2 (float), AARE (float) |
| `nn_errors_summary.csv` | Media de errores por modelo y estación | CSV     | Estacion (str), Modelo (str), Seleccion (str), MSE (float), RMSE (float), MAE (float), R2 (float), AARE (float) |

- **Salida del modelo neuronal**: Predicciones de ET₀ como array NumPy float (mm/día, denormalizado de 0-1 a valores reales con inverse_transform).
- **Análisis de datos**: Resumen en consola (DataFrame pd.round(3)), con métricas medias por estación/modelo/selección.

---

### 🧾 Fragmento clave del script (`train_nn_et0_fast.py`)

```python
input_combinations = {
    'ANN_Rs': ['Radiacion', 'TempMedia'],
    'ANN_Ra': ['TempMax', 'TempMin', 'TempMedia', 'Ra'],
    'ANN_HR': ['TempMax', 'TempMin', 'TempMedia', 'Ra', 'HumedadMedia']
}

# Ejemplo de entrenamiento
model = Sequential([
    Input(shape=(len(inputs),)),
    Dense(n_neurons, activation='tanh'),
    Dense(1, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.fit(X_train_scaled, y_train_scaled, epochs=30, batch_size=128, verbose=0)
```

---

## ✅ Conclusiones

✔ ANN_HR (con humedad) presenta mejor desempeño.  
✔ Se replicó la metodología del TFG con fidelidad en Python.  
✔ Los errores medios son comparables o mejores que los modelos empíricos Hargreaves y Valiantzas.  
✔ Resultados listos para ser integrados en dashboards o informes.

---

## 🚀 Mejoras futuras

- Implementar Levenberg-Marquardt (como en MATLAB) mediante `tensorflow-probability`.
- Exportar modelos `.h5` para predicción operativa.
- Aplicar redes LSTM para series temporales. 

Este tutorial es reproducible; ajusta `estaciones` para más datos. ¡Ejecuta y compárteme `nn_errors_summary.csv` para analizar! 😊
---

## ACTUALIZACIÓN 23/10/2026 — Contexto histórico de TensorFlow

La implementación ANN y sus resultados anteriores se conservan como **HISTÓRICO / LEGACY**. No constituyen nuevos resultados ML sobre Dataset Observational v1.

El dataset actual está construido/validado desde POST corregido y la ejecución oficial PM/HS está PASS, con [resultados reproducibles](scientific_pipeline/results/pm-vpd-20261023T194553Z/README.md). Valiantzas sigue pendiente de auditoría y baseline reproducible antes de una comparación de tres métodos y del avance a ML.

Las futuras redes se evaluarán con partición temporal sin leakage y parámetros/transformaciones ajustados exclusivamente en TRAIN. Esta actualización no instala TensorFlow, entrena modelos ni modifica artefactos científicos. [Estado vigente](docs/PROJECT_STATUS.md).


---

## ACTUALIZACIÓN 24/10/2026 — Referencias oficiales y prioridad de preservación

ANN/TensorFlow anterior permanece HISTÓRICO / LEGACY. Los cuatro baselines clásicos PM_FAO56_V1, HS_RA_V1, VALIANTZAS_V1 y HG_RS_V1 y su comparación son PASS; [evidencia](scientific_pipeline/results/four-classical-20261024T042250Z/README.md). EtPMon es referencia calculada, no medición independiente.

Antes de nuevos modelos: PROJECT-ARTIFACT-BACKUP-V1 → AHC-METHODOLOGY-REVIEW → contrato temporal → modelos calibrados simples → regresión/árboles/boosting → ANN. La revisión AHC no está cerrada y definirá estimación/validación en misión separada; no se decide aquí uso de todo el dataset frente a TRAIN. Ningún nuevo artefacto ML ni entrenamiento se genera en esta misión.
