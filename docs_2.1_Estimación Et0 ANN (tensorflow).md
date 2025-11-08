# 2.1 Estimación de ET₀ mediante Redes Neuronales Artificiales (ANN)

Este documento explica paso a paso cómo se implementaron las **Redes Neuronales Artificiales (ANN)** para estimar la evapotranspiración de referencia **ET₀**, replicando la metodología del **TFG original**, pero adaptada a **Python/Keras**.

Se integra como continuación natural del documento:

**`docs_1.2_Tratamiento de datos y cálculo de estimaciones.md`**

---

### 🎯 Objetivo

Desarrollar y evaluar **tres modelos de redes neuronales artificiales** que estimen ET₀ (mm/día), usando como referencia los valores calculados con **Penman-Monteith (PM)** en Python.

Estos modelos se comparan con sus equivalentes empíricos:

| Modelo ANN | Inputs utilizados                                  | Modelo empírico equivalente |
|------------|----------------------------------------------------|-----------------------------|
| **ANN_Rs** | Radiación solar medida (Rs), Temperatura media    | HGRₛ                        |
| **ANN_Ra** | TempMax, TempMin, TempMedia, Radiación extraterrestre (Ra) | HGRₐ          |
| **ANN_HR** | TempMax, TempMin, TempMedia, Ra, Humedad media    | HGHR                        |

---

### 📁 Estructura del proyecto

```
📂 datos_siar_baleares/
 ├─ IB01_et0_variants.csv
 ├─ IB02_et0_variants.csv
 ├─ ...
 ├─ train_nn_et0.py   ← Script principal ANNs
📂 outputs/
 ├─ nn_errors.csv         ← Errores por año
 ├─ nn_errors_summary.csv ← Errores medios resumen
```

---

### 🧠 Metodología aplicada

#### ✅ 1. Target (salida de la red)

- Se utiliza **ET0_calc**, calculado con la ecuación FAO-56 Penman-Monteith en Python.
- No se usa directamente EtPMon proporcionado por SIAR.

#### ✅ 2. Inputs según Tabla 4 del TFG

| Modelo | Inputs utilizados |
|--------|--------------------|
| ANN_Rs | Radiacion, TempMedia |
| ANN_Ra | TempMax, TempMin, TempMedia, Ra |
| ANN_HR | TempMax, TempMin, TempMedia, Ra, HumedadMedia |

- **Formato y tipo de datos**: Float (valores numéricos continuos, normalizados 0-1 con MinMaxScaler). El script verifica la disponibilidad de columnas y avisa si faltan (ej. "Advertencia: Faltan columnas {'Ra'}").

---

#### ✅ 3. Arquitectura de la red neuronal

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

- **Selección de modelo**: Por cada combinación, selecciona el mejor por MSE en validación (generalización) y test (ajuste).

---

#### ✅ 4. Validación cruzada por años (K-Fold temporal)

- Cada año del dataset se usa **una vez como test**.
- El resto de años se dividen en:
  - 85% entrenamiento
  - 15% validación
- Se repite para todas las estaciones y todos los modelos ANN.

- **Formato y tipo de datos**: Años como int (derivados de 'Fecha' datetime), test/train/val como DataFrames pandas con filas filtradas por año.

---

#### ✅ 5. Métricas evaluadas

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

- **Salida del modelo neuronal**: Predicciones de ET₀ como array float (mm/día, denormalizado de 0-1 a valores reales con inverse_transform).
- **Análisis de datos**: Resumen en consola (DataFrame pd.round(3)), con métricas medias por estación/modelo/selección.

---

### 🧾 Fragmento clave del script (`train_nn_et0_fast.py`)

```python:disable-run
input_combinations
```

