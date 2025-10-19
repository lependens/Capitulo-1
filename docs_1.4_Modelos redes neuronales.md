# 2.1 Estimación de ET₀ mediante Redes Neuronales Artificiales (ANN)

Este documento explica paso a paso cómo se implementaron las **Redes Neuronales Artificiales (ANN)** para estimar la evapotranspiración de referencia **ET₀**, replicando la metodología del **TFG original**, pero adaptada a **Python/Keras**.

Se integra como continuación natural del documento:

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

---

## 📁 Estructura del proyecto

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

## 🧠 Metodología aplicada

### ✅ 1. Target (salida de la red)

- Se utiliza **ET0_calc**, calculado con la ecuación FAO-56 Penman-Monteith en Python.
- No se usa directamente EtPMon proporcionado por SIAR.

### ✅ 2. Inputs según Tabla 4 del TFG

| Modelo | Inputs utilizados |
|--------|--------------------|
| ANN_Rs | Radiacion, TempMedia |
| ANN_Ra | TempMax, TempMin, TempMedia, Ra |
| ANN_HR | TempMax, TempMin, TempMedia, Ra, HumedadMedia |

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
| Épocas máximas   | 100                             |

---

### ✅ 4. Validación cruzada por años (K-Fold temporal)

- Cada año del dataset se usa **una vez como test**.
- El resto de años se dividen en:
  - 85% entrenamiento
  - 15% validación
- Se repite para todas las estaciones y todos los modelos ANN.

---

### ✅ 5. Métricas evaluadas

| Métrica | Descripción |
|---------|-------------|
| MSE     | Error cuadrático medio |
| RMSE    | Raíz de MSE |
| MAE     | Error absoluto medio |
| R²      | Coeficiente de determinación |
| AARE    | Error relativo absoluto medio |

---

## ⚙️ Ejecución del script

### 📌 1. Instalar dependencias

```bash
pip install pandas numpy tensorflow scikit-learn
```

### 📌 2. Ejecutar el script

```bash
python train_nn_et0.py
```

### 📌 3. Archivos generados

| Archivo           | Descripción                           |
|------------------|----------------------------------------|
| `nn_errors.csv`   | Métricas por estación, año, modelo     |
| `nn_errors_summary.csv` | Media de errores por modelo y estación |

---

## 🧾 Fragmento clave del script (`train_nn_et0.py`)

```python
input_combinations = {
    'ANN_Rs': ['Radiacion', 'TempMedia'],
    'ANN_Ra': ['TempMax', 'TempMin', 'TempMedia, 'Ra'],
    'ANN_HR': ['TempMax', 'TempMin', 'TempMedia', 'Ra', 'HumedadMedia']
}
```

---

## 📊 Resultados esperados (ejemplo)

| Estación | Modelo  | Selección | RMSE | MAE  | R²   |
|----------|---------|-----------|------|------|------|
| IB01     | ANN_HR  | Test      | 0.30 | 0.25 | 0.94 |
| IB01     | ANN_Ra  | Valid     | 0.40 | 0.32 | 0.91 |
| IB02     | ANN_Rs  | Test      | 0.52 | 0.41 | 0.88 |

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

