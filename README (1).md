# 📈 Predicción de Tendencias de Bitcoin (BTC/USD) a 10 Días Horizonte

**Proyecto de Machine Learning & Análisis Cuantitativo Multi-Output**  
**Autor:** Alejandro Karambasis Rodríguez  
**Fecha:** Diciembre 2025  

---

## 📋 Tabla de Contenidos
1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Objetivo y Enfoque del Problema](#-objetivo-y-enfoque-del-problema)
3. [Estructura del Repositorio](#-estructura-del-repositorio)
4. [Dataset y Datos Históricos](#-dataset-y-datos-históricos)
5. [Pipeline del Proyecto](#-pipeline-del-proyecto)
   - [Fase 1: Análisis Exploratorio de Datos (EDA)](#fase-1-análisis-exploratorio-de-datos-eda)
   - [Fase 2: Preprocesamiento e Ingeniería de Características](#fase-2-preprocesamiento-e-ingeniería-de-características)
   - [Fase 3: Entrenamiento y Selección de Modelos (Multi-Output)](#fase-3-entrenamiento-y-selección-de-modelos-multi-output)
6. [Resultados y Comparativa de Modelos](#-resultados-y-comparativa-de-modelos)
7. [Validación de Hipótesis](#-validación-de-hipótesis)
8. [Instalación y Requisitos](#-instalación-y-requisitos)
9. [Guía de Ejecución](#-guía-de-ejecución)
10. [Conclusiones y Trabajo Futuro](#-conclusiones-y-trabajo-futuro)

---

## 📌 Descripción del Proyecto

El mercado de las criptomonedas, y en particular **Bitcoin (BTC/USD)**, se caracteriza por una alta volatilidad, no linealidad y dependencia temporal estocástica. Este proyecto desarrolla un sistema integral de **Machine Learning para la clasificación de tendencias direccionales** (alcista / bajista) a lo largo de una ventana de **10 días hacia el futuro** mediante un esquema **Multi-Output**.

En lugar de predecir únicamente el precio puntual del día siguiente (enfoque propenso al ruido de microestructura) o un único horizonte agregado a 10 días, el sistema predice día por día si el precio de cierre en $T+n$ ($n \in \{1, 2, \dots, 10\}$) superará el precio actual en $T$.

---

## 🎯 Objetivo y Enfoque del Problema

- **Tipo de problema:** Clasificación Binaria Supervisada Multi-Output (10 variables objetivo independientes).
- **Variable Target ($Y$):**
  $$	ext{Tendencia\_dia}_n = egin{cases} 1 & 	ext{si } 	ext{Precio}_{t+n} > 	ext{Precio}_t 	ext{ (Alcista)} \ 0 & 	ext{si } 	ext{Precio}_{t+n} \le 	ext{Precio}_t 	ext{ (Bajista)} \end{cases} \quad orall n \in \{1, \dots, 10\}$$
- **Ventajas operativas:**
  - Permite identificar no sólo la dirección inmediata, sino el perfil de maduración temporal de la tendencia para estrategias de swing trading y gestión de riesgo.
  - Mitiga el sesgo de punto ciego de modelos univariables a $T+1$.

---

## 📂 Estructura del Repositorio

```text
├── RAW/
│   ├── DATOS_BITCOIN.csv             # Dataset histórico original
│   ├── Train_Bitcoin.csv             # Train split (80%) con targets
│   └── Test_Bitcoin.csv              # Test split (20%) con targets
├── PROCESSED/
│   ├── Train_Bitcoin_Processed.csv   # Train preprocesado y escalado
│   └── Test_Bitcoin_Processed.csv    # Test preprocesado y escalado
├── MODELS/
│   ├── best_model_random_forest.pkl  # Mejor modelo serializado
│   ├── model_baseline.pkl            # Modelo de referencia
│   ├── model_xgboost.pkl             # Modelo XGBoost Multi-Output
│   ├── model_lightgbm.pkl            # Modelo LightGBM Multi-Output
│   ├── model_config.json             # Metadatos y configuración
│   ├── model_comparison.csv          # Métricas comparativas
│   ├── metrics_random_forest.csv     # Rendimiento detallado RF día a día
│   ├── metrics_xgboost.csv           # Rendimiento detallado XGBoost día a día
│   ├── metrics_lightgbm.csv          # Rendimiento detallado LightGBM día a día
│   ├── metrics_baseline.csv          # Rendimiento baseline
│   ├── auc_comparison.png            # Visualización ROC-AUC por día
│   ├── confusion_matrices.png        # Matrices de confusión (10 días)
│   ├── feature_importance.png        # Importancia de características
│   ├── model_comparison.png          # Gráfico comparativo de modelos
│   └── roc_curves_10days.png         # Curvas ROC de los 10 días
├── 01_EDA_Bitcoin_COMPLETO_CORREGIDO_10DIAS.ipynb
├── 02_Preprocessing_Bitcoin_FINAL_10DIAS.ipynb
├── 03_ModelSelection_Bitcoin_FINAL_PREDICCION_FUTURA.ipynb
├── Requirements.txt                  # Dependencias de entorno
└── README.md                         # Documentación del proyecto
```

---

## 📊 Dataset y Datos Históricos

Los datos corresponden a series de tiempo diarias de Bitcoin en dólares/euros extraídas de fuentes de mercado (*CoinMarketCap / Yahoo Finance*).

- **Registros iniciales:** 398 observaciones diarias (período 2024–2025).
- **Atributos originales:** `timeOpen`, `timeClose`, `timeHigh`, `timeLow`, `name`, `open`, `high`, `low`, `close`, `volume`, `marketCap`, `circulatingSupply`, `timestamp`.
- **Precios observados:** Mínimo de €60,274.50, máximo de €124,752.53, con una media de €97,127.87 y una variación acumulada del 106.97%.
- **Manejo del horizonte terminal:** Se eliminaron las últimas 10 observaciones del histórico que carecían de horizonte futuro conocido a 10 días (reducción a 388 muestras en el dataset base).

---

## ⚙️ Pipeline del Proyecto

### Fase 1: Análisis Exploratorio de Datos (EDA)
- **Generación de Variables Target:** Creación sistemática de `Tendencia_dia1` a `Tendencia_dia10` con `shift(-n)`.
- **Balance de Clases:**
  - Día 1: 51.8% alcistas / 48.2% bajistas
  - Día 10: 57.0% alcistas / 43.0% bajistas
  - Balance promedio global del 55.9%, manteniéndose en rango equilibrado para clasificación sin necesidad estricta de sobremuestreo sintético.
- **Análisis Estadístico de Retornos y Volatilidad:**
  - Retorno diario medio: +0.17%, desviación estándar: 2.31%.
  - Asimetría (*Skewness*): +0.4070; Curtosis: 2.7890.
  - Volatilidad anualizada de ~44.12%.
- **Tests de Estacionariedad:**
  - **ADF (Augmented Dickey-Fuller):** $p$-valor $= 0.2715 > 0.05$ (se confirma que la serie de precios es **no estacionaria**).
  - **KPSS (Kwiatkowski-Phillips-Schmidt-Shin):** Estadístico de $2.2271$ ($p < 0.01$), corroborando la no estacionariedad y la necesidad de transformar hacia retornos e indicadores de ratio.
- **División Temporal Estricta:** División 80/20 cronológica (Train: 310 filas, Test: 78 filas) garantizando cero mezcla de información del futuro hacia el pasado (*Time Series Split*).

---

### Fase 2: Preprocesamiento e Ingeniería de Características

#### 1. Características Diseñadas (Feature Engineering)
Para dotar al modelo de representaciones con fundamento económico y cuantitativo:
1. `daily_return`: Retorno porcentual diario del precio de cierre (momentum a 1 día).
2. `MA_7`: Media móvil simple de 7 días (tendencia de corto plazo / 1 semana).
3. `MA_30`: Media móvil simple de 30 días (tendencia mensual).
4. `volatility_14`: Desviación estándar móvil de los retornos a 14 días (régimen de volatilidad).
5. `RSI_14`: Relative Strength Index a 14 periodos con suavizado exponencial para capturar condiciones de sobrecompra ($>70$) y sobreventa ($<30$).
6. `volume_ratio`: Ratio del volumen del día frente a su media de 7 días ($	ext{Volume}_t / 	ext{SMA}_7(	ext{Volume})$).

#### 2. Depuración de Columnas Redundantes
Se removieron columnas de alta colinealidad o meramente informativas: `timeOpen`, `timeClose`, `timeHigh`, `timeLow`, `timestamp`, `open`, `high`, `low`, `name`, `volume_MA_7`, `volume_cat` y `range_cat`.

#### 3. Manejo de Outliers mediante Winsorization
En series financieras, descartar filas extremas introduce huecos temporales y pérdida de información crítica de caídas o repuntes bruscos. Se aplicó **Winsorization** en percentiles $[1\%, 99\%]$ exclusivamente sobre las variables predictoras numéricas (excluyendo los targets binarios).

#### 4. Escalado Robusto (RobustScaler)
- **Técnica:** Utilización de `RobustScaler` basado en mediana e IQR:
  $$\hat{x} = rac{x - 	ext{mediana}}{	ext{IQR}}$$
- **Control de Data Leakage:** El escalador fue entrenado (`fit`) **únicamente con el conjunto de Train** y proyectado (`transform`) de forma separada sobre Train y Test.

---

### Fase 3: Entrenamiento y Selección de Modelos (Multi-Output)

Se empleó el meta-estimador `MultiOutputClassifier` de Scikit-Learn sobre 4 familias de algoritmos:

1. **Baseline (Dummy Classifier):** Estrategia de clase mayoritaria (`most_frequent`). Benchmark empírico base.
2. **Random Forest Multi-Output:**
   - 200 estimadores, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=4`, `random_state=42`.
3. **XGBoost Multi-Output:**
   - 200 estimadores, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='logloss'`.
4. **LightGBM Multi-Output:**
   - 200 estimadores, `max_depth=6`, `learning_rate=0.1`, `num_leaves=31`, `subsample=0.8`.

---

## 🏆 Resultados y Comparativa de Modelos

Evaluación de los modelos sobre el **conjunto de Test (76 observaciones fuera de muestra)**:

| Modelo | Accuracy Promedio | Precision Promedio | Recall Promedio | F1-Score Promedio | MCC Promedio | ROC-AUC Promedio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Ganador)** | **0.6355 (63.6%)** | **0.5796** | **0.8535** | **0.6866** | **0.3197** | **0.7268** |
| **XGBoost** | 0.6013 (60.1%) | 0.5503 | 0.8634 | 0.6677 | 0.2780 | 0.6850 |
| **LightGBM** | 0.5868 (58.7%) | 0.5401 | 0.7420 | 0.6404 | 0.2647 | 0.6620 |
| **Baseline (Mayoría)** | 0.4684 (46.8%) | 0.4684 | 1.0000 | 0.6370 | 0.0000 | 0.5000 |

### Rendimiento por Día (Random Forest):
- **Día 1:** Accuracy: 53.9%, ROC-AUC: 0.5392 (dificultad alta en micro-horizonte de 24h).
- **Día 3:** Accuracy: 63.2%, ROC-AUC: 0.6431.
- **Día 7:** Accuracy: 69.7%, ROC-AUC: 0.6965.
- **Día 8:** Accuracy: 71.1%, ROC-AUC: **0.8641** (excelente separación de tendencia).
- **Día 10:** Accuracy: **72.4%**, ROC-AUC: 0.7833.

> **Conclusión de rendimiento:** Random Forest superó al Baseline en un **+35.67% relativo de Accuracy**, demostrando capacidad para generalizar la inercia macro-temporal de Bitcoin a partir del día 5 hacia adelante.

### Variables Más Relevantes (Feature Importance):
1. **`MA_30` (Media Móvil a 30 días):** 12.89% de peso explicativo.
2. **`circulatingSupply`:** 12.71%.
3. **`volatility_14` (Volatilidad):** 10.50%.
4. **`marketCap`:** 10.26%.
5. **`close` (Precio actual):** 9.85%.
6. **`MA_7`:** 9.57%.

---

## 🔬 Validación de Hipótesis

- **H1 (Volumen vs Tendencia):** *Días con mayor volumen tienden a ser días alcistas.*  
  **Resultado: RECHAZADA.** En 10/10 días analizados, los días de volumen bajo tuvieron proporcionalmente mayor tasa de cierres alcistas que los días de volumen atípico (diferencia media de -7.99%). Los picos de volumen se correlacionaron fuertemente con capitulaciones y ventas por pánico.
- **H2 (Precio vs Market Cap):** *Existe correlación positiva lineal estricta entre Precio y Market Cap.*  
  **Resultado: CONFIRMADA.** Coeficiente de Pearson $= 1.0000$ ($p < 10^{-15}$).
- **H3 (Rango diario vs Alcismo):** *Rangos intradiarios mayores (High - Low) asocian a días alcistas.*  
  **Resultado: RECHAZADA/MIXTA.** Los rangos extremos se presentan de forma simétrica durante liquidaciones bajistas.
- **H4 (Velas Verdes previas vs Tendencia):** *Cierre previo superior a apertura predice mayor probabilidad de continuación.*  
  **Resultado: RECHAZADA.** Patrón de reversión a la media a corto plazo detectado en datos intradiarios (-5.66% de diferencia).

---

## 💻 Instalación y Requisitos

Se requiere **Python 3.9 o superior**. Se aconseja crear un entorno virtual limpio:

```bash
# Crear entorno virtual
python -m venv bitcoin_env

# Activar entorno
# En Windows:
bitcoin_env\Scripts\activate
# En Linux/macOS:
source bitcoin_env/bin/activate

# Instalar dependencias
pip install -r Requirements.txt
```

### Librerías Clave (`Requirements.txt`):
```text
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
statsmodels>=0.13.5
joblib>=1.2.0
optuna>=3.0.0
```

---

## 🚀 Guía de Ejecución

Para reproducir el pipeline completo de manera determinista:

1. **Paso 1: Exploración y Creación de Targets:**
   ```bash
   jupyter notebook 01_EDA_Bitcoin_COMPLETO_CORREGIDO_10DIAS.ipynb
   ```
   *Genera `RAW/Train_Bitcoin.csv` y `RAW/Test_Bitcoin.csv` con los 10 targets calculados.*

2. **Paso 2: Preprocesamiento y Escalado:**
   ```bash
   jupyter notebook 02_Preprocessing_Bitcoin_FINAL_10DIAS.ipynb
   ```
   *Crea las 6 variables predictoras, ejecuta Winsorization y `RobustScaler` sin data leakage. Exporta a `PROCESSED/`.*

3. **Paso 3: Entrenamiento y Evaluación de Modelos:**
   ```bash
   jupyter notebook 03_ModelSelection_Bitcoin_FINAL_PREDICCION_FUTURA.ipynb
   ```
   *Entrena los 4 modelos Multi-Output, evalúa con ROC-AUC y exporta los artefactos finales a `MODELS/`.*

---

## 🔮 Conclusiones y Trabajo Futuro

### Limitaciones Identificadas:
1. **Tamaño muestral:** Ventana temporal de ~388 días efectivos; vulnerable a cambios de ciclo macroeconómico (halvings, tasas de interés de la FED).
2. **Ausencia de datos exógenos:** El precio cripto responde fuertemente a eventos de noticias, liquidaciones de derivados y flujos de ETF.

### Próximas Líneas de Mejora:
- **Datos On-Chain y Derivados:** Integrar tasas de financiamiento (*Funding Rates*), interés abierto (*Open Interest*) y ratio NVT.
- **Modelos Secuenciales:** Implementar arquitecturas neuronales recurrentes (LSTM, GRU) y capas de atención temporal (*Temporal Fusion Transformers*).
- **Estrategia Cuantitativa de Backtesting:** Convertir las señales probabilísticas del modelo en reglas de entrada/salida y dimensionamiento de posición (*Kelly Criterion*), evaluando métricas financieras (Ratio de Sharpe, Sortino y Máximo Drawdown).
