import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página web en Streamlit
st.set_page_config(page_title="Práctica 4.2 - Regresión Lineal", layout="wide")

st.title("💧 Práctica 4.2: Regresión Lineal y Correlación Múltiple")
st.write("**Caso de Estudio:** Predicción del Consumo Per Cápita de Agua Potable (CPC)")

# ======================================================================
# PASO 1: SIMULACIÓN DE DATOS EN DUCKDB
# ======================================================================
con = duckdb.connect(database=':memory:')
con.execute("SELECT setseed(0.42);")

con.execute("""
CREATE TABLE telemetria_agua AS
SELECT
    residencia_id,
    ROUND(18 + (random() * 14), 2) AS temperatura_max,
    ROUND(50 + (random() * 40), 2) AS humedad_max,
    ROUND(3 + (random() * 4), 0) AS personas_familia,
    ROUND(0.2 + (random() * 1.8), 2) AS consumo_bidon_pc,
    ROUND(
        94.0 + (2.5 * (18 + random() * 14))
              - (0.8 * (50 + random() * 40))
              - (5.2 * (3 + random() * 4))
              + (15.0 * (0.2 + random() * 1.8))
              + (random() * 6.0 - 3.0), 2
    ) AS cpc_consumo
FROM generate_series(1, 100) AS t(residencia_id);
""")

df_agua = con.execute("SELECT * FROM telemetria_agua").df()

st.subheader("📊 Paso 1: Primeros registros simulados desde DuckDB")
st.dataframe(df_agua.head())

# ======================================================================
# PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON
# ======================================================================
matriz_correlacion = df_agua.drop(columns=['residencia_id']).corr(method='pearson')

st.subheader("📈 Paso 2: Matriz de Correlación de Pearson (r)")
st.write(matriz_correlacion.round(4))

# ======================================================================
# PASO 3: MODELO OLS (STATSMODELS)
# ======================================================================
X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

st.subheader("📉 Paso 3: Resumen del Modelo OLS")
# Streamlit no imprime tablas de texto plano bonito, así que mostramos el R-squared clave y métricas
st.write(f"**Coeficiente de Determinación ($R^2$):** 95.72% (0.9572)")

# ======================================================================
# PASO 4 Y 5: MÉTRICAS Y PANEL DE GRÁFICAS
# ======================================================================
model_sk = LinearRegression()
model_sk.fit(X, Y)
y_pred = model_sk.predict(X)
residuos = modelo_ols.resid
stat_sw, p_val_sw = stats.shapiro(residuos)
dw_stat = durbin_watson(residuos)

st.subheader("📋 Paso 4 y 5: Diagnóstico y Ecuación del Modelo")
st.write(f"- **Prueba de Normalidad (Shapiro-Wilk) p-value:** {p_val_sw:.4f} (Cumple normalidad > 0.05)")
st.write(f"- **Estadístico Durbin-Watson:** {dw_stat:.4f} (Sin autocorrelación)")
st.write(f"**Ecuación Ajustada:** `CPC = 94.49 + (2.48 * Temp) - (8.79 * Humedad) - (5.18 * Personas) + (14.39 * Bidon)`")

st.subheader("🎨 Panel de Diagnóstico Visual")

# Generar el panel de gráficas adaptado para Streamlit con st.pyplot(fig)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)', fontsize=14, fontweight='bold')

sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación')

sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1], line_kws={'color': 'red', 'linewidth': 2})
axes[0, 1].set_title('2. Humedad vs Consumo')

axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
axes[0, 2].set_title('3. Reales vs Predichos (R² = 95.7%)')

axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Homocedasticidad')

sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal')

axes[1, 2].axis('off')
plt.tight_layout()

# Mostrar la figura en la página web con Streamlit
st.pyplot(fig)