import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import qrcode
from io import BytesIO

st.set_page_config(page_title="Supply Chain Analytics", page_icon="📦", layout="wide")

st.title("📦 Supply Chain Analytics")
st.caption("Panel interactivo de análisis de demanda, regresión e inventarios")

with st.sidebar:
    st.header("Navegación")
    section = st.radio("Selecciona un modelo", [
        "Resumen general",
        "1. Series de tiempo — Winters",
        "2. Regresión explicativa — Price",
        "3. Rotación y cobertura de inventario",
        "4. Regresión OLS — Demanda",
        "Código QR del aplicativo"
    ])
    st.divider()
    st.caption("Dataset del caso: Supply Chain Analysis (100 SKU).")

def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)

def parse_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

if section == "Resumen general":
    st.subheader("Los cuatro enfoques del trabajo")
    st.write("El proyecto combina pronóstico temporal, regresión con variables explicativas e indicadores de inventario para apoyar la planificación de demanda y abastecimiento.")
    cards = st.columns(2)
    summaries = [
        ("1 · Winters", "Pronostica la demanda mensual usando nivel, tendencia y estacionalidad (α, β y γ)."),
        ("2 · Regresión explicativa", "Estima el precio a partir de variables de manufactura, defectos, lead time y cantidades pedidas."),
        ("3 · Inventario", "Calcula rotación y cobertura por categoría y permite evaluar alertas de quiebre."),
        ("4 · OLS", "Estima Number of Products Sold usando seis variables comerciales y logísticas.")
    ]
    for c, (title, desc) in zip(cards * 2, summaries):
        c.markdown(f"### {title}")
        c.write(desc)
    st.divider()
    st.subheader("Indicadores reportados en el informe")
    metric_row([("Muestra", "100 SKU"), ("SKUs con cobertura < 2 sem.", "21%"), ("Lead time medio", "17.08 días"), ("R² del OLS demanda", "0.021")])

elif section == "1. Series de tiempo — Winters":
    st.subheader("Pronóstico mensual de demanda — método de Winters")
    st.write("Serie histórica mensual 2024 y parámetros optimizables. El informe indica que Solver obtuvo α = β = γ = 0.1.")
    months = ["Ene-24","Feb-24","Mar-24","Abr-24","May-24","Jun-24","Jul-24","Ago-24","Sep-24","Oct-24","Nov-24","Dic-24"]
    demand = [2343,4394,3227,2197,3242,3420,4698,3274,6557,2519,3379,6849]
    df = pd.DataFrame({"Mes": months, "Demanda": demand})
    st.markdown("**Edita los valores históricos**")
    edited = st.data_editor(df, hide_index=True, use_container_width=True, num_rows="fixed")
    c1,c2,c3 = st.columns(3)
    alpha = c1.slider("Alfa (nivel)", 0.1, 0.9, 0.1, 0.05)
    beta = c2.slider("Beta (tendencia)", 0.1, 0.9, 0.1, 0.05)
    gamma = c3.slider("Gamma (estacionalidad)", 0.1, 0.9, 0.1, 0.05)
    horizon = st.slider("Meses a pronosticar", 1, 12, 3)
    vals = parse_numeric(edited["Demanda"]).dropna().values
    if len(vals) >= 4:
        try:
            model = ExponentialSmoothing(vals, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated")
            fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
            forecast = fit.forecast(horizon)
            dates = list(edited["Mes"]) + [f"Futuro +{i+1}" for i in range(horizon)]
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(range(len(vals)), vals, marker="o", label="Demanda ingresada")
            ax.plot(range(len(vals)-1, len(vals)+horizon), [vals[-1]] + list(forecast), marker="o", linestyle="--", label="Pronóstico")
            ax.set_xticks(range(len(dates))); ax.set_xticklabels(dates, rotation=45, ha="right")
            ax.set_ylabel("Unidades"); ax.grid(alpha=.25); ax.legend()
            st.pyplot(fig)
            st.dataframe(pd.DataFrame({"Periodo": [f"Futuro +{i+1}" for i in range(horizon)], "Pronóstico": np.round(forecast,2)}), use_container_width=True)
            st.caption("Nota: el informe utiliza inicialización propia (enero como periodo inicial) y Solver. Esta implementación de Holt-Winters usa inicialización estimada de statsmodels; por ello puede diferir de la tabla del informe.")
        except Exception:
            pass

elif section == "2. Regresión explicativa — Price":
    st.subheader("Regresión múltiple para estimar Price")
    st.write("El informe compara un Modelo 1 con cuatro predictores y un Modelo 2 que agrega Order Quantities.")
    st.markdown("**Coeficientes reportados**")
    coef = pd.DataFrame({
        "Variable":["Intercepto","Manufacturing lead time","Manufacturing costs","Defect rates","Lead time","Order quantities"],
        "Modelo 1":[73.490,-1.111,-0.230,-3.382,0.640,np.nan],
        "Modelo 2":[65.017,-1.167,-0.226,-3.490,0.693,0.172]
    })
    st.dataframe(coef, hide_index=True, use_container_width=True)
    st.markdown("**Modifica las variables para simular el precio estimado**")
    ranges = {
        "Manufacturing lead time": (1.0,30.0,15.0),
        "Manufacturing costs": (103.92,997.41,529.25),
        "Defect rates": (0.0,5.0,2.5),
        "Lead time": (1.0,30.0,17.08),
        "Order quantities": (1.0,1000.0,460.0)
    }
    x = {}
    cols = st.columns(2)
    for i,(name,(lo,hi,default)) in enumerate(ranges.items()):
        x[name] = cols[i%2].slider(name, float(lo), float(hi), float(default), step=0.1 if hi<50 else 1.0)
    p1 = 73.490 -1.111*x["Manufacturing lead time"] -0.230*x["Manufacturing costs"] -3.382*x["Defect rates"] +0.640*x["Lead time"]
    p2 = 65.017 -1.167*x["Manufacturing lead time"] -0.226*x["Manufacturing costs"] -3.490*x["Defect rates"] +0.693*x["Lead time"] +0.172*x["Order quantities"]
    metric_row([("Precio estimado · Modelo 1", f"{p1:,.2f}"),("Precio estimado · Modelo 2",f"{p2:,.2f}")])
    st.caption("Son estimaciones calculadas con los coeficientes publicados en las tablas 4.3 y 4.4. No se extrapolan coeficientes ni se reentrena el modelo.")
    st.markdown("**Comparación de ajuste reportada**")
    metric_row([("R² Modelo 1","0.186"),("R² ajustado M1","0.152"),("R² Modelo 2","0.207"),("R² ajustado M2","0.165")])

elif section == "3. Rotación y cobertura de inventario":
    st.subheader("Rotación y cobertura por categoría")
    st.write("Edita stock y demanda semanal para observar cómo cambian los indicadores.")
    df = pd.DataFrame({
        "Categoría":["Skincare","Haircare","Cosmetics"],
        "N.º SKUs":[40,34,26],
        "Stock total":[1608,1644,1525],
        "Rotación reportada":[12.89,8.28,7.71],
        "Cobertura reportada":[4.03,6.28,6.74],
        "Demanda semanal":[1608/4.03,1644/6.28,1525/6.74]
    })
    edited = st.data_editor(df, hide_index=True, use_container_width=True, column_config={
        "N.º SKUs": st.column_config.NumberColumn(min_value=1, step=1),
        "Stock total": st.column_config.NumberColumn(min_value=0, step=1),
        "Demanda semanal": st.column_config.NumberColumn(min_value=0.1, step=1)
    })
    threshold = st.slider("Umbral de alerta de cobertura (semanas)", 1.0, 8.0, 2.0, 0.5)
    out = edited.copy()
    # Se conserva como referencia el volumen de ventas implícito en la tabla original:
    # ventas = rotación reportada × stock original.
    stock_original = np.array([1608, 1644, 1525], dtype=float)
    rotacion_original = np.array([12.89, 8.28, 7.71], dtype=float)
    out["Ventas totales base (estimadas)"] = stock_original * rotacion_original
    out["Rotación recalculada"] = out["Ventas totales base (estimadas)"] / out["Stock total"].replace(0,np.nan)
    out["Cobertura recalculada (sem.)"] = out["Stock total"] / out["Demanda semanal"]
    out["Estado"] = np.where(out["Cobertura recalculada (sem.)"] < threshold, "ALERTA", "Normal")
    st.dataframe(out[["Categoría","Stock total","Ventas totales base (estimadas)","Rotación recalculada","Demanda semanal","Cobertura recalculada (sem.)","Estado"]], hide_index=True, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(out["Categoría"], out["Cobertura recalculada (sem.)"])
    ax.axhline(threshold, linestyle="--", label=f"Umbral {threshold} sem.")
    ax.set_ylabel("Semanas de cobertura"); ax.legend(); ax.grid(axis="y",alpha=.2)
    st.pyplot(fig)
    st.metric("SKUs con cobertura menor al umbral (referencia del informe)", "21 de 100 (21%)")

elif section == "4. Regresión OLS — Demanda":
    st.subheader("OLS para predecir Number of Products Sold")
    st.write("Resultados del modelo estimado en el informe y exploración interactiva de los rangos de las variables explicativas.")

    # Resultados publicados en el informe: se muestran directamente, sin pedir CSV.
    metric_row([
        ("R² reportado", "0.021"),
        ("R² ajustado", "-0.042"),
        ("MAE", "258.7"),
        ("RMSE", "299.0"),
        ("MAPE", "205.8%")
    ])

    st.markdown("#### Desempeño reportado del modelo")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        labels = ["R²", "R² ajustado"]
        values = [0.021, -0.042]
        ax.bar(labels, values)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel("Valor")
        ax.set_title("Capacidad explicativa")
        ax.grid(axis="y", alpha=.2)
        st.pyplot(fig, use_container_width=True)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        labels = ["MAE", "RMSE"]
        values = [258.7, 299.0]
        ax.bar(labels, values)
        ax.set_ylabel("Error (unidades)")
        ax.set_title("Errores de predicción")
        ax.grid(axis="y", alpha=.2)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
        st.pyplot(fig, use_container_width=True)

    st.caption("El informe reporta R² = 0.021, R² ajustado = -0.042, MAE = 258.7, RMSE = 299.0 y MAPE = 205.8%. Los gráficos anteriores representan esos indicadores, no observaciones individuales.")

    st.divider()
    st.markdown("#### Exploración interactiva de rangos de entrada")
    st.write("Modifica los límites inferior y superior de cada variable. El gráfico se actualiza automáticamente para comparar los intervalos seleccionados.")
    st.caption("Los rangos conocidos se inicializan con los estadísticos descriptivos del informe. Para las variables sin mínimo y máximo publicados, se muestran límites iniciales editables de exploración; no deben interpretarse como rangos observados del dataset.")

    range_defaults = {
        "Price": (1.70, 99.17),
        "Availability": (0.0, 100.0),
        "Stock levels": (0.0, 100.0),
        "Lead times": (1.0, 30.0),
        "Order quantities": (0.0, 100.0),
        "Shipping times": (0.0, 100.0)
    }
    range_values = {}
    range_cols = st.columns(2)
    for i, (name, (default_min, default_max)) in enumerate(range_defaults.items()):
        with range_cols[i % 2]:
            st.markdown(f"**{name}**")
            cmin, cmax = st.columns(2)
            lo = cmin.number_input("Mínimo", value=float(default_min), key=f"ols_min_{i}")
            hi = cmax.number_input("Máximo", value=float(default_max), key=f"ols_max_{i}")
            if lo <= hi:
                range_values[name] = (lo, hi)
            else:
                st.warning("El mínimo debe ser menor o igual al máximo.")

    valid_ranges = {k: v for k, v in range_values.items()}
    if valid_ranges:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        names = list(valid_ranges.keys())
        lows = [valid_ranges[n][0] for n in names]
        widths = [valid_ranges[n][1] - valid_ranges[n][0] for n in names]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, widths, left=lows)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Rango seleccionado")
        ax.set_title("Intervalos configurados para las variables OLS")
        ax.grid(axis="x", alpha=.2)
        st.pyplot(fig, use_container_width=True)
        range_table = pd.DataFrame({
            "Variable": names,
            "Límite inferior": lows,
            "Límite superior": [valid_ranges[n][1] for n in names]
        })
        st.dataframe(range_table, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("#### Interpretación del resultado")
    st.write("Con el R² reportado de 0.021, el modelo explica aproximadamente el 2.1% de la variabilidad observada en la demanda dentro de la muestra analizada. El R² ajustado negativo indica que, al considerar el número de predictores, el ajuste no supera al modelo que utiliza únicamente la media.")
    st.write("El informe señala que únicamente la constante fue estadísticamente significativa (p = 0.0068). Los coeficientes individuales de los seis predictores y los registros necesarios para reconstruir los gráficos de valores reales frente a predichos no están incluidos en el documento; por ello, esta sección presenta los indicadores disponibles y no inventa predicciones ni puntos de dispersión.")

else:
    st.subheader("Generar código QR para abrir el Streamlit")
    st.write("Primero publica la aplicación (por ejemplo, en Streamlit Community Cloud). Luego pega aquí la URL pública y genera el QR. Un QR no puede apuntar a tu computadora local si otras personas necesitan acceder desde sus teléfonos.")
    url = st.text_input("URL pública de tu aplicación", placeholder="https://tu-app.streamlit.app")
    if st.button("Generar QR", type="primary"):
        if url.startswith("https://") or url.startswith("http://"):
            qr = qrcode.QRCode(box_size=10,border=3)
            qr.add_data(url); qr.make(fit=True)
            img=qr.make_image(fill_color="black",back_color="white")
            buf=BytesIO(); img.save(buf,format="PNG")
            st.image(buf.getvalue(),caption="Escanea para abrir el aplicativo",width=260)
            st.download_button("Descargar QR (PNG)",buf.getvalue(),"qr_streamlit.png","image/png")

st.divider()
st.caption("Proyecto académico basado en el informe “Modelo de analítica predictiva para la planificación de la demanda y optimización de la gestión de inventarios en una cadena de suministro”.")
