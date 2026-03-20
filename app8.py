import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Fresh Meat AI", layout="wide")

# =========================
# Helpers
# =========================
def generate_demo_data(n: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    stores = ["MTY San Pedro", "MTY Cumbres", "MTY Contry", "Saltillo Centro", "Apodaca"]
    categories = ["Res", "Pollo", "Cerdo"]
    cuts = {
        "Res": ["Ribeye", "Diezmillo", "Milanesa", "Sirloin"],
        "Pollo": ["Pechuga", "Pierna", "Muslo", "Alitas"],
        "Cerdo": ["Chuleta", "Lomo", "Costilla", "Pierna"],
    }
    suppliers = ["Proveedor A", "Proveedor B", "Proveedor C"]

    rows = []
    for lot_id in range(1, n + 1):
        category = rng.choice(categories, p=[0.35, 0.40, 0.25])
        cut = rng.choice(cuts[category])
        store = rng.choice(stores)
        supplier = rng.choice(suppliers)

        shelf_life_days = {
            "Res": int(rng.integers(6, 12)),
            "Pollo": int(rng.integers(4, 8)),
            "Cerdo": int(rng.integers(5, 9)),
        }[category]

        age_days = int(rng.integers(0, shelf_life_days + 2))
        temp_avg = float(np.round(rng.normal(3.4, 1.4), 2))
        temp_max = float(np.round(temp_avg + abs(rng.normal(1.3, 0.8)), 2))
        hours_out_of_range = float(np.round(max(0, rng.normal(1.8, 2.0)), 2))
        inventory_units = int(rng.integers(5, 90))
        daily_sales = float(np.round(max(0.1, rng.normal(8, 4)), 2))
        price = float(np.round(rng.uniform(85, 320), 2))
        markdown_pct = float(np.round(rng.choice([0, 0, 0, 10, 15, 20, 25]), 2))
        shrink_hist = float(np.round(np.clip(rng.normal(7, 4), 0, 25), 2))

        remaining_days = shelf_life_days - age_days

        risk = (
            18 * (age_days / max(shelf_life_days, 1))
            + 12 * max(temp_avg - 4, 0)
            + 6 * max(temp_max - 5, 0)
            + 4.5 * hours_out_of_range
            + 0.12 * inventory_units
            - 1.3 * daily_sales
            + 0.6 * shrink_hist
            - 0.35 * markdown_pct
        )

        risk += rng.normal(0, 5)
        risk_score = float(np.clip(risk, 0, 100))
        at_risk = 1 if risk_score >= 55 else 0

        rows.append(
            {
                "lot_id": f"L{lot_id:04d}",
                "store": store,
                "supplier": supplier,
                "category": category,
                "cut": cut,
                "shelf_life_days": shelf_life_days,
                "age_days": age_days,
                "remaining_days": remaining_days,
                "temp_avg_c": temp_avg,
                "temp_max_c": temp_max,
                "hours_out_of_range": hours_out_of_range,
                "inventory_units": inventory_units,
                "daily_sales": daily_sales,
                "price": price,
                "markdown_pct": markdown_pct,
                "historical_shrink_pct": shrink_hist,
                "risk_score": round(risk_score, 2),
                "at_risk": at_risk,
            }
        )

    return pd.DataFrame(rows)


def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = generate_demo_data()

    expected_cols = [
        "lot_id", "store", "supplier", "category", "cut",
        "shelf_life_days", "age_days", "remaining_days",
        "temp_avg_c", "temp_max_c", "hours_out_of_range",
        "inventory_units", "daily_sales", "price",
        "markdown_pct", "historical_shrink_pct"
    ]

    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    # Recalcular métricas base por seguridad
    df = df.copy()
    df["remaining_days"] = df["shelf_life_days"] - df["age_days"]

    risk = (
        18 * (df["age_days"] / df["shelf_life_days"].replace(0, 1))
        + 12 * (df["temp_avg_c"] - 4).clip(lower=0)
        + 6 * (df["temp_max_c"] - 5).clip(lower=0)
        + 4.5 * df["hours_out_of_range"]
        + 0.12 * df["inventory_units"]
        - 1.3 * df["daily_sales"]
        + 0.6 * df["historical_shrink_pct"]
        - 0.35 * df["markdown_pct"]
    )

    if "risk_score" not in df.columns:
        df["risk_score"] = risk.clip(0, 100).round(2)

    if "at_risk" not in df.columns:
        df["at_risk"] = (df["risk_score"] >= 55).astype(int)

    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "temp_avg_c",
        "temp_max_c",
        "hours_out_of_range",
        "inventory_units",
        "daily_sales",
        "historical_shrink_pct",
        "age_days",
        "remaining_days",
    ]
    model = IsolationForest(
        n_estimators=150,
        contamination=0.10,
        random_state=42,
    )
    preds = model.fit_predict(df[numeric_cols])
    scores = model.decision_function(df[numeric_cols])

    out = df.copy()
    out["anomaly_flag"] = (preds == -1).astype(int)
    out["anomaly_score"] = (-scores).round(4)
    return out


def recommend_action(row: pd.Series) -> str:
    if row["remaining_days"] <= 0:
        return "Retirar / revisar inmediatamente"
    if row["temp_max_c"] > 7 or row["hours_out_of_range"] >= 4:
        return "Auditar cadena de frío"
    if row["risk_score"] >= 75 and row["inventory_units"] > row["daily_sales"]:
        return "Markdown fuerte hoy"
    if row["risk_score"] >= 55:
        return "Priorizar venta / FEFO"
    if row["anomaly_flag"] == 1:
        return "Revisar lote atípico"
    return "Operación normal"


def recommended_markdown(row: pd.Series) -> int:
    if row["remaining_days"] <= 0:
        return 0
    if row["risk_score"] >= 85:
        return 30
    if row["risk_score"] >= 75:
        return 25
    if row["risk_score"] >= 65:
        return 20
    if row["risk_score"] >= 55:
        return 10
    return 0


# =========================
# UI
# =========================
st.title("Fresh Meat AI")
st.subheader("MVP para monitoreo de riesgo, merma y cadena de frío en carnes")

with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader("Sube tu CSV", type=["csv"])
    st.caption("Si no subes archivo, se usarán datos demo.")

    risk_threshold = st.slider("Umbral de riesgo", 0, 100, 55, 5)
    anomaly_only = st.checkbox("Mostrar solo anomalías", value=False)

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

df = detect_anomalies(df)
df["at_risk"] = (df["risk_score"] >= risk_threshold).astype(int)
df["recommended_markdown_pct"] = df.apply(recommended_markdown, axis=1)
df["recommended_action"] = df.apply(recommend_action, axis=1)

# =========================
# Filters
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    selected_store = st.selectbox("Tienda", ["Todas"] + sorted(df["store"].unique().tolist()))
with col2:
    selected_category = st.selectbox("Categoría", ["Todas"] + sorted(df["category"].unique().tolist()))
with col3:
    selected_supplier = st.selectbox("Proveedor", ["Todos"] + sorted(df["supplier"].unique().tolist()))

filtered = df.copy()
if selected_store != "Todas":
    filtered = filtered[filtered["store"] == selected_store]
if selected_category != "Todas":
    filtered = filtered[filtered["category"] == selected_category]
if selected_supplier != "Todos":
    filtered = filtered[filtered["supplier"] == selected_supplier]
if anomaly_only:
    filtered = filtered[filtered["anomaly_flag"] == 1]

# =========================
# KPIs
# =========================
total_lots = len(filtered)
risk_lots = int(filtered["at_risk"].sum())
anomaly_lots = int(filtered["anomaly_flag"].sum())
avg_risk = round(float(filtered["risk_score"].mean()), 2) if total_lots else 0
estimated_inventory_value = round(float((filtered["inventory_units"] * filtered["price"]).sum()), 2)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Lotes", f"{total_lots}")
k2.metric("Lotes en riesgo", f"{risk_lots}")
k3.metric("Anomalías", f"{anomaly_lots}")
k4.metric("Riesgo promedio", f"{avg_risk}")
k5.metric("Valor inventario", f"${estimated_inventory_value:,.0f}")

# =========================
# Charts
# =========================
left, right = st.columns(2)

with left:
    st.markdown("### Riesgo promedio por tienda")
    if not filtered.empty:
        chart_df = (
            filtered.groupby("store", as_index=False)["risk_score"]
            .mean()
            .sort_values("risk_score", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(chart_df["store"], chart_df["risk_score"])
        ax.set_ylabel("Risk score promedio")
        ax.set_xlabel("Tienda")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
    else:
        st.info("No hay datos para mostrar.")

with right:
    st.markdown("### Inventario en riesgo por categoría")
    if not filtered.empty:
        cat_df = (
            filtered[filtered["at_risk"] == 1]
            .groupby("category", as_index=False)["inventory_units"]
            .sum()
            .sort_values("inventory_units", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cat_df["category"], cat_df["inventory_units"])
        ax.set_ylabel("Unidades")
        ax.set_xlabel("Categoría")
        st.pyplot(fig)
    else:
        st.info("No hay datos para mostrar.")

# =========================
# Priorities
# =========================
st.markdown("### Lotes prioritarios")
priority_cols = [
    "lot_id",
    "store",
    "supplier",
    "category",
    "cut",
    "age_days",
    "remaining_days",
    "temp_avg_c",
    "temp_max_c",
    "hours_out_of_range",
    "inventory_units",
    "daily_sales",
    "risk_score",
    "anomaly_flag",
    "recommended_markdown_pct",
    "recommended_action",
]

priority_df = filtered.sort_values(
    by=["risk_score", "anomaly_flag", "remaining_days"],
    ascending=[False, False, True]
)[priority_cols]

st.dataframe(priority_df, use_container_width=True)

# =========================
# Top insights
# =========================
st.markdown("### Hallazgos rápidos")

if not filtered.empty:
    top_store = (
        filtered.groupby("store")["risk_score"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    top_supplier = (
        filtered.groupby("supplier")["risk_score"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    hottest_lot = filtered.sort_values("temp_max_c", ascending=False).iloc[0]
    most_critical = filtered.sort_values("risk_score", ascending=False).iloc[0]

    st.write(f"**Tienda con mayor riesgo promedio:** {top_store}")
    st.write(f"**Proveedor con mayor riesgo promedio:** {top_supplier}")
    st.write(
        f"**Lote con mayor temperatura máxima:** {hottest_lot['lot_id']} "
        f"({hottest_lot['temp_max_c']}°C, {hottest_lot['store']})"
    )
    st.write(
        f"**Lote más crítico:** {most_critical['lot_id']} | "
        f"Riesgo {most_critical['risk_score']} | "
        f"Acción: {most_critical['recommended_action']}"
    )
else:
    st.info("No hay datos tras aplicar filtros.")

# =========================
# CSV format example
# =========================
with st.expander("Ver formato esperado del CSV"):
    sample = pd.DataFrame(
        [
            {
                "lot_id": "L0001",
                "store": "MTY San Pedro",
                "supplier": "Proveedor A",
                "category": "Res",
                "cut": "Ribeye",
                "shelf_life_days": 10,
                "age_days": 6,
                "remaining_days": 4,
                "temp_avg_c": 4.5,
                "temp_max_c": 6.8,
                "hours_out_of_range": 2.5,
                "inventory_units": 24,
                "daily_sales": 6.0,
                "price": 289.0,
                "markdown_pct": 10,
                "historical_shrink_pct": 8.5,
            }
        ]
    )
    st.dataframe(sample, use_container_width=True)