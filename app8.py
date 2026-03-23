import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Meat Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DATA GENERATION
# =========================================================
@st.cache_data
def generate_operational_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    stores = [
        "MTY San Pedro",
        "MTY Cumbres",
        "MTY Contry",
        "Saltillo Centro",
        "Apodaca",
    ]

    suppliers = [
        "Proveedor A",
        "Proveedor B",
        "Proveedor C",
        "Proveedor D",
    ]

    categories = ["Res", "Pollo", "Cerdo"]

    cuts = {
        "Res": ["Ribeye", "Diezmillo", "Milanesa", "Sirloin", "Arrachera"],
        "Pollo": ["Pechuga", "Pierna", "Muslo", "Alitas", "Milanesa Pollo"],
        "Cerdo": ["Chuleta", "Lomo", "Costilla", "Pierna", "Espinazo"],
    }

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
        remaining_days = shelf_life_days - age_days

        temp_avg_c = float(np.round(rng.normal(3.5, 1.3), 2))
        temp_max_c = float(np.round(temp_avg_c + abs(rng.normal(1.4, 0.9)), 2))
        hours_out_of_range = float(np.round(max(0, rng.normal(1.6, 1.8)), 2))

        inventory_units = int(rng.integers(5, 100))
        daily_sales = float(np.round(max(0.1, rng.normal(9, 4)), 2))
        price = float(np.round(rng.uniform(70, 380), 2))
        markdown_pct = int(rng.choice([0, 0, 0, 10, 15, 20, 25]))
        historical_shrink_pct = float(np.round(np.clip(rng.normal(7, 4), 0, 25), 2))

        purchase_qty = int(rng.integers(20, 150))
        received_qty = max(0, purchase_qty + int(rng.integers(-5, 6)))
        ordered_weight_kg = float(np.round(rng.uniform(50, 500), 2))
        received_weight_kg = float(np.round(ordered_weight_kg + rng.normal(0, 8), 2))

        theoretical_yield_pct = float(np.round(rng.uniform(65, 92), 2))
        actual_yield_pct = float(np.round(theoretical_yield_pct + rng.normal(0, 4), 2))

        production_plan_kg = float(np.round(rng.uniform(20, 250), 2))
        production_actual_kg = float(np.round(production_plan_kg + rng.normal(0, 18), 2))

        shelf_gaps = int(rng.integers(0, 8))
        expired_labels = int(rng.integers(0, 4))
        overfill_flag = int(rng.choice([0, 1], p=[0.8, 0.2]))

        audit_score = float(np.round(np.clip(rng.normal(84, 10), 40, 100), 2))

        risk = (
            18 * (age_days / max(shelf_life_days, 1))
            + 12 * max(temp_avg_c - 4, 0)
            + 6 * max(temp_max_c - 5, 0)
            + 4.5 * hours_out_of_range
            + 0.12 * inventory_units
            - 1.3 * daily_sales
            + 0.6 * historical_shrink_pct
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
                "temp_avg_c": temp_avg_c,
                "temp_max_c": temp_max_c,
                "hours_out_of_range": hours_out_of_range,
                "inventory_units": inventory_units,
                "daily_sales": daily_sales,
                "price": price,
                "markdown_pct": markdown_pct,
                "historical_shrink_pct": historical_shrink_pct,
                "purchase_qty": purchase_qty,
                "received_qty": received_qty,
                "ordered_weight_kg": ordered_weight_kg,
                "received_weight_kg": received_weight_kg,
                "theoretical_yield_pct": theoretical_yield_pct,
                "actual_yield_pct": actual_yield_pct,
                "production_plan_kg": production_plan_kg,
                "production_actual_kg": production_actual_kg,
                "shelf_gaps": shelf_gaps,
                "expired_labels": expired_labels,
                "overfill_flag": overfill_flag,
                "audit_score": audit_score,
                "risk_score": round(risk_score, 2),
                "at_risk": at_risk,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# CALCULATIONS
# =========================================================
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
        "actual_yield_pct",
        "audit_score",
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


def recommended_action(row: pd.Series) -> str:
    if row["remaining_days"] <= 0:
        return "Retirar / revisar inmediatamente"
    if row["temp_max_c"] > 7 or row["hours_out_of_range"] >= 4:
        return "Auditar cadena de frío"
    if row["actual_yield_pct"] < row["theoretical_yield_pct"] - 5:
        return "Revisar rendimiento y merma"
    if row["risk_score"] >= 75 and row["inventory_units"] > row["daily_sales"]:
        return "Markdown fuerte hoy"
    if row["risk_score"] >= 55:
        return "Priorizar venta / FEFO"
    if row.get("anomaly_flag", 0) == 1:
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


def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        df = generate_operational_data()

    required_cols = [
        "lot_id", "store", "supplier", "category", "cut",
        "shelf_life_days", "age_days", "remaining_days",
        "temp_avg_c", "temp_max_c", "hours_out_of_range",
        "inventory_units", "daily_sales", "price",
        "markdown_pct", "historical_shrink_pct",
        "purchase_qty", "received_qty",
        "ordered_weight_kg", "received_weight_kg",
        "theoretical_yield_pct", "actual_yield_pct",
        "production_plan_kg", "production_actual_kg",
        "shelf_gaps", "expired_labels", "overfill_flag", "audit_score"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df.copy()
    df["remaining_days"] = df["shelf_life_days"] - df["age_days"]

    if "risk_score" not in df.columns:
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
        df["risk_score"] = risk.clip(0, 100).round(2)

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    c1, c2, c3 = st.columns(3)

    with c1:
        selected_store = st.selectbox("Tienda", ["Todas"] + sorted(df["store"].unique().tolist()))
    with c2:
        selected_category = st.selectbox("Categoría", ["Todas"] + sorted(df["category"].unique().tolist()))
    with c3:
        selected_supplier = st.selectbox("Proveedor", ["Todos"] + sorted(df["supplier"].unique().tolist()))

    filtered = df.copy()

    if selected_store != "Todas":
        filtered = filtered[filtered["store"] == selected_store]
    if selected_category != "Todas":
        filtered = filtered[filtered["category"] == selected_category]
    if selected_supplier != "Todos":
        filtered = filtered[filtered["supplier"] == selected_supplier]

    return filtered


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Meat Intelligence System™")
st.sidebar.markdown("Plataforma integral para la industria cárnica")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo CSV o Excel",
    type=["csv", "xlsx"]
)

module = st.sidebar.radio(
    "Selecciona módulo",
    [
        "Inicio",
        "Dashboard Operativo",
        "Compras y Demanda",
        "Recibo y Almacén",
        "Producción y Rendimientos",
        "Exhibición y Venta",
        "Auditoría",
        "Dirección y Copiloto IA",
    ]
)

risk_threshold = st.sidebar.slider("Umbral de riesgo", 0, 100, 55, 5)

# =========================================================
# DATA PREP
# =========================================================
try:
    df = load_data(uploaded_file)
    df = detect_anomalies(df)
    df["at_risk"] = (df["risk_score"] >= risk_threshold).astype(int)
    df["recommended_markdown_pct"] = df.apply(recommended_markdown, axis=1)
    df["recommended_action"] = df.apply(recommended_action, axis=1)
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()


# =========================================================
# PAGES
# =========================================================
if module == "Inicio":
    st.title("Meat Intelligence System™")
    st.subheader("Plataforma de inteligencia operativa para la industria cárnica")

    st.markdown("""
    Esta solución integra módulos clave para la gestión del negocio cárnico:

    - Dashboard Operativo
    - Compras y Demanda
    - Recibo y Almacén
    - Producción y Rendimientos
    - Exhibición y Venta
    - Auditoría
    - Dirección y Copiloto IA
    """)

    st.info("Usa el menú lateral para navegar entre módulos.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lotes totales", len(df))
    k2.metric("Riesgo promedio", round(float(df["risk_score"].mean()), 2))
    k3.metric("Merma histórica prom.", round(float(df["historical_shrink_pct"].mean()), 2))
    k4.metric("Audit score prom.", round(float(df["audit_score"].mean()), 2))

    st.markdown("### Formato esperado del archivo")
    st.dataframe(df.head(10), use_container_width=True)

elif module == "Dashboard Operativo":
    st.title("Dashboard Operativo")
    filtered = apply_filters(df)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Lotes", len(filtered))
    k2.metric("Lotes en riesgo", int(filtered["at_risk"].sum()))
    k3.metric("Anomalías", int(filtered["anomaly_flag"].sum()))
    k4.metric("Riesgo promedio", round(float(filtered["risk_score"].mean()), 2))
    k5.metric("Valor inventario", f"${(filtered['inventory_units'] * filtered['price']).sum():,.0f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Riesgo promedio por tienda")
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

    with right:
        st.subheader("Inventario en riesgo por categoría")
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

    st.subheader("Lotes prioritarios")
    priority_cols = [
        "lot_id", "store", "supplier", "category", "cut",
        "risk_score", "temp_max_c", "remaining_days",
        "inventory_units", "anomaly_flag",
        "recommended_markdown_pct", "recommended_action"
    ]
    priority_df = filtered.sort_values(
        by=["risk_score", "anomaly_flag", "remaining_days"],
        ascending=[False, False, True],
    )[priority_cols]

    st.dataframe(priority_df, use_container_width=True)

elif module == "Compras y Demanda":
    st.title("Compras y Demanda")
    filtered = apply_filters(df)

    filtered["suggested_purchase_qty"] = (
        filtered["daily_sales"] * 3 - filtered["inventory_units"] * 0.25
    ).clip(lower=0).round(0)

    filtered["overinventory_risk"] = (
        (filtered["inventory_units"] > filtered["daily_sales"] * 5).astype(int)
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Demanda diaria promedio", round(float(filtered["daily_sales"].mean()), 2))
    k2.metric("Inventario promedio", round(float(filtered["inventory_units"].mean()), 2))
    k3.metric("Casos de sobreinventario", int(filtered["overinventory_risk"].sum()))

    st.subheader("Sugerencia de compra")
    cols = [
        "lot_id", "store", "supplier", "category", "cut",
        "inventory_units", "daily_sales", "suggested_purchase_qty", "overinventory_risk"
    ]
    st.dataframe(filtered[cols], use_container_width=True)

    summary = (
        filtered.groupby(["store", "category"], as_index=False)
        .agg(
            avg_daily_sales=("daily_sales", "mean"),
            avg_inventory=("inventory_units", "mean"),
            suggested_purchase_qty=("suggested_purchase_qty", "mean"),
        )
    )
    st.subheader("Resumen por tienda y categoría")
    st.dataframe(summary, use_container_width=True)

elif module == "Recibo y Almacén":
    st.title("Recibo y Almacén")
    filtered = apply_filters(df)

    filtered["qty_diff"] = filtered["received_qty"] - filtered["purchase_qty"]
    filtered["weight_diff_kg"] = (
        filtered["received_weight_kg"] - filtered["ordered_weight_kg"]
    ).round(2)
    filtered["temp_rejection_flag"] = (filtered["temp_max_c"] > 7).astype(int)

    k1, k2, k3 = st.columns(3)
    k1.metric("OC con diferencia en cantidad", int((filtered["qty_diff"] != 0).sum()))
    k2.metric("Recepciones con riesgo térmico", int(filtered["temp_rejection_flag"].sum()))
    k3.metric("Desviación promedio peso (kg)", round(float(filtered["weight_diff_kg"].mean()), 2))

    st.subheader("Validación de recibo")
    cols = [
        "lot_id", "store", "supplier", "purchase_qty", "received_qty",
        "qty_diff", "ordered_weight_kg", "received_weight_kg",
        "weight_diff_kg", "temp_avg_c", "temp_max_c", "temp_rejection_flag"
    ]
    st.dataframe(filtered[cols], use_container_width=True)

elif module == "Producción y Rendimientos":
    st.title("Producción y Rendimientos")
    filtered = apply_filters(df)

    filtered["yield_gap"] = (
        filtered["actual_yield_pct"] - filtered["theoretical_yield_pct"]
    ).round(2)

    filtered["production_gap_kg"] = (
        filtered["production_actual_kg"] - filtered["production_plan_kg"]
    ).round(2)

    filtered["low_yield_flag"] = (filtered["yield_gap"] < -5).astype(int)

    k1, k2, k3 = st.columns(3)
    k1.metric("Rendimiento teórico promedio", round(float(filtered["theoretical_yield_pct"].mean()), 2))
    k2.metric("Rendimiento real promedio", round(float(filtered["actual_yield_pct"].mean()), 2))
    k3.metric("Casos bajo rendimiento", int(filtered["low_yield_flag"].sum()))

    st.subheader("Detalle de rendimientos")
    cols = [
        "lot_id", "store", "supplier", "category", "cut",
        "theoretical_yield_pct", "actual_yield_pct", "yield_gap",
        "production_plan_kg", "production_actual_kg", "production_gap_kg", "low_yield_flag"
    ]
    st.dataframe(filtered[cols], use_container_width=True)

elif module == "Exhibición y Venta":
    st.title("Exhibición y Venta")
    filtered = apply_filters(df)

    filtered["shelf_attention_flag"] = (
        (filtered["shelf_gaps"] > 2)
        | (filtered["expired_labels"] > 0)
        | (filtered["overfill_flag"] == 1)
    ).astype(int)

    k1, k2, k3 = st.columns(3)
    k1.metric("Huecos totales", int(filtered["shelf_gaps"].sum()))
    k2.metric("Etiquetas vencidas", int(filtered["expired_labels"].sum()))
    k3.metric("Casos de sobrellenado", int(filtered["overfill_flag"].sum()))

    st.subheader("Control de exhibición")
    cols = [
        "lot_id", "store", "category", "cut", "inventory_units",
        "daily_sales", "shelf_gaps", "expired_labels",
        "overfill_flag", "shelf_attention_flag"
    ]
    st.dataframe(filtered[cols], use_container_width=True)

elif module == "Auditoría":
    st.title("Auditoría")
    filtered = apply_filters(df)

    filtered["audit_status"] = filtered["audit_score"].apply(
        lambda x: "Crítico" if x < 70 else "Atención" if x < 85 else "Aceptable"
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Audit score promedio", round(float(filtered["audit_score"].mean()), 2))
    k2.metric("Tiendas en atención", int((filtered["audit_score"] < 85).sum()))
    k3.metric("Casos críticos", int((filtered["audit_score"] < 70).sum()))

    st.subheader("Resultados de auditoría")
    cols = [
        "lot_id", "store", "supplier", "category", "cut",
        "audit_score", "audit_status"
    ]
    st.dataframe(filtered[cols], use_container_width=True)

elif module == "Dirección y Copiloto IA":
    st.title("Dirección y Copiloto IA")

    st.subheader("Resumen ejecutivo")
    summary = {
        "Lotes totales": len(df),
        "Riesgo promedio": round(float(df["risk_score"].mean()), 2),
        "Merma histórica promedio": round(float(df["historical_shrink_pct"].mean()), 2),
        "Rendimiento real promedio": round(float(df["actual_yield_pct"].mean()), 2),
        "Audit score promedio": round(float(df["audit_score"].mean()), 2),
    }

    for key, value in summary.items():
        st.write(f"**{key}:** {value}")

    st.subheader("Preguntas sugeridas al copiloto")
    questions = [
        "¿Qué tiendas están destruyendo margen?",
        "¿Qué proveedor tiene peor rendimiento real vs esperado?",
        "¿Qué categorías tienen mayor riesgo de merma?",
        "¿Dónde hay más oportunidades de mejora operativa?",
    ]
    for q in questions:
        st.write(f"- {q}")

    st.subheader("Simulación simple de insights")

    worst_store = (
        df.groupby("store")["risk_score"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    worst_supplier = (
        (df["theoretical_yield_pct"] - df["actual_yield_pct"])
        .groupby(df["supplier"])
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    worst_category = (
        df.groupby("category")["historical_shrink_pct"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    st.warning(f"La tienda con mayor riesgo promedio es: {worst_store}")
    st.warning(f"El proveedor con mayor gap de rendimiento es: {worst_supplier}")
    st.warning(f"La categoría con mayor merma histórica promedio es: {worst_category}")

    question = st.text_input("Escribe una pregunta ejecutiva")

    if question:
        q = question.lower()

        if "tienda" in q and "riesgo" in q:
            st.success(f"La tienda con mayor riesgo promedio es {worst_store}.")
        elif "proveedor" in q and "rendimiento" in q:
            st.success(f"El proveedor con peor diferencia entre rendimiento teórico y real es {worst_supplier}.")
        elif "merma" in q or "categoría" in q:
            st.success(f"La categoría con mayor merma histórica promedio es {worst_category}.")
        else:
            st.info("Versión base del copiloto: responde con reglas simples. Después se puede conectar a un LLM.")
