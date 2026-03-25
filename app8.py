import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Import seguro del SDK de OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(
    page_title="Meat Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# OPENAI / LLM
# =========================================================
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_openai_client():
    if OpenAI is None:
        return None, "La librería 'openai' no está instalada."

    api_key = get_api_key()
    if not api_key:
        return None, (
            "No encontré la API key. Agrega OPENAI_API_KEY en "
            ".streamlit/secrets.toml, en los Secrets de Streamlit Cloud "
            "o como variable de entorno."
        )

    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"No se pudo inicializar el cliente OpenAI: {e}"


def llm_status_text():
    client, err = get_openai_client()
    if client is not None:
        return "LLM listo"
    return f"LLM no disponible: {err}"


def build_llm_context(df: pd.DataFrame, filtered_df: pd.DataFrame | None = None) -> str:
    base_df = filtered_df if filtered_df is not None and len(filtered_df) > 0 else df

    total_lots = len(base_df)
    avg_risk = float(base_df["risk_score"].mean()) if "risk_score" in base_df.columns and len(base_df) else 0
    avg_pred_prob = float(base_df["predicted_shrink_prob"].mean()) if "predicted_shrink_prob" in base_df.columns and len(base_df) else 0
    avg_shrink = float(base_df["historical_shrink_pct"].mean()) if "historical_shrink_pct" in base_df.columns and len(base_df) else 0
    avg_yield = float(base_df["actual_yield_pct"].mean()) if "actual_yield_pct" in base_df.columns and len(base_df) else 0
    avg_audit = float(base_df["audit_score"].mean()) if "audit_score" in base_df.columns and len(base_df) else 0
    avg_margin = float(base_df["gross_margin_pct"].mean()) if "gross_margin_pct" in base_df.columns and len(base_df) else 0

    worst_store = "N/A"
    if {"store", "risk_score"}.issubset(base_df.columns) and len(base_df):
        worst_store = (
            base_df.groupby("store")["risk_score"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )

    worst_supplier = "N/A"
    if {"supplier", "theoretical_yield_pct", "actual_yield_pct"}.issubset(base_df.columns) and len(base_df):
        gap = (
            (base_df["theoretical_yield_pct"] - base_df["actual_yield_pct"])
            .groupby(base_df["supplier"])
            .mean()
            .sort_values(ascending=False)
        )
        if len(gap):
            worst_supplier = gap.index[0]

    worst_category = "N/A"
    if {"category", "historical_shrink_pct"}.issubset(base_df.columns) and len(base_df):
        worst_category = (
            base_df.groupby("category")["historical_shrink_pct"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )

    top_actions = ""
    if "recommended_action" in base_df.columns and len(base_df):
        s = base_df["recommended_action"].value_counts().head(10)
        top_actions = "\n".join([f"- {k}: {v}" for k, v in s.items()])

    top_predicted = ""
    if "predicted_shrink_prob" in base_df.columns and len(base_df):
        top_df = base_df.sort_values("predicted_shrink_prob", ascending=False).head(5)
        lines = []
        for _, row in top_df.iterrows():
            lines.append(
                f"- {row.get('lot_id','N/A')} | tienda {row.get('store','N/A')} | "
                f"cat {row.get('category','N/A')} | prob {row.get('predicted_shrink_prob',0):.2f}"
            )
        top_predicted = "\n".join(lines)

    context = f"""
Eres un copiloto ejecutivo para una plataforma de inteligencia operativa de carnes en autoservicio.
Responde únicamente con base en el contexto y los datos proporcionados.
Si no hay evidencia suficiente, dilo claramente.
No inventes KPIs ni cifras.
Responde en español, con tono ejecutivo y claro.

CONTEXTO GENERAL:
- Lotes analizados: {total_lots}
- Riesgo promedio: {avg_risk:.2f}
- Probabilidad predictiva promedio de merma/riesgo: {avg_pred_prob:.2f}
- Merma histórica promedio: {avg_shrink:.2f}
- Rendimiento real promedio: {avg_yield:.2f}
- Audit score promedio: {avg_audit:.2f}
- Margen bruto promedio: {avg_margin:.2f}

HALLAZGOS:
- Tienda con mayor riesgo promedio: {worst_store}
- Proveedor con mayor gap de rendimiento: {worst_supplier}
- Categoría con mayor merma histórica: {worst_category}

TOP LOTES PREDICTIVOS:
{top_predicted}

ACCIONES RECOMENDADAS MÁS FRECUENTES:
{top_actions}

DEFINICIONES:
- risk_score: indicador actual de riesgo operativo del lote
- predicted_shrink_prob: probabilidad predictiva de riesgo/merma
- predicted_shrink_flag: bandera predictiva de riesgo
- historical_shrink_pct: merma histórica porcentual
- actual_yield_pct vs theoretical_yield_pct: rendimiento real vs esperado
- audit_score: cumplimiento operativo
- markdown_pct: descuento aplicado
- recommended_action: acción sugerida por reglas
"""
    return context.strip()


def ask_llm(question: str, context: str) -> str:
    client, err = get_openai_client()
    if client is None:
        return err

    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {
                    "role": "system",
                    "content": (
                        "Eres un copiloto ejecutivo de Meat Intelligence System. "
                        "Responde solo con base en el contexto. "
                        "Sé claro, ejecutivo y preciso."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPregunta:\n{question}",
                },
            ],
        )
        return response.output_text
    except Exception as e:
        return f"Error al consultar el LLM: {e}"


# =========================================================
# DATA GENERATION
# =========================================================
@st.cache_data
def generate_operational_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    stores = ["MTY San Pedro", "MTY Cumbres", "MTY Contry", "Saltillo Centro", "Apodaca"]
    regions = {
        "MTY San Pedro": "Monterrey",
        "MTY Cumbres": "Monterrey",
        "MTY Contry": "Monterrey",
        "Saltillo Centro": "Saltillo",
        "Apodaca": "Monterrey",
    }
    clusters = {
        "MTY San Pedro": "Premium",
        "MTY Cumbres": "Masivo",
        "MTY Contry": "Masivo",
        "Saltillo Centro": "Regional",
        "Apodaca": "Masivo",
    }

    suppliers = ["Proveedor A", "Proveedor B", "Proveedor C", "Proveedor D"]
    categories = ["Res", "Pollo", "Cerdo"]

    cuts = {
        "Res": ["Ribeye", "Diezmillo", "Milanesa", "Sirloin", "Arrachera"],
        "Pollo": ["Pechuga", "Pierna", "Muslo", "Alitas", "Milanesa Pollo"],
        "Cerdo": ["Chuleta", "Lomo", "Costilla", "Pierna", "Espinazo"],
    }

    rows = []
    start_date = pd.Timestamp("2026-01-01")

    for lot_id in range(1, n + 1):
        category = rng.choice(categories, p=[0.35, 0.40, 0.25])
        cut = rng.choice(cuts[category])
        store = rng.choice(stores)
        region = regions[store]
        cluster = clusters[store]
        supplier = rng.choice(suppliers)

        shelf_life_days = {
            "Res": int(rng.integers(6, 12)),
            "Pollo": int(rng.integers(4, 8)),
            "Cerdo": int(rng.integers(5, 9)),
        }[category]

        age_days = int(rng.integers(0, shelf_life_days + 2))
        remaining_days = shelf_life_days - age_days

        recv_date = start_date + pd.Timedelta(days=int(rng.integers(0, 75)))
        production_date = recv_date + pd.Timedelta(days=int(rng.integers(0, 2)))
        expiry_date = recv_date + pd.Timedelta(days=shelf_life_days)
        snapshot_date = recv_date + pd.Timedelta(days=age_days)

        temp_avg_c = float(np.round(rng.normal(3.5, 1.3), 2))
        temp_max_c = float(np.round(temp_avg_c + abs(rng.normal(1.4, 0.9)), 2))
        hours_out_of_range = float(np.round(max(0, rng.normal(1.6, 1.8)), 2))

        inventory_units = int(rng.integers(5, 100))
        daily_sales_units = float(np.round(max(0.1, rng.normal(9, 4)), 2))
        price = float(np.round(rng.uniform(70, 380), 2))
        unit_cost = float(np.round(price * rng.uniform(0.62, 0.82), 2))
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
        hygiene_score = float(np.round(np.clip(rng.normal(88, 8), 50, 100), 2))
        audit_score = float(np.round(np.clip(rng.normal(84, 10), 40, 100), 2))

        gross_margin_pct = float(np.round(((price - unit_cost) / price) * 100, 2))
        theoretical_margin_pct = float(np.round(gross_margin_pct + rng.normal(0, 2), 2))

        risk = (
            18 * (age_days / max(shelf_life_days, 1))
            + 12 * max(temp_avg_c - 4, 0)
            + 6 * max(temp_max_c - 5, 0)
            + 4.5 * hours_out_of_range
            + 0.12 * inventory_units
            - 1.3 * daily_sales_units
            + 0.6 * historical_shrink_pct
            - 0.35 * markdown_pct
            + 0.25 * max(0, theoretical_yield_pct - actual_yield_pct)
        )

        risk += rng.normal(0, 5)
        risk_score = float(np.clip(risk, 0, 100))
        at_risk = 1 if risk_score >= 55 else 0

        rows.append(
            {
                "snapshot_date": snapshot_date.date(),
                "recv_date": recv_date.date(),
                "production_date": production_date.date(),
                "expiry_date": expiry_date.date(),
                "lot_id": f"L{lot_id:05d}",
                "po_number": f"PO-{100000 + lot_id}",
                "store": store,
                "region": region,
                "cluster": cluster,
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
                "daily_sales_units": daily_sales_units,
                "daily_sales": daily_sales_units,
                "price": price,
                "unit_cost": unit_cost,
                "gross_margin_pct": gross_margin_pct,
                "theoretical_margin_pct": theoretical_margin_pct,
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
                "hygiene_score": hygiene_score,
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


# =========================================================
# NORMALIZATION
# =========================================================
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "daily_sales_units": "daily_sales",
        "unit_price": "price",
        "sale_price": "price",
        "precio": "price",
        "purchase_units": "purchase_qty",
        "ordered_units": "purchase_qty",
        "qty_ordered": "purchase_qty",
        "cantidad_pedida": "purchase_qty",
        "received_units": "received_qty",
        "qty_received": "received_qty",
        "cantidad_recibida": "received_qty",
    }

    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    df = df.rename(columns=existing_renames)
    return df


def fill_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "daily_sales" not in df.columns and "daily_sales_units" in df.columns:
        df["daily_sales"] = df["daily_sales_units"]

    if "daily_sales_units" not in df.columns and "daily_sales" in df.columns:
        df["daily_sales_units"] = df["daily_sales"]

    if "price" not in df.columns:
        df["price"] = 150.0

    if "purchase_qty" not in df.columns:
        if "inventory_units" in df.columns:
            df["purchase_qty"] = pd.to_numeric(df["inventory_units"], errors="coerce").fillna(30) * 1.2
            df["purchase_qty"] = df["purchase_qty"].round().astype(int)
        else:
            df["purchase_qty"] = 50

    if "received_qty" not in df.columns:
        df["received_qty"] = df["purchase_qty"]

    if "unit_cost" not in df.columns and "price" in df.columns:
        df["unit_cost"] = (pd.to_numeric(df["price"], errors="coerce").fillna(150) * 0.72).round(2)

    if "gross_margin_pct" not in df.columns and {"price", "unit_cost"}.issubset(df.columns):
        safe_price = pd.to_numeric(df["price"], errors="coerce").replace(0, np.nan).fillna(150)
        safe_cost = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(108)
        df["gross_margin_pct"] = (((safe_price - safe_cost) / safe_price) * 100).round(2)

    if "theoretical_margin_pct" not in df.columns and "gross_margin_pct" in df.columns:
        df["theoretical_margin_pct"] = df["gross_margin_pct"]

    if "region" not in df.columns:
        df["region"] = "N/A"

    if "cluster" not in df.columns:
        df["cluster"] = "N/A"

    if "po_number" not in df.columns:
        df["po_number"] = [f"PO-{100000+i}" for i in range(len(df))]

    if "hygiene_score" not in df.columns:
        df["hygiene_score"] = 85.0

    return df


def ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    optional_defaults = {
        "production_date": pd.NaT,
        "recv_date": pd.NaT,
        "snapshot_date": pd.NaT,
        "expiry_date": pd.NaT,
        "po_number": "N/A",
        "region": "N/A",
        "cluster": "N/A",
        "hygiene_score": 85.0,
        "gross_margin_pct": 25.0,
        "theoretical_margin_pct": 25.0,
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = fill_missing_columns(df)

    for col in ["recv_date", "production_date", "expiry_date", "snapshot_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_defaults = {
        "shelf_life_days": 7,
        "age_days": 2,
        "temp_avg_c": 3.5,
        "temp_max_c": 5.0,
        "hours_out_of_range": 1.0,
        "inventory_units": 30,
        "daily_sales": 8.0,
        "markdown_pct": 0,
        "historical_shrink_pct": 7.0,
        "ordered_weight_kg": 100.0,
        "received_weight_kg": 100.0,
        "theoretical_yield_pct": 80.0,
        "actual_yield_pct": 78.0,
        "production_plan_kg": 50.0,
        "production_actual_kg": 48.0,
        "shelf_gaps": 1,
        "expired_labels": 0,
        "overfill_flag": 0,
        "audit_score": 85.0,
        "price": 150.0,
        "purchase_qty": 50,
        "received_qty": 50,
    }

    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    if "remaining_days" not in df.columns:
        df["remaining_days"] = df["shelf_life_days"] - df["age_days"]
    else:
        df["remaining_days"] = pd.to_numeric(df["remaining_days"], errors="coerce").fillna(
            df["shelf_life_days"] - df["age_days"]
        )

    return df


def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        df = generate_operational_data()

    df = normalize_columns(df)
    df = ensure_optional_columns(df)

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
            + 0.25 * (df["theoretical_yield_pct"] - df["actual_yield_pct"]).clip(lower=0)
        )
        df["risk_score"] = risk.clip(0, 100).round(2)

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filter_cols = st.columns(4)

    with filter_cols[0]:
        selected_store = st.selectbox("Tienda", ["Todas"] + sorted(df["store"].dropna().unique().tolist()))
    with filter_cols[1]:
        selected_category = st.selectbox("Categoría", ["Todas"] + sorted(df["category"].dropna().unique().tolist()))
    with filter_cols[2]:
        selected_supplier = st.selectbox("Proveedor", ["Todos"] + sorted(df["supplier"].dropna().unique().tolist()))
    with filter_cols[3]:
        selected_region = st.selectbox("Región", ["Todas"] + sorted(df["region"].dropna().unique().tolist()))

    filtered = df.copy()

    if selected_store != "Todas":
        filtered = filtered[filtered["store"] == selected_store]
    if selected_category != "Todas":
        filtered = filtered[filtered["category"] == selected_category]
    if selected_supplier != "Todos":
        filtered = filtered[filtered["supplier"] == selected_supplier]
    if selected_region != "Todas":
        filtered = filtered[filtered["region"] == selected_region]

    return filtered


def safe_show_dataframe(df: pd.DataFrame, cols: list[str]):
    available_cols = [c for c in cols if c in df.columns]
    if not available_cols:
        st.warning("No hay columnas disponibles para mostrar en esta vista.")
        return
    st.dataframe(df[available_cols], use_container_width=True)


# =========================================================
# PREDICTIVE MODEL
# =========================================================
PREDICTIVE_FEATURES = [
    "age_days",
    "remaining_days",
    "temp_avg_c",
    "temp_max_c",
    "hours_out_of_range",
    "inventory_units",
    "daily_sales",
    "markdown_pct",
    "historical_shrink_pct",
    "theoretical_yield_pct",
    "actual_yield_pct",
    "audit_score",
    "price",
    "purchase_qty",
    "received_qty",
    "shelf_gaps",
    "expired_labels",
    "overfill_flag",
]

@st.cache_resource
def train_predictive_model(model_df_hashable: tuple):
    # placeholder for cache signature only
    return None

def build_predictive_target(df: pd.DataFrame) -> pd.Series:
    # Target sintético/operativo: combina riesgo actual y señales de merma
    target = (
        (df["risk_score"] >= 55) |
        (df["historical_shrink_pct"] >= 10) |
        (df["remaining_days"] <= 1) |
        (df["hours_out_of_range"] >= 4) |
        ((df["theoretical_yield_pct"] - df["actual_yield_pct"]) >= 5)
    ).astype(int)
    return target


def fit_predictive_model(df: pd.DataFrame):
    model_df = df.copy()

    for col in PREDICTIVE_FEATURES:
        if col not in model_df.columns:
            model_df[col] = 0

    X = model_df[PREDICTIVE_FEATURES].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    y = build_predictive_target(model_df)

    # evitar error si solo hay una clase
    if y.nunique() < 2 or len(X) < 20:
        model_df["predicted_shrink_prob"] = model_df["risk_score"] / 100.0
        model_df["predicted_shrink_flag"] = (model_df["predicted_shrink_prob"] >= 0.55).astype(int)

        metrics = {
            "model_type": "fallback_rules",
            "accuracy": None,
            "roc_auc": None,
            "report": "No hubo suficiente diversidad de clases para entrenar el modelo. Se usó fallback basado en risk_score."
        }
        return model_df, None, metrics

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    model_df["predicted_shrink_prob"] = model.predict_proba(X)[:, 1]
    model_df["predicted_shrink_flag"] = (model_df["predicted_shrink_prob"] >= 0.55).astype(int)

    try:
        roc_auc = roc_auc_score(y_test, test_probs)
    except Exception:
        roc_auc = None

    report = classification_report(y_test, test_preds, zero_division=0)
    acc = accuracy_score(y_test, test_preds)

    metrics = {
        "model_type": "RandomForestClassifier",
        "accuracy": acc,
        "roc_auc": roc_auc,
        "report": report
    }

    return model_df, model, metrics


def get_feature_importance(model, feature_names):
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return imp


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Meat Intelligence System™")
st.sidebar.markdown("Plataforma integral para la industria cárnica")
st.sidebar.caption(llm_status_text())

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
        "Predictivo",
        "Dirección y Copiloto IA",
    ]
)

risk_threshold = st.sidebar.slider("Umbral de riesgo", 0, 100, 55, 5)
show_raw_data = st.sidebar.checkbox("Mostrar muestra de datos", value=False)

# =========================================================
# LOAD + MODEL
# =========================================================
try:
    df = load_data(uploaded_file)
    df = detect_anomalies(df)
    df["at_risk"] = (df["risk_score"] >= risk_threshold).astype(int)
    df["recommended_markdown_pct"] = df.apply(recommended_markdown, axis=1)
    df["recommended_action"] = df.apply(recommended_action, axis=1)

    df, predictive_model, predictive_metrics = fit_predictive_model(df)
    feature_importance_df = get_feature_importance(predictive_model, PREDICTIVE_FEATURES)

except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

if show_raw_data:
    st.subheader("Muestra de datos cargados")
    st.dataframe(df.head(20), use_container_width=True)

# =========================================================
# PAGES
# =========================================================
if module == "Inicio":
    st.title("Meat Intelligence System™")
    st.subheader("Plataforma de inteligencia operativa para la industria cárnica")
    st.info("Usa el menú lateral para navegar entre módulos.")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Lotes totales", len(df))
    k2.metric("Riesgo promedio", round(float(df["risk_score"].mean()), 2))
    k3.metric("Prob. predictiva prom.", round(float(df["predicted_shrink_prob"].mean()), 2))
    k4.metric("Merma histórica prom.", round(float(df["historical_shrink_pct"].mean()), 2))
    k5.metric("Audit score prom.", round(float(df["audit_score"].mean()), 2))
    k6.metric("Rendimiento real prom.", round(float(df["actual_yield_pct"].mean()), 2))

    st.dataframe(df.head(10), use_container_width=True)

elif module == "Dashboard Operativo":
    st.title("Dashboard Operativo")
    filtered = apply_filters(df)

    inventory_value = float((filtered["inventory_units"] * filtered["price"]).sum()) if len(filtered) else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Lotes", len(filtered))
    k2.metric("Lotes en riesgo", int(filtered["at_risk"].sum()) if len(filtered) else 0)
    k3.metric("Anomalías", int(filtered["anomaly_flag"].sum()) if len(filtered) else 0)
    k4.metric("Riesgo promedio", round(float(filtered["risk_score"].mean()), 2) if len(filtered) else 0)
    k5.metric("Valor inventario", f"${inventory_value:,.0f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Riesgo promedio por tienda")
        if len(filtered):
            chart_df = filtered.groupby("store", as_index=False)["risk_score"].mean().sort_values("risk_score", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(chart_df["store"], chart_df["risk_score"])
            ax.set_ylabel("Risk score promedio")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
        else:
            st.info("No hay datos para mostrar.")

    with right:
        st.subheader("Inventario en riesgo por categoría")
        if len(filtered):
            cat_df = filtered[filtered["at_risk"] == 1].groupby("category", as_index=False)["inventory_units"].sum()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(cat_df["category"], cat_df["inventory_units"])
            ax.set_ylabel("Unidades")
            st.pyplot(fig)
        else:
            st.info("No hay datos para mostrar.")

    st.subheader("Lotes prioritarios")
    priority_cols = [
        "lot_id", "store", "supplier", "category", "cut",
        "risk_score", "predicted_shrink_prob", "temp_max_c", "remaining_days",
        "inventory_units", "anomaly_flag",
        "recommended_markdown_pct", "recommended_action"
    ]
    priority_df = filtered.sort_values(
        by=["predicted_shrink_prob", "risk_score", "remaining_days"],
        ascending=[False, False, True],
    )
    safe_show_dataframe(priority_df, priority_cols)

elif module == "Compras y Demanda":
    st.title("Compras y Demanda")
    filtered = apply_filters(df).copy()

    filtered["suggested_purchase_qty"] = (
        filtered["daily_sales"] * 3 - filtered["inventory_units"] * 0.25
    ).clip(lower=0).round(0)

    filtered["overinventory_risk"] = (
        (filtered["inventory_units"] > filtered["daily_sales"] * 5).astype(int)
    )

    cols = [
        "po_number", "lot_id", "store", "supplier", "category", "cut",
        "inventory_units", "daily_sales", "purchase_qty",
        "suggested_purchase_qty", "overinventory_risk", "predicted_shrink_prob"
    ]
    safe_show_dataframe(filtered, cols)

elif module == "Recibo y Almacén":
    st.title("Recibo y Almacén")
    filtered = apply_filters(df).copy()

    filtered["qty_diff"] = filtered["received_qty"] - filtered["purchase_qty"]
    filtered["weight_diff_kg"] = (filtered["received_weight_kg"] - filtered["ordered_weight_kg"]).round(2)
    filtered["temp_rejection_flag"] = (filtered["temp_max_c"] > 7).astype(int)

    cols = [
        "recv_date", "po_number", "lot_id", "store", "supplier",
        "purchase_qty", "received_qty", "qty_diff",
        "ordered_weight_kg", "received_weight_kg", "weight_diff_kg",
        "temp_avg_c", "temp_max_c", "temp_rejection_flag", "predicted_shrink_prob"
    ]
    safe_show_dataframe(filtered, cols)

elif module == "Producción y Rendimientos":
    st.title("Producción y Rendimientos")
    filtered = apply_filters(df).copy()

    filtered["yield_gap"] = (filtered["actual_yield_pct"] - filtered["theoretical_yield_pct"]).round(2)
    filtered["production_gap_kg"] = (filtered["production_actual_kg"] - filtered["production_plan_kg"]).round(2)
    filtered["low_yield_flag"] = (filtered["yield_gap"] < -5).astype(int)

    cols = [
        "production_date", "lot_id", "store", "supplier", "category", "cut",
        "theoretical_yield_pct", "actual_yield_pct", "yield_gap",
        "production_plan_kg", "production_actual_kg", "production_gap_kg",
        "low_yield_flag", "predicted_shrink_prob"
    ]
    safe_show_dataframe(filtered, cols)

elif module == "Exhibición y Venta":
    st.title("Exhibición y Venta")
    filtered = apply_filters(df).copy()

    filtered["shelf_attention_flag"] = (
        (filtered["shelf_gaps"] > 2) |
        (filtered["expired_labels"] > 0) |
        (filtered["overfill_flag"] == 1)
    ).astype(int)

    cols = [
        "snapshot_date", "lot_id", "store", "category", "cut",
        "inventory_units", "daily_sales", "shelf_gaps",
        "expired_labels", "overfill_flag", "shelf_attention_flag",
        "markdown_pct", "gross_margin_pct", "predicted_shrink_prob"
    ]
    safe_show_dataframe(filtered, cols)

elif module == "Auditoría":
    st.title("Auditoría")
    filtered = apply_filters(df).copy()

    filtered["audit_status"] = filtered["audit_score"].apply(
        lambda x: "Crítico" if x < 70 else "Atención" if x < 85 else "Aceptable"
    )

    cols = [
        "snapshot_date", "lot_id", "store", "supplier", "category", "cut",
        "audit_score", "hygiene_score", "audit_status", "predicted_shrink_prob"
    ]
    safe_show_dataframe(filtered, cols)

elif module == "Predictivo":
    st.title("Modelo Predictivo de Riesgo / Merma")

    st.markdown(
        "Este módulo entrena un **RandomForestClassifier** para estimar la "
        "**probabilidad predictiva de riesgo/merma** por lote."
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Modelo", predictive_metrics.get("model_type"))
    k2.metric(
        "Accuracy",
        f"{predictive_metrics['accuracy']:.3f}" if predictive_metrics.get("accuracy") is not None else "N/A"
    )
    k3.metric(
        "ROC AUC",
        f"{predictive_metrics['roc_auc']:.3f}" if predictive_metrics.get("roc_auc") is not None else "N/A"
    )

    st.subheader("Importancia de variables")
    if len(feature_importance_df):
        fig, ax = plt.subplots(figsize=(10, 5))
        top_imp = feature_importance_df.head(10).sort_values("importance", ascending=True)
        ax.barh(top_imp["feature"], top_imp["importance"])
        ax.set_xlabel("Importancia")
        st.pyplot(fig)
        st.dataframe(feature_importance_df, use_container_width=True)
    else:
        st.info("No hay importancias disponibles. Se usó fallback.")

    st.subheader("Top lotes con mayor probabilidad predictiva")
    pred_cols = [
        "lot_id", "store", "supplier", "category", "cut",
        "risk_score", "predicted_shrink_prob", "predicted_shrink_flag",
        "remaining_days", "temp_max_c", "hours_out_of_range",
        "inventory_units", "daily_sales", "recommended_action"
    ]
    pred_df = df.sort_values("predicted_shrink_prob", ascending=False).head(50)
    safe_show_dataframe(pred_df, pred_cols)

    with st.expander("Ver reporte del modelo"):
        st.text(predictive_metrics.get("report", "Sin reporte"))

elif module == "Dirección y Copiloto IA":
    st.title("Dirección y Copiloto IA")

    worst_store = df.groupby("store")["risk_score"].mean().sort_values(ascending=False).index[0]
    worst_supplier = (
        (df["theoretical_yield_pct"] - df["actual_yield_pct"])
        .groupby(df["supplier"])
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    worst_category = df.groupby("category")["historical_shrink_pct"].mean().sort_values(ascending=False).index[0]

    st.write(f"**Tienda con mayor riesgo promedio:** {worst_store}")
    st.write(f"**Proveedor con mayor gap de rendimiento:** {worst_supplier}")
    st.write(f"**Categoría con mayor merma histórica promedio:** {worst_category}")

    st.markdown("### Copiloto IA")
    st.caption("Responde con base en los datos cargados y el contexto calculado por la plataforma.")

    filtered_for_llm = apply_filters(df)
    question = st.text_input("Escribe una pregunta ejecutiva")

    example_questions = [
        "¿Qué tiendas están destruyendo margen?",
        "¿Qué proveedor tiene peor rendimiento real vs esperado?",
        "Resume los 5 hallazgos más importantes del día.",
        "¿Dónde debo enfocar auditoría esta semana?",
        "¿Qué categoría tiene mayor riesgo y por qué?",
        "¿Qué lotes tienen la mayor probabilidad predictiva de merma?"
    ]
    st.markdown("**Preguntas sugeridas:**")
    for q in example_questions:
        st.write(f"- {q}")

    if question:
        with st.spinner("Consultando copiloto IA..."):
            context = build_llm_context(df, filtered_for_llm)
            answer = ask_llm(question, context)
        st.markdown("### Respuesta del copiloto")
        st.write(answer)

        with st.expander("Ver contexto enviado al LLM"):
            st.text(context)
