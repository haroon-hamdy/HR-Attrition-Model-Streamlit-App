# HR Employee Attrition Prediction - Ready Streamlit App
# Run: streamlit run app.py
# Data upload removed: the dashboard loads HR-Final.xlsx / artifacts automatically.

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import joblib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "HR-Final.xlsx"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "hr_attrition_model_bundle.joblib"


# -----------------------------
# Page config and custom style
# -----------------------------
st.set_page_config(
    page_title="HR Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

        :root{
            --bg:#050914;
            --panel:#0b1224;
            --panel2:#111a33;
            --stroke:rgba(103,232,249,.22);
            --text:#f8fafc;
            --muted:#9aa7bd;
            --cyan:#22d3ee;
            --violet:#8b5cf6;
            --pink:#f472b6;
            --green:#22c55e;
            --amber:#f59e0b;
            --red:#f43f5e;
        }

        html, body, [class*="css"] {font-family:'Inter', sans-serif !important;}

        .stApp{
            color:var(--text);
            background:
                radial-gradient(circle at 12% 15%, rgba(34,211,238,.18) 0, transparent 28%),
                radial-gradient(circle at 70% 18%, rgba(139,92,246,.22) 0, transparent 32%),
                radial-gradient(circle at 48% 88%, rgba(244,114,182,.12) 0, transparent 32%),
                linear-gradient(135deg,#020617 0%,#07111f 42%,#100f2e 100%);
        }

        .block-container{padding-top:2rem; padding-bottom:3rem; max-width:1360px;}
        header[data-testid="stHeader"]{background:transparent;}
        div[data-testid="stToolbar"]{display:none;}

        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg, rgba(3,7,18,.98), rgba(8,13,28,.98));
            border-right:1px solid rgba(34,211,238,.16);
        }
        section[data-testid="stSidebar"] *{color:#dbeafe !important;}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
            color:white !important; font-weight:900 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"]{
            background:rgba(15,23,42,.74);
            border:1px dashed rgba(34,211,238,.35);
            border-radius:16px;
            padding:14px;
        }

        .hero-card{
            position:relative;
            overflow:hidden;
            padding:34px 34px 30px;
            border-radius:28px;
            background:
                radial-gradient(circle at 96% 5%, rgba(34,211,238,.24), transparent 24%),
                linear-gradient(135deg, rgba(15,23,42,.95), rgba(18,26,51,.92) 58%, rgba(39,37,75,.88));
            border:1px solid rgba(34,211,238,.24);
            box-shadow:0 0 0 1px rgba(255,255,255,.035), 0 28px 70px rgba(0,0,0,.42);
            margin-bottom:18px;
        }
        .hero-pill{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:8px 14px;
            margin-bottom:18px;
            border-radius:999px;
            background:rgba(34,211,238,.12);
            border:1px solid rgba(34,211,238,.35);
            color:#67e8f9;
            font-size:12px;
            font-weight:900;
            letter-spacing:.04em;
            text-transform:uppercase;
        }
        .hero-card h1{
            font-size:46px;
            line-height:1.05;
            margin:0 0 18px;
            font-weight:950;
            color:#fff;
            letter-spacing:-.05em;
        }
        .gradient-text{
            background:linear-gradient(90deg,#67e8f9 0%,#8b5cf6 48%,#f472b6 100%);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            background-clip:text;
        }
        .hero-card p{font-size:15.5px; color:#cbd5e1; max-width:880px; margin:0; line-height:1.7;}
        .status-row{display:flex; gap:10px; flex-wrap:wrap; margin:0 0 20px;}
        .chip{padding:7px 12px; border-radius:999px; font-size:12px; font-weight:800; border:1px solid rgba(255,255,255,.1);}
        .chip-cyan{background:rgba(34,211,238,.14); color:#67e8f9; border-color:rgba(34,211,238,.32);}
        .chip-purple{background:rgba(139,92,246,.16); color:#c4b5fd; border-color:rgba(139,92,246,.35);}
        .chip-green{background:rgba(34,197,94,.14); color:#86efac; border-color:rgba(34,197,94,.32);}
        .chip-amber{background:rgba(245,158,11,.14); color:#fcd34d; border-color:rgba(245,158,11,.32);}

        div[data-testid="stMetric"], .metric-card{
            background:linear-gradient(180deg, rgba(17,24,39,.92), rgba(10,16,32,.94));
            border:1px solid rgba(148,163,184,.18);
            border-radius:20px;
            padding:18px 18px 14px;
            box-shadow:inset 0 1px 0 rgba(255,255,255,.03), 0 18px 46px rgba(0,0,0,.24);
            position:relative;
            overflow:hidden;
        }
        div[data-testid="stMetric"]:after, .metric-card:after{
            content:""; position:absolute; left:14px; right:14px; bottom:0; height:3px;
            background:linear-gradient(90deg,#22d3ee,#8b5cf6,#f43f5e);
            border-radius:99px;
        }
        div[data-testid="stMetricLabel"]{color:#a8b3c7 !important; font-weight:800 !important;}
        div[data-testid="stMetricValue"]{font-size:30px !important; color:#fff !important; font-weight:950 !important;}
        div[data-testid="stMetricDelta"]{color:#67e8f9 !important;}

        .stTabs [data-baseweb="tab-list"]{gap:18px; border-bottom:1px solid rgba(148,163,184,.18);}
        .stTabs [data-baseweb="tab"]{
            color:#cbd5e1; background:transparent; border-radius:0; padding:12px 4px; font-weight:800;
        }
        .stTabs [aria-selected="true"]{color:#fff !important; border-bottom:4px solid #22d3ee;}

        div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stExpander"]{
            background:rgba(15,23,42,.68) !important;
            border:1px solid rgba(148,163,184,.16) !important;
            border-radius:18px !important;
        }
        div[data-testid="stDataFrame"]{
            border:1px solid rgba(148,163,184,.16);
            border-radius:18px;
            overflow:hidden;
        }

        .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"], .stNumberInput input, .stTextInput input{
            background:#111827 !important;
            border-color:rgba(148,163,184,.22) !important;
            color:#f8fafc !important;
            border-radius:12px !important;
        }
        .stMultiSelect span[data-baseweb="tag"]{
            background:linear-gradient(90deg,#22d3ee,#8b5cf6) !important;
            color:white !important;
            border-radius:999px !important;
            font-weight:800 !important;
        }
        .stButton>button, .stDownloadButton>button, button[kind="primaryFormSubmit"]{
            width:100%; height:48px; border-radius:14px; border:0 !important;
            background:linear-gradient(90deg,#22d3ee,#8b5cf6,#f472b6) !important;
            color:white !important; font-weight:900 !important;
            box-shadow:0 12px 32px rgba(34,211,238,.18);
        }
        .stButton>button:hover, .stDownloadButton>button:hover{filter:brightness(1.12); transform:translateY(-1px);}
        hr{border-color:rgba(148,163,184,.18) !important;}
        h1,h2,h3{font-weight:950 !important; letter-spacing:-.03em; color:#fff !important;}
        p, label, .stMarkdown, .stCaption{color:#cbd5e1;}

        .risk-low{background:rgba(34,197,94,.13); color:#86efac; border:1px solid rgba(34,197,94,.3); padding:12px 14px; border-radius:14px; font-weight:900; text-align:center;}
        .risk-medium{background:rgba(245,158,11,.13); color:#fcd34d; border:1px solid rgba(245,158,11,.3); padding:12px 14px; border-radius:14px; font-weight:900; text-align:center;}
        .risk-high{background:rgba(244,63,94,.13); color:#fb7185; border:1px solid rgba(244,63,94,.3); padding:12px 14px; border-radius:14px; font-weight:900; text-align:center;}
    </style>
    """,
    unsafe_allow_html=True,
)

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#22d3ee", "#8b5cf6", "#f43f5e", "#22c55e", "#f59e0b", "#f472b6"]


# -----------------------------
# Helper functions
# -----------------------------
def risk_label(probability: float) -> str:
    """Convert attrition probability into a business-friendly risk level."""
    if probability >= 0.60:
        return "🔴 High"
    if probability >= 0.30:
        return "🟡 Medium"
    return "🟢 Low"


def risk_badge_html(probability: float) -> str:
    """Return a colored HTML badge for the prediction result."""
    label = risk_label(probability)
    if probability >= 0.60:
        return f'<div class="risk-high">{label} Risk</div>'
    if probability >= 0.30:
        return f'<div class="risk-medium">{label} Risk</div>'
    return f'<div class="risk-low">{label} Risk</div>'


@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> pd.DataFrame:
    """Load uploaded Excel file into a dataframe."""
    return pd.read_excel(BytesIO(file_bytes))


def clean_hr_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Apply the same cleaning logic used in the notebook."""
    df_clean = df.copy()
    dropped_columns: List[str] = []

    # Extract useful date features, then remove the raw date column from modeling.
    if "HireDate" in df_clean.columns:
        hire_date = pd.to_datetime(df_clean["HireDate"], errors="coerce")
        df_clean["HireYear"] = hire_date.dt.year.fillna(hire_date.dt.year.median()).fillna(0).astype(int)
        df_clean["HireMonth"] = hire_date.dt.month.fillna(hire_date.dt.month.median()).fillna(0).astype(int)

    # Drop columns that should not be used as model signals.
    for col in ["EmployeeID", "FullName", "Column25", "HireDate"]:
        if col in df_clean.columns:
            dropped_columns.append(col)
            df_clean = df_clean.drop(columns=[col])

    return df_clean, dropped_columns


def encode_hr_data(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], List[str]]:
    """Encode categorical columns and encode Attrition target as Yes=1 / No=0."""
    if "Attrition" not in df_clean.columns:
        raise ValueError("The uploaded file must contain an 'Attrition' column.")

    df_encoded = df_clean.copy()
    categorical_cols = df_encoded.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "Attrition"]

    encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str).fillna("Unknown"))
        encoders[col] = le

    attrition_text = df_encoded["Attrition"].astype(str).str.strip().str.lower()
    df_encoded["Attrition"] = attrition_text.map({"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0})

    if df_encoded["Attrition"].isna().any():
        raise ValueError("Attrition column must contain Yes/No values or 1/0 values.")

    df_encoded["Attrition"] = df_encoded["Attrition"].astype(int)
    return df_encoded, encoders, categorical_cols


@st.cache_resource(show_spinner=False)
def train_attrition_model(file_bytes: bytes, selected_features: Tuple[str, ...]):
    """Train XGBoost attrition model using selected features."""
    raw_df = load_excel(file_bytes)
    clean_df, dropped_columns = clean_hr_data(raw_df)
    encoded_df, encoders, categorical_cols = encode_hr_data(clean_df)

    selected_features = list(selected_features)
    if len(selected_features) < 2:
        raise ValueError("Please select at least two features to train the model.")

    X = encoded_df[selected_features].copy()
    y = encoded_df["Attrition"].copy()

    if y.nunique() < 2:
        raise ValueError("Attrition target must contain both classes: Yes and No.")

    # Use a safe split. Stratify only when both classes have enough samples.
    class_counts = y.value_counts()
    stratify_arg = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_arg,
    )

    # Apply SMOTE only when the minority class has enough samples.
    minority_count = y_train.value_counts().min()
    if minority_count >= 2:
        k_neighbors = min(5, minority_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_model, y_train_model = smote.fit_resample(X_train, y_train)
        balancing_status = f"SMOTE applied with k_neighbors={k_neighbors}"
    else:
        X_train_model, y_train_model = X_train, y_train
        balancing_status = "SMOTE skipped because the minority class is too small"

    model = XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train_model, y_train_model)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_prob)) if y_test.nunique() == 2 else np.nan,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["Stayed", "Left"], zero_division=0),
        "balancing_status": balancing_status,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    feature_importance = pd.DataFrame(
        {
            "Feature": selected_features,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return {
        "model": model,
        "raw_df": raw_df,
        "clean_df": clean_df,
        "encoded_df": encoded_df,
        "encoders": encoders,
        "categorical_cols": categorical_cols,
        "dropped_columns": dropped_columns,
        "selected_features": selected_features,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }


def build_single_input(clean_df: pd.DataFrame, encoders: Dict[str, LabelEncoder], selected_features: List[str]) -> pd.DataFrame:
    """Create a professional input form and return one encoded row for prediction."""
    values = {}
    left_col, right_col = st.columns(2)

    for index, feature in enumerate(selected_features):
        column_container = left_col if index % 2 == 0 else right_col
        with column_container:
            if feature in encoders:
                options = list(encoders[feature].classes_)
                values[feature] = st.selectbox(feature, options=options, key=f"input_{feature}")
            else:
                series = pd.to_numeric(clean_df[feature], errors="coerce")
                median_value = series.median()
                if pd.isna(median_value):
                    median_value = 0
                min_value = series.min()
                max_value = series.max()
                if pd.isna(min_value):
                    min_value = 0
                if pd.isna(max_value):
                    max_value = max(1, median_value)

                is_integer = pd.api.types.is_integer_dtype(clean_df[feature]) or float(median_value).is_integer()
                step = 1 if is_integer else 0.1
                value = int(round(median_value)) if is_integer else float(round(median_value, 2))
                min_clean = int(np.floor(min_value)) if is_integer else float(np.floor(min_value))
                max_clean = int(np.ceil(max_value)) if is_integer else float(np.ceil(max_value))

                values[feature] = st.number_input(
                    feature,
                    min_value=min_clean,
                    max_value=max_clean,
                    value=value,
                    step=step,
                    key=f"input_{feature}",
                )

    input_df = pd.DataFrame([values])
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col].astype(str))

    return input_df[selected_features]


def make_predictions(result: dict) -> pd.DataFrame:
    """Predict attrition probability for all employees in the uploaded dataset."""
    model = result["model"]
    encoded_df = result["encoded_df"]
    raw_df = result["raw_df"]
    selected_features = result["selected_features"]

    X_all = encoded_df[selected_features]
    probabilities = model.predict_proba(X_all)[:, 1]
    predictions = model.predict(X_all)

    display_cols = [col for col in ["EmployeeID", "FullName", "Department", "JobRole", "Age", "Salary", "Attrition"] if col in raw_df.columns]
    output = raw_df[display_cols].copy() if display_cols else raw_df.copy()
    output["Predicted_Leave"] = np.where(predictions == 1, "Yes", "No")
    output["Leave_Probability_%"] = (probabilities * 100).round(1)
    output["Risk_Level"] = [risk_label(p) for p in probabilities]
    return output.sort_values("Leave_Probability_%", ascending=False)


def dataframe_to_excel_bytes(df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    """Export predictions and summary to an Excel file in memory."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Predictions", index=False)
        df[df["Risk_Level"] == "🔴 High"].to_excel(writer, sheet_name="High Risk", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    buffer.seek(0)
    return buffer.read()



def polish_plotly(fig):
    """Apply the dark glass dashboard look to Plotly charts."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe", family="Inter"),
        title=dict(font=dict(color="#f8fafc", size=18)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#dbeafe")),
        margin=dict(l=20, r=20, t=55, b=25),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.13)", zerolinecolor="rgba(148,163,184,.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.13)", zerolinecolor="rgba(148,163,184,.2)")
    return fig

# -----------------------------
# App header
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-pill">AI POWERED · DARK WEB DASHBOARD · STREAMLIT</div>
        <h1>HR <span class="gradient-text">Intelligence</span></h1>
        <p>Predict employee attrition risk, detect workforce patterns, explore model insights, and export executive-ready HR actions from one dynamic web dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Ready model / data loading
# -----------------------------
def choose_default_features(encoded_df: pd.DataFrame) -> List[str]:
    """Choose the same fixed feature set used to build the ready model bundle."""
    available_features = [c for c in encoded_df.columns if c != "Attrition"]
    preferred_features = [
        "Age",
        "Salary",
        "OverTime",
        "YearsAtCompany",
        "DistanceFromHome (KM)",
        "DistanceFromHome",
        "JobSatisfaction",
        "Department",
        "JobRole",
        "MaritalStatus",
        "YearsInMostRecentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
        "HireYear",
        "HireMonth",
    ]
    selected = [f for f in preferred_features if f in available_features]
    return selected if len(selected) >= 2 else available_features


@st.cache_resource(show_spinner=False)
def load_ready_bundle() -> dict:
    """Load the ready trained bundle. If it is missing, rebuild it once from the bundled Excel file."""
    if ARTIFACT_PATH.exists():
        try:
            return joblib.load(ARTIFACT_PATH)
        except Exception:
            # Different package versions can occasionally break pickle/joblib loading.
            # In that case the app rebuilds the model from the included Excel file.
            pass

    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DEFAULT_DATA_PATH.name}. Put HR-Final.xlsx in the same folder as app.py."
        )

    file_bytes = DEFAULT_DATA_PATH.read_bytes()
    raw_preview = load_excel(file_bytes)
    clean_preview, _ = clean_hr_data(raw_preview)
    encoded_preview, _, _ = encode_hr_data(clean_preview)
    selected = choose_default_features(encoded_preview)
    bundle = train_attrition_model(file_bytes, tuple(selected))
    bundle["data_file"] = DEFAULT_DATA_PATH.name

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH)
    return bundle


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚡ HR Intelligence")
st.sidebar.caption("Ready ML web app for attrition prediction and workforce intelligence.")

with st.spinner("Loading ready HR model..."):
    try:
        result = load_ready_bundle()
    except Exception as exc:
        st.error(f"Ready model loading error: {exc}")
        st.stop()

selected_features = result["selected_features"]
metrics = result["metrics"]
predictions_df = make_predictions(result)

st.sidebar.success("Ready dashboard loaded")
st.sidebar.markdown(f"**Data file:** `{result.get('data_file', 'HR-Final.xlsx')}`")
st.sidebar.markdown(f"**Employees:** `{len(result['raw_df']):,}`")
st.sidebar.markdown(f"**Model features:** `{len(selected_features)}`")
st.sidebar.markdown(f"**Accuracy:** `{metrics['accuracy'] * 100:.2f}%`")
st.sidebar.info("No upload is needed. The app opens directly with the trained model and ready predictions.")

st.markdown(
    f"""
    <div class="status-row">
        <span class="chip chip-cyan">Target: Attrition Risk</span>
        <span class="chip chip-purple">Algorithm: XGBoost</span>
        <span class="chip chip-green">Features: {len(selected_features)}</span>
        <span class="chip chip-amber">Train/Test: {metrics['train_size']:,}/{metrics['test_size']:,}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Main tabs
# -----------------------------
tab_dashboard, tab_single, tab_batch, tab_model = st.tabs(
    ["⚡ Command Center", "🎯 Predict One", "🛡️ Workforce Radar", "🧠 Model Lab"]
)


with tab_dashboard:
    st.subheader("Executive Command Center")

    total_employees = len(predictions_df)
    predicted_leavers = int((predictions_df["Predicted_Leave"] == "Yes").sum())
    predicted_stayers = total_employees - predicted_leavers
    attrition_rate = predicted_leavers / total_employees * 100 if total_employees else 0
    high_risk_count = int((predictions_df["Risk_Level"] == "🔴 High").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", f"{total_employees:,}")
    col2.metric("Predicted to Leave", f"{predicted_leavers:,}", f"{attrition_rate:.1f}%")
    col3.metric("Predicted to Stay", f"{predicted_stayers:,}")
    col4.metric("High Risk Employees", f"{high_risk_count:,}")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        stay_leave_df = pd.DataFrame(
            {
                "Status": ["Stay", "Leave"],
                "Employees": [predicted_stayers, predicted_leavers],
            }
        )
        fig = px.pie(stay_leave_df, names="Status", values="Employees", hole=0.45, title="Predicted Attrition Split")
        st.plotly_chart(polish_plotly(fig), use_container_width=True)

    with chart_col2:
        risk_counts = predictions_df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Employees"]
        fig = px.bar(risk_counts, x="Risk Level", y="Employees", text="Employees", title="Risk Level Distribution")
        fig.update_traces(textposition="outside")
        st.plotly_chart(polish_plotly(fig), use_container_width=True)

    st.subheader("Top 10 Employees Most Likely to Leave")
    st.dataframe(predictions_df.head(10), use_container_width=True, hide_index=True)


with tab_single:
    st.subheader("Single Employee Prediction")
    st.caption("Fill the selected feature values, then click Predict.")

    with st.form("single_prediction_form"):
        single_input = build_single_input(result["clean_df"], result["encoders"], result["selected_features"])
        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        probability = float(result["model"].predict_proba(single_input)[:, 1][0])
        predicted_class = int(probability >= 0.50)

        result_col1, result_col2, result_col3 = st.columns([1, 1, 1])
        result_col1.metric("Prediction", "Leave" if predicted_class == 1 else "Stay")
        result_col2.metric("Leave Probability", f"{probability * 100:.1f}%")
        result_col3.markdown(risk_badge_html(probability), unsafe_allow_html=True)

        if probability >= 0.60:
            st.error("High attrition risk. Suggested HR action: review salary, workload, manager relationship, engagement, and retention plan.")
        elif probability >= 0.30:
            st.warning("Medium attrition risk. Suggested HR action: monitor the employee and schedule a check-in.")
        else:
            st.success("Low attrition risk. Keep monitoring normal engagement indicators.")


with tab_batch:
    st.subheader("Workforce Radar & Batch Intelligence")
    st.caption("Sorted by the highest leave probability.")

    st.dataframe(predictions_df, use_container_width=True, hide_index=True)

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Employees",
                "Predicted to Leave",
                "Predicted to Stay",
                "Predicted Attrition Rate (%)",
                "High Risk Employees",
                "Medium Risk Employees",
                "Low Risk Employees",
                "Model Accuracy (%)",
                "Model AUC (%)",
                "Model F1 (%)",
            ],
            "Value": [
                total_employees,
                predicted_leavers,
                predicted_stayers,
                round(attrition_rate, 1),
                int((predictions_df["Risk_Level"] == "🔴 High").sum()),
                int((predictions_df["Risk_Level"] == "🟡 Medium").sum()),
                int((predictions_df["Risk_Level"] == "🟢 Low").sum()),
                round(metrics["accuracy"] * 100, 2),
                "N/A" if np.isnan(metrics["auc"]) else round(metrics["auc"] * 100, 2),
                round(metrics["f1"] * 100, 2),
            ],
        }
    )

    excel_bytes = dataframe_to_excel_bytes(predictions_df, summary_df)
    st.download_button(
        label="Download Predictions Excel",
        data=excel_bytes,
        file_name="HR_Attrition_Streamlit_Predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


with tab_model:
    st.subheader("Model Lab")

    perf_col1, perf_col2, perf_col3 = st.columns(3)
    perf_col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    perf_col2.metric("AUC-ROC", "N/A" if np.isnan(metrics["auc"]) else f"{metrics['auc'] * 100:.2f}%")
    perf_col3.metric("F1-Score", f"{metrics['f1'] * 100:.2f}%")

    st.caption(metrics["balancing_status"])
    st.caption(f"Train samples: {metrics['train_size']} | Test samples: {metrics['test_size']}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        importance_df = result["feature_importance"].head(15)
        fig = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Feature Importances",
        )
        st.plotly_chart(polish_plotly(fig), use_container_width=True)

    with chart_col2:
        cm = np.array(metrics["confusion_matrix"])
        cm_df = pd.DataFrame(cm, index=["Actual Stay", "Actual Leave"], columns=["Predicted Stay", "Predicted Leave"])
        fig = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(polish_plotly(fig), use_container_width=True)

    with st.expander("Show detailed classification report"):
        st.code(metrics["classification_report"])

    with st.expander("Data cleaning summary"):
        st.write("Dropped columns:", result["dropped_columns"] if result["dropped_columns"] else "No dropped columns found")
        st.write("Categorical encoded columns:", result["categorical_cols"] if result["categorical_cols"] else "No categorical columns found")
        st.write("Selected model features:", result["selected_features"])
