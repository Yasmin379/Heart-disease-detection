"""
Heart Disease Prediction — Streamlit App
Run:  uv run streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, rgba(220,38,38,0.25), rgba(239,68,68,0.10));
    border: 1px solid rgba(220,38,38,0.4);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(8px);
}
.hero-card h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 .4rem 0;
}
.hero-card p {
    color: #cbd5e1;
    font-size: 1rem;
    margin: 0;
}

/* Model selector cards */
.model-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
    transition: border-color .25s;
    cursor: pointer;
}
.model-card:hover { border-color: #ef4444; }
.model-card.selected { border-color: #ef4444; background: rgba(239,68,68,0.12); }

/* Result boxes */
.result-positive {
    background: linear-gradient(135deg, rgba(220,38,38,0.30), rgba(239,68,68,0.15));
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    animation: pulse 2s infinite;
}
.result-negative {
    background: linear-gradient(135deg, rgba(16,185,129,0.30), rgba(52,211,153,0.15));
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.35); }
    50%      { box-shadow: 0 0 20px 6px rgba(239,68,68,0.15); }
}

.result-label {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
}
.result-sub {
    color: #cbd5e1;
    font-size: .9rem;
    margin-top: .3rem;
}

/* Metric pills */
.metric-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 50px;
    padding: .35rem .9rem;
    font-size: .82rem;
    font-weight: 500;
    color: #e2e8f0;
    margin: .2rem;
}

/* Streamlit overrides */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #94a3b8 !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    letter-spacing: .04em !important;
    text-transform: uppercase !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    border: none;
    border-radius: 10px;
    padding: .6rem 2.2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: opacity .2s, transform .15s;
    letter-spacing: .02em;
}
div.stButton > button:hover {
    opacity: .88;
    transform: translateY(-1px);
}

h2, h3 { color: #f1f5f9 !important; }

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 1.4rem 0 .6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: .4rem;
}

/* Probability bar */
.prob-bar-wrapper {
    background: rgba(255,255,255,0.08);
    border-radius: 50px;
    height: 10px;
    margin: .4rem 0 .2rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────────────────────
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Random Forest":        "random_forest.pkl",
        "SVM":                  "svm.pkl",
        "Decision Tree":        "decision_tree.pkl",
        "Logistic Regression":  "logistic_regression.pkl",
    }
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            models[name] = None
    return models

models = load_models()

# ── Feature Info ───────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure",
    "serum_cholesterol", "fasting_blood_sugar", "resting_electrocardiographic_results",
    "max_heart_rate_achieved", "exercise_induced_angina",
    "st_depression", "st_slope", "num_major_vessels", "thalassemia"
]

FEATURE_META = {
    "age":                                   {"label": "Age (years)",            "type": "int",   "min": 20,  "max": 80,   "default": 50},
    "sex":                                   {"label": "Sex",                    "type": "select","options": {0: "Female", 1: "Male"}},
    "chest_pain_type":                       {"label": "Chest Pain Type",        "type": "select","options": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}},
    "resting_blood_pressure":               {"label": "Resting BP (mm Hg)",     "type": "int",   "min": 80,  "max": 200,  "default": 120},
    "serum_cholesterol":                     {"label": "Serum Cholesterol (mg/dL)","type": "int", "min": 100, "max": 600,  "default": 200},
    "fasting_blood_sugar":                   {"label": "Fasting Blood Sugar > 120 mg/dL", "type": "select","options": {0: "No (≤120)", 1: "Yes (>120)"}},
    "resting_electrocardiographic_results": {"label": "Resting ECG Results",    "type": "select","options": {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}},
    "max_heart_rate_achieved":              {"label": "Max Heart Rate Achieved", "type": "int",   "min": 60,  "max": 220,  "default": 150},
    "exercise_induced_angina":              {"label": "Exercise Induced Angina", "type": "select","options": {0: "No", 1: "Yes"}},
    "st_depression":                        {"label": "ST Depression (Oldpeak)", "type": "float","min": 0.0, "max": 7.0,  "default": 1.0, "step": 0.1},
    "st_slope":                             {"label": "ST Slope",               "type": "select","options": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}},
    "num_major_vessels":                    {"label": "Num Major Vessels (0–3)", "type": "int",   "min": 0,   "max": 3,    "default": 0},
    "thalassemia":                          {"label": "Thalassemia",            "type": "select","options": {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}},
}

# ── Sidebar — Model Selection ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Choose Model")
    model_icons = {
        "Random Forest":       "🌲",
        "SVM":                 "🔷",
        "Decision Tree":       "🌿",
        "Logistic Regression": "📈",
    }
    model_descriptions = {
        "Random Forest":       "Ensemble of trees — robust & accurate",
        "SVM":                 "Finds optimal decision boundary",
        "Decision Tree":       "Intuitive rule-based decisions",
        "Logistic Regression": "Fast probabilistic classifier",
    }

    selected_model = st.radio(
        "Select algorithm",
        list(models.keys()),
        format_func=lambda m: f"{model_icons[m]}  {m}",
        label_visibility="collapsed",
    )

    st.markdown("---")
    if models[selected_model] is None:
        st.error(f"`{selected_model}` model file not found in `models/`.")
    else:
        st.success(f"**{selected_model}** loaded ✓")
        st.caption(model_descriptions[selected_model])

    st.markdown("---")
    st.markdown("### 📋 Feature Guide")
    st.markdown("""
- **ST Depression** — exercise vs rest ECG  
- **Chest Pain** — type matters most  
- **Thalassemia** — blood disorder type  
- **Major Vessels** — coloured by fluoroscopy  
    """)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <h1>❤️ Heart Disease Predictor</h1>
  <p>Enter patient clinical data and select a model to receive an instant risk assessment powered by trained ML models.</p>
</div>
""", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Patient Clinical Data</div>', unsafe_allow_html=True)

input_values = {}

# Layout: 3 columns — build rows dynamically so any number of features works
COLS_PER_ROW = 3
rows = []
for i in range(0, len(FEATURE_COLS), COLS_PER_ROW):
    rows.append(st.columns(COLS_PER_ROW))

for idx, feat in enumerate(FEATURE_COLS):
    meta = FEATURE_META[feat]
    row_idx = idx // COLS_PER_ROW
    col_idx = idx % COLS_PER_ROW
    col = rows[row_idx][col_idx]
    with col:
        if meta["type"] == "select":
            opts = meta["options"]
            keys   = list(opts.keys())
            labels = list(opts.values())
            choice = st.selectbox(meta["label"], options=keys,
                                  format_func=lambda k, o=opts: o[k],
                                  key=feat)
            input_values[feat] = choice
        elif meta["type"] == "int":
            val = st.number_input(meta["label"],
                                  min_value=meta["min"], max_value=meta["max"],
                                  value=meta["default"], step=1, key=feat)
            input_values[feat] = int(val)
        elif meta["type"] == "float":
            val = st.number_input(meta["label"],
                                  min_value=float(meta["min"]),
                                  max_value=float(meta["max"]),
                                  value=float(meta["default"]),
                                  step=meta.get("step", 0.1), key=feat)
            input_values[feat] = float(val)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ─────────────────────────────────────────────────────────────
left_col, _, right_col = st.columns([2, 3, 2])
with left_col:
    predict_clicked = st.button("🔍 Predict", key="predict_btn")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked:
    if models[selected_model] is None:
        st.error("Please select a model that has been loaded successfully.")
    else:
        model = models[selected_model]
        input_array = np.array([[input_values[f] for f in FEATURE_COLS]])

        prediction = model.predict(input_array)[0]
        has_disease = int(prediction) == 1

        # Try probability
        prob_disease = None
        try:
            proba = model.predict_proba(input_array)[0]
            prob_disease = proba[1]
        except AttributeError:
            pass

        st.markdown("---")
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        res_col, info_col = st.columns([1.2, 1])

        with res_col:
            if has_disease:
                st.markdown("""
                <div class="result-positive">
                  <div class="result-label">⚠️ Heart Disease Detected</div>
                  <div class="result-sub">High risk — please consult a cardiologist immediately.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                  <div class="result-label">✅ No Heart Disease</div>
                  <div class="result-sub">Low risk — maintain a healthy lifestyle.</div>
                </div>
                """, unsafe_allow_html=True)

        with info_col:
            st.markdown(f"""
            <span class="metric-pill">Model: {model_icons[selected_model]} {selected_model}</span>
            <span class="metric-pill">Prediction: {'Positive' if has_disease else 'Negative'}</span>
            """, unsafe_allow_html=True)

            if prob_disease is not None:
                pct = prob_disease * 100
                color = "#ef4444" if pct >= 50 else "#10b981"
                st.markdown(f"""
                <br>
                <div style="color:#94a3b8; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em;">
                  Disease Probability
                </div>
                <div style="font-size:2rem; font-weight:700; color:{color};">{pct:.1f}%</div>
                <div class="prob-bar-wrapper">
                  <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
                </div>
                <div style="color:#64748b; font-size:.75rem;">0% ─────────────────── 100%</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<br><span class='metric-pill'>Probability not available for this model</span>", unsafe_allow_html=True)

        # Input summary table
        with st.expander("📊 View Input Summary"):
            df_input = pd.DataFrame([input_values])
            df_input.columns = [FEATURE_META[c]["label"] for c in FEATURE_COLS]
            st.dataframe(df_input, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:.8rem; padding-bottom:1rem;">
  Heart Disease Predictor · Built with Streamlit · For educational purposes only<br>
  <span style="color:#ef4444;">⚠️</span> Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)
