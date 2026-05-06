# =============================================================================
# streamlit_app.py
# CTG Fetal Health Classifier – Interactive Streamlit UI
# =============================================================================

# ── 4.1  IMPORT LIBRARIES ─────────────────────────────────────────────────────
import streamlit as st          # UI framework
import joblib                   # Model / scaler deserialisation
import numpy as np              # Numerical array handling
import pandas as pd             # DataFrame construction for prediction input
import matplotlib.pyplot as plt # Matplotlib figures embedded in Streamlit
import matplotlib.patches as mpatches
import json                     # Reading stored classification report
import joblib                   # For loading model and scaler artifacts

# ── 4.2  PAGE CONFIGURATION & DOCUMENT TITLE ─────────────────────────────────
st.set_page_config(
    page_title="CTG Fetal Health Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8fafc; }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e86ab 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .header-banner h1 { color: white; margin: 0; font-size: 2rem; }
    .header-banner p  { color: #cfe8f3; margin: 0.4rem 0 0; font-size: 1rem; }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e3a5f;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Result boxes */
    .result-normal     { background:#d4edda; border-left:5px solid #28a745;
                         padding:1rem 1.4rem; border-radius:8px; color:#155724; }
    .result-suspect    { background:#fff3cd; border-left:5px solid #ffc107;
                         padding:1rem 1.4rem; border-radius:8px; color:#856404; }
    .result-pathologic { background:#f8d7da; border-left:5px solid #dc3545;
                         padding:1rem 1.4rem; border-radius:8px; color:#721c24; }

    /* Metric pills */
    .metric-pill {
        display:inline-block;
        background:#e8f4fd;
        color:#1e3a5f;
        border-radius:20px;
        padding:0.25rem 0.8rem;
        font-size:0.82rem;
        font-weight:600;
        margin:0.2rem;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] { background:#1e3a5f; }
    section[data-testid="stSidebar"] * { color: #cfe8f3 !important; }
    section[data-testid="stSidebar"] .stSlider > label { color:#cfe8f3 !important; }
</style>
""", unsafe_allow_html=True)

# ── 4.3  LOAD MODEL, SCALER & FEATURE LIST ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model        = joblib.load("rf_best_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

@st.cache_data(show_spinner=False)
def load_report():
    try:
        with open("classification_report.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

model, scaler, FEATURES = load_artifacts()
report_data = load_report()

# ── LABEL MAP ─────────────────────────────────────────────────────────────────
LABEL_MAP = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
LABEL_EMOJI = {"Normal": "✅", "Suspect": "⚠️", "Pathologic": "🚨"}
LABEL_CSS   = {"Normal": "result-normal",
               "Suspect": "result-suspect",
               "Pathologic": "result-pathologic"}

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  – navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 CTG Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔬 Predict", "📊 Model Performance", "📖 Dataset Info"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model:** Tuned Random Forest")
    st.markdown("**Dataset:** CTG – UCI ML Repository")
    st.markdown("**Classes:**")
    st.markdown("- 🟢 Normal (NSP=1)")
    st.markdown("- 🟡 Suspect (NSP=2)")
    st.markdown("- 🔴 Pathologic (NSP=3)")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 – PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔬 Predict":

    # Header
    st.markdown("""
    <div class="header-banner">
      <h1>🫀 CTG Fetal Health Classifier</h1>
      <p>Enter CTG signal features below to predict the fetal state in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 4.4  GET INPUT VIA STREAMLIT INPUT FUNCTIONS ─────────────────────────
    st.markdown('<div class="section-card"><div class="section-title">📥 CTG Signal Inputs</div>',
                unsafe_allow_html=True)

    # Default reference values (approximate dataset medians)
    defaults = {
        "LB": 133.0, "AC": 0.003, "FM": 0.09,  "UC": 0.004,
        "ASTV": 47.0, "MSTV": 1.35, "ALTV": 10.0, "MLTV": 8.3,
        "Width": 70.0, "Min": 93.0,  "Max": 164.0,
        "Nmax": 4,    "Nzeros": 0,   "Mode": 137.0,
        "Mean": 134.0, "Median": 139.0, "Variance": 18.0, "Tendency": 0,
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Heart Rate Baseline**")
        LB     = st.slider("LB – FHR Baseline (bpm)",
                           min_value=50.0,  max_value=200.0, value=defaults["LB"],  step=0.5,
                           help="Fetal heart rate baseline in beats per minute")
        AC     = st.number_input("AC – Accelerations / sec",
                                 min_value=0.0, max_value=0.05, value=defaults["AC"],
                                 step=0.001, format="%.4f")
        FM     = st.number_input("FM – Fetal Movements / sec",
                                 min_value=0.0, max_value=1.0,  value=defaults["FM"],
                                 step=0.01, format="%.3f")
        UC     = st.number_input("UC – Uterine Contractions / sec",
                                 min_value=0.0, max_value=0.05, value=defaults["UC"],
                                 step=0.001, format="%.4f")
        Tendency = st.selectbox("Tendency – FHR Trend",
                                options=[-1, 0, 1],
                                index=1,
                                format_func=lambda x: {-1:"↓ Decreasing",
                                                        0:"→ Stable",
                                                        1:"↑ Increasing"}[x])

    with col2:
        st.markdown("**Variability Metrics**")
        ASTV   = st.slider("ASTV – % Abnormal Short-Term Variability",
                           0.0, 100.0, defaults["ASTV"], 0.5)
        MSTV   = st.number_input("MSTV – Mean Short-Term Variability",
                                 min_value=-5.0, max_value=10.0, value=defaults["MSTV"],
                                 step=0.1, format="%.2f")
        ALTV   = st.slider("ALTV – % Abnormal Long-Term Variability",
                           0.0, 100.0, defaults["ALTV"], 0.5)
        MLTV   = st.number_input("MLTV – Mean Long-Term Variability",
                                 min_value=-20.0, max_value=50.0, value=defaults["MLTV"],
                                 step=0.1, format="%.1f")

    with col3:
        st.markdown("**Histogram Statistics**")
        Width    = st.slider("Width – FHR Histogram Width",
                             0.0, 300.0, defaults["Width"], 1.0)
        Min      = st.slider("Min – Minimum FHR",
                             50.0, 250.0, defaults["Min"],   0.5)
        Max      = st.slider("Max – Maximum FHR",
                             50.0, 300.0, defaults["Max"],   0.5)
        Nmax     = st.number_input("Nmax – No. of Histogram Peaks",
                                   min_value=0, max_value=20, value=defaults["Nmax"])
        Nzeros   = st.number_input("Nzeros – No. of Histogram Zeros",
                                   min_value=0, max_value=10, value=defaults["Nzeros"])
        Mode     = st.number_input("Mode – Histogram Mode",
                                   min_value=50.0, max_value=250.0, value=defaults["Mode"],
                                   step=0.5, format="%.1f")
        Mean_val = st.number_input("Mean – Mean FHR",
                                   min_value=50.0, max_value=250.0, value=defaults["Mean"],
                                   step=0.5, format="%.1f")
        Median   = st.number_input("Median – Median FHR",
                                   min_value=50.0, max_value=250.0, value=defaults["Median"],
                                   step=0.5, format="%.1f")
        Variance = st.number_input("Variance – FHR Variance",
                                   min_value=0.0, max_value=300.0, value=defaults["Variance"],
                                   step=0.5, format="%.1f")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 4.5  MAKE PREDICTIONS & DISPLAY RESULTS ──────────────────────────────
    predict_btn = st.button("🔍 Predict Fetal State", type="primary", use_container_width=True)

    if predict_btn:
        # Build input dict aligned to feature order used during training
        input_dict = {
            "LB": LB, "AC": AC, "FM": FM, "UC": UC,
            "ASTV": ASTV, "MSTV": MSTV, "ALTV": ALTV, "MLTV": MLTV,
            "Width": Width, "Min": Min, "Max": Max,
            "Nmax": Nmax, "Nzeros": Nzeros, "Mode": Mode,
            "Mean": Mean_val, "Median": Median,
            "Variance": Variance, "Tendency": Tendency,
        }
        # Align to the exact feature order the model was trained on
        input_row = np.array([[input_dict[f] for f in FEATURES]])
        input_scaled = scaler.transform(input_row)

        prediction      = model.predict(input_scaled)[0]
        probabilities   = model.predict_proba(input_scaled)[0]
        label           = LABEL_MAP[prediction]
        emoji           = LABEL_EMOJI[label]
        css_class       = LABEL_CSS[label]

        # Result banner
        st.markdown(f"""
        <div class="{css_class}">
          <h2 style="margin:0">{emoji} Prediction: <strong>{label}</strong></h2>
          <p style="margin:0.3rem 0 0">
            NSP Class {prediction} &nbsp;|&nbsp;
            Confidence: <strong>{max(probabilities)*100:.1f}%</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar chart
        st.markdown('<div class="section-card">'
                    '<div class="section-title">📊 Class Probabilities</div>',
                    unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Class":       ["Normal", "Suspect", "Pathologic"],
            "Probability": probabilities,
        })

        colors = ["#28a745" if c == label else "#adb5bd" for c in prob_df["Class"]]
        fig, ax = plt.subplots(figsize=(7, 2.5))
        bars = ax.barh(prob_df["Class"], prob_df["Probability"],
                       color=colors, edgecolor="white", height=0.5)
        for bar, prob in zip(bars, probabilities):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{prob*100:.1f}%", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Probability", fontsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(left=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        # Clinical guidance
        guidance = {
            "Normal":     "The fetal heart rate pattern appears normal. Routine monitoring is advised.",
            "Suspect":    "The pattern is suspicious. Enhanced monitoring and clinical review are recommended.",
            "Pathologic": "The pattern is pathological. Immediate clinical assessment and intervention may be required.",
        }
        st.info(f"🩺 **Clinical note:** {guidance[label]}")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 – MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":

    st.markdown("""
    <div class="header-banner">
      <h1>📊 Model Performance</h1>
      <p>Classification metrics for the Tuned Random Forest model.</p>
    </div>
    """, unsafe_allow_html=True)

    if report_data:
        rf_report = report_data.get("Random Forest", {})

        # Top-line metrics
        st.markdown('<div class="section-card">'
                    '<div class="section-title">🏆 Overall Metrics – Random Forest (tuned)</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        acc = rf_report.get("accuracy", 0)
        wa  = rf_report.get("weighted avg", {})
        m1.metric("Accuracy",  f"{acc*100:.2f}%")
        m2.metric("Precision", f"{wa.get('precision',0)*100:.2f}%")
        m3.metric("Recall",    f"{wa.get('recall',0)*100:.2f}%")
        m4.metric("F1-Score",  f"{wa.get('f1-score',0)*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # Per-class table
        st.markdown('<div class="section-card">'
                    '<div class="section-title">📋 Per-Class Metrics</div>',
                    unsafe_allow_html=True)
        rows = []
        for cls in ["Normal", "Suspect", "Pathologic"]:
            d = rf_report.get(cls, {})
            rows.append({
                "Class":     cls,
                "Precision": f"{d.get('precision',0)*100:.2f}%",
                "Recall":    f"{d.get('recall',0)*100:.2f}%",
                "F1-Score":  f"{d.get('f1-score',0)*100:.2f}%",
                "Support":   int(d.get("support", 0)),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Confusion matrix image
        try:
            st.markdown('<div class="section-card">'
                        '<div class="section-title">🔲 Confusion Matrix</div>',
                        unsafe_allow_html=True)
            st.image("confusion_matrix.png", use_container_width=False, width=480)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            st.info("Confusion matrix image not found. Run model_training.py first.")

    else:
        st.warning("classification_report.json not found. Run model_training.py first.")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 – DATASET INFO
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖 Dataset Info":

    st.markdown("""
    <div class="header-banner">
      <h1>📖 Dataset Information</h1>
      <p>Cardiotocography (CTG) Dataset – UCI Machine Learning Repository</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">📌 Overview</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        | Attribute | Value |
        |-----------|-------|
        | **Source** | UCI ML Repository |
        | **Instances** | 2,126 |
        | **Features** | 21 numeric |
        | **Target** | NSP (3 classes) |
        | **License** | CC BY 4.0 |
        """)
    with col_b:
        # Class balance pie chart
        labels = ["Normal (78%)", "Suspect (14%)", "Pathologic (9%)"]
        sizes  = [0.778, 0.136, 0.086]
        colors = ["#28a745", "#ffc107", "#dc3545"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=140, textprops={"fontsize": 9})
        ax.set_title("Class Distribution", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-title">🔢 Feature Dictionary</div>',
                unsafe_allow_html=True)
    feature_info = {
        "LB":       "FHR baseline (bpm)",
        "AC":       "Accelerations / sec",
        "FM":       "Fetal movements / sec",
        "UC":       "Uterine contractions / sec",
        "ASTV":     "% Abnormal short-term variability",
        "MSTV":     "Mean short-term variability",
        "ALTV":     "% Abnormal long-term variability",
        "MLTV":     "Mean long-term variability",
        "Width":    "Width of FHR histogram",
        "Min":      "Minimum FHR",
        "Max":      "Maximum FHR",
        "Nmax":     "Number of histogram peaks",
        "Nzeros":   "Number of zeros in histogram",
        "Mode":     "Mode of FHR histogram",
        "Mean":     "Mean FHR",
        "Median":   "Median FHR",
        "Variance": "Variance of FHR",
        "Tendency": "Trend of FHR  (-1 = ↓, 0 = →, 1 = ↑)",
        "NSP":      "Target: 1=Normal, 2=Suspect, 3=Pathologic",
    }
    feat_df = pd.DataFrame(
        [(k, v) for k, v in feature_info.items()],
        columns=["Feature", "Description"],
    )
    st.dataframe(feat_df.set_index("Feature"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)