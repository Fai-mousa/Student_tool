"""
student_tool/student.py
=======================
Streamlit app: Individual Student Grade Predictor.

Enter G1, G2, and study/lifestyle features → predict G3 →
compute total score → show PASS / FAIL verdict.

Run:
    streamlit run student.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Import shared helpers from train.py (same directory) ──────────────────────
# This guarantees preprocessing is 100% identical to training.
# The model now trains natively on the school scale, so G1/G2 are passed
# directly as 0-25 and the model outputs G3 directly as 0-50.
sys.path.insert(0, os.path.dirname(__file__))
from train import (  # noqa: E402
    preprocess_inference,
    compute_verdict,
    PASS_THRESHOLD,
    G3_MAX,
    MODEL_PATH,
    SCALER_PATH,
    COLUMNS_PATH,
)
from recommender import get_recommendations  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🧑‍🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --navy:   #1e3a5f;
    --teal:   #0f766e;
    --teal2:  #14b8a6;
    --amber:  #d97706;
    --ink:    #111827;
    --muted:  #6b7280;
    --bg:     #f8fafc;
    --white:  #ffffff;
    --pass-dark: #064e3b;
    --fail-dark: #7f1d1d;
    --shadow: 0 2px 16px rgba(0,0,0,.07);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg);
    color: var(--ink);
}
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--navy) !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stMarkdown a { color: var(--teal2) !important; }

/* ── Page header ── */
.page-header {
    background: linear-gradient(120deg, #1e3a5f 0%, #134e4a 100%);
    border-radius: 14px;
    padding: 32px 36px;
    margin-bottom: 24px;
}
.page-header h1 {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px;
}
.page-header p {
    color: #94d5cf;
    font-size: .95rem;
    margin: 0;
    font-weight: 300;
}

/* ── Section card ── */
.section {
    background: var(--white);
    border-radius: 12px;
    padding: 20px 20px 12px;
    box-shadow: var(--shadow);
    margin-bottom: 16px;
}
.section-title {
    font-family: 'Sora', sans-serif;
    font-size: .95rem;
    font-weight: 600;
    color: var(--navy);
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 8px;
    margin-bottom: 14px;
}

/* ── Result — PASS ── */
.result-pass {
    background: linear-gradient(135deg, var(--pass-dark) 0%, #065f46 100%);
    border-radius: 16px;
    padding: 36px 28px;
    text-align: center;
    box-shadow: 0 8px 28px rgba(6,78,59,.35);
    animation: pop .4s cubic-bezier(.175,.885,.32,1.275);
}
/* ── Result — FAIL ── */
.result-fail {
    background: linear-gradient(135deg, var(--fail-dark) 0%, #991b1b 100%);
    border-radius: 16px;
    padding: 36px 28px;
    text-align: center;
    box-shadow: 0 8px 28px rgba(127,29,29,.35);
    animation: pop .4s cubic-bezier(.175,.885,.32,1.275);
}
@keyframes pop {
    0%   { transform: scale(.88); opacity: 0; }
    100% { transform: scale(1);   opacity: 1; }
}
.res-emoji  { font-size: 3.2rem; margin-bottom: 8px; }
.res-label  { font-family:'Sora',sans-serif; font-size:2.4rem;
              font-weight:700; color:#fff; margin:0; }
.res-sub    { font-size:.88rem; color:rgba(255,255,255,.7);
              margin-top:4px; }
.res-pill   { display:inline-block; background:rgba(255,255,255,.15);
              color:#fff; border-radius:100px; padding:4px 16px;
              font-size:.82rem; font-weight:600; margin-top:12px; }

/* ── Score grid ── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 16px 0;
}
.score-box {
    background: var(--white);
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
    box-shadow: var(--shadow);
}
.score-lbl  { font-size:.68rem; color:var(--muted); text-transform:uppercase;
              letter-spacing:.07em; margin-bottom:3px; }
.score-val  { font-family:'Sora',sans-serif; font-size:1.5rem;
              font-weight:700; color:var(--navy); }
.score-max  { font-size:.7rem; color:var(--muted); }

/* ── Tip box ── */
.tip {
    background: #fefce8;
    border-left: 4px solid var(--amber);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: .82rem;
    color: #92400e;
    margin-top: 12px;
    line-height: 1.65;
}

/* ── Placeholder ── */
.placeholder {
    background: var(--white);
    border-radius: 14px;
    padding: 50px 24px;
    text-align: center;
    box-shadow: var(--shadow);
}

/* ── Button ── */
.stButton > button {
    background: var(--teal) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    width: 100%;
    padding: 12px !important;
    font-size: .95rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── Importance bar ── */
.fi-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.fi-lbl { width:140px; font-size:.79rem; font-weight:500;
          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.fi-bg  { flex:1; background:#f1f5f9; border-radius:100px; height:7px; }
.fi-bar { height:7px; border-radius:100px; background:var(--teal); }
.fi-val { width:36px; text-align:right; font-size:.74rem; color:var(--muted); }

/* ── Recommendations ── */
.rec-section { margin-top: 28px; }
.rec-header {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: var(--navy);
    margin-bottom: 16px;
}
.rec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.rec-card {
    background: var(--white);
    border-radius: 12px;
    padding: 16px;
    box-shadow: var(--shadow);
    border-top: 3px solid var(--teal2);
}
.rec-card-title {
    font-family: 'Sora', sans-serif;
    font-size: .75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    color: var(--teal); margin-bottom: 10px;
}
.rec-item {
    display: flex; gap: 8px; align-items: flex-start;
    margin-bottom: 8px; font-size: .82rem;
    line-height: 1.6; color: #374151;
}
.rec-item:last-child { margin-bottom: 0; }
.rec-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.rec-card.academic   { border-top-color: #3b82f6; }
.rec-card.study      { border-top-color: #8b5cf6; }
.rec-card.lifestyle  { border-top-color: #10b981; }
.rec-card.motivation { border-top-color: #f59e0b; }
.rec-card.academic   .rec-card-title { color: #2563eb; }
.rec-card.study      .rec-card-title { color: #7c3aed; }
.rec-card.lifestyle  .rec-card-title { color: #059669; }
.rec-card.motivation .rec-card-title { color: #d97706; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 20px;">
        <div style="font-size:2.6rem;">🧑‍🎓</div>
        <div style="font-family:'Sora',sans-serif;font-size:1.2rem;
                    font-weight:700;color:#fff;margin-top:6px;">
            Student Tool
        </div>
        <div style="font-size:.76rem;color:#94a3b8;margin-top:2px;">
            Grade Predictor
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Grading rules ────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:.7rem;font-weight:600;letter-spacing:.12em;
                text-transform:uppercase;color:#64748b;margin-bottom:8px;">
        Grading Rules
    </div>
    <div style="font-size:.82rem;color:#cbd5e1;line-height:1.9;">
        G1, G2 → scale <strong style="color:#fff">0 – 25</strong><br>
        G3 (predicted) → <strong style="color:#fff">0 – 50</strong><br>
        Total = G1 + G2 + G3<br>
        <span style="color:#4ade80;">✓ PASS</span> if total
        <strong style="color:#fff">≥ 60</strong><br>
        <span style="color:#f87171;">✗ FAIL</span> otherwise
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size:.76rem;color:#94a3b8;line-height:1.7;">
        🔒 <strong style="color:#cbd5e1">Privacy-first</strong><br>
        School, address, alcohol &amp; relationship
        data are never used in predictions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.7rem;color:#475569;text-align:center;">
        UCI Student Performance Dataset<br>
        Streamlit &amp; scikit-learn
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artefacts():
    """Load and cache model, scaler, and feature columns."""
    missing = [p for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]
               if not os.path.exists(p)]
    if missing:
        return None, None, None
    return (joblib.load(MODEL_PATH),
            joblib.load(SCALER_PATH),
            joblib.load(COLUMNS_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-header">
    <h1>🧑‍🎓 Student Grade Predictor</h1>
    <p>Enter your grades and study profile to get an instant G3 prediction with pass/fail verdict.</p>
</div>
""", unsafe_allow_html=True)

model, scaler, feature_columns = load_artefacts()

if model is None:
    st.error(
        "⚠️ Model files not found. "
        "Please run `python train.py` first, then restart the app."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FORM  +  RESULT  (side-by-side layout)
# ─────────────────────────────────────────────────────────────────────────────

col_form, col_result = st.columns([1.1, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT: INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────

with col_form:

    # ── Current grades ────────────────────────────────────────────────────
    st.markdown('<div class="section"><div class="section-title">📝 Current Grades</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        g1_display = st.number_input(
            "G1 – First Period (0–25)", min_value=0.0, max_value=25.0,
            value=13.0, step=0.5,
            help="Your first period grade on a 0-25 scale"
        )
    with c2:
        g2_display = st.number_input(
            "G2 – Second Period (0–25)", min_value=0.0, max_value=25.0,
            value=13.0, step=0.5,
            help="Your second period grade on a 0-25 scale"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Personal profile ──────────────────────────────────────────────────
    st.markdown('<div class="section"><div class="section-title">👤 Personal Profile</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sex  = st.selectbox("Sex", ["M", "F"])
        age  = st.slider("Age", 15, 22, 17)
        Medu = st.selectbox("Mother's Education",
            [0,1,2,3,4], index=2,
            format_func=lambda x: {0:"None",1:"Primary (4th)",
                2:"Primary (9th)",3:"Secondary",4:"Higher"}[x])
        Fedu = st.selectbox("Father's Education",
            [0,1,2,3,4], index=2,
            format_func=lambda x: {0:"None",1:"Primary (4th)",
                2:"Primary (9th)",3:"Secondary",4:"Higher"}[x])
    with c2:
        Mjob     = st.selectbox("Mother's Job",
                                ["teacher","health","services","at_home","other"])
        Fjob     = st.selectbox("Father's Job",
                                ["teacher","health","services","at_home","other"])
        guardian = st.selectbox("Guardian", ["mother","father","other"])
        reason   = st.selectbox("School Choice Reason",
                                ["home","reputation","course","other"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Study habits ──────────────────────────────────────────────────────
    st.markdown('<div class="section"><div class="section-title">📚 Study Habits</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        studytime = st.select_slider("Weekly Study Time", options=[1,2,3,4],
            format_func=lambda x:{1:"<2 hrs",2:"2–5 hrs",
                                   3:"5–10 hrs",4:">10 hrs"}[x], value=2)
        failures  = st.selectbox("Past Failures", [0,1,2,3],
            format_func=lambda x: f"{x} failure{'s' if x!=1 else ''}")
        absences  = st.number_input("Absences", 0, 93, 4, step=1)
    with c2:
        traveltime = st.select_slider("Travel Time to School",
            options=[1,2,3,4],
            format_func=lambda x:{1:"<15 min",2:"15–30 min",
                                   3:"30–60 min",4:">60 min"}[x])
        schoolsup  = st.radio("School Support", ["no","yes"], horizontal=True)
        famsup     = st.radio("Family Support",  ["no","yes"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Lifestyle ─────────────────────────────────────────────────────────
    st.markdown('<div class="section"><div class="section-title">🌿 Lifestyle & Wellbeing</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        internet   = st.radio("Internet at Home",       ["no","yes"], horizontal=True, index=1)
        higher     = st.radio("Wants Higher Education", ["no","yes"], horizontal=True, index=1)
        activities = st.radio("Extra-Curricular",       ["no","yes"], horizontal=True)
    with c2:
        famrel   = st.slider("Family Relations (1–5)", 1, 5, 4)
        freetime = st.slider("Free Time (1–5)",        1, 5, 3)
        health   = st.slider("Health Status (1–5)",    1, 5, 4)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────
    predict_clicked = st.button("🔮  Predict My Final Grade")

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT: RESULT PANEL
# ─────────────────────────────────────────────────────────────────────────────

with col_result:
    st.markdown("### 📋 Your Prediction")

    if predict_clicked:

        # ── Validate ──────────────────────────────────────────────────────
        errors = []
        if not 0 <= g1_display <= 25:
            errors.append("G1 must be between 0 and 25.")
        if not 0 <= g2_display <= 25:
            errors.append("G2 must be between 0 and 25.")
        for e in errors:
            st.error(e)

        if not errors:
            # G1/G2 are already on the school scale (0-25).
            # The model was trained on the school scale, so pass them directly.
            raw_input = {
                "sex":sex, "age":age, "Medu":Medu, "Fedu":Fedu,
                "Mjob":Mjob, "Fjob":Fjob, "reason":reason, "guardian":guardian,
                "traveltime":traveltime, "studytime":studytime,
                "failures":failures, "schoolsup":schoolsup, "famsup":famsup,
                "activities":activities, "higher":higher, "internet":internet,
                "famrel":famrel, "freetime":freetime, "health":health,
                "absences":absences,
                "G1": g1_display,   # school scale 0-25, passed directly
                "G2": g2_display,   # school scale 0-25, passed directly
            }

            df_input = pd.DataFrame([raw_input])

            # Preprocess using the exact same pipeline as training
            X = preprocess_inference(df_input, scaler, feature_columns)

            # Model predicts G3 directly on school scale (0-50) — no conversion needed
            g3_predicted = round(float(np.clip(model.predict(X)[0], 0, G3_MAX)), 1)
            total, passed = compute_verdict(g1_display, g2_display, g3_predicted)

            # ── Score breakdown ────────────────────────────────────────
            st.markdown(f"""
            <div class="score-grid">
              <div class="score-box">
                <div class="score-lbl">G1</div>
                <div class="score-val">{g1_display:.1f}</div>
                <div class="score-max">/ 25</div>
              </div>
              <div class="score-box">
                <div class="score-lbl">G2</div>
                <div class="score-val">{g2_display:.1f}</div>
                <div class="score-max">/ 25</div>
              </div>
              <div class="score-box">
                <div class="score-lbl">G3 (predicted)</div>
                <div class="score-val">{g3_predicted:.1f}</div>
                <div class="score-max">/ 50</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Verdict banner ─────────────────────────────────────────
            shortage = round(PASS_THRESHOLD - total, 1)
            if passed:
                st.markdown(f"""
                <div class="result-pass">
                  <div class="res-emoji">🎉</div>
                  <div class="res-label">PASS</div>
                  <div class="res-sub">
                    Final Grade: <strong>{g3_predicted:.1f} / 50</strong>
                  </div>
                  <div class="res-sub" style="margin-top:4px;">
                    Total Score: <strong>{total:.1f} / 75</strong>
                  </div>
                  <div class="res-pill">Total ≥ {PASS_THRESHOLD} ✓</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fail">
                  <div class="res-emoji">⚠️</div>
                  <div class="res-label">FAIL</div>
                  <div class="res-sub">
                    Final Grade: <strong>{g3_predicted:.1f} / 50</strong>
                  </div>
                  <div class="res-sub" style="margin-top:4px;">
                    Total Score: <strong>{total:.1f} / 100</strong>
                  </div>
                  <div class="res-pill">
                    {shortage} pts needed to pass
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── AI-Powered Personalised Recommendations ────────────────
            st.markdown('<div class="rec-section">', unsafe_allow_html=True)
            st.markdown(
                '<div class="rec-header">🤖 Your Personalised Recommendations</div>',
                unsafe_allow_html=True,
            )

            recs = get_recommendations(raw_input, g3_predicted, total, passed)

            if recs:
                CATEGORIES = [
                    ("academic",   "📖 Academic",      "academic"),
                    ("study",      "⏰ Study Habits",   "study"),
                    ("lifestyle",  "🌿 Lifestyle",      "lifestyle"),
                    ("motivation", "🌟 Motivation",     "motivation"),
                ]

                # Build the 2x2 card grid
                html_cards = '<div class="rec-grid">'
                for key, label, css_class in CATEGORIES:
                    items = recs.get(key, [])
                    items_html = "".join(
                        f'<div class="rec-item">'
                        f'<span class="rec-icon">{item.get("icon","💡")}</span>'
                        f'<span>{item.get("tip","")}</span>'
                        f'</div>'
                        for item in items
                    )
                    html_cards += (
                        f'<div class="rec-card {css_class}">'
                        f'<div class="rec-card-title">{label}</div>'
                        f'{items_html}'
                        f'</div>'
                    )
                html_cards += '</div>'
                st.markdown(html_cards, unsafe_allow_html=True)
            else:
                st.info("Could not generate recommendations. Please try again.")

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── Placeholder ────────────────────────────────────────────────
        st.markdown("""
        <div class="placeholder">
          <div style="font-size:2.8rem;margin-bottom:10px;">🔮</div>
          <div style="font-family:'Sora',sans-serif;font-size:1.25rem;
                      font-weight:600;color:#1e3a5f;margin-bottom:6px;">
            Ready to predict
          </div>
          <div style="font-size:.87rem;color:#6b7280;line-height:1.65;">
            Complete the form and click<br>
            <strong>Predict My Final Grade</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature importance (always visible) ───────────────────────────────
    if os.path.exists("feature_importance.pkl"):
        import joblib as _jl
        fi = _jl.load("feature_importance.pkl")
        st.markdown("---")
        st.markdown("### 📈 What Drives G3")
        top = fi.head(10)
        mx  = top.iloc[0]
        rows = ""
        for feat, val in top.items():
            pct   = val / mx * 100
            label = feat.replace("_", " ").title()
            rows += f"""
            <div class="fi-row">
              <div class="fi-lbl" title="{label}">{label}</div>
              <div class="fi-bg">
                <div class="fi-bar" style="width:{pct:.0f}%"></div>
              </div>
              <div class="fi-val">{val:.3f}</div>
            </div>"""
        st.markdown(rows, unsafe_allow_html=True)