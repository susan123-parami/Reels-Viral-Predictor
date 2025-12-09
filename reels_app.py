import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# -----------------------
# model load (your style)
# -----------------------
def load_viral_model():
    with open("viral_model.pkl", "rb") as f:
        pkg = pickle.load(f)
    # return in order: model, scaler, ohe, numeric_cols, categorical_cols
    return pkg["model"], pkg["scaler"], pkg["ohe"], pkg["numeric_cols"], pkg["categorical_cols"]

# -----------------------
# page config + sidebar
# -----------------------
st.set_page_config(page_title="Reel Viral Predictor", page_icon="🌸", layout="wide")

with st.sidebar:
    st.write("**Student Name:** May Thaw Tar")
    st.write("**Student ID:** PIUS20230081")
    st.write("**Professor:** Nwe Nwe Htay Win")
    st.write("---")
    st.write("Reel Analysis Dashboard")

# -----------------------
# title & inputs (your simple style)
# -----------------------
st.title("🌸 Reel Viral Predictor")
st.write("Enter your reel info to predict virality (before posting).")

col1, col2 = st.columns(2)
with col1:
    duration = st.slider("Reel length (seconds)", 1, 60, 15)
    hashtags = st.slider("Number of hashtags", 0, 20, 3)

    hook_strength = st.slider("Hook Strength (0.1 low → 0.9 strong)", 0.1, 0.9, 0.5, step=0.1)
with col2:
    niche = st.selectbox("Niche", ["Motivation","Tech","Travel","Gaming","Music","Education","Fitness","Comedy","Food","Beauty"])
    music_type = st.selectbox("Music type", ["viral track","trending","remix","original","no music"])

# -----------------------
# small helper to build df (same keys as training)
# -----------------------
def build_input_df():
    return pd.DataFrame([{
        "duration_sec": float(duration),
        "hook_strength_score": float(hook_strength),
        "niche": niche,
        "music_type": music_type
    }])

# -----------------------
# Predict (exactly like your preprocessing)
# -----------------------
def predict():
    # load model package
    model, scaler, ohe, numeric_cols, categorical_cols = load_viral_model()

    # build raw df
    X = build_input_df()

    # numeric scaling (your variable name)
    x_numeric = scaler.transform(X[numeric_cols])   # shape (1, n_numeric)

    # categorical OHE
    x_cat = ohe.transform(X[categorical_cols])      # shape (1, n_ohe)

    # combine exactly as in training
    x_final = np.concatenate([x_numeric, x_cat], axis=1)  # shape (1, total_features)

    # predict probability of class 1
    proba = model.predict_proba(x_final)[:,1][0]
    return proba, x_numeric, x_cat, numeric_cols, categorical_cols

# -----------------------
# UI: button + output
# -----------------------
if st.button("Predict Viral"):
    try:
        proba, x_numeric, x_cat, numeric_cols, categorical_cols = predict()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pct = proba * 100
    if proba >= 0.7:
        verdict = "High chance — Amplify & post!"
        color = "#16a34a"
    elif proba >= 0.4:
        verdict = "Medium chance — Tweak & re-test"
        color = "#f59e0b"
    else:
        verdict = "Low chance — Rework before posting"
        color = "#ef4444"

    # big colored badge (simple HTML)
    st.markdown(
        f"<div style='padding:18px;border-radius:12px;background:{color};color:white;text-align:center'>"
        f"<h1 style='margin:0'>{pct:.1f}%</h1>"
        f"<div style='opacity:0.95'>{verdict}</div></div>",
        unsafe_allow_html=True
    )

       # -----------------------------
    # Suggestions (ordered, one-line)
    # -----------------------------
    # Build rule-based suggestions with priorities
    suggestions = []  # list of (priority, title, short_description)

    # feature rules (higher number = higher priority)
    if hook_strength < 0.4:
        suggestions.append((10, "Improve your hook", "Show the outcome first or add a curiosity trigger in the first 3 seconds."))
    elif hook_strength >= 0.8:
        suggestions.append((6, "Amplify your hook", "Pin, crosspost, and engage early to maximise momentum."))

    if music_type in ["original", "no music"]:
        suggestions.append((8, "Try trending or viral audio", "Use a trending sound to boost discoverability."))

    if duration > 20:
        suggestions.append((7, "Shorten the reel", "Trim to ~10-15s to improve completion/retention."))

    # optional hashtags check — only if variable exists
    try:
        if hashtags <= 2:
            suggestions.append((4, "Add targeted hashtags", "Use 3–7 niche + broad tags to increase reach."))
    except Exception:
        # hashtags not in UI — skip
        pass

    # band-based suggestions
    if proba >= 0.70:
        suggestions.append((5, "Amplify & post", "Your reel is ready — pin the reel and engage early."))
    elif proba < 0.40:
        suggestions.append((9, "Rework before posting", "Consider redoing the hook and testing trending audio."))

    # Order suggestions by priority (desc) and remove duplicates by title
    ordered = []
    seen = set()
    for pr, title, desc in sorted(suggestions, key=lambda x: -x[0]):
        if title not in seen:
            ordered.append({"priority": pr, "title": title, "desc": desc})
            seen.add(title)

    # top_actions = first two titles
    top_actions = [s["title"] for s in ordered[:2]]

    # Display verdict and short bullets
    st.markdown("### Suggestions")
    if len(ordered) == 0:
        st.write("Your reel looks good — try testing different audio or hooks for marginal gains.")
    else:
        # show top actions bolded
        if top_actions:
            st.markdown("**Top actions:** " + ", ".join(top_actions))
        # show ordered one-line suggestions
        for s in ordered:
            st.markdown(f"- **{s['title']}** — {s['desc']}")

   
with st.expander("Why? (explanation)"):
    st.write("Your result is based on:")
    st.write("""
    • Your hook strength  
    • Reel duration  
    • Music type  
    • Niche category  
    • Logistic regression model trained on 400 reels  
    """)
