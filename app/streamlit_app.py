from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

from inference import LEADS, load_models, load_thresholds, predict_ecg

st.set_page_config(page_title="CardioAI | ECG Research", page_icon="❤️", layout="wide")
st.title("❤️ CardioAI")
st.caption("12-lead ECG · MI detection and multi-label territory estimation")
st.warning(
    "Research prototype only. Not a medical device. A negative model output cannot "
    "exclude acute MI. Suspected heart-attack symptoms require emergency medical care."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def initialize_models():
    detector, localizer, _, _ = load_models(PROJECT_ROOT, DEVICE)
    detection_threshold, localization_thresholds = load_thresholds(PROJECT_ROOT)
    return detector, localizer, detection_threshold, localization_thresholds

with st.sidebar:
    st.header("System")
    st.write(f"Compute device: `{DEVICE}`")
    st.write("Input: 12 leads · 500 Hz · 10 seconds")
    st.caption("CSV columns must be ordered I, II, III, aVR, aVL, aVF, V1–V6.")

ecg_tab, symptom_tab, about_tab = st.tabs(["ECG analysis", "Patient-reported symptoms", "About"])

with ecg_tab:
    uploaded = st.file_uploader("Upload ECG CSV (5000 × 12 numeric samples)", type=["csv"])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            numeric = frame.select_dtypes(include=[np.number])
            signal = numeric.to_numpy(dtype=np.float32)
            if signal.shape == (12, 5000):
                signal = signal.T
            if signal.shape != (5000, 12):
                st.error(f"CSV numeric shape is {signal.shape}; expected (5000, 12) or (12, 5000).")
            elif not np.isfinite(signal).all():
                st.error("ECG contains NaN or infinite values.")
            else:
                st.success("ECG shape and numeric values validated.")
                time = np.arange(5000) / 500.0
                fig = make_subplots(rows=6, cols=2, subplot_titles=LEADS, vertical_spacing=0.07)
                for i, lead in enumerate(LEADS):
                    fig.add_trace(
                        go.Scatter(x=time, y=signal[:, i], mode="lines", name=lead, showlegend=False),
                        row=i // 2 + 1, col=i % 2 + 1,
                    )
                    fig.update_xaxes(title_text="Time (s)", row=i // 2 + 1, col=i % 2 + 1)
                fig.update_layout(height=950, title="12-lead ECG waveform")
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Run MI detection and localization", type="primary"):
                    try:
                        detector, localizer, detection_threshold, localization_thresholds = initialize_models()
                        result = predict_ecg(
                            signal.T, detector, localizer, DEVICE,
                            detection_threshold, localization_thresholds,
                        )
                        st.subheader("MI detection")
                        c1, c2 = st.columns(2)
                        c1.metric("Model probability", f"{result['mi_probability']:.1%}")
                        c2.metric("Validation-selected threshold", f"{result['mi_threshold']:.2f}")
                        if result["mi_positive"]:
                            st.warning(result["mi_prediction"])
                        else:
                            st.info(result["mi_prediction"] + " — this does not rule out MI.")

                        st.subheader("Territory estimates")
                        output_df = pd.DataFrame(result["territories"])
                        output_df["probability"] = output_df["probability"].map(lambda x: f"{x:.1%}")
                        output_df["threshold"] = output_df["threshold"].map(lambda x: f"{x:.2f}")
                        output_df["model output"] = output_df["positive"].map({True: "Above threshold", False: "Below threshold"})
                        st.dataframe(output_df[["territory", "probability", "threshold", "model output"]], use_container_width=True, hide_index=True)
                        st.caption(
                            "Territory estimates are multi-label outputs from the trained PTB-XL model; "
                            "they are not imaging-confirmed anatomical diagnoses."
                        )
                    except Exception as exc:
                        st.error(f"Inference could not run: {exc}")
                        st.exception(exc)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

with symptom_tab:
    st.subheader("Patient-reported symptoms")
    st.caption("These are self-reported observations and are not detected by the ECG model.")
    symptoms = st.multiselect(
        "Select reported symptoms",
        ["Chest discomfort or pressure", "Arm/shoulder/back/neck/jaw pain",
         "Shortness of breath", "Cold sweat", "Nausea or vomiting",
         "Dizziness", "Unusual fatigue"],
    )
    onset = st.selectbox("Symptom onset", ["Unknown", "Within 1 hour", "1–6 hours", "More than 6 hours", "Intermittent"])
    if st.button("Record symptoms"):
        st.write("Reported symptoms:", ", ".join(symptoms) if symptoms else "None selected")
        st.write("Reported onset:", onset)
        st.warning("If a heart attack is possible, seek emergency medical care immediately. Do not wait for this app.")

with about_tab:
    st.markdown("""
    **Project:** Myocardial Infarction Detection and Localization using Deep Learning on PTB-XL.

    - Detection: ResNet1D binary classifier.
    - Localization: compact ResNet1D with seven sigmoid outputs.
    - Detection preprocessing: raw WFDB signal in physical units.
    - Localization preprocessing: per-lead z-score normalization.
    - Decision thresholds: selected using validation data in the training notebooks.

    PTB-XL SCP-derived labels are ECG diagnostic annotations, not independent anatomical ground truth.
    """)
