import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.inference import InferenceError, estimate_thresholds_from_data

st.set_page_config(page_title="VT1/VT2 Threshold Estimator", page_icon="\U0001FAC0")

st.title("Exercise Threshold Estimation from HRV")
st.write(
    "Upload raw RR-interval data from a graded exercise test to estimate VT1 and VT2 "
    "(ventilatory thresholds) directly from heart rate variability — no gas exchange "
    "measurement needed."
)

uploaded = st.file_uploader("RR interval data (CSV or Excel)", type=["csv", "xlsx", "xls"])
model_name = st.radio(
    "Model",
    ["rf", "xgboost"],
    format_func=lambda m: "Random Forest" if m == "rf" else "XGBoost",
    horizontal=True,
)
st.caption("Required columns: `time` (s), `RR` (ms), `power` (W)")

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            data = pd.read_excel(uploaded)
        else:
            data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    if st.button("Estimate thresholds", type="primary"):
        with st.spinner("Extracting HRV features and classifying windows..."):
            try:
                vt1_power, vt1_hr, vt2_power, vt2_hr = estimate_thresholds_from_data(
                    data, model_name=model_name
                )
            except InferenceError as e:
                st.error(str(e))
                st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("VT1 power", f"{vt1_power:.0f} W" if vt1_power is not None else "not detected")
            st.metric("VT1 heart rate", f"{vt1_hr:.0f} bpm" if vt1_hr is not None else "—")
        with col2:
            st.metric("VT2 power", f"{vt2_power:.0f} W" if vt2_power is not None else "not detected")
            st.metric("VT2 heart rate", f"{vt2_hr:.0f} bpm" if vt2_hr is not None else "—")

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(data["time"], data["RR"], color="tab:blue", linewidth=0.7)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("RR (ms)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(data["time"], data["power"], color="tab:red")
        ax2.set_ylabel("Power (W)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        if vt1_power is not None:
            ax2.axhline(y=vt1_power, color="tab:purple", linestyle="--", linewidth=1.5, label="VT1")
        if vt2_power is not None:
            ax2.axhline(y=vt2_power, color="tab:orange", linestyle="--", linewidth=1.5, label="VT2")
        if vt1_power is not None or vt2_power is not None:
            ax2.legend(loc="upper left")

        fig.tight_layout()
        st.pyplot(fig)
