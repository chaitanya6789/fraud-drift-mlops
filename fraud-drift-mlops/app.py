import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cloud MLOps Concept Drift Mitigation Engine",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Cloud-Native Adaptive MLOps Concept Drift Mitigation Engine")
st.markdown("**Event-Driven Feature-Partitioned Sub-Ensemble Selective Retraining**")

# Automatically locate the artifacts folder regardless of folder nesting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE_DIR, "artifacts")):
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
elif os.path.exists(os.path.join(BASE_DIR, "fraud-drift-mlops", "artifacts")):
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "fraud-drift-mlops", "artifacts")
else:
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

@st.cache_resource
def load_all_artifacts():
    m1 = joblib.load(os.path.join(ARTIFACTS_DIR, "submodel_1_rf.joblib"))
    m2 = joblib.load(os.path.join(ARTIFACTS_DIR, "submodel_2_xgb.joblib"))
    m3 = joblib.load(os.path.join(ARTIFACTS_DIR, "submodel_3_dt.joblib"))
    m3_up = joblib.load(os.path.join(ARTIFACTS_DIR, "submodel_3_dt_updated.joblib"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "robust_scaler.joblib"))
    return m1, m2, m3, m3_up, scaler

try:
    m1, m2, m3, m3_up, scaler = load_all_artifacts()
    st.sidebar.success("✅ Models & Registry Loaded")
    models_ready = True
except Exception as err:
    st.sidebar.error(f"Artifact loading error: {err}")
    models_ready = False

# Sidebar Controls
st.sidebar.header("⚙️ Streaming & Drift Simulation")
uploaded_file = st.sidebar.file_uploader("Upload Transaction Batch (CSV)", type=["csv"])
selected_partition = st.sidebar.selectbox(
    "Inject Drift into Feature Group",
    ["Partition 3 (Sub-Model 3)", "Partition 1 (Sub-Model 1)", "Partition 2 (Sub-Model 2)"]
)
drift_mult = st.sidebar.slider("Drift Severity Multiplier", 1.0, 5.0, 2.0)
psi_threshold = st.sidebar.slider("PSI Alarm Threshold", 0.10, 0.50, 0.25)
execute_btn = st.sidebar.button("🚀 Execute Live MLOps Pipeline", type="primary")

# Main Interface Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Live Streaming Inference", "📊 Research Benchmarks & Diagrams", "🏗️ Architecture Blueprint"])

with tab1:
    st.subheader("Simulate Incoming Transaction Stream")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### **Partition 1: User Identity (V1–V9)**")
        v1 = st.slider("V1 (Profile Habit)", -5.0, 5.0, 0.0)
        v2 = st.slider("V2 (Behavior Index)", -5.0, 5.0, 0.0)
        v3 = st.slider("V3 (Spending Baseline)", -5.0, 5.0, 0.0)
        g1_input = [v1, v2, v3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    with col2:
        st.markdown("##### **Partition 2: Terminal & Channel (V10–V19)**")
        v10 = st.slider("V10 (Channel Trust)", -5.0, 5.0, 0.0)
        v11 = st.slider("V11 (Terminal Reliability)", -5.0, 5.0, 0.0)
        v12 = st.slider("V12 (Token Validity)", -5.0, 5.0, 0.0)
        g2_input = [v10, v11, v12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    with col3:
        st.markdown("##### **Partition 3: Amount & Velocity (V20–V28)**")
        raw_amt = st.number_input("Transaction Amount ($/₹)", 0.0, 50000.0, 120.0)
        v20 = st.slider("V20 (Transaction Frequency)", -5.0, 5.0, 0.0)
        
    use_retrained = st.toggle("Active Model Registry: Use Retrained Sub-Model 3 (Strategy C)", value=True)
    
    if models_ready:
        scaled_amount = scaler.transform([[raw_amt]])[0][0]
        g3_input = [v20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, scaled_amount]
        active_m3 = m3_up if use_retrained else m3

        if st.button("Evaluate Transaction Risk", type="primary"):
            p1 = m1.predict_proba([g1_input])[0, 1]
            p2 = m2.predict_proba([g2_input])[0, 1]
            p3 = active_m3.predict_proba([g3_input])[0, 1]

            final_risk = (0.30 * p1) + (0.45 * p2) + (0.25 * p3)

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Sub-Model 1 (RF)", f"{p1:.4f}")
            m_col2.metric("Sub-Model 2 (XGB)", f"{p2:.4f}")
            m_col3.metric("Sub-Model 3 (DT)", f"{p3:.4f}")
            m_col4.metric("Ensemble Final Risk", f"{final_risk:.4f}")

            if final_risk >= 0.50:
                st.error(f"🚨 **PREDICTION: FRAUD DETECTED (Risk: {final_risk * 100:.2f}%) — Transaction Intercepted!**")
            else:
                st.success(f"🟢 **PREDICTION: GENUINE TRANSACTION (Risk: {final_risk * 100:.2f}%) — Approved.**")
    else:
        st.warning("⚠️ Waiting for model artifacts to initialize.")

with tab2:
    st.subheader("IEEE Research Benchmark Results (Table I)")
    benchmark_df = pd.DataFrame([
        {"MLOps Strategy": "Strategy A: Static Model (No Retraining)", "Precision": "0.9130", "Recall": "0.7500", "F1-Score": "0.8235", "ROC-AUC": "0.9535", "Compute Latency": "0.0000 s"},
        {"MLOps Strategy": "Strategy B: Global Retraining (29 Features)", "Precision": "1.0000", "Recall": "1.0000", "F1-Score": "1.0000", "ROC-AUC": "1.0000", "Compute Latency": "5.7067 s"},
        {"MLOps Strategy": "Strategy C: Proposed Selective Retraining", "Precision": "0.9038", "Recall": "0.8393", "F1-Score": "0.8704", "ROC-AUC": "0.9937", "Compute Latency": "0.4623 s"}
    ])
    st.table(benchmark_df)
    st.info("⚡ **Total Cloud Compute & Execution Latency Reduction: 91.90%** (0.4623 s vs 5.7067 s)")

    st.subheader("Publication Figures")
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        img_p1 = os.path.join(ARTIFACTS_DIR, "diagram_1_class_distribution.png")
        img_p3 = os.path.join(ARTIFACTS_DIR, "diagram_3_drift_audit.png")
        if os.path.exists(img_p1):
            st.image(img_p1, caption="Fig 1: Class Imbalance Distribution")
        if os.path.exists(img_p3):
            st.image(img_p3, caption="Fig 3: Partition-Level PSI Drift Audit")
    with fig_col2:
        img_p2 = os.path.join(ARTIFACTS_DIR, "diagram_2_baseline_models.png")
        img_p4 = os.path.join(ARTIFACTS_DIR, "diagram_4_master_benchmarks.png")
        if os.path.exists(img_p2):
            st.image(img_p2, caption="Fig 2: Baseline Sub-Model vs Ensemble Performance")
        if os.path.exists(img_p4):
            st.image(img_p4, caption="Fig 4: Master Evaluation Suite (Latency & Confusion Matrix)")

with tab3:
    st.subheader("MLOps Architecture & Drift Detection Workflow")
    st.markdown("""
    * **Feature Decoupling**: 29 transaction features segregated into Partition 1 ($V_1$–$V_9$), Partition 2 ($V_{10}$–$V_{19}$), and Partition 3 ($V_{20}$–$V_{28}$ + Amount).
    * **Independent PSI Monitoring**: The statistical engine computes Population Stability Index per partition. When $\\text{PSI} \\ge 0.25$, only the affected sub-model is retrained asynchronously.
    * **High Availability**: Non-drifted sub-models continue serving live traffic with zero pipeline downtime.
    """)
