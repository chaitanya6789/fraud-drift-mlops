import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MLOps Fraud Drift Monitor",
    page_icon="💳",
    layout="wide",
)

st.title("🛡️ Selective Concept Drift Mitigation in Financial Fraud Detection")
st.markdown(
    "**Heterogeneous Feature-Partitioned Sub-Ensemble with PSI Statistical Drift"
    " Monitoring**"
)

# Resolve Base Path Dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(
    BASE_DIR, "fraud-drift-mlops", "artifacts"
) if os.path.exists(
    os.path.join(BASE_DIR, "fraud-drift-mlops", "artifacts")
) else os.path.join(BASE_DIR, "artifacts")


@st.cache_resource
def load_models():
  m1 = joblib.load(os.path.join(ARTIFACTS_DIR, "submodel_1_rf.joblib"))
  m2 = joblib.load(os.path.join(ARTIFACTS_DIR, "submodel_2_xgb.joblib"))
  m3 = joblib.load(os.path.join(ARTIFACTS_DIR, "submodel_3_dt.joblib"))
  m3_up = joblib.load(
      os.path.join(ARTIFACTS_DIR, "submodel_3_dt_updated.joblib")
  )
  scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "robust_scaler.joblib"))
  return m1, m2, m3, m3_up, scaler


try:
  m1, m2, m3, m3_up, scaler = load_models()
  st.sidebar.success("✅ Models Loaded Successfully")
  models_loaded = True
except Exception as e:
  st.sidebar.error(f"Error loading models: {e}")
  models_loaded = False

tab1, tab2 = st.tabs(["🔍 Live Prediction", "📊 Research Benchmarks"])

with tab1:
  st.subheader("Simulate a Transaction")
  raw_amount = st.number_input(
      "Transaction Amount ($/₹)", min_value=0.0, max_value=50000.0, value=120.0
  )
  v1 = st.slider("V1 (User Profile Index)", -5.0, 5.0, 0.0)
  v10 = st.slider("V10 (Terminal Security Score)", -5.0, 5.0, 0.0)
  v20 = st.slider("V20 (Velocity / Frequency)", -5.0, 5.0, 0.0)

  if models_loaded:
    scaled_amt = scaler.transform([[raw_amount]])[0][0]

    g1_vals = [v1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    g2_vals = [v10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    g3_vals = [v20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, scaled_amt]

    if st.button("Check Transaction", type="primary"):
      p1 = m1.predict_proba([g1_vals])[0, 1]
      p2 = m2.predict_proba([g2_vals])[0, 1]
      p3 = m3_up.predict_proba([g3_vals])[0, 1]

      final_score = (0.30 * p1) + (0.45 * p2) + (0.25 * p3)

      st.metric("Ensemble Fraud Probability", f"{final_score * 100:.2f}%")

      if final_score >= 0.50:
        st.error("🚨 FRAUD DETECTED: Transaction Blocked!")
      else:
        st.success("🟢 GENUINE: Transaction Approved!")
  else:
    st.warning("⚠️ Please upload the 'artifacts' folder to GitHub to run predictions.")

with tab2:
  st.subheader("IEEE Table I: Benchmark Results")
  benchmark_df = pd.DataFrame([
      {
          "Strategy": "Strategy A: Static (No Retraining)",
          "Precision": "0.9130",
          "Recall": "0.7500",
          "F1-Score": "0.8235",
          "Latency": "0.0000 s",
      },
      {
          "Strategy": "Strategy B: Global Retraining",
          "Precision": "1.0000",
          "Recall": "1.0000",
          "F1-Score": "1.0000",
          "Latency": "5.7067 s",
      },
      {
          "Strategy": "Strategy C: Proposed Selective",
          "Precision": "0.9038",
          "Recall": "0.8393",
          "F1-Score": "0.8704",
          "Latency": "0.4623 s",
      },
  ])
  st.table(benchmark_df)
  st.info("⚡ Compute Latency Reduction: 91.90%")
