# 🛡️ Selective Concept Drift Mitigation in Financial Fraud Detection

An event-driven, feature-partitioned heterogeneous sub-ensemble framework designed to selectively detect and mitigate localized concept drift in financial transaction streams with minimal cloud computational overhead[cite: 1].

---

## 📌 Problem Overview
In production financial fraud detection pipelines, consumer spending patterns and adversarial fraud behaviors evolve dynamically, causing **Concept Drift**. Traditional MLOps architectures trigger global retraining across all features when drift occurs, leading to:
- High cloud compute expenses and extended latency ($5.7067\text{ s}$ execution time on baseline models)[cite: 1].
- Pipeline downtime and service disruption during full artifact replacement.
- Unnecessary re-computation across stationary, non-drifted feature spaces.

---

## 🏗️ Proposed Architecture

The 29-dimensional feature space is decomposed into three isolated, specialized sub-models aggregated via weighted soft voting


[ Incoming Transaction: 29 Features ]
                                     │
    ┌────────────────────────────────┼────────────────────────────────┐
    ▼                                ▼                                ▼
[ Partition 1: V1–V9 ]       [ Partition 2: V10–V19 ]      [ Partition 3: V20–V28 + Amount ]
│                                │                                │
Sub-Model 1:                     Sub-Model 2:                     Sub-Model 3:
Random Forest                       XGBoost                      Decision Tree
(User Profile)                   (Channel/Terminal)            (Velocity & Amount)
│                                │                                │
└────────────────────────────────┼────────────────────────────────┘
▼
[ Weighted Soft-Voting Aggregator ]
P_final = (0.30 * P1) + (0.45 * P2) + (0.25 * P3)


### Statistical Drift Engine (PSI)
Population Stability Index (PSI) evaluates each partition independently against the historical reference baseline:

$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - R_b \right) \times \ln\left( \frac{P_b}{R_b} \right)$$

- **$\text{PSI} < 0.10$**: 🟢 Stable
- **$0.10 \le \text{PSI} < 0.25$**: 🟡 Moderate Shift
- **$\text{PSI} \ge 0.25$**: 🚨 Severe Drift $\rightarrow$ Triggers targeted retraining of **only the affected sub-model**.

---

## 📊 Experimental Results & Benchmarks

Evaluated on $57,530$ transactions ($34,518$ reference baseline / $23,012$ live streaming test set).

### Drift Audit Diagnostics
- **Partition 1 ($V_1$–$V_9$):** $\text{PSI} = 0.0301$ (🟢 Stable)
- **Partition 2 ($V_{10}$–$V_{19}$):** $\text{PSI} = 0.1743$ (🟢 Stable)
- **Partition 3 ($V_{20}$–$V_{28}$ + Amount):** $\text{PSI} = 5.6628$ (🚨 Severe Drift Detected)

### Benchmark Comparison (IEEE Table I)

| MLOps Strategy | Precision | Recall | F1-Score | ROC-AUC | Compute Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Static Model (No Retraining)** | 0.9130 | 0.7500 | 0.8235 | 0.9535 | 0.0000 s |
| **Strategy B: Global Retraining (29 Features)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5.7067 s |
| **Strategy C: Proposed Selective Retraining** | **0.9038** | **0.8393** | **0.8704** | **0.9937** | **0.4623 s** |

> ⚡ **Cloud Compute & Latency Savings:** **91.90% reduction** ($0.4623\text{ s}$ vs $5.7067\text{ s}$) while restoring fraud detection $F_1$-score to $0.8704$.

---

## 📁 Repository Structure

```text
├── artifacts/
│   ├── submodel_1_rf.joblib              # Base Model 1 (Random Forest)
│   ├── submodel_2_xgb.joblib             # Base Model 2 (XGBoost)
│   ├── submodel_3_dt.joblib              # Base Model 3 (Decision Tree)
│   ├── submodel_3_dt_updated.joblib      # Retrained Model 3 (Strategy C)
│   ├── robust_scaler.joblib              # Robust Amount Scaler
│   ├── diagram_1_class_distribution.png  # Fig 1: Class Imbalance Breakdown
│   ├── diagram_2_baseline_models.png     # Fig 2: Baseline Model Scores
│   ├── diagram_3_drift_audit.png         # Fig 3: Partition-Level PSI Drift
│   └── diagram_4_master_benchmarks.png   # Fig 4: Latency & Confusion Matrix
├── app.py                                # Streamlit Web Application
├── Project_SRM.ipynb                     # Research Implementation Notebook
├── requirements.txt                      # Project Dependencies
└── README.md                             # Repository Documentation
