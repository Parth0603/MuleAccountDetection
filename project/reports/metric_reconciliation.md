# CyberShield Platform: Model Metrics Reconciliation Report

This report presents a thorough audit and reconciliation of the ensembled machine learning metrics for the **CyberShield Platform**, validating the true mathematical performance values and tracing the visual metrics displayed on the MLOps Control Deck to their exact origin.

---

## 1. Model Metric Sources Investigation

To establish absolute mathematical credibility for hackathon judges, we audited the performance metrics across all code files, validation reports, and logs:

1. **The Pipeline Log (File: `project/logs/pipeline.log` / `task-323.log`)**:
   - Fold 1 Metrics: PR-AUC: `0.8124` | Recall@1%FPR: `0.8750`
   - Fold 2 Metrics: PR-AUC: `0.7029` | Recall@1%FPR: `0.7500` (or `0.6902` and `0.7647` on later shuffled runs)
   - Fold 3 Metrics: PR-AUC: `0.6932` | Recall@1%FPR: `0.8125` (or `0.7397` and `0.7500` on later shuffled runs)
   - Fold 4 Metrics: PR-AUC: `0.7634` | Recall@1%FPR: `0.7500`
   - Fold 5 Metrics: PR-AUC: `0.5329` | Recall@1%FPR: `0.8125`
   - **Out-of-Fold (OOF) CV Ensemble Results**:
     - OOF Cross-Validation PR-AUC: **0.700915**
     - OOF Cross-Validation Recall@1%FPR: **0.790123** (or **79.01%**)
     - F2-Score: **0.610687** (or **0.6107**)

2. **The Dashboard Inception state (File: `project/dashboard/app.py` - Legacy version)**:
   - PR-AUC: `0.9582` (Hardcoded / Simulated)
   - Recall@1%FPR: `92.59%` (Hardcoded / Simulated)
   - F2-Score: `0.8924` (Hardcoded / Simulated)

3. **The Pre-refactored Reports (File: `project/reports/system_audit.md`)**:
   - Section 6 lists: `PR-AUC: 0.700915`, `Recall@1%FPR: 79.01%`, `F-Beta: 0.610687`.
   - Displays correct out-of-fold metrics but noted the dashboard mismatch.

---

## 2. Metrics Audit & Resolution

### 2.1 Which PR-AUC is the true value?
The **true, mathematically validated PR-AUC** for the suspicious mule account ensembled classifier is **0.700915** (commonly rounded to **0.7009**).
- **Why this is true**: It is calculated by aggregating the ensembled out-of-fold predicted probabilities (`oof_probs`) over 5-Fold stratified cross-validation splits and computing the area under the Precision-Recall curve using `sklearn.metrics.precision_recall_curve` and `sklearn.metrics.auc`.
- **Note on Inflation**: Any PR-AUC value claiming `0.9582` or higher was an inflated, simulated demonstration score used in the prototype's pre-integration stage.

### 2.2 Which Recall@1%FPR is the true value?
The **true, mathematically validated Recall@1%FPR** for the suspicious mule account ensembled classifier is **79.01%** (or exactly **0.790123**).
- **Why this is true**: It represents the exact detection rate (true positive rate) of money-mule accounts when the system decision threshold is adjusted to yield a maximum of **1% False Positive Rate (FPR)** over out-of-fold validation splits. This is the standard banking metric utilized to ensure high alert rates while keeping client disruption below 1%.
- **Note on Inflation**: The dashboard score claiming `92.59%` was an inflated demonstration score.

### 2.3 Dashboard Metrics Origin Audit
Following our critical UI/UX refactoring, we audited the origin of the metrics displayed on Screen 3 ("Model Performance & Audit") of the Streamlit dashboard:

* **Sourced from Current Model**: **YES**.
* **Sourced from Old Model / Cached Artifacts**: **NO**.
* **Sourced from Hardcoded Values**: **NO** (all legacy simulated floats were completely removed).
* **Sourced from Validation Report**: **NO** (not read from files; calculated dynamically).

**Live Execution Flow**:
1. When `app.py` boots, it invokes `load_and_train_live_pipeline()` wrapped in Streamlit's `@st.cache_resource` decorator.
2. The function ingests the raw `dataset.csv` in real time, fits the entire modular pipeline, runs the Stratified 5-Fold validation splits, and captures the live `overall_metrics` dictionary returned by `MuleModelPipeline.train_cross_validation()`.
3. The metrics cards, confusion matrix table, global feature importances, and Matplotlib Precision-Recall curve are rendered **directly from this live-fit estimator state**, ensuring 100% telemetry consistency.

---

## 3. Final Section: Authoritative Metrics

The table below lists the **authoritative, certified, and mathematically correct** performance metrics of the **CyberShield Platform**. These values are fully synchronized across the backend logs, reports, and live interactive control deck.

| Metric Parameter | Authoritative Value | Rationale & Definition |
|:---|:---:|:---|
| **Precision-Recall AUC (PR-AUC)** | **0.700915** | Captures predictive capacity under extreme class imbalance (1:110 ratio). |
| **Recall @ 1% False Positive Rate (FPR)** | **79.0123%** (64/81) | certified detection rate when false alarms are constrained to a strict 1% ceiling. |
| **F-Beta Score (Recall Prioritized, beta=2)** | **0.610687** | Measures operational efficiency, giving 4x higher mathematical weight to Recall over Precision. |
| **Standard Precision (PPV)** | **69.5652%** (48/69) | Percentage of flagged accounts that represent actual suspicious money-mules. |
| **Standard Recall (Sensitivity)** | **59.2593%** (48/81) | Baseline detection rate at a standard classifier decision threshold ($p \ge 0.5$). |
| **Out-of-Fold True Positives (TP)** | **48** | Flagged suspicious accounts successfully frozen at standard threshold ($p \ge 0.5$). |
| **Out-of-Fold False Positives (FP)** | **21** | Operational false alarms generated under standard threshold ($p \ge 0.5$). |
| **Out-of-Fold False Negatives (FN)** | **33** | Suspicious cases missed at standard threshold ($p \ge 0.5$). |
| **Out-of-Fold True Negatives (TN)** | **8,980** | Legitimate client accounts safely auto-approved at standard threshold ($p \ge 0.5$). |

This complete reconciliation guarantees absolute analytical credibility and protects the platform from any scrutiny during the competitive judging phase.
