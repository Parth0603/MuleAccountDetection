# CyberShield Mule Account Detection Platform: Model Validation & Pipeline Audit

This report documents the rigorous model validation, preprocessing architecture, feature engineering layers, and leakage prevention rules implemented inside the **CyberShield Detection Pipeline**.

---

## 1. Feature Engineering & Date Parsing Pipeline

Due to the highly sparse and high-dimensional nature of the dataset (3925 columns), standard ML estimators fail without extensive behavior extraction. The system incorporates custom pre-encoders and anomaly engineers.

### 1.1 Account Opening Timeline Parsing
The raw dataset contains account registration date strings inside column `F3888` (e.g. `9-19-2025`).
1. **Gate 1**: The date parser extracts these strings using `pd.to_datetime` with robust fallback mappings.
2. **Gate 2**: Using a standard fintech baseline date (`2025-12-31`), the engineer computes the exact operational duration in days:
   $$\text{Account Age Days} = \text{Baseline Date} - \text{Opening Date}$$
3. **Imputation**: Any faulty or empty dates are programmatically filled with a standard default active period of **365 days** to prevent training drift.

---

### 1.2 Unsupervised Behavioral Outlier Profiling (Isolation Forest)
To enhance supervised XGBoost decision trees, we train an unsupervised **Isolation Forest** to capture multi-dimensional transaction anomalies.
- **Strict Separation of Legitimate Behaviors**: The Isolation Forest is fitted strictly on legitimate account profiles ($y = 0$). This ensures the outlier boundary represents the true operational profile of normal Bank of India clients.
- **Fitted Metrics**:
  - `n_estimators`: 150
  - `contamination`: 0.01 (1% outlier sensitivity)
  - Output Score: The model decision function outputs a behavior index (`F_unsupervised_anomaly_score`) representing proximity to normal transaction bounds (lower score = higher behavioral anomaly).

---

### 1.3 Fintech Behavioral Velocity Indices
We engineer two key velocity metrics representing standard mule activity patterns:
- **Balance velocity per month**: Volume (`F3836`) divided by active months (`F3887`).
- **Balance velocity per day**: Volume (`F3836`) divided by calculated account age days (`F_account_age_days`).

These indicators safely flag accounts that have massive transaction volumes paired with extremely short active duration, which is a classic money-mule signature.

---

## 2. Leakage Protection & Columns Excluded

To prevent "artificial performance" where a model memorizes ordering or system-generated helper tags, we programmatically prune all high-risk proxy columns.

### 2.1 Collinear Proxy Columns Dropped:
1. **`Unnamed: 0` (Index)**: The raw dataset was perfectly sorted by label (legitimate accounts first, then mule accounts). Any model that learned the index would artificially score 100% precision on historical data while completely failing in real-time production.
2. **`F2230` (Month proxy)**: Highly collinear with ordering due to sequential data collection. Safely dropped.
3. **`F3912` (System-generated helper label)**: Over 96% correlated with the target variable `F3924`. Dropped at the preprocessing gate to force the models to learn genuine transactional behavior.

---

## 3. Preprocessing Grid & Dimensionality Reduction

| Stage | Action | Input Columns | Output Columns | Purpose |
|:---|:---|:---:|:---:|:---|
| **Phase 1** | Leakage Drop | 3925 | 3922 | Excludes `Unnamed: 0`, `F2230`, `F3912`. |
| **Phase 2** | Empty Drop | 3922 | 3859 | Excludes 63 columns that are 100% empty. |
| **Phase 3** | Zero-Var Drop | 3859 | 3563 | Excludes 296 columns with zero variation. |
| **Phase 4** | Sparse Filter | 3563 | 2733 | Excludes 830 columns with $>80\%$ NaNs. |
| **Phase 5** | NaN Flags | 2733 | 3192 | Creates indicator columns for 459 features. |
| **Phase 6** | ANOVA F-Test | 3192 | 168 | Selects 150 best features + protects 18 BOI features. |

---

## 4. Single-Row Type Safety (The XGBoost Dtype Fix)

In production REST APIs, client POST requests often pass single-row JSON payloads. When converting a single-row dict into a pandas DataFrame:
1. **The Pandas Problem**: If certain engineered ratios (`F_balance_velocity_per_day` or daily balances) are missing or passed as `null`/`None`, pandas infers the entire column as an `object` type (rather than `float`).
2. **The XGBoost Crash**: During inference, passing an `object` dtype column into XGBoost causes an immediate internal matrix translation error (400 Bad Request), crashing the server thread.
3. **The Solution**: CyberShield implements a strict type-casting middleware layer across all three endpoints:
   ```python
   df_feat = df_select.drop(columns=["F3924", "target"], errors="ignore").astype(float)
   ```
   This guarantees that every attribute passed to the estimators is represented as a high-precision `float`, completely resolving dtype mismatch crashes.

The pipeline is mathematically stable, structurally secure against data leaks, and production-hardened for single-record REST queries.
