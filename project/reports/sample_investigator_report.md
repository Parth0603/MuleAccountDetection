# CyberShield Fraud Investigation Report
-----------------------------------------
## EXECUTIVE RISK SUMMARY
* **Investigation Target**: Suspicious Mule Account
* **Calibrated Risk Score**: **669 / 900** (MEDIUM RISK TIER)
* **Fraud Probability**: **82.52%**
* **Operational Action Required**: **Soft Hold / Verification**

## AUDIT ANALYSIS
This account has been flagged by the hybrid XGBoost-Isolation Forest detection pipeline due to suspicious transaction patterns matching money-mule activity profiles. Under cross-validation auditing, the alert was triggered based on critical risk factors:

   1. **F3898** (value: 0.00): increased fraud risk score by 1.2005 SHAP units.
   2. **F3914** (value: 0.00): increased fraud risk score by 0.7501 SHAP units.
   3. **F3908** (value: 1.00): increased fraud risk score by 0.7122 SHAP units.

## DETAILED BEHAVIORAL DIAGNOSTICS
The unsupervised Isolation Forest mapped this transaction sequence to a highly anomalous region of normal banking distributions (Anomaly Score: 0.0663). 

Traditional rules fail to catch this, but the ML attributions indicate standard mule velocity patterns: high incoming funds immediately withdrawn, paired with zero-variance static history.

## RECONSTRUCTIVE ACTION STEPS
Pursuant to Bank of India cyber safety directives:
1. **Immediate Action**: Execute operational mandate (**Soft Hold / Verification**).
2. **Operations Order**: Triggers automated OTP check. Flag account for 48-hour velocity review.
3. **Audit Trail**: File this report automatically with the fraud registry, logging SHAP feature attribution metrics for regulatory compliance.
