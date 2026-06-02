# CyberShield Mule Account Detection API: Test Results & Contract Specifications

This document outlines the API specifications, request-response schemas, and live captured results for the REST endpoints exposed by the **CyberShield API Gateway**. All tests were validated against the live ASGI server running at `http://localhost:8000`.

---

## 1. API Endpoint Inventory & Contract Specifications

### 1.1 `GET /health`
- **Purpose**: High-speed health status checker validating pipeline fit readiness.
- **HTTP Method**: `GET`
- **Request Schema**: None
- **Response Schema**:
```json
{
  "status": "string",
  "pipeline": "string",
  "features_count": "integer"
}
```
- **Example Response**:
```json
{
  "status": "healthy",
  "pipeline": "fitted_and_ready",
  "features_count": 168
}
```

---

### 1.2 `POST /api/v1/predict`
- **Purpose**: Preprocesses client attributes, engineers daily/monthly velocity parameters, runs classification ensembles, and returns calibrated credit-style risk score (300-900).
- **HTTP Method**: `POST`
- **Request Schema**:
```json
{
  "account_id": "string",
  "features": {
    "F1": "float / null",
    "F2": "float / null",
    "...": "...",
    "F3923": "float / null"
  }
}
```
- **Response Schema**:
```json
{
  "account_id": "string",
  "status": "string",
  "risk_profile": {
    "calibrated_score": "integer",
    "risk_tier": "string",
    "operational_action": "string",
    "color": "string",
    "instructions": "string",
    "raw_probability": "float"
  }
}
```

---

### 1.3 `POST /api/v1/explain`
- **Purpose**: Computes local additive feature attributions utilizing high-speed SHAP TreeExplainer, sorting drivers by absolute impact.
- **HTTP Method**: `POST`
- **Request Schema**: Identical to `POST /api/v1/predict`
- **Response Schema**:
```json
{
  "account_id": "string",
  "status": "string",
  "shap_contributions": [
    {
      "feature": "string",
      "value": "float / null",
      "shap_value": "float"
    }
  ]
}
```

---

### 1.4 `POST /api/v1/report`
- **Purpose**: Generates dynamic plain-text investigator briefs detailing executive metrics, anomaly scoring indexes, SHAP diagnostics, and step-by-step cybersecurity action directives.
- **HTTP Method**: `POST`
- **Request Schema**: Identical to `POST /api/v1/predict`
- **Response Schema**:
```json
{
  "account_id": "string",
  "status": "string",
  "report_markdown": "string"
}
```

---

## 2. Live Captured Test Results

During the automated contract audit execution, the following exact JSON payloads were captured.

### 2.1 Health Check Output (`GET /health`)
```json
{
  "status": "healthy",
  "pipeline": "fitted_and_ready",
  "features_count": 168
}
```

---

### 2.2 Legitimate Account Scenario (`POST /api/v1/predict` - Test A)
**Payload Extract**:
```json
{
  "account_id": "BOI-LEGI-001",
  "features": {
    "Unnamed: 0": 1,
    "F13": 0.59,
    "F14": 0.74,
    "F15": 0.38,
    "F115": 0.53,
    "F321": 1.09,
    "F527": 1.32,
    "F531": 1.12,
    "F3836": 29814.53,
    "F3887": 170
  }
}
```

**Response Output**:
```json
{
  "account_id": "BOI-LEGI-001",
  "status": "success",
  "risk_profile": {
    "calibrated_score": 300,
    "risk_tier": "LOW",
    "operational_action": "Auto-Approved",
    "color": "#10B981",
    "instructions": "No manual actions required. Account remains under standard rule monitoring.",
    "raw_probability": 0.0005198127008043229
  }
}
```

---

### 2.3 Mule Account Scenario (`POST /api/v1/predict` - Test B)
**Payload Extract**:
```json
{
  "account_id": "BOI-MULE-999",
  "features": {
    "Unnamed: 0": 9002,
    "F13": 0.31,
    "F14": 0.44,
    "F15": 0.19,
    "F115": 0.53,
    "F321": 1.09,
    "F527": 1.32,
    "F531": 1.12,
    "F3836": 29814.53,
    "F3887": 167
  }
}
```

**Response Output**:
```json
{
  "account_id": "BOI-MULE-999",
  "status": "success",
  "risk_profile": {
    "calibrated_score": 669,
    "risk_tier": "MEDIUM",
    "operational_action": "Soft Hold / Verification",
    "color": "#F59E0B",
    "instructions": "Triggers automated OTP check. Flag account for 48-hour velocity review.",
    "raw_probability": 0.8251693546772003
  }
}
```

---

### 2.4 Explainer Output (`POST /api/v1/explain` - Test B)
```json
{
  "account_id": "BOI-MULE-999",
  "status": "success",
  "shap_contributions": [
    {
      "feature": "F3898",
      "value": 0.0,
      "shap_value": 1.2005401849746704
    },
    {
      "feature": "F3914",
      "value": 0.0,
      "shap_value": 0.7500945329666138
    },
    {
      "feature": "F3908",
      "value": 1.0,
      "shap_value": 0.7121983170509338
    },
    {
      "feature": "F2956",
      "value": 191.0,
      "shap_value": -0.7119092345237732
    },
    {
      "feature": "F158",
      "value": 0.76,
      "shap_value": 0.3145557641983032
    }
  ]
}
```

All JSON inputs and outputs perfectly match the documentation contract, with float values, null representations, and calibrated thresholds executing flawlessly.
