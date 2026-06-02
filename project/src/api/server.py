from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import yaml
import uvicorn
import os
import json

from project.src.utils.logger import logger
from project.src.preprocessing.cleaning import DataCleaner
from project.src.features.engineering import FeatureEngineer
from project.src.features.selection import FeatureSelector
from project.src.models.pipeline import MuleModelPipeline
from project.src.risk_engine.scoring import RiskScoreCalibrator
from project.src.explainability.describer import FraudExplainer
from project.src.utils.database import db_manager
from project.src.utils.gemini import gemini_service

# Initialize FastAPI App with metadata for Swagger UI docs
app = FastAPI(
    title="CyberShield Mule Account Detection API",
    description=(
        "Production-grade REST API backend for Suspicious Mule Account Detection, "
        "calibrated risk scoring, and automated SHAP explainability reporting. "
        "Created for the Bank of India + IIT Hyderabad CyberShield Hackathon."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local cross-origin connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold loaded assets
config = None
cleaner = None
engineer = None
selector = None
model_pipeline = None
risk_calibrator = None
explainer = None
dtypes_map = {}
is_ready = False


class TransactionData(BaseModel):
    """
    Validation request schema for incoming client transaction records.
    Allows passing flexible key-value attributes matching the F1-F3923 columns.
    """
    account_id: str
    features: Dict[str, Any]

@app.on_event("startup")
def startup_event():
    """
    Triggered on API server launch. Fits the core modular pipelines 
    on the raw dataset, keeping estimators active in-memory.
    """
    global config, cleaner, engineer, selector, model_pipeline, risk_calibrator, explainer, is_ready, dtypes_map
    logger.info("Starting up CyberShield API Server...")
    
    try:
        # 1. Load Configurations
        config_path = "project/configs/config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # 2. Ingest Raw Data to fit modular services
        raw_path = config["data"]["raw_path"]
        logger.info(f"Startup Ingestion: Loading {raw_path} to fit pipeline...")
        raw_df = pd.read_csv(raw_path)
        
        # Save dtypes map
        dtypes_map = raw_df.dtypes.to_dict()
        
        target_col = config["pipeline"]["target_col"]
        y_raw = raw_df[target_col]
        X_raw = raw_df.drop(columns=[target_col], errors="ignore")
        
        # 3. Fit Data Cleaner
        cleaner = DataCleaner(config)
        X_clean = cleaner.fit_transform(X_raw)
        
        # 4. Fit Feature Engineer
        engineer = FeatureEngineer(config)
        X_eng = engineer.fit_transform(X_clean, y=y_raw)
        
        # 5. Fit Feature Selector
        selector = FeatureSelector(config)
        X_select = selector.fit_transform(X_eng, y=y_raw)
        
        # Concat target back to train the XGBoost ensemble models
        train_df = pd.concat([X_select, y_raw], axis=1)
        
        # 6. Fit Model Pipeline
        model_pipeline = MuleModelPipeline(config)
        _ = model_pipeline.train_cross_validation(train_df, n_splits=5)
        
        # 7. Initialize scoring & explainer engines
        risk_calibrator = RiskScoreCalibrator()
        explainer = FraudExplainer(model_pipeline, model_pipeline.feature_names_)
        
        is_ready = True
        logger.info("CyberShield API Pipeline fit successfully! Server is ready.")
    except Exception as e:
        logger.error(f"Failed to initialize server pipeline: {e}")
        is_ready = False

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    """
    Redirects root requests to standard Swagger UI interactive documentation.
    """
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """
    Health check status endpoint verifying pipeline readiness.
    """
    if is_ready:
        return {"status": "healthy", "pipeline": "fitted_and_ready", "features_count": len(model_pipeline.feature_names_)}
    else:
        return {"status": "degraded", "pipeline": "initializing_or_failed"}

def prepare_input_dataframe(features: dict) -> pd.DataFrame:
    df_row = pd.DataFrame([features])
    if dtypes_map:
        for col in df_row.columns:
            if col in dtypes_map:
                try:
                    df_row[col] = df_row[col].astype(dtypes_map[col])
                except Exception:
                    pass
    return df_row

@app.post("/api/v1/predict", summary="Predict Fraud & Calibrate Risk Score")
def predict_fraud(data: TransactionData):
    """
    Exposes end-to-end inference logic: Preprocesses raw fields, engineers anomaly indexes, 
    predicts ensemble probabilities, and calibrates credit-style 300-900 risk scores.
    """
    if not is_ready:
        raise HTTPException(status_code=503, detail="API pipeline has not initialized successfully.")
        
    try:
        # Convert JSON feature map to single-row DataFrame and cast types
        df_row = prepare_input_dataframe(data.features)
        
        # Apply preprocessing
        df_clean = cleaner.transform(df_row)
        
        # Apply feature engineering
        df_eng = engineer.transform(df_clean)
        
        # Apply feature selection alignment
        df_select = selector.transform(df_eng)
        df_feat = df_select.drop(columns=["F3924", "target"], errors="ignore").astype(float)
        
        # Predict continuous probability
        prob = model_pipeline.predict_proba(df_feat)[0]
        
        # Calibrate risk score & procedures profile
        risk_profile = risk_calibrator.generate_risk_profile(prob)
        
        return {
            "account_id": data.account_id,
            "status": "success",
            "risk_profile": risk_profile
        }
    except Exception as e:
        logger.error(f"Error predicting instance: {e}")
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.post("/api/v1/explain", summary="Get Local SHAP Risk Drivers")
def explain_attributions(data: TransactionData):
    """
    Exposes local SHAP Tree attributions for a transaction record.
    Returns sorted list of features that contributed to the final score.
    """
    if not is_ready:
        raise HTTPException(status_code=503, detail="API pipeline has not initialized successfully.")
        
    try:
        df_row = prepare_input_dataframe(data.features)
        df_clean = cleaner.transform(df_row)
        df_eng = engineer.transform(df_clean)
        df_select = selector.transform(df_eng)
        df_feat = df_select.drop(columns=["F3924", "target"], errors="ignore").astype(float)
        
        # Extract SHAP contributions
        contributions = explainer.explain_instance(df_feat)
        return {
            "account_id": data.account_id,
            "status": "success",
            "shap_contributions": contributions
        }
    except Exception as e:
        logger.error(f"Error generating SHAP explainability: {e}")
        raise HTTPException(status_code=400, detail=f"Explainability error: {str(e)}")

@app.post("/api/v1/report", summary="Generate AI CyberShield Case Brief")
def generate_narrative_report(data: TransactionData):
    """
    Auto-generates a detailed plain-text case audit report and operational 
    orders suitable for bank Cyber Cells and fraud investigation units.
    """
    if not is_ready:
        raise HTTPException(status_code=503, detail="API pipeline has not initialized successfully.")
        
    try:
        df_row = prepare_input_dataframe(data.features)
        df_clean = cleaner.transform(df_row)
        df_eng = engineer.transform(df_clean)
        df_select = selector.transform(df_eng)
        df_feat = df_select.drop(columns=["F3924", "target"], errors="ignore").astype(float)
        
        # Predict probability and score
        prob = model_pipeline.predict_proba(df_feat)[0]
        risk_profile = risk_calibrator.generate_risk_profile(prob)
        
        # Generate executive narrative report
        report = explainer.generate_investigator_report(df_feat, risk_profile)
        return {
            "account_id": data.account_id,
            "status": "success",
            "report_markdown": report
        }
    except Exception as e:
        logger.error(f"Error compiling AI investigator report: {e}")
        raise HTTPException(status_code=400, detail=f"Reporting error: {str(e)}")

# =========================================================
# NEW PERSISTENT CASE MANAGEMENT & AI ASSISTANT APIs
# =========================================================

class AnalystNoteRequest(BaseModel):
    analyst: str
    note: str

class EscalateRequest(BaseModel):
    status: str
    escalation_level: str
    analyst: str
    log_msg: str

class AssistantRequest(BaseModel):
    case_id: str
    question: str

@app.get("/api/v1/cases", summary="Get all cases")
def list_cases():
    return db_manager.get_cases()

@app.get("/api/v1/cases/{id}", summary="Get case details by ID")
def get_case(id: str):
    case = db_manager.get_case_by_id(id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
        
    try:
        # Extract features from case account
        account_features = json.loads(case["account"]["behavioral_features"])
        df_row = prepare_input_dataframe(account_features)
        
        # Process features through ML pipeline to get fresh live score & attributions
        df_clean = cleaner.transform(df_row)
        df_eng = engineer.transform(df_clean)
        df_select = selector.transform(df_eng)
        df_feat = df_select.drop(columns=["F3924", "target"], errors="ignore").astype(float)
        
        prob = model_pipeline.predict_proba(df_feat)[0]
        risk_profile = risk_calibrator.generate_risk_profile(prob)
        
        # Evaluate local rule-based score
        rule_score = 0
        triggers = []
        
        val_velocity_per_month = df_eng.iloc[0].get("F_balance_velocity_per_month", 0.0)
        val_velocity_per_day = df_eng.iloc[0].get("F_balance_velocity_per_day", 0.0)
        val_anomaly = df_eng.iloc[0].get("F_unsupervised_anomaly_score", 0.0)
        val_longevity = case["account"].get("account_longevity_months", 24)
        val_balance = case["account"].get("balance_volume", 0.0)
        
        if val_velocity_per_month and val_velocity_per_month > 10000:
            rule_score += 25
            triggers.append({"name": "High transaction velocity", "score": 25})
        if val_balance and val_balance > 100000:
            rule_score += 20
            triggers.append({"name": "Large cash movement", "score": 20})
        if val_velocity_per_day and val_velocity_per_day > 1000:
            rule_score += 20
            triggers.append({"name": "Rapid ledger withdrawals", "score": 20})
        if val_longevity and val_longevity < 12:
            rule_score += 10
            triggers.append({"name": "New account profile", "score": 10})
        if val_anomaly and val_anomaly > 0.015:
            rule_score += 10
            triggers.append({"name": "Behavior Anomaly Index trigger", "score": 10})
            
        if rule_score == 0:
            rule_score = 10
            triggers.append({"name": "Baseline account audit score", "score": 10})
            
        ml_score = risk_profile["calibrated_score"]
        final_score = int(np.clip((ml_score * 0.7) + (rule_score * 3.5), 300, 900))
        
        final_tier = risk_calibrator.get_risk_tier(final_score)
        final_meta = risk_calibrator.get_recommends_and_actions(final_tier)
        
        full_risk_profile = {
            "calibrated_score": final_score,
            "risk_tier": final_tier,
            "operational_action": final_meta["action"],
            "color": final_meta["color"],
            "instructions": final_meta["instructions"],
            "raw_probability": float(prob),
            "rule_score": rule_score,
            "triggers": triggers,
            "shap_contributions": explainer.explain_instance(df_feat)[:10]
        }
    except Exception as e:
        logger.error(f"Error executing real-time prediction for case {id}: {e}")
        # Default fallback risk profile if ML pipeline fails for some reason
        full_risk_profile = {
            "calibrated_score": 600,
            "risk_tier": "MEDIUM",
            "operational_action": "Verification Inquiry",
            "color": "#F59E0B",
            "instructions": "Audit baseline features locally.",
            "raw_probability": 0.50,
            "rule_score": 10,
            "triggers": [{"name": "Pipeline inference fallback", "score": 10}],
            "shap_contributions": []
        }
        
    timeline = db_manager.get_investigations_by_case(id)
    notes = db_manager.get_notes_by_case(id)
    alerts = db_manager.get_alerts_by_account(case["account_id"])
    
    return {
        "case": case,
        "risk_profile": full_risk_profile,
        "timeline": timeline,
        "notes": notes,
        "alerts": alerts
    }

@app.post("/api/v1/cases/{id}/notes", summary="Add analyst note")
def add_note(id: str, data: AnalystNoteRequest):
    note_id = db_manager.add_analyst_note(id, data.analyst, data.note)
    return {"status": "success", "note_id": note_id}

@app.post("/api/v1/cases/{id}/escalate", summary="Update status and escalate case")
def escalate_case(id: str, data: EscalateRequest):
    db_manager.update_case_workflow(id, data.status, data.escalation_level, data.analyst, data.log_msg)
    return {"status": "success"}

@app.get("/api/v1/cases/{id}/report", summary="Generate Deloitte-Style compliance report")
def get_or_generate_report(id: str, analyst: str = "Forensic Analyst"):
    existing = db_manager.get_report_by_case(id)
    if existing:
        return {"status": "success", "report": existing["report_content"]}
        
    case_details = get_case(id)
    case = case_details["case"]
    risk_profile = case_details["risk_profile"]
    timeline = case_details["timeline"]
    notes = case_details["notes"]
    alerts = case_details["alerts"]
    
    report_markdown = gemini_service.generate_deloitte_report(case, risk_profile, timeline, notes, alerts, analyst)
    db_manager.save_report(id, analyst, report_markdown)
    return {"status": "success", "report": report_markdown}

@app.post("/api/v1/assistant", summary="Ask floating AI assistant")
def ask_assistant(data: AssistantRequest):
    case_details = get_case(data.case_id)
    case = case_details["case"]
    risk_profile = case_details["risk_profile"]
    timeline = case_details["timeline"]
    notes = case_details["notes"]
    alerts = case_details["alerts"]
    
    response = gemini_service.ask_investigation_assistant(data.question, case, risk_profile, timeline, notes, alerts)
    return {"status": "success", "answer": response}

@app.post("/api/v1/demo/generate", summary="Trigger database seeding manually")
def trigger_generate():
    db_manager.seed_data_if_empty(raw_dataset_path="dataset.csv")
    return {"status": "success", "message": "Database seeded with synthetic data."}

if __name__ == "__main__":
    # Start ASGI server on port 8000
    uvicorn.run("project.src.api.server:app", host="0.0.0.0", port=8000, reload=False)

