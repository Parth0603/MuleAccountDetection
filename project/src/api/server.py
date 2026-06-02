from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import yaml
import uvicorn
import os

from project.src.utils.logger import logger
from project.src.preprocessing.cleaning import DataCleaner
from project.src.features.engineering import FeatureEngineer
from project.src.features.selection import FeatureSelector
from project.src.models.pipeline import MuleModelPipeline
from project.src.risk_engine.scoring import RiskScoreCalibrator
from project.src.explainability.describer import FraudExplainer

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

if __name__ == "__main__":
    # Start ASGI server on port 8000
    uvicorn.run("project.src.api.server:app", host="0.0.0.0", port=8000, reload=False)
