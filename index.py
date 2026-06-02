import pandas as pd
import numpy as np
import os
import yaml
from project.src.utils.logger import logger
from project.src.preprocessing.cleaning import DataCleaner
from project.src.features.engineering import FeatureEngineer
from project.src.features.selection import FeatureSelector
from project.src.models.pipeline import MuleModelPipeline
from project.src.risk_engine.scoring import RiskScoreCalibrator
from project.src.explainability.describer import FraudExplainer

def load_yaml_config(config_path="project/configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    logger.info("======================================================================")
    st_header = "CYBERSHIELD: SUSPICIOUS MULE ACCOUNT DETECTION PIPELINE"
    logger.info(st_header)
    logger.info("======================================================================")

    # 1. Load Configurations
    config_path = "project/configs/config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}. Aborting.")
        return
        
    config = load_yaml_config(config_path)
    logger.info("Successfully loaded system configurations.")

    # 2. Ingest Raw Dataset
    raw_path = config["data"]["raw_path"]
    if not os.path.exists(raw_path):
        logger.error(f"Raw dataset file not found at {raw_path}. Aborting.")
        return
        
    logger.info(f"Ingesting raw dataset from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    logger.info(f"Raw Ingestion Complete. Ingested Shape: {raw_df.shape}")

    # 3. Step 1: Run Data Quality Cleaning Pipeline
    logger.info("Initializing DataCleaner...")
    cleaner = DataCleaner(config)
    
    # Exclude target from fit, fit cleaner, and transform
    target_col = config["pipeline"]["target_col"]
    features_only = raw_df.drop(columns=[target_col], errors="ignore")
    
    cleaner.fit(features_only)
    cleaned_df = cleaner.transform(raw_df)
    
    # 4. Step 2: Run Behavioral & Outlier Feature Engineering
    logger.info("Initializing FeatureEngineer...")
    engineer = FeatureEngineer(config)
    
    # We pass the target column y to fit Isolation Forest strictly on legitimate behaviors (y==0)
    y_raw = cleaned_df[target_col]
    X_cleaned = cleaned_df.drop(columns=[target_col], errors="ignore")
    
    engineer.fit(X_cleaned, y=y_raw)
    engineered_df = engineer.transform(cleaned_df)
    
    # 5. Step 3: Run High-Dimensional Feature Selection
    logger.info("Initializing FeatureSelector...")
    selector = FeatureSelector(config)
    
    X_engineered = engineered_df.drop(columns=[target_col], errors="ignore")
    selector.fit(X_engineered, y=y_raw)
    final_df = selector.transform(engineered_df)
    
    logger.info(f"Feature Selection complete. Modeling features dimension: {final_df.shape}")

    # 6. Save Cleaned Processed Dataset for Modeling Sandbox
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "engineered_features.csv")
    
    logger.info(f"Persisting engineered feature matrix to: {processed_path}")
    final_df.to_csv(processed_path, index=False)
    logger.info("Successfully persisted processed dataset.")

    # 7. Model Auditing & Cross-Validation Stratification
    logger.info("Initializing Model Pipeline Sandbox...")
    model_pipeline = MuleModelPipeline(config)
    
    # Train Stratified Cross-Validation on the hybrid ensemble
    oof_probs, overall_metrics = model_pipeline.train_cross_validation(final_df, n_splits=5)
    
    logger.info("======================================================================")
    logger.info("OOF CROSS-VALIDATION PERFORMANCE METRICS")
    logger.info("======================================================================")
    logger.info(f"PR-AUC Score (Area Under Precision-Recall): {overall_metrics['pr_auc']:.6f}")
    logger.info(f"Recall at 1% False Positive Rate (FPR):    {overall_metrics['recall_at_1_fpr'] * 100:.2f}%")
    logger.info(f"F-Beta Score (Recall weighted beta=2):     {overall_metrics['f2_score']:.6f}")
    logger.info(f"Legitimate Case Count (Class 0):          {overall_metrics['confusion_matrix']['tn'] + overall_metrics['confusion_matrix']['fp']}")
    logger.info(f"Fraudulent Mule Case Count (Class 1):      {overall_metrics['confusion_matrix']['tp'] + overall_metrics['confusion_matrix']['fn']}")
    logger.info(f"Successfully Detected Mule Accounts (TP):  {overall_metrics['confusion_matrix']['tp']}")
    logger.info(f"False Alarms (FP):                        {overall_metrics['confusion_matrix']['fp']}")
    logger.info("======================================================================")

    # 8. Calibrated Risk Score Demonstration
    logger.info("Demonstrating Calibrated Risk Score & AI Investigator Reporting...")
    risk_calibrator = RiskScoreCalibrator()
    explainer = FraudExplainer(model_pipeline, model_pipeline.feature_names_)
    
    # Pick a real suspicious account (y == 1) for report demonstration
    mule_idx = final_df[final_df[target_col] == 1].index[0]
    mule_sample = final_df.iloc[[mule_idx]]
    mule_feat = mule_sample.drop(columns=[target_col], errors="ignore")
    
    # Predict raw probability using ensemble models
    mule_prob = model_pipeline.predict_proba(mule_feat)[0]
    
    # Calibrate risk score and tier
    risk_profile = risk_calibrator.generate_risk_profile(mule_prob)
    
    # Generate SHAP attributions and plain-text forensic investigator report
    report = explainer.generate_investigator_report(mule_feat, risk_profile)
    
    # Print the report preview
    logger.info("Generated CyberShield AI Investigator Report Demonstration:\n")
    print(report)
    
    # Save the report demonstration to project reports dir
    reports_dir = "project/reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, "sample_investigator_report.md")
    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write(report)
    logger.info(f"Demo CyberShield report successfully saved to: {report_file}")

if __name__ == "__main__":
    main()
