import pytest
import pandas as pd
import numpy as np
import yaml
from project.src.preprocessing.cleaning import DataCleaner
from project.src.features.engineering import FeatureEngineer
from project.src.features.selection import FeatureSelector
from project.src.risk_engine.scoring import RiskScoreCalibrator

# Mock config dictionary
MOCK_CONFIG = {
    "pipeline": {
        "target_col": "F3924",
        "leakage_cols": ["Unnamed: 0", "F2230", "F3912"],
        "boi_features": ["F115", "F321", "F3836"],
        "null_threshold": 0.8,
        "date_baseline": "2025-12-31"
    },
    "features": {
        "selection": {
            "n_features_to_select": 5,
            "variance_threshold": 0.0
        }
    },
    "anomaly_detection": {
        "n_estimators": 10,
        "contamination": 0.05,
        "random_state": 42
    }
}

def test_data_cleaner_leakage_and_null_removal():
    """
    Tests that target leakage, indices, and fully null columns are correctly identified and pruned.
    """
    # Create simple mock data
    mock_data = pd.DataFrame({
        "Unnamed: 0": [1, 2, 3],
        "F2230": ["Oct25", "Oct25", "Sep25"],
        "F3912": [0, 0, 1],
        "F115": [0.5, np.nan, 0.4], # Protected BOI feature
        "F321": [1.2, 1.3, np.nan], # Protected BOI feature
        "F3836": [1000.0, 2000.0, 3000.0], # Protected BOI feature
        "F_empty": [np.nan, np.nan, np.nan], # Fully null
        "F_const": [1, 1, 1], # Constant
        "F3924": [0, 0, 1] # Target
    })
    
    cleaner = DataCleaner(MOCK_CONFIG)
    cleaned_df = cleaner.fit_transform(mock_data)
    
    # Assert leakage columns are dropped
    assert "Unnamed: 0" not in cleaned_df.columns
    assert "F2230" not in cleaned_df.columns
    assert "F3912" not in cleaned_df.columns
    
    # Assert fully empty and constant columns are dropped
    assert "F_empty" not in cleaned_df.columns
    assert "F_const" not in cleaned_df.columns
    
    # Assert protected features are preserved
    assert "F115" in cleaned_df.columns
    assert "F321" in cleaned_df.columns
    
    # Assert missing value imputation filled the NaNs
    assert cleaned_df["F115"].isnull().sum() == 0
    assert cleaned_df["F321"].isnull().sum() == 0

def test_data_cleaner_date_parsing():
    """
    Tests that opening date F3888 is parsed into correct formats.
    """
    mock_data = pd.DataFrame({
        "F3888": ["8-1-2011", "9-15-2025", "INVALID"],
        "F3924": [0, 0, 1]
    })
    
    cleaner = DataCleaner(MOCK_CONFIG)
    cleaned_df = cleaner.fit_transform(mock_data)
    
    assert "F3888_parsed" in cleaned_df.columns
    assert not cleaned_df["F3888_parsed"].isnull().any()

def test_feature_engineer_and_anomaly_scores():
    """
    Tests feature age calculations and Isolation Forest anomaly injection.
    """
    mock_data = pd.DataFrame({
        "F3888_parsed": pd.to_datetime(["2011-08-01", "2025-09-15", "2025-10-15"]),
        "F3836": [10000.0, 500.0, 120000.0],
        "F3887": [12, 1, 3],
        "F3924": [0, 0, 1]
    })
    
    engineer = FeatureEngineer(MOCK_CONFIG)
    engineered_df = engineer.fit_transform(mock_data, y=mock_data["F3924"])
    
    assert "F_account_age_days" in engineered_df.columns
    assert engineered_df["F_account_age_days"].iloc[0] > 3000
    
    assert "F_unsupervised_anomaly_score" in engineered_df.columns
    assert "F_balance_velocity_per_month" in engineered_df.columns
    assert "F_balance_velocity_per_day" in engineered_df.columns

def test_risk_score_calibrator():
    """
    Tests that probability is successfully calibrated into credit-style FICO scores.
    """
    calibrator = RiskScoreCalibrator()
    
    low_score = calibrator.probability_to_score(0.01)
    high_score = calibrator.probability_to_score(0.99)
    med_score = calibrator.probability_to_score(0.5)
    
    assert 300 <= low_score < med_score < high_score <= 900
    
    profile = calibrator.generate_risk_profile(0.95)
    assert profile["risk_tier"] in ["HIGH", "CRITICAL"]
    assert profile["operational_action"] != "Auto-Approved"
