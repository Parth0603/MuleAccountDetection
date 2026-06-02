import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from project.src.utils.logger import logger

class FeatureEngineer:
    def __init__(self, config):
        """
        Initializes the feature engineering component.
        """
        self.config = config
        self.date_baseline_str = config.get("pipeline", {}).get("date_baseline", "2025-12-31")
        self.date_baseline = pd.to_datetime(self.date_baseline_str)
        self.boi_features = config.get("pipeline", {}).get("boi_features", [])
        
        # State saved during fit() and applied in transform()
        self.categorical_cols_ = []
        self.ohe_encoder_ = None
        self.isolation_forest_ = None
        self.fitted_numeric_cols_ = []
        self.is_fit = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits category encoders, handles baseline dates, and fits an unsupervised Isolation Forest anomaly scorer.
        """
        logger.info(f"Fitting FeatureEngineer on dataset with shape: {X.shape}")
        
        # 1. Identify Categorical Columns for One-Hot Encoding
        # Any object/string column that is NOT parsed date
        self.categorical_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if "F3888" in self.categorical_cols_:
            self.categorical_cols_.remove("F3888") # Keep original string date out of OHE
            
        logger.info(f"Identified categorical columns for encoding: {self.categorical_cols_}")
        
        # Fit OneHotEncoder
        if self.categorical_cols_:
            self.ohe_encoder_ = OneHotEncoder(
                handle_unknown="ignore", 
                sparse_output=False, 
                dtype=np.float32
            )
            self.ohe_encoder_.fit(X[self.categorical_cols_])
            logger.info("Fitted OneHotEncoder on categorical columns.")

        # 2. Setup Isolation Forest for Behavior Outlier Profiling
        # Isolation Forest is fitted strictly on numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if "F3924" in numeric_cols:
            numeric_cols.remove("F3924") # Target is not a feature
            
        self.fitted_numeric_cols_ = numeric_cols
        
        # To avoid high-null issues (which were handled in preprocessing), we filter only complete columns
        # Fit Isolation Forest on a subsample of typical clean samples
        if numeric_cols:
            anomaly_cfg = self.config.get("anomaly_detection", {})
            self.isolation_forest_ = IsolationForest(
                n_estimators=anomaly_cfg.get("n_estimators", 150),
                contamination=anomaly_cfg.get("contamination", 0.01),
                random_state=anomaly_cfg.get("random_state", 42),
                n_jobs=-1
            )
            # If target y is provided, fit strictly on legitimate samples (y == 0) to capture normal bank behaviors!
            if y is not None:
                legit_mask = (y == 0)
                fit_data = X.loc[legit_mask, numeric_cols]
                # Fallback if too few samples
                if len(fit_data) < 100:
                    fit_data = X[numeric_cols]
            else:
                fit_data = X[numeric_cols]
                
            # Fill remaining NaNs with 0 if any leaked
            fit_data_clean = fit_data.fillna(0.0)
            self.isolation_forest_.fit(fit_data_clean)
            logger.info("Fitted unsupervised Isolation Forest on legitimate behavior samples.")
            
        self.is_fit = True
        logger.info("FeatureEngineer fit complete.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw cleaned features into engineered transaction and behavior feature tensors.
        """
        if not self.is_fit:
            raise ValueError("FeatureEngineer must be fitted before transforming.")
            
        logger.info(f"Engineering features on dataset with shape: {X.shape}")
        df_feat = X.copy()
        
        # 1. Parse Account Age in Days (from F3888)
        if "F3888_parsed" in df_feat.columns:
            # Days elapsed between opening date and baseline date
            df_feat["F_account_age_days"] = (self.date_baseline - df_feat["F3888_parsed"]).dt.days
            # Drop date columns to avoid model issues
            df_feat = df_feat.drop(columns=["F3888", "F3888_parsed"], errors="ignore")
            logger.info("Engineered 'F_account_age_days' feature.")
        elif "F3888" in df_feat.columns:
            # Fallback if cleaning didn't parse it
            parsed_dates = pd.to_datetime(df_feat["F3888"], errors="coerce")
            df_feat["F_account_age_days"] = (self.date_baseline - parsed_dates).dt.days.fillna(365.0)
            df_feat = df_feat.drop(columns=["F3888"], errors="ignore")

        # 2. Unsupervised Anomaly Feature (Outlier Score)
        if self.isolation_forest_ is not None and self.fitted_numeric_cols_:
            # Align numeric columns
            numeric_data = df_feat[self.fitted_numeric_cols_].fillna(0.0)
            # decision_function outputs anomaly score (lower means more anomalous)
            df_feat["F_unsupervised_anomaly_score"] = self.isolation_forest_.decision_function(numeric_data)
            logger.info("Engineered 'F_unsupervised_anomaly_score' outlier feature.")

        # 3. Fintech Domain Behavioral & Velocity Ratio Features
        # F3836 is balance/transaction volume.
        # F3887 is account duration / months active. Let's create balance-per-month velocity.
        if "F3836" in df_feat.columns and "F3887" in df_feat.columns:
            # Avoid division by zero
            months_active = df_feat["F3887"].replace(0, 1)
            df_feat["F_balance_velocity_per_month"] = df_feat["F3836"] / months_active
            logger.info("Engineered balance-per-month velocity feature.")

        # F3836 divided by F_account_age_days (balance-per-day velocity)
        if "F3836" in df_feat.columns and "F_account_age_days" in df_feat.columns:
            days_active = df_feat["F_account_age_days"].replace(0, 1)
            df_feat["F_balance_velocity_per_day"] = df_feat["F3836"] / days_active
            logger.info("Engineered balance-per-day velocity feature.")

        # 4. Apply OneHotEncoder
        if self.categorical_cols_ and self.ohe_encoder_ is not None:
            # Encode categorical features
            ohe_features = self.ohe_encoder_.transform(df_feat[self.categorical_cols_])
            ohe_col_names = self.ohe_encoder_.get_feature_names_out(self.categorical_cols_)
            
            # Create DataFrame with OHE features
            ohe_df = pd.DataFrame(ohe_features, columns=ohe_col_names, index=df_feat.index)
            
            # Drop original categorical columns and concat OHE features
            df_feat = df_feat.drop(columns=self.categorical_cols_)
            df_feat = pd.concat([df_feat, ohe_df], axis=1)
            logger.info(f"One-Hot encoded categorical columns. Generated {len(ohe_col_names)} binary columns.")
            
        logger.info(f"Feature engineering complete. Engineered shape: {df_feat.shape}")
        return df_feat

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
