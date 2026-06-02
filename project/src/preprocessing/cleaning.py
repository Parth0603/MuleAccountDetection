import pandas as pd
import numpy as np
import os
from project.src.utils.logger import logger

class DataCleaner:
    def __init__(self, config):
        """
        Initializes the data cleaning pipeline with configuration parameters.
        """
        self.config = config
        self.pipeline_cfg = config.get("pipeline", {})
        self.target_col = self.pipeline_cfg.get("target_col", "F3924")
        self.leakage_cols = self.pipeline_cfg.get("leakage_cols", [])
        self.boi_features = self.pipeline_cfg.get("boi_features", [])
        self.null_threshold = self.pipeline_cfg.get("null_threshold", 0.8)
        
        # State saved during fit() and applied during transform()
        self.dropped_leakage_cols_ = []
        self.fully_null_cols_ = []
        self.constant_cols_ = []
        self.high_null_cols_ = []
        self.numerical_imputation_values_ = {}
        self.categorical_imputation_values_ = {}
        self.missingness_columns_to_track_ = []
        self.is_fit = False

    def fit(self, X: pd.DataFrame):
        """
        Learns the data quality issues (fully nulls, constant features, high missing values, medians).
        X should be the raw training dataset containing features (and target if it's there, but we won't mutate target).
        """
        logger.info(f"Fitting DataCleaner on dataset with shape: {X.shape}")
        
        # 1. Identify Target Leakage columns
        self.dropped_leakage_cols_ = [col for col in self.leakage_cols if col in X.columns]
        logger.info(f"Identified {len(self.dropped_leakage_cols_)} target leakage/index columns to drop: {self.dropped_leakage_cols_}")
        
        # Working with a temporary DataFrame for analysis, excluding leakage cols
        temp_df = X.drop(columns=self.dropped_leakage_cols_, errors="ignore")
        
        # 2. Identify 100% empty columns
        n_rows = len(temp_df)
        null_counts = temp_df.isnull().sum()
        self.fully_null_cols_ = null_counts[null_counts == n_rows].index.tolist()
        logger.info(f"Identified {len(self.fully_null_cols_)} fully empty columns.")
        
        # 3. Identify zero-variance (constant) columns
        remaining_cols = [c for c in temp_df.columns if c not in self.fully_null_cols_]
        self.constant_cols_ = []
        for col in remaining_cols:
            if col == self.target_col or col in self.boi_features:
                continue
            # For numerical/categorical check unique count excluding NaN
            if temp_df[col].nunique(dropna=True) <= 1:
                self.constant_cols_.append(col)
        logger.info(f"Identified {len(self.constant_cols_)} constant (zero-variance) columns to drop.")
        
        # 4. Identify high-null columns (> null_threshold) except BOI features
        active_cols = [c for c in remaining_cols if c not in self.constant_cols_]
        self.high_null_cols_ = []
        for col in active_cols:
            if col == self.target_col or col in self.boi_features:
                continue
            null_ratio = temp_df[col].isnull().sum() / n_rows
            if null_ratio > self.null_threshold:
                self.high_null_cols_.append(col)
        logger.info(f"Identified {len(self.high_null_cols_)} columns exceeding null threshold of {self.null_threshold * 100}%.")
        
        # Pruned list of columns for imputation profiling
        pruned_cols = [c for c in active_cols if c not in self.high_null_cols_]
        
        # 5. Track columns for predictive missingness indicators (> 10% nulls)
        self.missingness_columns_to_track_ = []
        for col in pruned_cols:
            if col == self.target_col or col == "F3888":  # Exclude target and date
                continue
            null_ratio = temp_df[col].isnull().sum() / n_rows
            if null_ratio > 0.10:
                self.missingness_columns_to_track_.append(col)
        logger.info(f"Will create missingness indicator columns for {len(self.missingness_columns_to_track_)} features.")

        # 6. Profile Imputation Values
        for col in pruned_cols:
            if col == self.target_col or col == "F3888":
                continue
            if pd.api.types.is_numeric_dtype(temp_df[col]):
                # Median for numericals
                self.numerical_imputation_values_[col] = temp_df[col].median()
            else:
                # UNKNOWN mode for categoricals
                self.categorical_imputation_values_[col] = "UNKNOWN"
                
        self.is_fit = True
        logger.info("DataCleaner fit complete.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies data cleaning, leakage removal, missing value indicators, and imputations.
        """
        if not self.is_fit:
            raise ValueError("DataCleaner must be fitted before transforming data.")
            
        logger.info(f"Transforming dataset with shape: {X.shape}")
        
        # Work on a copy to prevent SettingWithCopy warnings
        df_clean = X.copy()
        
        # 1. Drop Target Leakage
        df_clean = df_clean.drop(columns=self.dropped_leakage_cols_, errors="ignore")
        
        # 2. Drop Fully Null, Constant, and High-Null Columns
        drop_quality_cols = list(set(self.fully_null_cols_ + self.constant_cols_ + self.high_null_cols_))
        # Ensure we do NOT drop BOI features or target
        drop_quality_cols = [c for c in drop_quality_cols if c != self.target_col and c not in self.boi_features]
        df_clean = df_clean.drop(columns=drop_quality_cols, errors="ignore")
        
        # 3. Create Missingness Indicator Columns (F_col_isnan)
        for col in self.missingness_columns_to_track_:
            if col in df_clean.columns:
                df_clean[f"{col}_isnan"] = df_clean[col].isnull().astype(int)
                
        # 4. Impute Missing Values
        # Impute numeric features
        for col, val in self.numerical_imputation_values_.items():
            if col in df_clean.columns:
                # Ensure no NaN remains. If column has all NaN (e.g. rare case in test set), fill with 0
                fill_val = val if not pd.isna(val) else 0.0
                df_clean[col] = df_clean[col].fillna(fill_val)
                
        # Impute categorical features
        for col, val in self.categorical_imputation_values_.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(val).astype(str)

        # Handle remaining NaNs in BOI features that might have exceeded threshold or were not caught
        for col in self.boi_features:
            if col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median() if not pd.isna(df_clean[col].median()) else 0.0)
                    else:
                        df_clean[col] = df_clean[col].fillna("UNKNOWN")
                        
        # 5. Parse Date Column F3888 (Account Opening Date)
        if "F3888" in df_clean.columns:
            # Parse dates safely, coercing errors
            df_clean["F3888_parsed"] = pd.to_datetime(df_clean["F3888"], errors="coerce")
            
            # Fill unparseable dates with a logical historical fallback (e.g. median date)
            median_date = df_clean["F3888_parsed"].dropna().iloc[len(df_clean["F3888_parsed"].dropna()) // 2]
            df_clean["F3888_parsed"] = df_clean["F3888_parsed"].fillna(median_date)
            
            logger.info("Successfully parsed and imputed date column F3888.")
            
        logger.info(f"Transformation complete. Cleaned shape: {df_clean.shape}")
        return df_clean

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
