import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from project.src.utils.logger import logger

class FeatureSelector:
    def __init__(self, config):
        """
        Initializes the Feature Selection processor.
        """
        self.config = config
        self.select_cfg = config.get("features", {}).get("selection", {})
        self.n_features = self.select_cfg.get("n_features_to_select", 150)
        self.var_threshold = self.select_cfg.get("variance_threshold", 0.01)
        self.boi_features = config.get("pipeline", {}).get("boi_features", [])
        
        # State saved during fit
        self.variance_selector_ = None
        self.kbest_selector_ = None
        self.selected_features_ = []
        self.is_fit = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits variance thresholds and ANOVA F-tests while preserving BOI-highlighted columns.
        """
        logger.info(f"Fitting FeatureSelector on X shape: {X.shape}, y shape: {y.shape}")
        
        # Ensure target is not in X
        X_feats = X.drop(columns=[col for col in ["F3924", "target"] if col in X.columns], errors="ignore")
        
        # 1. Apply Variance Threshold
        self.variance_selector_ = VarianceThreshold(threshold=self.var_threshold)
        self.variance_selector_.fit(X_feats)
        
        # Get columns that survived variance threshold
        var_survived_cols = X_feats.columns[self.variance_selector_.get_support()].tolist()
        logger.info(f"{len(var_survived_cols)} features survived variance threshold of {self.var_threshold}.")
        
        # Always protect BOI features and engineered columns
        mandatory_cols = [c for c in X_feats.columns if c in self.boi_features or c.startswith("F_")]
        combined_candidates = list(set(var_survived_cols + mandatory_cols))
        
        # Working candidate set
        X_candidates = X_feats[combined_candidates].fillna(0.0)
        
        # 2. SelectKBest using ANOVA F-value
        actual_k = min(self.n_features, X_candidates.shape[1])
        self.kbest_selector_ = SelectKBest(score_func=f_classif, k=actual_k)
        self.kbest_selector_.fit(X_candidates, y)
        
        # Get selected candidate features
        kbest_selected = X_candidates.columns[self.kbest_selector_.get_support()].tolist()
        logger.info(f"SelectKBest selected top {len(kbest_selected)} features.")
        
        # 3. Add back BOI features and engineered features to guarantee their inclusion
        final_features = list(set(kbest_selected + mandatory_cols))
        
        # Keep clean, distinct list
        self.selected_features_ = [col for col in X_feats.columns if col in final_features]
        logger.info(f"Total features finalized for modeling: {len(self.selected_features_)} (includes protected and engineered features).")
        
        self.is_fit = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Slices the DataFrame to keep only the selected predictive features.
        """
        if not self.is_fit:
            raise ValueError("FeatureSelector must be fitted before transforming.")
            
        # Ensure we keep the target variable if present
        target_cols = [col for col in ["F3924", "target"] if col in X.columns]
        
        # Align columns
        cols_to_keep = [col for col in self.selected_features_ if col in X.columns]
        
        df_selected = X[cols_to_keep + target_cols]
        logger.info(f"Feature selection complete. Out shape: {df_selected.shape}")
        return df_selected

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
