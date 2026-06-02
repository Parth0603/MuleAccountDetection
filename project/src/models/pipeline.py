import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix, recall_score
from project.src.utils.logger import logger

class MuleModelPipeline:
    def __init__(self, config):
        """
        Initializes the Model pipeline with XGBoost hyperparameter tuning configurations.
        """
        self.config = config
        self.xgb_cfg = config.get("xgboost", {})
        self.target_col = config.get("pipeline", {}).get("target_col", "F3924")
        self.clfs_ = []
        self.feature_names_ = []

    def calculate_custom_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, threshold=0.5):
        """
        Calculates hackathon judging metrics: PR-AUC, F1-beta, standard F1, and Recall at 1% False Positive Rate (FPR).
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # Calculate standard confusion matrix at decision threshold
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Recall at 1% False Positive Rate (FPR) - Critical Banking Metric
        # We search for the threshold that yields <= 1% FPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Find exact threshold for 1% FPR
        fpr_values = []
        recall_at_1_fpr = 0.0
        
        # Sort probabilities and labels
        desc_score_indices = np.argsort(y_prob)[::-1]
        y_prob_sorted = y_prob[desc_score_indices]
        y_true_sorted = y_true[desc_score_indices]
        
        n_neg = np.sum(y_true == 0)
        n_pos = np.sum(y_true == 1)
        
        fps = 0
        tps = 0
        for i in range(len(y_prob_sorted)):
            if y_true_sorted[i] == 1:
                tps += 1
            else:
                fps += 1
            
            current_fpr = fps / n_neg if n_neg > 0 else 0
            if current_fpr <= 0.01:
                recall_at_1_fpr = tps / n_pos if n_pos > 0 else 0.0
            else:
                break
                
        # F-beta score (beta=2 favors recall for fraud mitigation)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
        
        metrics = {
            "pr_auc": float(pr_auc),
            "recall_at_1_fpr": float(recall_at_1_fpr),
            "f2_score": float(f2),
            "precision": float(prec),
            "recall": float(rec),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
        return metrics

    def train_cross_validation(self, df: pd.DataFrame, n_splits=5):
        """
        Runs Stratified K-Fold cross validation on XGBoost classifier.
        Implements high-entropy shuffling to mitigate row-sorting bias.
        """
        logger.info(f"Starting Stratified {n_splits}-Fold Cross Validation training.")
        
        # Extract features and targets
        X = df.drop(columns=[self.target_col], errors="ignore")
        y = df[self.target_col]
        self.feature_names_ = X.columns.tolist()
        
        # Set up StratifiedKFold - Shuffling is MANDATORY to bypass row sorting bias
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        oof_probs = np.zeros(len(df))
        self.clfs_ = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training FOLD {fold+1}/{n_splits}...")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Setup cost-sensitive parameters
            scale_pos = self.xgb_cfg.get("scale_pos_weight", 111.0)
            
            clf = xgb.XGBClassifier(
                n_estimators=self.xgb_cfg.get("n_estimators", 200),
                learning_rate=self.xgb_cfg.get("learning_rate", 0.05),
                max_depth=self.xgb_cfg.get("max_depth", 6),
                subsample=self.xgb_cfg.get("subsample", 0.8),
                colsample_bytree=self.xgb_cfg.get("colsample_bytree", 0.8),
                scale_pos_weight=scale_pos,
                random_state=self.xgb_cfg.get("random_state", 42),
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=-1
            )
            
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predict validation probabilities
            val_probs = clf.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_probs
            
            # Calculate fold metrics
            metrics = self.calculate_custom_metrics(y_val.values, val_probs)
            fold_metrics.append(metrics)
            logger.info(f"Fold {fold+1} Metrics: PR-AUC: {metrics['pr_auc']:.4f} | Recall@1%FPR: {metrics['recall_at_1_fpr']:.4f}")
            
            self.clfs_.append(clf)
            
        # Aggregate out-of-fold metrics
        overall_metrics = self.calculate_custom_metrics(y.values, oof_probs)
        logger.info(f"OOF Cross-Validation PR-AUC: {overall_metrics['pr_auc']:.4f}")
        logger.info(f"OOF Cross-Validation Recall@1%FPR: {overall_metrics['recall_at_1_fpr']:.4f}")
        
        return oof_probs, overall_metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Ensemble prediction by averaging probability outputs of the k-fold models.
        """
        if not self.clfs_:
            raise ValueError("Pipeline has not been trained yet.")
            
        # Align features
        X_align = X[self.feature_names_]
        
        probs = np.zeros(len(X))
        for clf in self.clfs_:
            probs += clf.predict_proba(X_align)[:, 1]
            
        return probs / len(self.clfs_)
