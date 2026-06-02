import numpy as np
from project.src.utils.logger import logger

class RiskScoreCalibrator:
    def __init__(self, score_min=300, score_max=900):
        """
        Calibrates model transaction probabilities into a standardized Credit-Style Risk Score.
        Ranges from 300 (legitimate) to 900 (highly critical mule behavior).
        """
        self.score_min = score_min
        self.score_max = score_max

    def probability_to_score(self, prob: float) -> int:
        """
        Calibrates a single probability into a 300-900 score using a non-linear log-odds transformation
        similar to banking FICO architectures.
        """
        # Constrain probability to avoid domain errors
        p = np.clip(prob, 1e-6, 1.0 - 1e-6)
        
        # Calculate log odds (logit)
        logit = np.log(p / (1.0 - p))
        
        # Logit usually spans from -13.8 (1e-6) to +13.8 (1 - 1e-6)
        # Shift logit range into the calibrated range [300, 900]
        # Standard logistic sigmoid maps -5 to +5 beautifully
        # Use simple min-max scaling of the log odds with a scaling constant
        scale_factor = 600.0 / (2 * 10.0) # Mapping log odds of -10 to +10 into 600-point width
        
        # Linear shift logit to center normal around 500-600
        score = 600.0 + (logit * 45.0) # logit = 0 (prob = 0.5) maps to 600
        
        # Constrain to hard bounds
        calibrated_score = int(np.clip(score, self.score_min, self.score_max))
        return calibrated_score

    def get_risk_tier(self, score: int) -> str:
        """
        Categorizes risk scores into banking operational action tiers.
        """
        if score <= 500:
            return "LOW"
        elif score <= 700:
            return "MEDIUM"
        elif score <= 850:
            return "HIGH"
        else:
            return "CRITICAL"

    def get_recommends_and_actions(self, tier: str) -> dict:
        """
        Returns actionable fraud procedures for the bank's operations desk.
        """
        actions = {
            "LOW": {
                "action": "Auto-Approved",
                "color": "#10B981", # Green
                "instructions": "No manual actions required. Account remains under standard rule monitoring."
            },
            "MEDIUM": {
                "action": "Soft Hold / Verification",
                "color": "#F59E0B", # Orange
                "instructions": "Triggers automated OTP check. Flag account for 48-hour velocity review."
            },
            "HIGH": {
                "action": "Hard Hold / Analyst Inquest",
                "color": "#EF4444", # Red
                "instructions": "Place debit freeze immediately. Route to Mule Account Investigation queue."
            },
            "CRITICAL": {
                "action": "Freeze Account & Police Alert",
                "color": "#7F1D1D", # Deep Red
                "instructions": "Suspend all transactions. Freeze linked funds. Draft Cyber Cell FIR report."
            }
        }
        return actions.get(tier, actions["LOW"])

    def generate_risk_profile(self, prob: float) -> dict:
        """
        Generates a comprehensive risk profile dictionary for downstream APIs or dashboards.
        """
        score = self.probability_to_score(prob)
        tier = self.get_risk_tier(score)
        meta = self.get_recommends_and_actions(tier)
        
        return {
            "calibrated_score": score,
            "risk_tier": tier,
            "operational_action": meta["action"],
            "color": meta["color"],
            "instructions": meta["instructions"],
            "raw_probability": float(prob)
        }
