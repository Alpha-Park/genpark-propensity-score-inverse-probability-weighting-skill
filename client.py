"""
Propensity Score Inverse Probability Weighting (IPW) Skill Client
Pure Python Standard Library implementation of propensity score modeling and IPW ATE estimation.
Fits a logistic regression model for treatment assignment probabilities and computes the stabilized
Horvitz-Thompson / Hajek IPW estimator for Average Treatment Effect (ATE).
"""

import math
from typing import List, Dict, Any


def sigmoid(z: float) -> float:
    z_clip = max(min(z, 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-z_clip))


class PropensityScoreIPW:
    def __init__(self):
        self.w0 = 0.0
        self.w1 = 0.0

    def fit_propensity(self, x: List[float], t: List[int], lr: float = 0.2, epochs: int = 600):
        """Fit univariate logistic regression for P(T=1 | X)."""
        n = len(x)
        for _ in range(epochs):
            grad0 = 0.0
            grad1 = 0.0
            for xi, ti in zip(x, t):
                pred = sigmoid(self.w0 + self.w1 * xi)
                err = pred - ti
                grad0 += err
                grad1 += err * xi
            self.w0 -= lr * (grad0 / n)
            self.w1 -= lr * (grad1 / n)

    def predict_propensity(self, x: List[float]) -> List[float]:
        """Predict propensity scores with probability trimming [0.02, 0.98]."""
        scores = []
        for xi in x:
            p = sigmoid(self.w0 + self.w1 * xi)
            p = min(max(p, 0.02), 0.98)
            scores.append(p)
        return scores

    def estimate_ate(self, x: List[float], t: List[int], y: List[float]) -> Dict[str, float]:
        """Calculate Hajek-stabilized Inverse Probability Weighting ATE estimate."""
        self.fit_propensity(x, t)
        propensities = self.predict_propensity(x)

        sum_w_treated = 0.0
        sum_wy_treated = 0.0
        sum_w_control = 0.0
        sum_wy_control = 0.0

        for ti, yi, pi in zip(t, y, propensities):
            if ti == 1:
                w = 1.0 / pi
                sum_w_treated += w
                sum_wy_treated += w * yi
            else:
                w = 1.0 / (1.0 - pi)
                sum_w_control += w
                sum_wy_control += w * yi

        mu1 = sum_wy_treated / sum_w_treated if sum_w_treated != 0 else 0.0
        mu0 = sum_wy_control / sum_w_control if sum_w_control != 0 else 0.0
        ate_ipw = mu1 - mu0

        # Naive unadjusted difference in means
        n_t = sum(t)
        n_c = len(t) - n_t
        naive_treated = sum(yi for ti, yi in zip(t, y) if ti == 1) / n_t
        naive_control = sum(yi for ti, yi in zip(t, y) if ti == 0) / n_c
        naive_ate = naive_treated - naive_control

        return {
            "ate_ipw": ate_ipw,
            "naive_ate": naive_ate,
            "mu1": mu1,
            "mu0": mu0,
            "w0": self.w0,
            "w1": self.w1
        }
