"""
Demonstration of Propensity Score Inverse Probability Weighting (IPW) Skill
"""

import random
import math
from client import PropensityScoreIPW, sigmoid

def main():
    print("=== Propensity Score Inverse Probability Weighting (IPW) ATE Estimation ===")
    random.seed(42)
    n = 500

    # Covariate X (e.g., customer lifetime or baseline risk)
    X = [random.gauss(0, 1.0) for _ in range(n)]

    # Confounded treatment assignment: higher X -> higher likelihood of treatment
    T = [1 if random.random() < sigmoid(0.8 * xi) else 0 for xi in X]

    # Outcome Y: True causal effect = 5.0 * T, with strong confounding + 3.0 * X
    Y = [5.0 * ti + 3.0 * xi + random.gauss(0, 0.2) for ti, xi in zip(T, X)]

    ipw = PropensityScoreIPW()
    results = ipw.estimate_ate(X, T, Y)

    print(f"Naive Confounded Difference in Means: {results['naive_ate']:.4f}")
    print(f"IPW Causal ATE Estimate:              {results['ate_ipw']:.4f} (True Target: 5.0000)")

    assert 4.6 <= results["ate_ipw"] <= 5.4
    assert results["naive_ate"] > 6.0  # Biased upward by positive confounder

    print("\nPropensity Score IPW Estimation Verification PASS!")

if __name__ == "__main__":
    main()
