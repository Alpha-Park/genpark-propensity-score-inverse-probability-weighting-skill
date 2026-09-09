# GenPark Propensity Score Inverse Probability Weighting (IPW) Skill

Propensity score matching and Inverse Probability Weighting (IPW) estimator calculating Average Treatment Effect (ATE).

Read more at [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph TD
    X[Observed Covariates X] -->|Logistic Regression| P[Propensity Score e X = P T=1|X]
    P --> W[Inverse Weighting: 1/e for T=1, 1/1-e for T=0]
    W --> H[Hajek Pseudo-Population Balancing]
    H --> A[Unbiased Average Treatment Effect ATE]
    style X fill:#e1f5fe
    style P fill:#fff9c4
    style W fill:#ffcdd2
    style H fill:#c8e6c9
    style A fill:#bbdefb
```

## Features
- Pure Python gradient descent logistic regression for propensity score modeling.
- Hajek-stabilized Inverse Probability Weighting (IPW) with boundary trimming.
- Zero external dependencies.
