# Advanced Statistical Analysis

A comprehensive Python framework for statistical hypothesis testing, inference, and data analysis with emphasis on proper methodology and effect size reporting.

## Features

### Hypothesis Testing

| Category | Tests |
|----------|-------|
| **Parametric** | Student's t-test, Welch's t-test, Paired t-test, One-way ANOVA |
| **Non-parametric** | Mann-Whitney U, Kruskal-Wallis H, Wilcoxon Signed-Rank |
| **Correlation** | Pearson, Spearman, Kendall's tau |
| **Normality** | Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling |

### Statistical Inference

- Confidence intervals (parametric and bootstrap)
- Power analysis and sample size calculation
- Effect size computation (Cohen's d, η², r)

## Installation

```bash
git clone https://github.com/AbdallahAbou/Advanced_Statistical_Analysis.git
cd Advanced_Statistical_Analysis
pip install -r requirements.txt
```

## Quick Start

### Hypothesis Testing

```python
from src.tests import TTest, MannWhitneyU, ShapiroWilk

# Independent samples t-test
group1 = [23, 25, 28, 29, 31, 35]
group2 = [31, 32, 35, 38, 42, 45]

ttest = TTest(equal_var=False)  # Welch's t-test
result = ttest.test(group1, group2)

print(result)
# Welch's t-test
#   Statistic: -3.4821
#   P-value: 0.0062
#   α: 0.05
#   Decision: reject H₀
#   Effect size: -1.9438
#   95% CI: [-18.23, -3.77]
```

### Check Assumptions

```python
# Test normality before parametric tests
normality = ShapiroWilk()
print(normality.test(group1))

# If non-normal, use non-parametric alternative
if result.p_value > 0.05:
    mann_whitney = MannWhitneyU()
    result = mann_whitney.test(group1, group2)
```

### Power Analysis

```python
from src.inference import power_analysis, sample_size_ttest

# What power do we have?
power = power_analysis(effect_size=0.5, n=30)
print(f"Power: {power:.2%}")

# How many subjects needed?
n = sample_size_ttest(effect_size=0.5, power=0.80)
print(f"Need {n} per group")
```

### Confidence Intervals

```python
from src.inference import confidence_interval_mean, bootstrap_ci
import numpy as np

data = [23, 25, 28, 30, 32, 35, 100]  # Note outlier

# Parametric CI
ci = confidence_interval_mean(data, confidence=0.95)
print(f"Parametric 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

# Bootstrap CI (robust to outliers)
ci_boot = bootstrap_ci(data, np.median, method='bca')
print(f"Bootstrap 95% CI for median: [{ci_boot[0]:.2f}, {ci_boot[1]:.2f}]")
```

## Project Structure

```
src/
├── tests/
│   ├── parametric.py      # t-test, ANOVA
│   ├── nonparametric.py   # Mann-Whitney, Kruskal-Wallis
│   ├── correlation.py     # Pearson, Spearman, Kendall
│   └── normality.py       # Shapiro-Wilk, KS, Anderson-Darling
├── inference/
│   ├── confidence.py      # CI calculations, bootstrap
│   └── power.py           # Power analysis, sample size
├── scripts/
│   ├── statistical_analysis.py
│   └── data_processor.py
└── configs/
    └── config.py
```

## Test Selection Guide

```
                    ┌─────────────────────┐
                    │ Comparing Groups?   │
                    └─────────┬───────────┘
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        2 groups        >2 groups       Correlation
              │               │               │
     ┌────────┴────────┐     │        ┌──────┴──────┐
     ▼                 ▼     ▼        ▼             ▼
Independent      Paired   ANOVA    Linear    Monotonic
     │               │       │        │             │
     ▼               ▼       ▼        ▼             ▼
┌─────────┐   ┌──────────┐  │   ┌─────────┐  ┌──────────┐
│Normal?  │   │Normal    │  │   │Pearson  │  │Spearman  │
│         │   │diff?     │  │   └─────────┘  │Kendall   │
└────┬────┘   └────┬─────┘  │                └──────────┘
 Yes │ No      Yes │ No     │
     ▼             ▼        ▼
┌────────┐   ┌─────────┐ ┌────────────┐
│t-test  │   │Paired   │ │Kruskal-    │
│        │   │t-test   │ │Wallis      │
└────────┘   └─────────┘ └────────────┘
     │             │
     ▼             ▼
┌────────┐   ┌──────────┐
│Mann-   │   │Wilcoxon  │
│Whitney │   │Signed-   │
└────────┘   │Rank      │
             └──────────┘
```

## Effect Size Interpretation

| Measure | Small | Medium | Large |
|---------|-------|--------|-------|
| Cohen's d | 0.2 | 0.5 | 0.8 |
| η² (eta-squared) | 0.01 | 0.06 | 0.14 |
| r (correlation) | 0.1 | 0.3 | 0.5 |

## License

MIT License
