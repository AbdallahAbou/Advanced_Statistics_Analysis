"""Statistical power analysis."""

import numpy as np
from scipy import stats
from typing import Optional


def power_analysis(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    test_type: str = 'two-sided'
) -> float:
    """
    Calculate statistical power for a t-test.
    
    Power = P(reject H₀ | H₀ is false)
    
    Parameters
    ----------
    effect_size : float
        Cohen's d (standardized mean difference)
    n : int
        Sample size per group
    alpha : float
        Significance level
    test_type : str
        'two-sided' or 'one-sided'
    
    Returns
    -------
    float
        Statistical power (0 to 1)
    
    Example
    -------
    >>> # What's the power to detect d=0.5 with n=30?
    >>> power = power_analysis(effect_size=0.5, n=30)
    >>> print(f"Power: {power:.2%}")
    """
    df = 2 * n - 2  # Degrees of freedom for independent t-test
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt(n / 2)
    
    if test_type == 'two-sided':
        # Critical value for two-sided test
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # Power = P(|T| > t_crit) under alternative
        # Using non-central t distribution
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    else:
        t_crit = stats.t.ppf(1 - alpha, df)
        power = 1 - stats.nct.cdf(t_crit, df, ncp)
    
    return power


def sample_size_ttest(
    effect_size: float,
    power: float = 0.80,
    alpha: float = 0.05,
    test_type: str = 'two-sided'
) -> int:
    """
    Calculate required sample size per group for t-test.
    
    Parameters
    ----------
    effect_size : float
        Cohen's d (standardized effect size)
        Small: 0.2, Medium: 0.5, Large: 0.8
    power : float
        Desired statistical power (default 0.80)
    alpha : float
        Significance level (default 0.05)
    test_type : str
        'two-sided' or 'one-sided'
    
    Returns
    -------
    int
        Required sample size per group (rounded up)
    
    Example
    -------
    >>> # Sample size needed to detect medium effect (d=0.5)
    >>> n = sample_size_ttest(effect_size=0.5, power=0.80)
    >>> print(f"Need {n} per group")
    """
    # Binary search for sample size
    n_low = 2
    n_high = 1000
    
    # Check if even max is insufficient
    if power_analysis(effect_size, n_high, alpha, test_type) < power:
        n_high = 10000
    
    while n_high - n_low > 1:
        n_mid = (n_low + n_high) // 2
        current_power = power_analysis(effect_size, n_mid, alpha, test_type)
        
        if current_power < power:
            n_low = n_mid
        else:
            n_high = n_mid
    
    return n_high


def effect_size_cohens_d(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int
) -> float:
    """
    Calculate Cohen's d effect size.
    
    Uses pooled standard deviation.
    
    Parameters
    ----------
    mean1, mean2 : float
        Group means
    std1, std2 : float
        Group standard deviations
    n1, n2 : int
        Group sample sizes
    
    Returns
    -------
    float
        Cohen's d
    
    Interpretation:
        |d| < 0.2: negligible
        |d| < 0.5: small
        |d| < 0.8: medium
        |d| >= 0.8: large
    """
    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    )
    
    return (mean1 - mean2) / pooled_std


def effect_size_from_ttest(
    t_statistic: float,
    n1: int,
    n2: int
) -> float:
    """
    Calculate Cohen's d from t-statistic.
    
    Parameters
    ----------
    t_statistic : float
        T-test statistic
    n1, n2 : int
        Group sample sizes
    
    Returns
    -------
    float
        Cohen's d
    """
    return t_statistic * np.sqrt(1/n1 + 1/n2)
