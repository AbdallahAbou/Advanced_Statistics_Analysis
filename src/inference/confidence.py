"""Confidence interval calculations."""

import numpy as np
from scipy import stats
from typing import Tuple, Union, List


def confidence_interval_mean(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean.
    
    Uses t-distribution for unknown population variance.
    
    Parameters
    ----------
    data : array-like
        Sample data
    confidence : float
        Confidence level (default 0.95 for 95% CI)
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    
    Example
    -------
    >>> data = [23, 25, 28, 30, 32, 35]
    >>> ci = confidence_interval_mean(data, 0.95)
    >>> print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    """
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    
    # t critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin = t_crit * se
    return (mean - margin, mean + margin)


def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    statistic_func=np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    method: str = 'percentile',
    random_state: int = None
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Non-parametric method that makes no distributional assumptions.
    
    Parameters
    ----------
    data : array-like
        Sample data
    statistic_func : callable
        Function to compute statistic (default: mean)
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    method : str
        'percentile', 'basic', or 'bca' (bias-corrected accelerated)
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    
    Example
    -------
    >>> data = [23, 25, 28, 30, 32, 35, 100]  # With outlier
    >>> ci = bootstrap_ci(data, np.median, method='percentile')
    """
    data = np.asarray(data)
    n = len(data)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate bootstrap samples and compute statistic
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_func(boot_sample)
    
    alpha = 1 - confidence
    
    if method == 'percentile':
        # Simple percentile method
        lower = np.percentile(boot_stats, 100 * alpha/2)
        upper = np.percentile(boot_stats, 100 * (1 - alpha/2))
    
    elif method == 'basic':
        # Basic bootstrap (reverse percentile)
        original_stat = statistic_func(data)
        lower = 2 * original_stat - np.percentile(boot_stats, 100 * (1 - alpha/2))
        upper = 2 * original_stat - np.percentile(boot_stats, 100 * alpha/2)
    
    elif method == 'bca':
        # Bias-corrected and accelerated
        original_stat = statistic_func(data)
        
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < original_stat))
        
        # Acceleration (jackknife estimate)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic_func(jack_sample)
        
        jack_mean = np.mean(jackknife_stats)
        numerator = np.sum((jack_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = numerator / denominator if denominator != 0 else 0
        
        # Adjusted percentiles
        z_alpha = stats.norm.ppf(alpha/2)
        z_1_alpha = stats.norm.ppf(1 - alpha/2)
        
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
        
        lower = np.percentile(boot_stats, 100 * p_lower)
        upper = np.percentile(boot_stats, 100 * p_upper)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return (lower, upper)
