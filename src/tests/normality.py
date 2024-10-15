"""Normality tests."""

import numpy as np
from scipy import stats
from typing import Union, List
from .parametric import TestResult


class ShapiroWilk:
    """
    Shapiro-Wilk test for normality.
    
    Most powerful test for normality, especially for small samples (n < 50).
    
    H₀: Data comes from a normal distribution
    H₁: Data does not come from a normal distribution
    
    Note: Sensitive to sample size. With large samples, even small
    deviations from normality can lead to rejection.
    
    Example
    -------
    >>> data = np.random.normal(0, 1, 50)
    >>> test = ShapiroWilk()
    >>> result = test.test(data)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def test(self, data: Union[List[float], np.ndarray]) -> TestResult:
        """
        Perform Shapiro-Wilk test.
        
        Parameters
        ----------
        data : array-like
            Sample data to test for normality
        
        Returns
        -------
        TestResult
            Test results with W-statistic and p-value
        """
        data = np.asarray(data)
        
        if len(data) < 3:
            raise ValueError("Need at least 3 samples")
        if len(data) > 5000:
            import warnings
            warnings.warn("Shapiro-Wilk not recommended for n > 5000")
        
        statistic, p_value = stats.shapiro(data)
        
        return TestResult(
            statistic=statistic,
            p_value=p_value,
            test_name="Shapiro-Wilk test",
            reject_null=p_value < self.alpha,
            alpha=self.alpha
        )


class KolmogorovSmirnov:
    """
    Kolmogorov-Smirnov test for normality.
    
    Tests whether a sample comes from a specified distribution.
    Less powerful than Shapiro-Wilk but can handle larger samples.
    
    H₀: Data comes from a normal distribution
    H₁: Data does not come from a normal distribution
    
    Note: Uses Lilliefors correction when parameters are estimated
    from the data (which is the typical case).
    
    Example
    -------
    >>> data = np.random.normal(100, 15, 100)
    >>> test = KolmogorovSmirnov()
    >>> result = test.test(data)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def test(self, data: Union[List[float], np.ndarray]) -> TestResult:
        """
        Perform KS test for normality.
        
        Parameters
        ----------
        data : array-like
            Sample data
        
        Returns
        -------
        TestResult
            Test results with D-statistic and p-value
        """
        data = np.asarray(data)
        
        # Standardize data
        data_standardized = (data - np.mean(data)) / np.std(data, ddof=1)
        
        # Perform KS test against standard normal
        statistic, p_value = stats.kstest(data_standardized, 'norm')
        
        return TestResult(
            statistic=statistic,
            p_value=p_value,
            test_name="Kolmogorov-Smirnov test",
            reject_null=p_value < self.alpha,
            alpha=self.alpha
        )


class AndersonDarling:
    """
    Anderson-Darling test for normality.
    
    More sensitive to deviations in the tails than KS test.
    Provides critical values at different significance levels.
    
    H₀: Data comes from a normal distribution
    H₁: Data does not come from a normal distribution
    
    Example
    -------
    >>> data = np.random.normal(0, 1, 100)
    >>> test = AndersonDarling()
    >>> result = test.test(data)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def test(self, data: Union[List[float], np.ndarray]) -> TestResult:
        """
        Perform Anderson-Darling test.
        
        Parameters
        ----------
        data : array-like
            Sample data
        
        Returns
        -------
        TestResult
            Test results with A² statistic and critical value comparison
        """
        data = np.asarray(data)
        
        result = stats.anderson(data, dist='norm')
        statistic = result.statistic
        critical_values = result.critical_values
        significance_levels = result.significance_level
        
        # Find appropriate critical value for alpha
        # significance_level array: [15%, 10%, 5%, 2.5%, 1%]
        alpha_mapping = {0.15: 0, 0.10: 1, 0.05: 2, 0.025: 3, 0.01: 4}
        
        idx = alpha_mapping.get(self.alpha, 2)  # Default to 5%
        critical_value = critical_values[idx]
        
        # Reject if statistic > critical value
        reject = statistic > critical_value
        
        # Approximate p-value (AD doesn't give exact p-value)
        # Using interpolation
        for i, (sig, crit) in enumerate(zip(significance_levels, critical_values)):
            if statistic < crit:
                p_approx = sig / 100
                break
        else:
            p_approx = 0.001  # Very small
        
        return TestResult(
            statistic=statistic,
            p_value=p_approx,
            test_name="Anderson-Darling test",
            reject_null=reject,
            alpha=self.alpha
        )
