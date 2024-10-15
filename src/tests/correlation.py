"""Correlation tests and measures."""

import numpy as np
from scipy import stats
from typing import Union, List, Tuple
from .parametric import TestResult


class PearsonCorrelation:
    """
    Pearson product-moment correlation coefficient.
    
    Measures linear relationship between two continuous variables.
    
    Assumptions:
    - Continuous data
    - Linear relationship
    - Bivariate normal distribution (for inference)
    - No significant outliers
    
    H₀: ρ = 0 (no linear relationship)
    H₁: ρ ≠ 0
    
    Example
    -------
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> y = [2.1, 4.0, 5.9, 8.2, 10.1, 11.8, 14.2, 16.0, 18.1, 19.9]
    >>> corr = PearsonCorrelation()
    >>> result = corr.test(x, y)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def test(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray]
    ) -> TestResult:
        """
        Compute Pearson correlation and test significance.
        
        Parameters
        ----------
        x : array-like
            First variable
        y : array-like
            Second variable
        
        Returns
        -------
        TestResult
            Test results with r, p-value, and confidence interval
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Compute correlation
        r, p_value = stats.pearsonr(x, y)
        
        # Fisher z-transformation for CI
        n = len(x)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(0.975)
        
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        # Transform back
        ci = (np.tanh(z_lower), np.tanh(z_upper))
        
        return TestResult(
            statistic=r,
            p_value=p_value,
            test_name="Pearson correlation",
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=r,  # r itself is effect size
            confidence_interval=ci,
            degrees_freedom=n - 2
        )
    
    @staticmethod
    def interpret_strength(r: float) -> str:
        """
        Interpret correlation strength (Cohen's conventions).
        
        |r| < 0.1: negligible
        |r| < 0.3: small
        |r| < 0.5: medium
        |r| >= 0.5: large
        """
        r_abs = abs(r)
        if r_abs < 0.1:
            return "negligible"
        elif r_abs < 0.3:
            return "small"
        elif r_abs < 0.5:
            return "medium"
        else:
            return "large"


class SpearmanCorrelation:
    """
    Spearman rank correlation coefficient.
    
    Non-parametric measure of monotonic relationship.
    
    Assumptions:
    - Ordinal or continuous data
    - Monotonic relationship (not necessarily linear)
    
    H₀: ρₛ = 0 (no monotonic relationship)
    H₁: ρₛ ≠ 0
    
    Example
    -------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [5, 6, 7, 8, 7]  # Non-linear but monotonic increasing
    >>> corr = SpearmanCorrelation()
    >>> result = corr.test(x, y)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def test(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray]
    ) -> TestResult:
        """
        Compute Spearman correlation and test significance.
        
        Parameters
        ----------
        x : array-like
            First variable
        y : array-like
            Second variable
        
        Returns
        -------
        TestResult
            Test results with rho and p-value
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Compute correlation
        rho, p_value = stats.spearmanr(x, y)
        
        return TestResult(
            statistic=rho,
            p_value=p_value,
            test_name="Spearman rank correlation",
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=rho,
            degrees_freedom=len(x) - 2
        )


class KendallTau:
    """
    Kendall's tau rank correlation coefficient.
    
    Non-parametric measure of ordinal association.
    More robust to ties than Spearman's rho.
    
    τ = (concordant pairs - discordant pairs) / total pairs
    
    Example
    -------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [1, 3, 2, 5, 4]
    >>> corr = KendallTau()
    >>> result = corr.test(x, y)
    """
    
    def __init__(self, variant: str = 'b', alpha: float = 0.05):
        """
        Parameters
        ----------
        variant : str
            'b' for tau-b (handles ties), 'c' for tau-c
        alpha : float
            Significance level
        """
        self.variant = variant
        self.alpha = alpha
    
    def test(
        self,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray]
    ) -> TestResult:
        """
        Compute Kendall's tau and test significance.
        
        Parameters
        ----------
        x : array-like
            First variable
        y : array-like
            Second variable
        
        Returns
        -------
        TestResult
            Test results with tau and p-value
        """
        x = np.asarray(x)
        y = np.asarray(y)
        
        tau, p_value = stats.kendalltau(x, y, variant=self.variant)
        
        return TestResult(
            statistic=tau,
            p_value=p_value,
            test_name=f"Kendall's tau-{self.variant}",
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=tau
        )
