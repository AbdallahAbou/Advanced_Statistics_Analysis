"""Non-parametric statistical tests."""

import numpy as np
from scipy import stats
from typing import List, Union
from .parametric import TestResult


class MannWhitneyU:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).
    
    Non-parametric alternative to independent samples t-test.
    Tests whether one distribution is stochastically greater than another.
    
    Assumptions:
    - Independent observations
    - Ordinal or continuous data
    - Similar distribution shapes (for median comparison)
    
    H₀: The distributions of both groups are equal
    H₁: The distributions differ
    
    Example
    -------
    >>> group1 = [23, 25, 28, 29, 31, 35]
    >>> group2 = [31, 32, 35, 38, 42, 45]
    >>> test = MannWhitneyU()
    >>> result = test.test(group1, group2)
    """
    
    def __init__(self, alternative: str = 'two-sided', alpha: float = 0.05):
        self.alternative = alternative
        self.alpha = alpha
    
    def test(
        self,
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray]
    ) -> TestResult:
        """
        Perform Mann-Whitney U test.
        
        Parameters
        ----------
        group1 : array-like
            First sample
        group2 : array-like
            Second sample
        
        Returns
        -------
        TestResult
            Test results with U-statistic and effect size (rank-biserial r)
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        
        # Perform test
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative=self.alternative
        )
        
        # Rank-biserial correlation (effect size)
        n1, n2 = len(group1), len(group2)
        # r = 1 - (2*U) / (n1*n2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return TestResult(
            statistic=statistic,
            p_value=p_value,
            test_name="Mann-Whitney U test",
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size  # rank-biserial r
        )


class KruskalWallis:
    """
    Kruskal-Wallis H test.
    
    Non-parametric alternative to one-way ANOVA.
    Tests whether samples originate from the same distribution.
    
    Assumptions:
    - Independent observations
    - Ordinal or continuous data
    
    H₀: All groups have equal medians
    H₁: At least one group differs
    
    Example
    -------
    >>> g1 = [85, 86, 88, 75, 78]
    >>> g2 = [90, 91, 89, 92, 87]
    >>> g3 = [80, 81, 82, 79, 78]
    >>> test = KruskalWallis()
    >>> result = test.test([g1, g2, g3])
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def test(self, groups: List[Union[List[float], np.ndarray]]) -> TestResult:
        """
        Perform Kruskal-Wallis H test.
        
        Parameters
        ----------
        groups : list of array-like
            List of samples from each group
        
        Returns
        -------
        TestResult
            Test results with H-statistic and effect size (epsilon-squared)
        """
        groups = [np.asarray(g) for g in groups]
        
        # Perform test
        statistic, p_value = stats.kruskal(*groups)
        
        # Epsilon-squared effect size
        # ε² = H / (n² - 1) / (n - 1)
        n = sum(len(g) for g in groups)
        k = len(groups)
        effect_size = (statistic - k + 1) / (n - k)
        
        return TestResult(
            statistic=statistic,
            p_value=p_value,
            test_name="Kruskal-Wallis H test",
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size,  # ε²
            degrees_freedom=k - 1
        )


class WilcoxonSignedRank:
    """
    Wilcoxon signed-rank test.
    
    Non-parametric alternative to paired samples t-test.
    Tests whether the median of paired differences is zero.
    
    Assumptions:
    - Paired/matched observations
    - Symmetric distribution of differences around median
    
    H₀: Median difference = 0
    H₁: Median difference ≠ 0
    
    Example
    -------
    >>> before = [200, 210, 220, 230, 240]
    >>> after = [180, 190, 185, 200, 205]
    >>> test = WilcoxonSignedRank()
    >>> result = test.test(before, after)
    """
    
    def __init__(self, alternative: str = 'two-sided', alpha: float = 0.05):
        self.alternative = alternative
        self.alpha = alpha
    
    def test(
        self,
        before: Union[List[float], np.ndarray],
        after: Union[List[float], np.ndarray]
    ) -> TestResult:
        """
        Perform Wilcoxon signed-rank test.
        
        Parameters
        ----------
        before : array-like
            Pre-treatment measurements
        after : array-like
            Post-treatment measurements
        
        Returns
        -------
        TestResult
            Test results with W-statistic and effect size (r)
        """
        before = np.asarray(before)
        after = np.asarray(after)
        
        # Perform test
        statistic, p_value = stats.wilcoxon(
            before, after, alternative=self.alternative
        )
        
        # Effect size r = Z / sqrt(n)
        # Approximate Z from p-value
        z = stats.norm.ppf(p_value / 2)
        n = len(before)
        effect_size = abs(z) / np.sqrt(n)
        
        return TestResult(
            statistic=statistic,
            p_value=p_value,
            test_name="Wilcoxon signed-rank test",
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size  # r
        )
