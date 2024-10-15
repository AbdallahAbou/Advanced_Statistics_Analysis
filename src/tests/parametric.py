"""Parametric statistical tests."""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass


@dataclass
class TestResult:
    """Container for hypothesis test results."""
    statistic: float
    p_value: float
    test_name: str
    reject_null: bool
    alpha: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    degrees_freedom: Optional[float] = None
    
    def __str__(self) -> str:
        result = "reject" if self.reject_null else "fail to reject"
        s = f"{self.test_name}\n"
        s += f"  Statistic: {self.statistic:.4f}\n"
        s += f"  P-value: {self.p_value:.4f}\n"
        s += f"  α: {self.alpha}\n"
        s += f"  Decision: {result} H₀\n"
        if self.effect_size is not None:
            s += f"  Effect size: {self.effect_size:.4f}\n"
        if self.confidence_interval is not None:
            s += f"  95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]\n"
        return s


class TTest:
    """
    Independent samples t-test.
    
    Tests whether two groups have different population means.
    
    Assumptions:
    - Independent observations
    - Normal distribution in both groups
    - Homogeneity of variance (or use Welch's correction)
    
    H₀: μ₁ = μ₂
    H₁: μ₁ ≠ μ₂ (two-tailed)
    
    Parameters
    ----------
    equal_var : bool
        If True, perform standard t-test assuming equal variances.
        If False, perform Welch's t-test (default, more robust).
    alpha : float
        Significance level (default 0.05)
    
    Example
    -------
    >>> group1 = [23, 25, 28, 29, 31]
    >>> group2 = [31, 32, 35, 38, 42]
    >>> test = TTest(equal_var=False)
    >>> result = test.test(group1, group2)
    >>> print(result)
    """
    
    def __init__(self, equal_var: bool = False, alpha: float = 0.05):
        self.equal_var = equal_var
        self.alpha = alpha
    
    def test(
        self, 
        group1: Union[List[float], np.ndarray],
        group2: Union[List[float], np.ndarray]
    ) -> TestResult:
        """
        Perform independent samples t-test.
        
        Parameters
        ----------
        group1 : array-like
            First sample
        group2 : array-like
            Second sample
        
        Returns
        -------
        TestResult
            Test results with statistic, p-value, effect size
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(
            group1, group2, equal_var=self.equal_var
        )
        
        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(group1) - 1) * np.var(group1, ddof=1) + 
             (len(group2) - 1) * np.var(group2, ddof=1)) /
            (len(group1) + len(group2) - 2)
        )
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Confidence interval for mean difference
        mean_diff = np.mean(group1) - np.mean(group2)
        se = np.sqrt(np.var(group1, ddof=1)/len(group1) + 
                     np.var(group2, ddof=1)/len(group2))
        df = len(group1) + len(group2) - 2
        t_crit = stats.t.ppf(0.975, df)
        ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
        
        test_name = "Welch's t-test" if not self.equal_var else "Student's t-test"
        
        return TestResult(
            statistic=statistic,
            p_value=p_value,
            test_name=test_name,
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=ci,
            degrees_freedom=df
        )


class PairedTTest:
    """
    Paired samples t-test (dependent samples).
    
    Tests whether the mean difference between paired observations is zero.
    
    Assumptions:
    - Paired/matched observations
    - Differences are normally distributed
    
    H₀: μ_d = 0 (mean difference is zero)
    H₁: μ_d ≠ 0
    
    Example
    -------
    >>> before = [200, 210, 220, 230, 240]
    >>> after = [180, 190, 185, 200, 205]
    >>> test = PairedTTest()
    >>> result = test.test(before, after)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def test(
        self,
        before: Union[List[float], np.ndarray],
        after: Union[List[float], np.ndarray]
    ) -> TestResult:
        """
        Perform paired samples t-test.
        
        Parameters
        ----------
        before : array-like
            Pre-treatment measurements
        after : array-like
            Post-treatment measurements
        
        Returns
        -------
        TestResult
            Test results
        """
        before = np.asarray(before)
        after = np.asarray(after)
        
        if len(before) != len(after):
            raise ValueError("Arrays must have same length for paired test")
        
        # Compute differences
        differences = before - after
        
        # Perform test
        statistic, p_value = stats.ttest_rel(before, after)
        
        # Cohen's d for paired samples
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Confidence interval
        se = stats.sem(differences)
        df = len(differences) - 1
        t_crit = stats.t.ppf(0.975, df)
        mean_diff = np.mean(differences)
        ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
        
        return TestResult(
            statistic=statistic,
            p_value=p_value,
            test_name="Paired samples t-test",
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=effect_size,
            confidence_interval=ci,
            degrees_freedom=df
        )


class ANOVA:
    """
    One-way Analysis of Variance (ANOVA).
    
    Tests whether multiple groups have equal population means.
    
    Assumptions:
    - Independent observations
    - Normal distribution within each group
    - Homogeneity of variance across groups
    
    H₀: μ₁ = μ₂ = ... = μₖ
    H₁: At least one mean differs
    
    Example
    -------
    >>> group1 = [85, 86, 88, 75, 78]
    >>> group2 = [90, 91, 89, 92, 87]
    >>> group3 = [80, 81, 82, 79, 78]
    >>> test = ANOVA()
    >>> result = test.test([group1, group2, group3])
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def test(self, groups: List[Union[List[float], np.ndarray]]) -> TestResult:
        """
        Perform one-way ANOVA.
        
        Parameters
        ----------
        groups : list of array-like
            List of samples from each group
        
        Returns
        -------
        TestResult
            Test results with F-statistic and effect size (eta-squared)
        """
        groups = [np.asarray(g) for g in groups]
        
        # Perform ANOVA
        statistic, p_value = stats.f_oneway(*groups)
        
        # Eta-squared (effect size)
        # SS_between / SS_total
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_squared = ss_between / ss_total
        
        # Degrees of freedom
        df_between = len(groups) - 1
        df_within = sum(len(g) - 1 for g in groups)
        
        return TestResult(
            statistic=statistic,
            p_value=p_value,
            test_name="One-way ANOVA",
            reject_null=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=eta_squared,  # η²
            degrees_freedom=df_between
        )
    
    def posthoc_tukey(
        self, 
        groups: List[Union[List[float], np.ndarray]],
        group_names: Optional[List[str]] = None
    ):
        """
        Perform Tukey's HSD post-hoc test.
        
        Parameters
        ----------
        groups : list of array-like
            List of samples from each group
        group_names : list of str, optional
            Names for each group
        
        Returns
        -------
        ndarray
            Tukey HSD results
        """
        from scipy.stats import tukey_hsd
        
        groups = [np.asarray(g) for g in groups]
        result = tukey_hsd(*groups)
        
        return result
