import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, permutation_test, ttest_ind


def chi2_test(df: pd.DataFrame, variable: str, target_var: str = 'd30') -> dict:
    """Perform chi-square test between a variable and target variable."""
    contingency_table = pd.crosstab(df[variable], df[target_var])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return {'chi2_statistic': chi2, 'p_value': p_value, 'degrees_of_freedom': dof, 'expected_frequencies': expected, 'contingency_table': contingency_table}


def chi2_test_multiple(df: pd.DataFrame, variables: list, target_var: str = 'd30') -> pd.DataFrame:
    """Run chi-square tests on multiple variables against a target variable."""
    return pd.DataFrame([{'Variable': var, 'Chi2 P-Value': chi2_test(df, var, target_var)['p_value']} for var in variables])


def chi2_test_pair(df: pd.DataFrame, var1: str, var2: str, min_obs: int = 30) -> dict:
    """Perform chi-square test between two variables, filtering modalities with less than min_obs observations."""
    var1_counts = df[var1].value_counts()
    var2_counts = df[var2].value_counts()
    valid_var1 = var1_counts[var1_counts >= min_obs].index
    valid_var2 = var2_counts[var2_counts >= min_obs].index
    filtered_df = df[(df[var1].isin(valid_var1)) & (df[var2].isin(valid_var2))]
    contingency_table = pd.crosstab(filtered_df[var1], filtered_df[var2])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return {'chi2_statistic': chi2, 'p_value': p_value, 'degrees_of_freedom': dof, 'expected_frequencies': expected, 'contingency_table': contingency_table}


def correlation_tests(df: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Run chi-square correlation tests on specific variable pairs, filtering modalities with less than min_obs observations."""
    r1 = chi2_test_pair(df, 'session_length_segmented', 'had_meaningful', min_obs)
    results = [{'Variable 1': 'session_length_segmented', 'Variable 2': 'had_meaningful', 'Chi2 Statistic': r1['chi2_statistic'], 'P-Value': r1['p_value']}]
    for var1, var2 in [('country', 'locale'), ('country', 'timezone'), ('locale', 'timezone')]:
        r = chi2_test_pair(df, var1, var2, min_obs)
        results.append({'Variable 1': var1, 'Variable 2': var2, 'Chi2 Statistic': r['chi2_statistic'], 'P-Value': r['p_value']})
    return pd.DataFrame(results)


def student_test(df: pd.DataFrame, variable: str, group_column: str = 'ab_test_cohort') -> float:
    """Perform Student's t-test to compare means between test and control groups, removing top 1% outliers."""
    from scripts.preprocessor import remove_top_1_percent
    df_without_outlier = remove_top_1_percent(df, [variable])
    test_data = df_without_outlier[df_without_outlier[group_column] == 'test'][variable].values
    control_data = df_without_outlier[df_without_outlier[group_column] == 'control'][variable].values
    _, p_value = ttest_ind(test_data, control_data)
    return p_value


def permutation_test_func(df: pd.DataFrame, variable: str, group_column: str = 'ab_test_cohort') -> float:
    """Perform permutation test to compare means between test and control groups, removing top 1% outliers."""
    from scripts.preprocessor import remove_top_1_percent
    df_without_outlier = remove_top_1_percent(df, [variable])
    test_data = df_without_outlier[df_without_outlier[group_column] == 'test'][variable].values
    control_data = df_without_outlier[df_without_outlier[group_column] == 'control'][variable].values
    result = permutation_test((test_data, control_data), statistic=lambda a, b: np.mean(a) - np.mean(b), 
                               n_resamples=2000, alternative='two-sided')
    return result.pvalue
