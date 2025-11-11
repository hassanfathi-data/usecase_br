import pandas as pd
from sklearn.preprocessing import LabelEncoder


def age_segmentation(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add 'age_segmented' column with age bins: 5-year bins until 35, then 10-year bins until 100."""
    df = dataset.copy()
    bins_5yr = list(range(0, 36, 5))
    bins_10yr = list(range(35, 101, 10))
    age_bins = sorted(set(bins_5yr + bins_10yr))
    df['age_segmented'] = pd.cut(df['age'], bins=age_bins, include_lowest=True)
    return df


def session_length_segmentation(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add 'session_length_segmented' column with bins: 60s until 10min, then 5min, then hourly."""
    df = dataset.copy()
    max_val = int(df['session_length'].max())
    bins_60s = list(range(0, 601, 60))
    bins_5min = list(range(600, 3601, 300))
    bins_hourly = list(range(3600, max_val + 3601, 3600))
    session_bins = sorted(set(bins_60s + bins_5min + bins_hourly))

    df['session_length_segmented'] = pd.cut(df['session_length'], bins=session_bins, include_lowest=True)
    return df


def remove_top_1_percent(dataset: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Remove the top 1% of values for each specified variable."""
    df = dataset.copy()
    for var in variables:
        threshold = df[var].quantile(0.99)
        df = df[df[var] <= threshold]
    return df


def prepare_features(df: pd.DataFrame, variables: list, target_var: str = 'd30'):
    """Prepare features by encoding categorical variables and selecting columns."""
    df = df[variables + [target_var]].copy()

    for var in variables:
        if df[var].dtype == 'object' or pd.api.types.is_categorical_dtype(df[var]):
            encoder = LabelEncoder()
            df[var] = encoder.fit_transform(df[var].astype(str))

    return df
