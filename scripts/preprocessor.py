import pandas as pd


class PreProcessor:
    def __init__(self):
        pass
    
    def merge(self, df1: pd.DataFrame, df2: pd.DataFrame, merge_key: str = 'keychain_udid') -> pd.DataFrame:
        """Merge two datasets."""
        return df1.merge(df2, on=merge_key, how='left')
    
    def age_segmentation(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add 'age_segmented' column with age bins: 5-year bins until 35, then 10-year bins until 100."""
        df = dataset.copy()
        age_bins = sorted(list(set(list(range(0, 36, 5)) + list(range(35, 101, 10)))))
        df['age_segmented'] = pd.cut(df['age'], bins=age_bins, include_lowest=True)
        return df
    
    def session_length_segmentation(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add 'session_length_segmented' column with bins: 60s until 10min, then 5min, then hourly."""
        df = dataset.copy()
        max_val = int(df['session_length'].max())
        session_bins = sorted(list(set(list(range(0, 601, 60)) + list(range(600, 3601, 300)) + list(range(3600, max_val + 3601, 3600)))))
        df['session_length_segmented'] = pd.cut(df['session_length'], bins=session_bins, include_lowest=True)
        return df
    
    def remove_top_1_percent(self, dataset: pd.DataFrame, variables: list) -> pd.DataFrame:
        """Remove the top 1% of values for each specified variable."""
        df = dataset.copy()
        for var in variables:
            threshold = df[var].quantile(0.99)
            df = df[df[var] <= threshold]
        return df

