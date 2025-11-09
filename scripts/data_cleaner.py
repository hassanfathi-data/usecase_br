import pandas as pd
import numpy as np


class DataCleaner():
    def __init__(self, onboarding_dataset: pd.DataFrame = None, retention_dataset: pd.DataFrame = None, ab_test_dataset: pd.DataFrame = None) -> None:
        self.df = onboarding_dataset.copy() if onboarding_dataset is not None else None
        self.dfr = retention_dataset.copy() if retention_dataset is not None else None
        self.dfab = ab_test_dataset.copy() if ab_test_dataset is not None else None
    
    def onboarding_dataset_cleaning_steps(self) -> pd.DataFrame:
        """Execute all cleaning steps: remove negatives, fill nulls, and verify no nulls remain."""
        self.drop_negative_values('session_length')
        [self.fill_null_with_average(col) for col in ['age', 'session_length']]
        [self.fill_null_with_mode(col) for col in ['country', 'locale', 'device_model']]
        self.fill_os_with_device_model()
        self.fill_null_with_binary('had_meaningful')
        self.test_no_nulls()
        return self.df
    
    def retention_dataset_cleaning_steps(self, retention_columns: list = ['d0', 'd3', 'd7', 'd14', 'd30']) -> pd.DataFrame:
        """Execute cleaning steps for retention dataset: fill nulls with zero."""
        self.fill_null_with_zero(retention_columns, use_retention=True)
        return self.dfr
    
    def retrieve_datasets_cleanse(self, retention_columns: list = ['d0', 'd3', 'd7', 'd14', 'd30']) -> tuple:
        """Clean both onboarding and retention datasets."""
        onboarding_dataset_cleaned = self.onboarding_dataset_cleaning_steps()
        retention_dataset_cleaned = self.retention_dataset_cleaning_steps(retention_columns)
        return onboarding_dataset_cleaned, retention_dataset_cleaned
    
    def drop_negative_values(self, column_name: str) -> None:
        """Remove rows where the specified column has negative values."""
        print(f"\n-- Managing negative outliers for column: {column_name}")
        initial_count = len(self.df)
        self.df = self.df[(self.df[column_name] >= 0) | (self.df[column_name].isna())].copy()
        dropped = initial_count - len(self.df)
        if dropped > 0:
            print(f"Dropped {dropped} rows")

    def fill_null_with_average(self, column_name: str) -> None:
        """Fill null values in a quantitative column with the average (as integer)."""
        print(f"\n-- Managing nulls for quantitative column: {column_name}")
        null_count = self.df[column_name].isnull().sum()
        if null_count > 0:
            fill_value = int(round(self.df[column_name].mean(), 0))
            print(f"Number of null values: {null_count}\nFilled with average value: {fill_value}")
            self.df[column_name] = self.df[column_name].fillna(fill_value)

    def fill_null_with_mode(self, column_name: str) -> None:
        """Fill null values in a qualitative column with the most frequent value."""
        print(f"\n-- Managing nulls for qualitative column: {column_name}")
        null_count = self.df[column_name].isnull().sum()
        if null_count > 0:
            most_frequent = self.df[column_name].mode().iloc[0]
            print(f"Number of null values: {null_count}\nFilled nulls with most frequent value: {most_frequent}")
            self.df[column_name] = self.df[column_name].fillna(most_frequent)

    def fill_os_with_device_model(self) -> None:
        """Fill OS column using device_model: iOS if contains iPhone/iPad, otherwise Android."""
        os_null = self.df['os'].isna()
        print(f"\n-- Managing nulls for OS column using device_model\nNumber of null values: {os_null.sum()}")
        contains_apple = self.df['device_model'].str.contains('iPhone|iPad', case=False, na=False, regex=True)
        ios_count = (os_null & contains_apple).sum()
        self.df.loc[os_null & contains_apple, 'os'] = 'iOS'
        self.df.loc[os_null & ~contains_apple, 'os'] = 'Android'
        print(f"Replaced {ios_count} null(s) with iOS (found iPhone or iPad in device_model)\nReplaced {os_null.sum() - ios_count} null(s) with Android (the rest)")

    def fill_null_with_binary(self, column_name: str, seed: int = 33) -> None:
        """Fill null values in a binary column with random values based on column probability."""
        null_mask = self.df[column_name].isna()
        null_count = null_mask.sum()
        print(f"\n-- Managing nulls for binary column: {column_name}\nNumber of null values: {null_count}")
        if null_count > 0:
            prob = self.df[column_name].mean()
            self.df.loc[null_mask, column_name] = np.random.default_rng(seed).binomial(1, prob, size=null_count)
            print(f"Filled nulls with randomly generated binary values with a probability of {prob:.4f} (seed fixed to: {seed})")
    
    def fill_null_with_zero(self, columns: list, use_retention: bool = False) -> None:
        """Fill null values in specified columns with zero."""
        df_to_use = self.dfr if use_retention else self.df
        df_to_use[columns] = df_to_use[columns].fillna(0)
    
    def test_no_nulls(self) -> None:
        """Quality test: check for null values in the entire dataframe."""
        total_nulls = self.df.isnull().sum().sum()
        print(f"\n-- Quality test to check for remaining null")
        print(f"✓ No nulls were found in this dataset" if total_nulls == 0 else f"✗ Found {total_nulls} null(s) in this dataset")
    
    def remove_rows_with_missing_user_id(self, user_id_column: str = 'keychain') -> None:
        """Remove rows where the user ID column has null values from the A/B test dataset."""
        null_count = self.dfab[user_id_column].isnull().sum()
        print(f"\n-- Removing rows with missing {user_id_column}\nNumber of null values in {user_id_column}: {null_count}")
        if null_count > 0:
            initial_count = len(self.dfab)
            self.dfab = self.dfab[self.dfab[user_id_column].notna()].copy()
            print(f"Dropped {initial_count - len(self.dfab)} rows with missing {user_id_column}")
    

