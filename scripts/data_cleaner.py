import pandas as pd
import numpy as np


def drop_negative_values(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Remove rows where the specified column has negative values."""
    print(f"\n-- Managing negative outliers for column: {column_name}")
    initial_count = len(df)
    df = df[(df[column_name] >= 0) | (df[column_name].isna())].copy()
    dropped = initial_count - len(df)
    print(f"Dropped {dropped} rows")
    return df


def fill_null_with_average(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Fill null values in a quantitative column with the average (as integer)."""
    print(f"\n-- Managing nulls for quantitative column: {column_name}")
    null_count = df[column_name].isnull().sum()
    fill_value = int(round(df[column_name].mean(), 0))
    print(f"Number of null values: {null_count}\nFilled with average value: {fill_value}")
    df[column_name] = df[column_name].fillna(fill_value)
    return df


def fill_null_with_mode(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Fill null values in a qualitative column with the most frequent value."""
    print(f"\n-- Managing nulls for qualitative column: {column_name}")
    null_count = df[column_name].isnull().sum()
    most_frequent = df[column_name].mode().iloc[0]
    print(f"Number of null values: {null_count}\nFilled nulls with most frequent value: {most_frequent}")
    df[column_name] = df[column_name].fillna(most_frequent)
    return df


def fill_os_with_device_model(df: pd.DataFrame) -> pd.DataFrame:
    """Fill OS column using device_model: iOS if contains iPhone/iPad, otherwise Android."""
    os_null = df['os'].isna()
    print(f"\n-- Managing nulls for OS column using device_model\nNumber of null values: {os_null.sum()}")
    contains_apple = df['device_model'].str.contains('iPhone|iPad', case=False, na=False, regex=True)
    ios_count = (os_null & contains_apple).sum()

    df.loc[os_null & contains_apple, 'os'] = 'iOS'
    df.loc[os_null & ~contains_apple, 'os'] = 'Android'
    print(f"Replaced {ios_count} null(s) with iOS (found iPhone or iPad in device_model)\nReplaced {os_null.sum() - ios_count} null(s) with Android (the rest)")
    return df


def fill_null_with_random_bernouilli(df: pd.DataFrame, column_name: str, seed: int = 33) -> pd.DataFrame:
    """Fill null values in a binary column with random values based on column probability."""
    null_mask = df[column_name].isna()
    null_count = null_mask.sum()
    print(f"\n-- Managing nulls for binary column: {column_name}\nNumber of null values: {null_count}")
    prob = df[column_name].mean()

    df.loc[null_mask, column_name] = np.random.default_rng(seed).binomial(1, prob, size=null_count)
    print(f"Filled nulls with randomly generated binary values with a probability of {prob:.4f} (seed fixed to: {seed})")
    return df


def remove_rows_with_missing_user_id(df: pd.DataFrame, user_id_column: str = 'keychain') -> pd.DataFrame:
    """Remove rows where the user ID column has null values from the A/B test dataset."""
    null_count = df[user_id_column].isnull().sum()
    print(f"\n-- Removing rows with missing {user_id_column}\nNumber of null values in {user_id_column}: {null_count}")
    initial_count = len(df)
    df = df[df[user_id_column].notna()].copy()
    print(f"Dropped {initial_count - len(df)} rows with missing {user_id_column}")
    return df
