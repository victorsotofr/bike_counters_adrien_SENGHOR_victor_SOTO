import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin


### TRAINING SET
### FEATURE ENGINEERING
def _encode_dates(X):
    """
    We will do feature engineering related to dates, including:
    - Weekend: it has a big influence on the pattern
    - Public holiday detection: it can have a big influence on the pattern
    - We introduce Cyclical encoding for hours, weekdays, months and seasons
    for Python to understand these features as cyclical and not linear
    
    Parameters:
    X (pd.DataFrame): Input DataFrame with a "date" column.

    Returns:
    pd.DataFrame: Transformed DataFrame with new features.
    """

    X = X.copy()  # modify a copy of X

    # Encode the date information from the DateOfDeparture columns
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour

    # Identify weekends
    X["is_weekend"] = X["weekday"].isin([5, 6]).astype(int)
    
    # Public holidays
    french_holidays = [
        # 2020 Holidays
        datetime(2020, 1, 1),   # New Year's Day
        datetime(2020, 4, 13),  # Easter Monday
        datetime(2020, 5, 1),   # Labor Day
        datetime(2020, 5, 8),   # Victory in Europe Day
        datetime(2020, 5, 21),  # Ascension Day
        datetime(2020, 6, 1),   # Whit Monday
        datetime(2020, 7, 14),  # Bastille Day
        datetime(2020, 8, 15),  # Assumption Day
        datetime(2020, 11, 1),  # All Saints' Day
        datetime(2020, 11, 11), # Armistice Day
        datetime(2020, 12, 25), # Christmas Day

        # 2021 Holidays
        datetime(2021, 1, 1),   # New Year's Day
        datetime(2021, 4, 5),   # Easter Monday
        datetime(2021, 5, 1),   # Labor Day
        datetime(2021, 5, 8),   # Victory in Europe Day
        datetime(2021, 5, 13),  # Ascension Day
        datetime(2021, 5, 24),  # Whit Monday
        datetime(2021, 7, 14),  # Bastille Day
        datetime(2021, 8, 15),  # Assumption Day
        datetime(2021, 11, 1),  # All Saints' Day
        datetime(2021, 11, 11), # Armistice Day
        datetime(2021, 12, 25), # Christmas Day
    ]
    X["is_holiday"] = X["date"].isin(french_holidays).astype(int)

    # Let's set a dateindex
    X = X.set_index(X["date"])

    # Seasons
    def get_season(date):
        if date.month in [12, 1, 2]:  # Winter
            return 0
        elif date.month in [3, 4, 5]:  # Spring
            return 2
        elif date.month in [6, 7, 8]:  # Summer
            return 3
        else:  # Autumn
            return 1
    
    
    # Cyclical encoding for months
    X["season"] = X["date"].apply(get_season)
    X["season_sin"] = np.sin(2 * np.pi * X["season"] / 4)
    X["season_cos"] = np.cos(2 * np.pi * X["season"] / 4)

    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    X["weekday_sin"] = np.sin(2 * np.pi * X["weekday"] / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * X["weekday"] / 7)
    
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

    return X.drop(columns=["date", "hour", "weekday", "month", "season"])

### ENCODER FOR PIPELINE
class FeatureEngineer:
    """
    A combined feature engineering class to apply _encode_dates
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _encode_dates(X)
        return X
    
    def _encode_dates(self, X):
        X = _encode_dates(X)
        return X