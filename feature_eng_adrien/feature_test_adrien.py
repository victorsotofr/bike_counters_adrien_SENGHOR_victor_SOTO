from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Pour merge mes deux datasets en ajoutant certains features préselectionnés de l'external data
def merge_external_data(X, df_ext, col_ext):
    df_ext = df_ext.copy()
    X = X.copy()

    # Conciliate date type
    df_ext["date"] = pd.to_datetime(df_ext["date"]).astype("datetime64[us]")

    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[col_ext].sort_values("date"), on="date"
    )
    X = X.ffill() # replacing NaN by the last non missing value
    # when there is no last value
    for col in X.columns:
        if X[col].isna().any():  
            mode_value = X[col].mode().iloc[0]
            X[col].fillna(mode_value, inplace=True)

    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def _encode_dates(X):
    """
    We will do feature engineering related to dates, including:
    - Weekend: it has a big influence on the pattern
    - Public holiday detection: it can have a big influence on the pattern
    - Seasons: it can have a big influence on the pattern
    - We introduce Cyclical encoding for month and weekday,
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
    
    # separate night from day
    X["is_night"] = ((X["hour"] <= 5) | (X["hour"] >= 22)).astype(int)

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

    # Covid
    # Périodes de confinement à Paris (2020-2021)
    covid_periods = [
        (datetime(2020, 3, 17), datetime(2020, 5, 11)),  # Premier confinement
        (datetime(2020, 10, 30), datetime(2020, 12, 15)),  # Deuxième confinement
        (datetime(2021, 4, 3), datetime(2021, 5, 3)),  # Troisième confinement
    ]
    def is_covid_period(date):
        for start, end in covid_periods:
            if start <= date <= end:
                return 1
        return 0
    X["is_covid"] = X["date"].apply(is_covid_period)

    # Cyclical encoding for months
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    # Seasons
    def get_season(date):
        if date.month in [12, 1, 2]:  # Winter
            return 0
        elif date.month in [3, 4, 5]:  # Spring
            return 1
        elif date.month in [6, 7, 8]:  # Summer
            return 2
        else:  # Autumn
            return 4
    

    X["season"] = X["date"].apply(get_season)

    #return X.drop(columns=["date", "month", "weekday"])
    return X
