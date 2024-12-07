import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin

### FEATURE ENGINEERING
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

    # Seasons
    def get_season(date):
        if date.month in [12, 1, 2]:  # Winter
            return 'Winter'
        elif date.month in [3, 4, 5]:  # Spring
            return 'Spring'
        elif date.month in [6, 7, 8]:  # Summer
            return 'Summer'
        else:  # Autumn
            return 'Autumn'

    X["season"] = X["date"].apply(get_season)

    # Cyclical encoding for months
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    X["weekday_sin"] = np.sin(2 * np.pi * X["weekday"] / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * X["weekday"] / 7)

    return X.drop(columns=["date", "month", "weekday"])


def _encode_lat_lon(X):
    """
    We will do feature engineering related to latitude and longitude, including:
    - The parisian arrondissement the counter is located
    
    Parameters:
    X (pd.DataFrame): Input DataFrame with a "latitude" and a "longitude" column.

    Returns:
    pd.DataFrame: Transformed DataFrame with new features.
    """

    X = X.copy()  # modify a copy of X

    # We took some help to find the list of the coordinates of the Parisian arrondissements:
    arrondissement_ranges = {
    1: ((48.861992, 48.865215), (2.332125, 2.336405)),  # 1st arrondissement
    2: ((48.863206, 48.866611), (2.341881, 2.347173)),  # 2nd arrondissement
    3: ((48.861255, 48.865822), (2.360033, 2.367505)),  # 3rd arrondissement
    4: ((48.843799, 48.861194), (2.354573, 2.365420)),  # 4th arrondissement
    5: ((48.831783, 48.843197), (2.342185, 2.354164)),  # 5th arrondissement
    6: ((48.841360, 48.851082), (2.322960, 2.335156)),  # 6th arrondissement
    7: ((48.855246, 48.861785), (2.292324, 2.304169)),  # 7th arrondissement
    8: ((48.871865, 48.876433), (2.298935, 2.316489)),  # 8th arrondissement
    9: ((48.878293, 48.886988), (2.332082, 2.342776)),  # 9th arrondissement
    10: ((48.867081, 48.877137), (2.354785, 2.368476)), # 10th arrondissement
    11: ((48.860539, 48.868028), (2.368059, 2.377489)), # 11th arrondissement
    12: ((48.838929, 48.853496), (2.372685, 2.395029)), # 12th arrondissement
    13: ((48.832859, 48.845874), (2.364123, 2.377156)), # 13th arrondissement
    14: ((48.835038, 48.844455), (2.308681, 2.334001)), # 14th arrondissement
    15: ((48.846062, 48.868057), (2.285918, 2.314678)), # 15th arrondissement
    16: ((48.846780, 48.876165), (2.246473, 2.296048)), # 16th arrondissement
    17: ((48.873200, 48.887113), (2.284953, 2.319586)), # 17th arrondissement
    18: ((48.877165, 48.895797), (2.324915, 2.363556)), # 18th arrondissement
    19: ((48.868440, 48.886066), (2.377394, 2.396128)), # 19th arrondissement
    20: ((48.855109, 48.873755), (2.382319, 2.411434))  # 20th arrondissement
    }
    
    # We will use a function to determine, based on the lat & lon of the counter,
    # in which arrondissement it is located:
    def find_arrondissement(lat, lon):
        for arr, ((lat_min, lat_max), (lon_min, lon_max)) in arrondissement_ranges.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return arr
        return None
    
    X['arrondissement'] = X.apply(lambda row: find_arrondissement(row['latitude'], row['longitude']), axis=1)

    return X.drop(columns=["latitude", "longitude"])

### ENCODER FOR PIPELINE
class FeatureEngineer:
    """
    A combined feature engineering class to apply _encode_dates and _encode_lat_lon
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _encode_dates(X)
        X = _encode_lat_lon(X)
        return X