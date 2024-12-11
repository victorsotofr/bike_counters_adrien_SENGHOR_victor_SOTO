from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# DATA MERGING
# ==============================================================================
def _merge_external_data(X):
    
    X = X.copy()

    external_data = pd.read_csv("./external_data/external_data.csv").copy()

    columns_of_interest = [
        'date', 
        't', # temperature
        'raf10', # rafales sur les 10 dernières minutes
        'etat_sol', # etat du sol
        'nnuage1', # nébulosité couche de nuage 1
        'rr12', # précipitations dans les 12 dernières heures
        'rr24',
        'cl', # type des nuages de l'étage inférieur
        'ssfrai', # hauteur de la neige fraîche
        'w2' # temps passé 2
        ]
    
    # Conciliate date type
    external_data["date"] = pd.to_datetime(external_data["date"]).astype("datetime64[us]")

    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), external_data[columns_of_interest].sort_values("date"), on="date"
    )
    X = X.ffill() # replacing NaN by the last non missing value
    # when there is no last value
    for col in X.columns:
        if X[col].isna().any():  
            mode_value = X[col].mode().iloc[0]
            X[col] = X[col].fillna(mode_value)

    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

# DATA ENCODING
# ==============================================================================
class RepeatingBasisFunction:
    def __init__(self, n_periods=12, column="day_of_year", input_range=(1, 365), remainder="drop"):
        self.n_periods = n_periods
        self.column = column
        self.input_range = input_range
        self.remainder = remainder
        self.centers = None
        self.widths = None
        
    def fit(self, X):
        """Fit the RBF based on the input data."""
        # Create the centers of the basis functions
        self.centers = np.linspace(self.input_range[0], self.input_range[1], self.n_periods)
        # Set the widths (this can be customized)
        self.widths = np.ones_like(self.centers) * ((self.input_range[1] - self.input_range[0]) / self.n_periods) / 2

    def transform(self, X):
        """Transform the input data using radial basis functions."""
        transformed_data = pd.DataFrame(index=X.index)

        # Calculate the radial basis functions for each center
        for center, width in zip(self.centers, self.widths):
            rbf_values = np.exp(-((X[self.column] - center) ** 2) / (2 * (width ** 2)))
            transformed_data[f"rbf_{center:.2f}"] = rbf_values
            
        return transformed_data
    

def _encode_dates(X):
    """
    Feature engineering related to dates, including:
    - Weekend detection
    - Public holiday detection
    - Radial basis encoding for days of the year
    
    Parameters:
    X (pd.DataFrame): Input DataFrame with a "date" column.

    Returns:
    pd.DataFrame: Transformed DataFrame with new features.
    """
    
    X = X.copy()  # Modify a copy of X

    # Encode the date information from the "date" column
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["day_of_year"] = X["date"].dt.dayofyear  # Add day of year for RBF

    # WEEKENDS
    X["is_weekend"] = X["weekday"].isin([5, 6]).astype(int)

    # HOLIDAYS
    french_holidays = [
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

    # COVID periods
    covid_periods = [
        (datetime(2020, 3, 17), datetime(2020, 5, 11)),  # First lockdown
        (datetime(2020, 10, 30), datetime(2020, 12, 15)),  # Second lockdown
        (datetime(2021, 4, 3), datetime(2021, 5, 3)),  # Third lockdown
    ]
    
    def is_covid_period(date):
        return any(start <= date <= end for start, end in covid_periods)

    X["is_covid"] = X["date"].apply(is_covid_period).astype(int)

    # Radial Basis Function Transformation
    rbf = RepeatingBasisFunction(n_periods=12, column="day_of_year", input_range=(1, 365))
    rbf.fit(X)
    X_rbf = rbf.transform(X)

    # Combine the original DataFrame with RBF features
    X = pd.concat([X, X_rbf], axis=1)

    return X

# CLASS CREATION
# ==============================================================================
class FeatureEngineering:
    """
    A combined feature engineering class to apply _merge_external_data and _encode_dates
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _merge_external_data(X)
        X = _encode_dates(X)
        return X