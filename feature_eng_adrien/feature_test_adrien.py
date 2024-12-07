
from feature_eng_adrien import feature_training_adrien
import pandas as pd

feature_engineer = feature_training_adrien.FeatureEngineer()

# same as training but without the 'bike count' column
def _delete_columns_test(X):
    X = X.copy()  # modify a copy of X
    col_delete = [
        "counter_id",  # I only keep the site_id
        "counter_name",  # same
        "site_name",  # same
        "counter_technical_id",  # same
        "coordinates",  # I prefer to get latitude and longitude
        "counter_installation_date",
    ]  # for my example I remove it because can't fit the model with that but still a data to use
    X = X.drop(columns=col_delete)
    return X

class FeatureEngineerTest:
    """
    A combined feature engineering class to apply _encode_dates and _encode_lat_lon
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = feature_engineer._encode_dates(X)
        #X = feature_engineer._encode_lat_lon(X)
        X = _delete_columns_test(X)
        return X