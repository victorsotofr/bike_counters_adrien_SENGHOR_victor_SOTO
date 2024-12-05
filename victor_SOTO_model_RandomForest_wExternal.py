from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # To handle the NaNs
from sklearn.pipeline import make_pipeline

from victor_SOTO_model_Feature_engineering import FeatureEngineer
import utils

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 

### LOAD THE DATA, MERGE THE EXTERNAL DATA
# Load
X, y = utils.get_train_data()
external_data = pd.read_csv(Path("external_data") / "external_data.csv")

# Merge
X['date'] = pd.to_datetime(X['date'])
external_data['date'] = pd.to_datetime(external_data['date'])
external_data_cleaned = external_data.drop_duplicates(subset='date')

X = pd.merge(X, external_data_cleaned, on='date', how='left') # Left join on the 'date' column

### PIPELINE CREATION
# Columns of interest:
numeric_features = [
    'hour', 'is_weekend', 'is_holiday', 'month_sin', 'month_cos',
    'weekday_sin', 'weekday_cos', 'arrondissement',
    't', 'ww', 'cl', 'tend24', 'ff', 'etat_sol', 'rr3' # external_data features
]
categorical_features = ['counter_name', 'site_name', 'season']

# We replace NaN values for numeric and categorical features since they are
# not well handled by regression models:
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Encoder:
categorical_encoder = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_imputer, numeric_features),
        ("categorical", categorical_encoder, categorical_features),
    ],
    remainder="drop"  # Drop columns not specified
)

# We create the full pipeline:
pipeline = make_pipeline(
    FeatureEngineer(),     # Apply FeaturEngineering
    preprocessor,          # Apply imputation and encoding
    RandomForestRegressor() # RandomForest regression model
)

### TRAIN_TEST_split and RMSE measures:
def train_test_split_temporal(X, y, delta_threshold="30 days"):
    
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid

X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

pipeline.fit(X_train, y_train)

# We print the RMSE obtained on the train and test sets:
print(
    f"Train set, RMSE={mean_squared_error(y_train, pipeline.predict(X_train), squared=False):.2f}"
)
print(
    f"Validation set, RMSE={mean_squared_error(y_valid, pipeline.predict(X_valid), squared=False):.2f}"
)

### PREDICTION
test_data = pd.read_parquet(Path("data") / "final_test.parquet")

# Merge the test set with the external data:
test_data['date'] = pd.to_datetime(test_data['date'])
external_data['date'] = pd.to_datetime(external_data['date'])
external_data_cleaned = external_data.drop_duplicates(subset='date')

merged_test_data = pd.merge(test_data, external_data_cleaned, on='date', how='left')

# Feature engineering:

predictions = pipeline.predict(merged_test_data)

### SUBMISSION
output_df = pd.DataFrame({
    'Id': test_data.index,  # Use the original index or a specific ID column if it exists
    'log_bike_count': predictions
})

# Format log_bike_count:
output_df['log_bike_count'] = output_df['log_bike_count'].map(lambda x: f"{x:.4f}")

# Save to CSV:
output_df.to_csv('victor_SOTO_submission_RandomForest_wExternal_v0.csv', index=False)
print("Predictions saved to 'victor_SOTO_submission_RandomForest_wExternal_v0.csv'.")