import utils
import pandas as pd
import numpy as np
import FINAL_FeatureEngineering

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# DATA IMPORT
# ==============================================================================
X, y = utils.get_train_data()

# DATA PREPROCESSING
# ==============================================================================
numeric_features = [### DATA
                    # Relative to dates
                    'year',
                    'season_sin',
                    'season_cos',
                    'month_sin',
                    'month_cos',
                    'weekday_sin',
                    'weekday_cos',
                    'hour_sin',
                    'hour_cos',
                    'is_weekend',
                    'is_holiday',
                    'is_covid',
                    # Relative to location:
                    'site_id',
                    'latitude',
                    'longitude',

                    ### EXTERNAL DATA
                    't',
                    'raf10',
                    'etat_sol',
                    'nnuage1', 
                    'rr12', 
                    'rr24',
                    'cl', 
                    'ssfrai', 
                    'w2'
                    ]


preprocessor = ColumnTransformer(
    [('standard-scaler', StandardScaler(), numeric_features)],
    remainder='drop'
    )

# PIPELINE CREATION
# ==============================================================================
model = make_pipeline(
    FINAL_FeatureEngineering.FeatureEngineering(),
    preprocessor,
    HistGradientBoostingRegressor(
        learning_rate=0.05, 
        max_iter=300, 
        max_depth=10) # default values
)

# MODEL FITTING
# ==============================================================================
def train_test_split_temporal(X, y, delta_threshold="30 days"):
    
    cutoff_date = X["date"].max() - pd.Timedelta(delta_threshold)
    mask = (X["date"] <= cutoff_date)
    X_train, X_valid = X.loc[mask], X.loc[~mask]
    y_train, y_valid = y[mask], y[~mask]

    return X_train, y_train, X_valid, y_valid

X_train, y_train, X_valid, y_valid = train_test_split_temporal(X, y)

model.fit(X_train, y_train)

# MODEL ERROR
# ==============================================================================
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

# Calculate and print RMSE for train and validation sets
print(f"Train set, RMSE={mean_squared_error(y_train, y_train_pred, squared=False):.2f}")
print(f"Validation set, RMSE={mean_squared_error(y_valid, y_valid_pred, squared=False):.2f}")

# MODEL PREDICT
# ==============================================================================
test_set = pd.read_parquet("./data/final_test.parquet")

predictions = model.predict(test_set)

# SUBMISSION
# ==============================================================================
output_df = pd.DataFrame({
    'Id': test_set.index,
    'log_bike_count': predictions
})

# Format log_bike_count:
output_df['log_bike_count'] = output_df['log_bike_count'].map(lambda x: f"{x:.4f}")

# Save to CSV:
output_df.to_csv('FINAL_predictions_HGB.csv', index=False)
print("Predictions saved to 'FINAL_predictions_HGB.csv'.")