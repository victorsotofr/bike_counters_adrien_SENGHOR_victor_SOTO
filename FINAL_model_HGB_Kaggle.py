import utils
import pandas as pd
import numpy as np
import FINAL_FeatureEngineering

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

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
        learning_rate=0.09610866770801975, 
        max_iter=484, 
        max_depth=6
        ) # parameters from Optuna
)

# MODEL FITTING
# ==============================================================================
model.fit(X, y)

# MODEL PREDICT
# ==============================================================================
test_set = pd.read_parquet("../msdb-2024/final_test.parquet")

y_pred = model.predict(test_set)

# SUBMISSION
# ==============================================================================
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)