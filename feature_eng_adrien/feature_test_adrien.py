from pathlib import Path
import pandas as pd
import numpy as np

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
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X