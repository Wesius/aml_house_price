# pip install catboost==1.2.5 pandas numpy scikit-learn
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool

# --------------------------
# Config
# --------------------------
DATA_DIR = "data"
ID_COL   = "Id"
TARGET   = "SalePrice"

SEEDS = [42, 1337, 2025]   # increase to 7–10 for a touch more stability
FOLDS = 5
SENTINEL = "__NA__"
RARE_MIN_COUNT = 10

# --------------------------
# Utils
# --------------------------
def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Compact, high-signal engineered features (no label leakage)."""
    df = df.copy()
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"] +
        df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )
    df["AgeSold"] = df["YrSold"] - df["YearBuilt"]
    df["AgeRemod"] = df["YrSold"] - df["YearRemodAdd"]
    # mild interaction that often helps
    df["OverallQual_SF"] = df["OverallQual"] * np.log1p(df["TotalSF"])
    return df

def prepare_categoricals(X: pd.DataFrame, X_test: pd.DataFrame):
    """
    Convert categorical columns to string, fill NaNs, collapse rare levels,
    and return cat_features indices.
    """
    from pandas.api.types import is_object_dtype

    # integer-coded categoricals to force
    force_cat = {"MSSubClass", "MoSold", "YrSold"}

    # identify categorical columns (object OR explicitly forced)
    cat_cols = [c for c in X.columns if (c in force_cat) or is_object_dtype(X[c])]

    # cast to string + fill NaNs
    for c in cat_cols:
        X[c] = X[c].astype("object").fillna(SENTINEL).astype(str)
        X_test[c] = X_test[c].astype("object").fillna(SENTINEL).astype(str)

    # collapse rare categories based on TRAIN frequencies only
    for c in cat_cols:
        vc = X[c].value_counts()
        rare = set(vc[vc < RARE_MIN_COUNT].index)
        if rare:
            X[c] = X[c].where(~X[c].isin(rare), "__RARE__")
            X_test[c] = X_test[c].where(~X_test[c].isin(rare), "__RARE__")

    # indices for CatBoost
    cat_features = [X.columns.get_loc(c) for c in cat_cols]
    return X, X_test, cat_features, cat_cols

def rmsle_from_log(log_preds, y_true_log) -> float:
    """RMSLE equals RMSE in log-space (since we train on log1p)."""
    diff = log_preds - y_true_log
    return float(np.sqrt(np.mean(diff * diff)))

# --------------------------
# Load
# --------------------------
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# --------------------------
# Light cleanup (Ames-specific outlier clip)
# --------------------------
mask_outlier = (train["GrLivArea"] > 4000) & (train[TARGET] < 300000)
train = train.loc[~mask_outlier].reset_index(drop=True)

# --------------------------
# Target (log-space for RMSLE)
# --------------------------
y_log = np.log1p(train[TARGET].astype(float))

# --------------------------
# Features
# --------------------------
X = add_feats(train.drop(columns=[TARGET, ID_COL]))
X_test = add_feats(test.drop(columns=[ID_COL]))

# Ensure identical column order
X_test = X_test.reindex(columns=X.columns)

# Categorical processing (strings + sentinel + rare collapse)
X, X_test, cat_features, cat_cols = prepare_categoricals(X, X_test)

# --------------------------
# Model params (GPU, early stopping, wider trees budget)
# --------------------------
params = dict(
    loss_function="RMSE",          # RMSE in log-space ~= RMSLE
    eval_metric="RMSE",
    learning_rate=0.03,            # lower LR, more trees
    iterations=5000,               # early stopping will cut this
    depth=8,                       # try {6,8,10} if tuning
    l2_leaf_reg=6.0,               # try {3,6,10,20}
    random_strength=1.5,           # try {1.0,1.5,2.0}
    bootstrap_type="Bayesian",
    bagging_temperature=0.5,
    one_hot_max_size=12,
    border_count=254,
    rsm=0.8,                       # feature subsampling
    task_type="GPU",               # use H100
    devices="0",
    od_type="Iter",
    od_wait=500,                   # give room before stopping
    verbose=200
)

# --------------------------
# CV + seed bagging
# --------------------------
oof_log_pred = np.zeros(len(X), dtype=float)
test_log_pred_accum = np.zeros(len(X_test), dtype=float)

for seed in SEEDS:
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    fold_test_log = np.zeros(len(X_test), dtype=float)
    fold_oof = np.zeros(len(X), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y_log), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]

        tr_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
        va_pool = Pool(X_va, label=y_va, cat_features=cat_features)
        te_pool = Pool(X_test,           cat_features=cat_features)

        model = CatBoostRegressor(**params, random_seed=seed)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        va_pred_log = model.predict(va_pool)
        fold_oof[va_idx] = va_pred_log
        fold_test_log += model.predict(te_pool) / FOLDS

        fold_rmsle = rmsle_from_log(va_pred_log, y_va.values)
        print(f"[seed {seed}] fold {fold} RMSLE: {fold_rmsle:.5f}")

    # accumulate across folds and seeds
    oof_log_pred += fold_oof / len(SEEDS)
    test_log_pred_accum += fold_test_log / len(SEEDS)

# --------------------------
# OOF evaluation (RMSLE)
# --------------------------
oof_rmsle = rmsle_from_log(oof_log_pred, y_log.values)
print(f"OOF RMSLE: {oof_rmsle:.5f}")

# --------------------------
# Submission
# --------------------------
sub = pd.read_csv("data/sample_submission.csv")
sub["SalePrice"] = np.clip(np.expm1(test_log_pred_accum), 0, None)
sub.to_csv("submission_catboost.csv", index=False)
print("Wrote submission_catboost.csv")
