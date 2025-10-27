# pip install catboost==1.2.5 pandas numpy scikit-learn
import os, warnings, math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor, Pool

# --- Load
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
ID_COL = "Id"
TARGET = "SalePrice"

# --- Light feature cleanup (Ames-specific, safe defaults)
# Known leverage points: very large living area with relatively low price
mask_outlier = (train["GrLivArea"] > 4000) & (train[TARGET] < 300000)
train = train.loc[~mask_outlier].reset_index(drop=True)

# --- Target: log1p for RMSLE optimization
y = np.log1p(train[TARGET].astype(float))

# --- Features
X = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])

# numeric-looking columns you still want treated as categories
force_cat = {"MSSubClass", "MoSold", "YrSold"}

# Make CatBoost’s life easy: object->category; force some ints to category
for c in X.columns:
    if c in force_cat:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    elif X[c].dtype == "object":
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

# CatBoost needs categorical feature indices (0-based)
cat_features = [i for i, c in enumerate(X.columns) if str(X[c].dtype) == "category"]

# --- CV + bagging across seeds
SEEDS = [42, 1337, 2025]   # bump to 5–10 for a bit more stability
FOLDS = 5

def rmsle_from_log(log_preds, y_true):
    """Compute RMSLE after inverse transform."""
    preds = np.expm1(log_preds)
    true  = np.expm1(y_true)
    # Clip to avoid negatives from numeric noise
    preds = np.clip(preds, 0, None)
    return mean_squared_log_error(true, preds, squared=False)

params = dict(
    loss_function="RMSE",         # we optimized in log-space (RMSLE proxy)
    eval_metric="RMSE",
    learning_rate=0.05,
    depth=8,                      # 6–10 is a good band
    l2_leaf_reg=6.0,              # 3–10 usually works well
    random_strength=1.5,
    bootstrap_type="Bayesian",
    bagging_temperature=0.5,      # mild stochasticity
    one_hot_max_size=12,
    border_count=254,             # good for GPU
    task_type="GPU",              # use H100
    devices="0",
    od_type="Iter",
    od_wait=200,
    verbose=100
)

oof_pred = np.zeros(len(X))
test_pred_accum = np.zeros(len(X_test))

for seed in SEEDS:
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    fold_preds = np.zeros(len(X_test))
    fold_oof   = np.zeros(len(X))
    for fold, (tr, va) in enumerate(kf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        tr_pool = Pool(X_tr, label=y_tr, cat_features=cat_features)
        va_pool = Pool(X_va, label=y_va, cat_features=cat_features)
        te_pool = Pool(X_test,          cat_features=cat_features)

        model = CatBoostRegressor(**params, random_seed=seed)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        va_pred_log = model.predict(va_pool)
        fold_oof[va] = va_pred_log
        fold_preds  += model.predict(te_pool) / FOLDS

        # Optional: print fold RMSLE
        fold_rmsle = rmsle_from_log(va_pred_log, y_va.values)
        print(f"[seed {seed}] fold {fold} RMSLE: {fold_rmsle:.5f}")

    # accumulate
    oof_pred += fold_oof / len(SEEDS)
    test_pred_accum += fold_preds / len(SEEDS)

# --- Evaluate OOF RMSLE
oof_rmsle = rmsle_from_log(oof_pred, y.values)
print(f"OOF RMSLE: {oof_rmsle:.5f}")

# --- Make submission
sub = pd.read_csv("sample_submission.csv")
sub["SalePrice"] = np.clip(np.expm1(test_pred_accum), 0, None)
sub.to_csv("submission_catboost.csv", index=False)
print("Wrote submission_catboost.csv")
