# pip install catboost==1.2.5 pandas numpy scikit-learn
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from catboost import CatBoostRegressor, Pool

# --------------------------
# Config
# --------------------------
DATA_DIR = "data"
ID_COL   = "Id"
TARGET   = "SalePrice"

# More folds + stratification to stabilize OOF
FOLDS = 10
SEEDS = [13, 42, 73, 101, 233, 777, 1337, 2025]  # bag more on H100

SENTINEL = "__NA__"
RARE_MIN_COUNT = 20  # more aggressive
MAX_ITERS = 6000

# --------------------------
# Utilities
# --------------------------
def drop_low_info(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are constant or (almost) empty."""
    keep = []
    for c in df.columns:
        nun = df[c].nunique(dropna=False)
        if nun <= 1:
            continue
        # Extremely sparse mostly-NaN object columns are often noise
        if df[c].isna().mean() > 0.98:
            continue
        keep.append(c)
    return df[keep]

def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    """High-value engineered features without label leakage."""
    df = df.copy()
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["TotalBath"] = (
        df["FullBath"] + 0.5*df["HalfBath"] +
        df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
    )
    df["AgeSold"] = df["YrSold"] - df["YearBuilt"]
    df["AgeRemod"] = df["YrSold"] - df["YearRemodAdd"]
    df["OverallQual_SF"] = df["OverallQual"] * np.log1p(np.maximum(df["TotalSF"], 0))
    df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["RoomsPerSF"] = df["TotRmsAbvGrd"] / np.clip(df["GrLivArea"], 1, None)
    df["GarageAreaPerCar"] = df["GarageArea"] / np.clip(df["GarageCars"], 1, None)
    return df

def impute_by_neighborhood(train_df, test_df):
    """LotFrontage imputation by Neighborhood median (classic Ames trick)."""
    both = pd.concat([train_df, test_df], axis=0, copy=True)
    med = both.groupby("Neighborhood")["LotFrontage"].median()
    train_df["LotFrontage"] = train_df.apply(
        lambda r: med[r["Neighborhood"]] if pd.isna(r["LotFrontage"]) else r["LotFrontage"], axis=1
    )
    test_df["LotFrontage"] = test_df.apply(
        lambda r: med[r["Neighborhood"]] if pd.isna(r["LotFrontage"]) else r["LotFrontage"], axis=1
    )
    return train_df, test_df

def add_ordinal_scores(df):
    """Add numeric ordinal versions of quality columns (keep originals for CTR)."""
    qual_map = {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5, SENTINEL:0}
    cols = ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]
    for c in cols:
        if c in df.columns:
            col = df[c].fillna(SENTINEL).astype(str)
            df[c+"_ord"] = col.map(qual_map).fillna(0).astype(float)
    return df

def prepare_categoricals(X: pd.DataFrame, X_test: pd.DataFrame):
    """Cast cats to strings, fill NaNs, collapse rare levels; return cat_features indices."""
    from pandas.api.types import is_object_dtype

    force_cat = {"MSSubClass", "MoSold", "YrSold"}
    cat_cols = [c for c in X.columns if (c in force_cat) or is_object_dtype(X[c])]

    # cast to string + fill NaNs
    for c in cat_cols:
        X[c] = X[c].astype("object").fillna(SENTINEL).astype(str)
        X_test[c] = X_test[c].astype("object").fillna(SENTINEL).astype(str)

    # collapse rare categories (based on TRAIN counts only)
    for c in cat_cols:
        vc = X[c].value_counts()
        rare = set(vc[vc < RARE_MIN_COUNT].index)
        if rare:
            X[c] = X[c].where(~X[c].isin(rare), "__RARE__")
            X_test[c] = X_test[c].where(~X_test[c].isin(rare), "__RARE__")

    cat_features = [X.columns.get_loc(c) for c in cat_cols]
    return X, X_test, cat_features, cat_cols

def stratify_bins(y_log, n_bins=10):
    """Stratify by target distribution to stabilize folds."""
    bins = pd.qcut(y_log, q=n_bins, duplicates="drop")
    # map to ints
    codes = pd.factorize(bins, sort=True)[0]
    return codes

def rmsle_from_log(log_preds, y_true_log) -> float:
    """RMSLE equals RMSE in log-space."""
    diff = log_preds - y_true_log
    return float(np.sqrt(np.mean(diff * diff)))

# --------------------------
# Load
# --------------------------
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# --------------------------
# Light cleanup
# --------------------------
# Outlier policy: large GrLivArea with suspiciously low price
mask_outlier = (train["GrLivArea"] > 4000) & (train[TARGET] < 300000)
train = train.loc[~mask_outlier].reset_index(drop=True)

# Drop constants/near-empty
train = drop_low_info(train)
test  = test[train.columns.drop([TARGET], errors="ignore").tolist() + [c for c in test.columns if c not in train.columns]]

# Neighborhood-aware LotFrontage impute
train, test = impute_by_neighborhood(train, test)

# Target in log-space
y_log = np.log1p(train[TARGET].astype(float))

# --------------------------
# Features
# --------------------------
X = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])

# Add engineered features
X = add_feats(X)
X_test = add_feats(X_test)

# Ordinal numeric scores (keep original cat columns too)
X = add_ordinal_scores(X)
X_test = add_ordinal_scores(X_test)

# Align columns
X_test = X_test.reindex(columns=X.columns)

# Categorical processing
X, X_test, cat_features, cat_cols = prepare_categoricals(X, X_test)

# --------------------------
# Model params (GPU)
# --------------------------
params = dict(
    loss_function="RMSE",         # RMSE in log-space ~= RMSLE
    eval_metric="RMSE",
    learning_rate=0.03,
    iterations=MAX_ITERS,
    depth=8,                      # tune {6,8,10}
    l2_leaf_reg=10.0,             # stronger L2
    random_strength=2.0,          # a bit more noise
    bootstrap_type="Bayesian",
    bagging_temperature=0.75,     # slightly stronger bagging
    one_hot_max_size=32,          # one-hot more small-card cats
    border_count=254,
    task_type="GPU",
    devices="0",
    od_type="Iter",
    od_wait=700,                  # patience; early stop will cut
    verbose=250
)

# --------------------------
# CV + seed bagging with stratified folds
# --------------------------
strata = stratify_bins(y_log, n_bins=10)

oof_log_pred = np.zeros(len(X), dtype=float)
test_log_pred_accum = np.zeros(len(X_test), dtype=float)

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    fold_test_log = np.zeros(len(X_test), dtype=float)
    fold_oof = np.zeros(len(X), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strata), 1):
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

        print(f"[seed {seed}] fold {fold} RMSLE: {rmsle_from_log(va_pred_log, y_va.values):.5f}")

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
sub = pd.read_csv("sample_submission.csv")
sub["SalePrice"] = np.clip(np.expm1(test_log_pred_accum), 0, None)
sub.to_csv("submission_catboost.csv", index=False)
print("Wrote submission_catboost.csv")

# --------------------------
# Optional: minimal grid to try after the above
# --------------------------
# depth: {6, 8, 10}
# l2_leaf_reg: {6, 10, 20}
# random_strength: {1.5, 2.0}
# learning_rate: {0.025, 0.03}
# bagging_temperature: {0.5, 0.75, 1.0}
