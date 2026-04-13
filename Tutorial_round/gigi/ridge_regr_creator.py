import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ==========================================================
# 1) LOAD DATA
# ==========================================================
prices_m2 = pd.read_csv("../Data/prices_round_0_day_-2.csv", sep=";")
prices_m1 = pd.read_csv("../Data/prices_round_0_day_-1.csv", sep=";")

df_prices = pd.concat([prices_m2, prices_m1], ignore_index=True)
df_prices = df_prices.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)


# ==========================================================
# 2) FEATURE ENGINEERING
# ==========================================================
def build_features(df_prices: pd.DataFrame, product_name: str, h: int = 1) -> pd.DataFrame:
    df = (
        df_prices[df_prices["product"] == product_name]
        .copy()
        .sort_values(["day", "timestamp"])
        .reset_index(drop=True)
    )

    df["mid"] = (df["bid_price_1"] + df["ask_price_1"]) / 2

    df["ask_volume_1_abs"] = df["ask_volume_1"].abs()
    df["ask_volume_2_abs"] = df["ask_volume_2"].abs() if "ask_volume_2" in df.columns else 0

    den1 = df["bid_volume_1"] + df["ask_volume_1_abs"]
    df["microprice"] = np.where(
        den1 > 0,
        (df["bid_price_1"] * df["ask_volume_1_abs"] + df["ask_price_1"] * df["bid_volume_1"]) / den1,
        df["mid"]
    )

    df["spread"] = df["ask_price_1"] - df["bid_price_1"]

    df["imbalance_1"] = np.where(
        den1 > 0,
        (df["bid_volume_1"] - df["ask_volume_1_abs"]) / den1,
        0.0
    )

    # Imbalance level 2
    if "bid_volume_2" in df.columns and "ask_volume_2" in df.columns:
        den2 = df["bid_volume_2"].fillna(0) + df["ask_volume_2_abs"].fillna(0)
        df["imbalance_2"] = np.where(
            den2 > 0,
            (df["bid_volume_2"].fillna(0) - df["ask_volume_2_abs"].fillna(0)) / den2,
            0.0
        )
    else:
        df["imbalance_2"] = 0.0


    df["bid_depth_tot"] = df["bid_volume_1"].fillna(0) + df["bid_volume_2"].fillna(0)
    df["ask_depth_tot"] = df["ask_volume_1_abs"].fillna(0) + df["ask_volume_2_abs"].fillna(0)

    den_tot = df["bid_depth_tot"] + df["ask_depth_tot"]
    df["imbalance_tot"] = np.where(
        den_tot > 0,
        (df["bid_depth_tot"] - df["ask_depth_tot"]) / den_tot,
        0.0
    )

    df["micro_edge"] = df["microprice"] - df["mid"]

    g = df.groupby("day", group_keys=False)

    df["mid_fut"] = g["mid"].shift(-h)
    df["y_micro"] = df["mid_fut"] - df["microprice"]

    return df


# ==========================================================
# 3) METRICS
# ==========================================================
def metrics_dict(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(name, m):
    print(f"{name:25s} | RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | R2={m['r2']:.6f}")


# ==========================================================
# 4) FIT RIDGECV LITE
# ==========================================================

def fit_ridge_lite_for_product(df_prices: pd.DataFrame, product_name: str, h: int = 1):
    df_feat = build_features(df_prices, product_name, h=h)

    # SOLO VERSIONE LITE
    feature_cols = [
        "micro_edge",
        "spread",
        "imbalance_1",
        "imbalance_2",
        "imbalance_tot",
    ]

    needed_cols = ["day", "timestamp", "mid", "microprice", "mid_fut", "y_micro"] + feature_cols
    model_df = df_feat[needed_cols].dropna().copy()

    train = model_df[model_df["day"] == -2].copy()
    test = model_df[model_df["day"] == -1].copy()

    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"No train/test rows for {product_name}")

    X_train = train[feature_cols]
    y_train = train["y_micro"]

    X_test = test[feature_cols]
    y_test = test["y_micro"]

    alphas = np.logspace(-4, 4, 100)
    tscv = TimeSeriesSplit(n_splits=5)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=tscv))
    ])

    model.fit(X_train, y_train)

    ridge = model.named_steps["ridge"]
    scaler = model.named_steps["scaler"]

    yhat_test = model.predict(X_test)
    pred_mid_fut_ridge = test["microprice"].values + yhat_test

    pred_mid_fut_rw = test["mid"].values
    pred_mid_fut_micro = test["microprice"].values
    true_mid_fut = test["mid_fut"].values

    metrics_rw = metrics_dict(true_mid_fut, pred_mid_fut_rw)
    metrics_micro = metrics_dict(true_mid_fut, pred_mid_fut_micro)
    metrics_ridge = metrics_dict(true_mid_fut, pred_mid_fut_ridge)

    # coefficienti nella scala originale
    coef_original_scale = ridge.coef_ / scaler.scale_
    intercept_original_scale = ridge.intercept_ - np.sum((scaler.mean_ / scaler.scale_) * ridge.coef_)

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coef": coef_original_scale
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    improvement_vs_micro = 1 - (metrics_ridge["rmse"] / metrics_micro["rmse"])

    print("" * 50)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train, i) for i in range(len(X_train.columns))]
    print(vif_data)

    print("\n" + "=" * 70)
    print(f"{product_name} | h={h} | selected alpha={ridge.alpha_}")
    print("=" * 70)
    print(f"Train rows: {len(train)} | Test rows: {len(test)}\n")

    print_metrics("RW baseline (mid_t)", metrics_rw)
    print_metrics("Microprice", metrics_micro)
    print_metrics("RidgeCV lite", metrics_ridge)

    print(f"\nImprovement vs Microprice RMSE: {improvement_vs_micro:.4%}")

    print("\nCoefficients (original scale):")
    print(coef_df[["feature", "coef"]])

    print("\nIntercept (original scale):")
    print(intercept_original_scale)

    return {
        "product": product_name,
        "h": h,
        "selected_alpha": float(ridge.alpha_),
        "df_feat": df_feat,
        "model_df": model_df,
        "train": train,
        "test": test,
        "model": model,
        "ridge": ridge,
        "coef_df": coef_df,
        "intercept": float(intercept_original_scale),
        "feature_cols_sq": feature_cols,
        "metrics_rw": metrics_rw,
        "metrics_micro": metrics_micro,
        "metrics_ridge": metrics_ridge,
        "improvement_vs_micro": improvement_vs_micro,
    }





# ==========================================================
# 5) HELPER TO PRINT HARDCODE FORMULA
# ==========================================================
def print_formula_for_trader(res):
    print("\n" + "=" * 70)
    print(f"HARDCODE FORMULA FOR {res['product']}")
    print("=" * 70)

    print(f"intercept = {res['intercept']:.12f}")
    for _, row in res["coef_df"].iterrows():
        print(f"{row['feature']}: {row['coef']:.12f}")


# ==========================================================
# 6) RUN
# ==========================================================
res_emeralds = fit_ridge_lite_for_product(df_prices, "EMERALDS", h=1)
res_tomatoes = fit_ridge_lite_for_product(df_prices, "TOMATOES", h=1)

print_formula_for_trader(res_emeralds)
print_formula_for_trader(res_tomatoes)