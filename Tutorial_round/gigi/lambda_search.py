import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
    df["ask_volume_3_abs"] = df["ask_volume_3"].abs() if "ask_volume_3" in df.columns else 0

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

    if "bid_volume_2" in df.columns and "ask_volume_2" in df.columns:
        bid_v2 = df["bid_volume_2"].fillna(0)
        ask_v2 = df["ask_volume_2_abs"].fillna(0)
        den2 = bid_v2 + ask_v2
        df["imbalance_2"] = np.where(
            den2 > 0,
            (bid_v2 - ask_v2) / den2,
            0.0
        )
    else:
        bid_v2 = 0
        ask_v2 = 0
        df["imbalance_2"] = 0.0

    if "bid_volume_3" in df.columns and "ask_volume_3" in df.columns:
        bid_v3 = df["bid_volume_3"].fillna(0)
        ask_v3 = df["ask_volume_3_abs"].fillna(0)
        den3 = bid_v3 + ask_v3
        df["imbalance_3"] = np.where(
            den3 > 0,
            (bid_v3 - ask_v3) / den3,
            0.0
        )
    else:
        bid_v3 = 0
        ask_v3 = 0
        df["imbalance_3"] = 0.0

    df["bid_depth_tot"] = df["bid_volume_1"].fillna(0) + bid_v2 + bid_v3
    df["ask_depth_tot"] = df["ask_volume_1_abs"].fillna(0) + ask_v2 + ask_v3

    den_tot = df["bid_depth_tot"] + df["ask_depth_tot"]
    df["imbalance_tot"] = np.where(
        den_tot > 0,
        (df["bid_depth_tot"] - df["ask_depth_tot"]) / den_tot,
        0.0
    )

    df["micro_edge"] = df["microprice"] - df["mid"]

    g = df.groupby("day", group_keys=False)
    df["mid_fut"] = g["mid"].shift(-h)

    # target della ridge
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
    print(f"{name:30s} | RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | R2={m['r2']:.6f}")


# ==========================================================
# 4) FIT RIDGE
# ==========================================================
def fit_ridge_and_opt_lambda(df_prices: pd.DataFrame, product_name: str, h: int = 1):
    df_feat = build_features(df_prices, product_name, h=h)

    feature_cols = [
        "micro_edge",
        "spread",
        "imbalance_1",
        "imbalance_2",
        "imbalance_3",
        "imbalance_tot",
    ]

    needed_cols = ["day", "timestamp", "mid", "microprice", "mid_fut", "y_micro"] + feature_cols
    model_df = df_feat[needed_cols].dropna().copy()

    train = model_df[model_df["day"] == -2].copy()
    test = model_df[model_df["day"] == -1].copy()

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

    # prediction of y_hat on test
    yhat_test = model.predict(X_test)

    # true mid future
    true_mid_fut = test["mid_fut"].to_numpy()
    micro_test = test["microprice"].to_numpy()

    # ------------------------------------------------------
    # lambda = 1 benchmark
    # ------------------------------------------------------
    pred_lambda_1 = micro_test + yhat_test

    # ------------------------------------------------------
    # lambda* closed form
    # minimize sum( (mid_fut - micro - lambda*yhat)^2 )
    # ------------------------------------------------------
    a = true_mid_fut - micro_test
    denom = np.sum(yhat_test ** 2)

    if denom <= 1e-12:
        lambda_closed = 0.0
    else:
        lambda_closed = float(np.sum(yhat_test * a) / denom)

    pred_lambda_closed = micro_test + lambda_closed * yhat_test

    # ------------------------------------------------------
    # grid search lambda
    # ------------------------------------------------------
    lambda_grid = np.linspace(-1.0, 2.0, 301)
    best_lambda_grid = None
    best_rmse = np.inf

    for lam in lambda_grid:
        pred = micro_test + lam * yhat_test
        rmse = np.sqrt(mean_squared_error(true_mid_fut, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lambda_grid = float(lam)

    pred_lambda_grid = micro_test + best_lambda_grid * yhat_test

    # ------------------------------------------------------
    # baselines
    # ------------------------------------------------------
    pred_rw = test["mid"].to_numpy()
    pred_micro = micro_test

    metrics_rw = metrics_dict(true_mid_fut, pred_rw)
    metrics_micro = metrics_dict(true_mid_fut, pred_micro)
    metrics_lambda_1 = metrics_dict(true_mid_fut, pred_lambda_1)
    metrics_lambda_closed = metrics_dict(true_mid_fut, pred_lambda_closed)
    metrics_lambda_grid = metrics_dict(true_mid_fut, pred_lambda_grid)

    print("\n" + "=" * 80)
    print(f"{product_name} | h={h}")
    print("=" * 80)
    print(f"Train rows: {len(train)} | Test rows: {len(test)}\n")

    print_metrics("RW baseline (mid_t)", metrics_rw)
    print_metrics("Microprice", metrics_micro)
    print_metrics("Ridge with lambda = 1", metrics_lambda_1)
    print_metrics("Ridge with lambda* closed", metrics_lambda_closed)
    print_metrics("Ridge with lambda* grid", metrics_lambda_grid)

    print("\nSelected alpha for ridge:", model.named_steps["ridge"].alpha_)
    print("Optimal lambda (closed form):", lambda_closed)
    print("Optimal lambda (grid):      ", best_lambda_grid)

    return {
        "product": product_name,
        "model": model,
        "feature_cols": feature_cols,
        "train": train,
        "test": test,
        "yhat_test": yhat_test,
        "lambda_closed": lambda_closed,
        "lambda_grid": best_lambda_grid,
        "metrics_rw": metrics_rw,
        "metrics_micro": metrics_micro,
        "metrics_lambda_1": metrics_lambda_1,
        "metrics_lambda_closed": metrics_lambda_closed,
        "metrics_lambda_grid": metrics_lambda_grid,
    }


# ==========================================================
# 5) RUN
# ==========================================================
res_emeralds = fit_ridge_and_opt_lambda(df_prices, "EMERALDS", h=1)
res_tomatoes = fit_ridge_and_opt_lambda(df_prices, "TOMATOES", h=1)