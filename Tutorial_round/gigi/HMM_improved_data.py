import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from hmmlearn.hmm import GaussianHMM


# =============================================================================
# 1) LOAD DATA
# =============================================================================
prices_m2 = pd.read_csv("../Data/prices_round_0_day_-2.csv", sep=";")
prices_m1 = pd.read_csv("../Data/prices_round_0_day_-1.csv", sep=";")

df_prices = pd.concat([prices_m2, prices_m1], ignore_index=True)
df_prices = df_prices.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)


# =============================================================================
# 2) FEATURE ENGINEERING
# =============================================================================
def build_hmm_features(
    df_prices: pd.DataFrame,
    product_name: str,
    h: int = 3,
    tick_size: float = 1.0
) -> pd.DataFrame:
    df = (
        df_prices[df_prices["product"] == product_name]
        .copy()
        .sort_values(["day", "timestamp"])
        .reset_index(drop=True)
    )

    # Mid
    df["mid"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0

    # Ask abs volumes
    for k in [1, 2, 3]:
        ask_col = f"ask_volume_{k}"
        abs_col = f"ask_volume_{k}_abs"
        if ask_col in df.columns:
            df[abs_col] = df[ask_col].abs()
        else:
            df[abs_col] = 0.0

    # Fill missing bid/ask volumes
    for col in [
        "bid_volume_1", "bid_volume_2", "bid_volume_3",
        "ask_volume_1_abs", "ask_volume_2_abs", "ask_volume_3_abs"
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Level-1 microprice
    den1 = df["bid_volume_1"] + df["ask_volume_1_abs"]
    df["microprice"] = np.where(
        den1 > 0,
        (df["bid_price_1"] * df["ask_volume_1_abs"] + df["ask_price_1"] * df["bid_volume_1"]) / den1,
        df["mid"]
    )

    # Spread
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["spread_ticks"] = df["spread"] / tick_size

    # Imbalance L1
    df["imbalance_1"] = np.where(
        den1 > 0,
        (df["bid_volume_1"] - df["ask_volume_1_abs"]) / den1,
        0.0
    )

    # Imbalance L2
    den2 = df["bid_volume_2"] + df["ask_volume_2_abs"]
    df["imbalance_2"] = np.where(
        den2 > 0,
        (df["bid_volume_2"] - df["ask_volume_2_abs"]) / den2,
        0.0
    )

    # Imbalance L3
    den3 = df["bid_volume_3"] + df["ask_volume_3_abs"]
    df["imbalance_3"] = np.where(
        den3 > 0,
        (df["bid_volume_3"] - df["ask_volume_3_abs"]) / den3,
        0.0
    )

    # Total imbalance
    total_bid = df["bid_volume_1"] + df["bid_volume_2"] + df["bid_volume_3"]
    total_ask = df["ask_volume_1_abs"] + df["ask_volume_2_abs"] + df["ask_volume_3_abs"]
    den_tot = total_bid + total_ask

    df["imbalance_tot"] = np.where(
        den_tot > 0,
        (total_bid - total_ask) / den_tot,
        0.0
    )

    # Micro edge
    df["micro_edge"] = df["microprice"] - df["mid"]
    df["micro_edge_ticks"] = df["micro_edge"] / tick_size

    # Group by day to avoid leakage across day boundary
    g = df.groupby("day", group_keys=False)

    # Lagged / future values
    df["mid_prev"] = g["mid"].shift(1)
    df["mid_fut"] = g["mid"].shift(-h)

    df["ret_1"] = df["mid"] - df["mid_prev"]
    df["ret_fut"] = df["mid_fut"] - df["mid"]

    df["ret_1_ticks"] = df["ret_1"] / tick_size
    df["ret_fut_ticks"] = df["ret_fut"] / tick_size

    # Rolling features: these are slower and more regime-like
    df["roll_ret_mean_10"] = (
        g["ret_1_ticks"].rolling(10).mean().reset_index(level=0, drop=True)
    )
    df["roll_ret_mean_20"] = (
        g["ret_1_ticks"].rolling(20).mean().reset_index(level=0, drop=True)
    )
    df["roll_vol_10"] = (
        g["ret_1_ticks"].rolling(10).std().reset_index(level=0, drop=True)
    )
    df["roll_vol_20"] = (
        g["ret_1_ticks"].rolling(20).std().reset_index(level=0, drop=True)
    )
    df["roll_edge_mean_10"] = (
        g["micro_edge_ticks"].rolling(10).mean().reset_index(level=0, drop=True)
    )
    df["roll_edge_mean_20"] = (
        g["micro_edge_ticks"].rolling(20).mean().reset_index(level=0, drop=True)
    )
    df["roll_imb1_mean_10"] = (
        g["imbalance_1"].rolling(10).mean().reset_index(level=0, drop=True)
    )
    df["roll_imbtot_mean_10"] = (
        g["imbalance_tot"].rolling(10).mean().reset_index(level=0, drop=True)
    )

    # Activity / local motion proxy
    df["abs_ret_1_ticks"] = df["ret_1_ticks"].abs()
    df["roll_abs_ret_10"] = (
        g["abs_ret_1_ticks"].rolling(10).mean().reset_index(level=0, drop=True)
    )

    return df


# =============================================================================
# 3) HMM FIT + PROBABILITIES
# =============================================================================
def fit_hmm_states_prob(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    n_states: int = 3,
    train_days: list[int] = [-2],
    test_days: list[int] = [-1],
):
    base_cols = [
        "day",
        "timestamp",
        "mid",
        "mid_fut",
        "ret_fut",
        "ret_1",
        "micro_edge",
        "imbalance_1",
        "imbalance_tot",
        "spread",
        "spread_ticks",
        "ret_fut_ticks",
    ]

    keep_cols = list(dict.fromkeys(base_cols + feature_cols))
    model_df = df_feat[keep_cols].dropna().copy()

    train = model_df[model_df["day"].isin(train_days)].copy()
    test = model_df[model_df["day"].isin(test_days)].copy()

    if train.empty:
        raise ValueError("Train set is empty.")
    if test.empty:
        raise ValueError("Test set is empty.")

    scaler = RobustScaler()

    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=500,
        random_state=42,
        min_covar=1e-3,
        init_params="stmc",
        params="stmc"
    )

    hmm.fit(X_train)

    train["state_raw"] = hmm.predict(X_train)
    test["state_raw"] = hmm.predict(X_test)

    train_probs = hmm.predict_proba(X_train)
    test_probs = hmm.predict_proba(X_test)

    for k in range(n_states):
        train[f"p_state_{k}"] = train_probs[:, k]
        test[f"p_state_{k}"] = test_probs[:, k]

    return hmm, scaler, train, test


# =============================================================================
# 4) STATE LABELING
# =============================================================================
def label_states_by_future_return(train_df: pd.DataFrame, n_states: int = 3):
    state_summary = (
        train_df.groupby("state_raw")
        .agg(
            count=("state_raw", "size"),
            avg_ret_fut=("ret_fut", "mean"),
            median_ret_fut=("ret_fut", "median"),
            std_ret_fut=("ret_fut", "std"),
            hit_up=("ret_fut", lambda x: (x > 0).mean()),
            hit_down=("ret_fut", lambda x: (x < 0).mean()),
            avg_ret_1=("ret_1", "mean"),
            avg_micro_edge=("micro_edge", "mean"),
            avg_imb1=("imbalance_1", "mean"),
            avg_imbtot=("imbalance_tot", "mean"),
            avg_spread=("spread", "mean"),
        )
        .reset_index()
        .sort_values("avg_ret_fut")
        .reset_index(drop=True)
    )

    if len(state_summary) != n_states:
        raise ValueError(f"Expected exactly {n_states} states, got {len(state_summary)}.")

    bearish_state = int(state_summary.iloc[0]["state_raw"])
    neutral_state = int(state_summary.iloc[1]["state_raw"])
    bullish_state = int(state_summary.iloc[2]["state_raw"])

    label_map = {
        bearish_state: "bearish",
        neutral_state: "neutral",
        bullish_state: "bullish",
    }

    state_summary["label"] = state_summary["state_raw"].map(label_map)
    return label_map, state_summary


# =============================================================================
# 5) ATTACH PROBABILITIES
# =============================================================================
def attach_regime_probabilities(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    df = df.copy()

    inv_map = {v: k for k, v in label_map.items()}
    bear_idx = inv_map["bearish"]
    neu_idx = inv_map["neutral"]
    bull_idx = inv_map["bullish"]

    df["state_label"] = df["state_raw"].map(label_map)
    df["p_bear"] = df[f"p_state_{bear_idx}"]
    df["p_neutral"] = df[f"p_state_{neu_idx}"]
    df["p_bull"] = df[f"p_state_{bull_idx}"]

    df["regime_score"] = df["p_bull"] - df["p_bear"]
    df["regime_confidence"] = df[["p_bear", "p_neutral", "p_bull"]].max(axis=1)

    return df


# =============================================================================
# 6) RUN LENGTHS
# =============================================================================
def state_run_lengths(df: pd.DataFrame, state_col: str = "state_label") -> pd.DataFrame:
    runs = []

    for day, d in df.groupby("day"):
        s = d[state_col].reset_index(drop=True)
        if len(s) == 0:
            continue

        current = s.iloc[0]
        run_len = 1

        for x in s.iloc[1:]:
            if x == current:
                run_len += 1
            else:
                runs.append((day, current, run_len))
                current = x
                run_len = 1

        runs.append((day, current, run_len))

    return pd.DataFrame(runs, columns=["day", state_col, "run_length"])


def summarize_run_lengths(run_df: pd.DataFrame, state_col: str = "state_label") -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    return (
        run_df.groupby(state_col)
        .agg(
            n_runs=("run_length", "size"),
            avg_run_length=("run_length", "mean"),
            median_run_length=("run_length", "median"),
            max_run_length=("run_length", "max"),
        )
        .reset_index()
    )


# =============================================================================
# 7) TRANSITION MATRIX
# =============================================================================
def transition_matrix_labeled(hmm: GaussianHMM, label_map: dict) -> pd.DataFrame:
    raw_trans = hmm.transmat_

    ordered_labels = ["bearish", "neutral", "bullish"]
    label_to_raw = {v: k for k, v in label_map.items()}
    ordered_raw = [label_to_raw[lbl] for lbl in ordered_labels]

    trans_labeled = raw_trans[np.ix_(ordered_raw, ordered_raw)]
    return pd.DataFrame(trans_labeled, index=ordered_labels, columns=ordered_labels)


# =============================================================================
# 8) STATE SUMMARY
# =============================================================================
def summarize_states(df: pd.DataFrame, crossing_cost_ticks: float = 1.0) -> pd.DataFrame:
    out = (
        df.groupby("state_label")
        .agg(
            count=("state_label", "size"),
            avg_ret_fut=("ret_fut", "mean"),
            median_ret_fut=("ret_fut", "median"),
            std_ret_fut=("ret_fut", "std"),
            hit_up=("ret_fut", lambda x: (x > 0).mean()),
            hit_down=("ret_fut", lambda x: (x < 0).mean()),
            avg_abs_ret_fut=("ret_fut", lambda x: np.abs(x).mean()),
            avg_ret_1=("ret_1", "mean"),
            avg_micro_edge=("micro_edge", "mean"),
            avg_imb1=("imbalance_1", "mean"),
            avg_imbtot=("imbalance_tot", "mean"),
            avg_spread=("spread", "mean"),
            avg_spread_ticks=("spread_ticks", "mean"),
            avg_p_bear=("p_bear", "mean"),
            avg_p_neutral=("p_neutral", "mean"),
            avg_p_bull=("p_bull", "mean"),
            avg_regime_score=("regime_score", "mean"),
            avg_confidence=("regime_confidence", "mean"),
        )
        .reset_index()
    )

    out["edge_after_crossing_cost_ticks"] = out["avg_ret_fut"] - crossing_cost_ticks
    out["sharpe_like"] = out["avg_ret_fut"] / out["std_ret_fut"].replace(0, np.nan)

    return out.sort_values("avg_ret_fut").reset_index(drop=True)


# =============================================================================
# 9) PROBABILITY DIAGNOSTICS
# =============================================================================
def summarize_probability_softness(df: pd.DataFrame) -> pd.DataFrame:
    def second_largest(row):
        vals = sorted([row["p_bear"], row["p_neutral"], row["p_bull"]], reverse=True)
        return vals[1]

    tmp = df.copy()
    tmp["second_prob"] = tmp.apply(second_largest, axis=1)

    bins = pd.cut(
        tmp["regime_confidence"],
        bins=[0.0, 0.55, 0.70, 0.85, 0.95, 1.000001],
        labels=["<=0.55", "0.55-0.70", "0.70-0.85", "0.85-0.95", ">0.95"]
    )

    conf_table = (
        tmp.groupby(bins)
        .agg(
            count=("regime_confidence", "size"),
            avg_conf=("regime_confidence", "mean"),
            avg_second_prob=("second_prob", "mean"),
        )
        .reset_index()
    )

    return conf_table


# =============================================================================
# 10) ALPHA MODEL
# =============================================================================
def build_soft_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Softer alpha, less dominated by instantaneous jumps.
    """
    df = df.copy()

    df["alpha_raw"] = (
        0.45 * df["micro_edge"]
        + 0.20 * df["imbalance_1"]
        + 0.20 * df["imbalance_tot"]
        + 0.10 * df["roll_edge_mean_10"].fillna(0.0)
        + 0.05 * df["roll_imb1_mean_10"].fillna(0.0)
    )

    return df


def combine_alpha_and_regime(
    df: pd.DataFrame,
    regime_weight: float = 0.25,
    confidence_floor: float = 0.60
) -> pd.DataFrame:
    df = df.copy()

    regime_multiplier = np.where(
        df["regime_confidence"] >= confidence_floor,
        1.0,
        0.0
    )

    df["alpha_adj"] = df["alpha_raw"] + regime_weight * regime_multiplier * df["regime_score"]
    return df


# =============================================================================
# 11) POLICY
# =============================================================================
def choose_action(
    alpha_adj: float,
    regime_score: float,
    confidence: float,
    inventory: float,
    spread_ticks: float,
    buy_take_threshold: float = 0.60,
    sell_take_threshold: float = 0.60,
    inv_limit: float = 20.0,
    conf_min: float = 0.60,
    strong_regime_threshold: float = 0.35,
    max_spread_for_taking: float = 8.0,
):
    """
    More conservative than before.
    """
    # Uncertain regime: stay symmetric unless inventory is large
    if confidence < conf_min:
        if abs(inventory) > 0.7 * inv_limit:
            return "inventory_reduction"
        return "mm_symmetric"

    # Do not take aggressively on too-wide spread
    taking_allowed = spread_ticks <= max_spread_for_taking

    if taking_allowed and alpha_adj > buy_take_threshold and inventory < inv_limit:
        return "take_buy"

    if taking_allowed and alpha_adj < -sell_take_threshold and inventory > -inv_limit:
        return "take_sell"

    # Otherwise skew quoting
    if regime_score > strong_regime_threshold:
        if inventory < 0.5 * inv_limit:
            return "mm_skew_long"
        return "mm_light_long"

    if regime_score < -strong_regime_threshold:
        if inventory > -0.5 * inv_limit:
            return "mm_skew_short"
        return "mm_light_short"

    if abs(inventory) > 0.8 * inv_limit:
        return "inventory_reduction"

    return "mm_symmetric"


def attach_policy_actions(
    df: pd.DataFrame,
    inventory_col: str = None,
    buy_take_threshold: float = 0.60,
    sell_take_threshold: float = 0.60,
    inv_limit: float = 20.0,
    conf_min: float = 0.60,
    strong_regime_threshold: float = 0.35,
    max_spread_for_taking: float = 8.0,
) -> pd.DataFrame:
    df = df.copy()

    if inventory_col is None or inventory_col not in df.columns:
        df["inventory_proxy"] = 0.0
        inventory_col = "inventory_proxy"

    df["action"] = df.apply(
        lambda row: choose_action(
            alpha_adj=row["alpha_adj"],
            regime_score=row["regime_score"],
            confidence=row["regime_confidence"],
            inventory=row[inventory_col],
            spread_ticks=row["spread_ticks"],
            buy_take_threshold=buy_take_threshold,
            sell_take_threshold=sell_take_threshold,
            inv_limit=inv_limit,
            conf_min=conf_min,
            strong_regime_threshold=strong_regime_threshold,
            max_spread_for_taking=max_spread_for_taking,
        ),
        axis=1
    )

    return df


def summarize_actions(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["state_label", "action"])
        .agg(
            count=("action", "size"),
            avg_ret_fut=("ret_fut", "mean"),
            hit_up=("ret_fut", lambda x: (x > 0).mean()),
            avg_alpha_adj=("alpha_adj", "mean"),
            avg_conf=("regime_confidence", "mean"),
            avg_regime_score=("regime_score", "mean"),
            avg_spread_ticks=("spread_ticks", "mean"),
        )
        .reset_index()
        .sort_values(["state_label", "count"], ascending=[True, False])
    )


# =============================================================================
# 12) MAIN RUN
# =============================================================================
PRODUCT = "TOMATOES"
HORIZON = 3
TICK_SIZE = 1.0

# IMPORTANT:
# ret_1 REMOVED from the HMM feature set on purpose
feature_cols = [
    "micro_edge",
    "imbalance_1",
    "imbalance_tot",
    "spread_ticks",
    "roll_vol_10",
    "roll_vol_20",
    "roll_edge_mean_10",
    "roll_edge_mean_20",
    "roll_imb1_mean_10",
    "roll_imbtot_mean_10",
    "roll_ret_mean_10",
    "roll_ret_mean_20",
    "roll_abs_ret_10",
]

df_feat = build_hmm_features(
    df_prices=df_prices,
    product_name=PRODUCT,
    h=HORIZON,
    tick_size=TICK_SIZE
)

hmm, scaler, train_df, test_df = fit_hmm_states_prob(
    df_feat=df_feat,
    feature_cols=feature_cols,
    n_states=3,
    train_days=[-2],
    test_days=[-1],
)

label_map, train_state_summary = label_states_by_future_return(train_df, n_states=3)

train_df = attach_regime_probabilities(train_df, label_map)
test_df = attach_regime_probabilities(test_df, label_map)

train_summary_full = summarize_states(train_df, crossing_cost_ticks=1.0)
test_summary_full = summarize_states(test_df, crossing_cost_ticks=1.0)

train_runs = state_run_lengths(train_df, state_col="state_label")
test_runs = state_run_lengths(test_df, state_col="state_label")

train_run_summary = summarize_run_lengths(train_runs, state_col="state_label")
test_run_summary = summarize_run_lengths(test_runs, state_col="state_label")

trans_df = transition_matrix_labeled(hmm, label_map)

train_prob_softness = summarize_probability_softness(train_df)
test_prob_softness = summarize_probability_softness(test_df)

# Build softer alpha and policy
train_df = build_soft_alpha(train_df)
test_df = build_soft_alpha(test_df)

train_df = combine_alpha_and_regime(train_df, regime_weight=0.25, confidence_floor=0.60)
test_df = combine_alpha_and_regime(test_df, regime_weight=0.25, confidence_floor=0.60)

train_df = attach_policy_actions(
    train_df,
    inventory_col=None,
    buy_take_threshold=0.60,
    sell_take_threshold=0.60,
    inv_limit=20.0,
    conf_min=0.60,
    strong_regime_threshold=0.35,
    max_spread_for_taking=8.0,
)

test_df = attach_policy_actions(
    test_df,
    inventory_col=None,
    buy_take_threshold=0.60,
    sell_take_threshold=0.60,
    inv_limit=20.0,
    conf_min=0.60,
    strong_regime_threshold=0.35,
    max_spread_for_taking=8.0,
)

train_action_summary = summarize_actions(train_df)
test_action_summary = summarize_actions(test_df)


# =============================================================================
# 13) OUTPUT
# =============================================================================
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 120)

print("\n" + "=" * 110)
print("STATE LABEL MAP")
print("=" * 110)
print(label_map)

print("\n" + "=" * 110)
print("TRAIN RAW STATE SUMMARY (PRE-LABEL INTERPRETATION)")
print("=" * 110)
print(train_state_summary)

print("\n" + "=" * 110)
print("LABELED TRANSITION MATRIX")
print("=" * 110)
print(trans_df)

print("\n" + "=" * 110)
print("TRAIN STATE SUMMARY")
print("=" * 110)
print(train_summary_full)

print("\n" + "=" * 110)
print("TEST STATE SUMMARY")
print("=" * 110)
print(test_summary_full)

print("\n" + "=" * 110)
print("TRAIN STATE RUN LENGTH SUMMARY")
print("=" * 110)
print(train_run_summary)

print("\n" + "=" * 110)
print("TEST STATE RUN LENGTH SUMMARY")
print("=" * 110)
print(test_run_summary)

print("\n" + "=" * 110)
print("TRAIN PROBABILITY SOFTNESS")
print("=" * 110)
print(train_prob_softness)

print("\n" + "=" * 110)
print("TEST PROBABILITY SOFTNESS")
print("=" * 110)
print(test_prob_softness)

print("\n" + "=" * 110)
print("TRAIN ACTION SUMMARY")
print("=" * 110)
print(train_action_summary)

print("\n" + "=" * 110)
print("TEST ACTION SUMMARY")
print("=" * 110)
print(test_action_summary)

print("\n" + "=" * 110)
print("HEAD OF TEST DF")
print("=" * 110)
print(
    test_df[
        [
            "day",
            "timestamp",
            "mid",
            "ret_fut",
            "ret_fut_ticks",
            "state_raw",
            "state_label",
            "p_bear",
            "p_neutral",
            "p_bull",
            "regime_score",
            "regime_confidence",
            "spread_ticks",
            "alpha_raw",
            "alpha_adj",
            "action",
        ]
    ].head(30)
)