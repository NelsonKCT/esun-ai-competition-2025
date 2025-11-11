"""
Advanced pipeline for the 2025 Esun AI competition preliminary task.

This script builds rich account-level features from the raw transaction table
and trains an ensemble of CatBoost and XGBoost models optimized for F1-score.
It outputs a submission-ready CSV (acct, label) that can be uploaded to T-Brain.
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

pd.options.mode.chained_assignment = None


# --------------------------------------------------------------------------------------
# Data loading and feature engineering
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost+XGBoost ensemble for Esun AI Cup.")
    parser.add_argument(
        "--data-dir",
        default="T-Brain Competition Preliminary Data V3/初賽資料",
        help="Directory that contains acct_transaction.csv, acct_alert.csv, acct_predict.csv.",
    )
    parser.add_argument("--output", default="result.csv", help="Submission CSV path.")
    parser.add_argument("--probs-output", default=None, help="Optional CSV to dump acct + blended probability.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--obs-day",
        type=int,
        default=105,
        help="Only transactions with txn_date <= obs_day are used for feature engineering. "
        "Use -1 to disable time filtering.",
    )
    parser.add_argument(
        "--future-window",
        type=int,
        default=16,
        help="Alerts with event_date in (obs_day, obs_day + future_window] are treated as positives. "
        "Use -1 to include all future alerts.",
    )
    parser.add_argument(
        "--min-train-last-day",
        type=int,
        default=-1,
        help="Optional minimum last_day_all for training accounts to mimic candidate recency. Use -1 to disable.",
    )
    parser.add_argument(
        "--esun-threshold",
        type=float,
        default=0.0,
        help="Optional minimum esun_ratio_total for training accounts.",
    )
    parser.add_argument(
        "--target-positive-rate",
        type=float,
        default=-1.0,
        help="Desired positive fraction for final predictions. Set to a negative value to disable calibration.",
    )
    parser.add_argument(
        "--neg-multiplier",
        type=float,
        default=10.0,
        help="Negative sampling ratio. Final negatives = ratio * positives (capped by availability). "
        "Set <=0 to keep all negatives.",
    )
    return parser.parse_args()


def load_raw_data(dir_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx_path = os.path.join(dir_path, "acct_transaction.csv")
    alert_path = os.path.join(dir_path, "acct_alert.csv")
    predict_path = os.path.join(dir_path, "acct_predict.csv")

    dtype_map = {
        "from_acct_type": "Int8",
        "to_acct_type": "Int8",
        "txn_amt": "float32",
        "txn_date": "Int16",
        "is_self_txn": "string",
        "currency_type": "string",
        "channel_type": "string",
    }
    usecols = [
        "from_acct",
        "from_acct_type",
        "to_acct",
        "to_acct_type",
        "is_self_txn",
        "txn_amt",
        "txn_date",
        "txn_time",
        "currency_type",
        "channel_type",
    ]
    df_txn = pd.read_csv(tx_path, dtype=dtype_map, usecols=usecols, low_memory=False)
    df_alert = pd.read_csv(alert_path)
    df_predict = pd.read_csv(predict_path)

    print(f"[Data] Loaded {len(df_txn):,} transactions, {len(df_alert):,} alerts, {len(df_predict):,} targets.")
    return df_txn, df_alert, df_predict


def _time_to_seconds(series: pd.Series) -> pd.Series:
    hh = series.str.slice(0, 2).astype("int16")
    mm = series.str.slice(3, 5).astype("int16")
    ss = series.str.slice(6, 8).astype("int16")
    return (hh * 3600 + mm * 60 + ss).astype("int32")


def _category_ratios(long_df: pd.DataFrame, column: str, top_values: Sequence[str], prefix: str) -> pd.DataFrame:
    if not len(top_values):
        return pd.DataFrame({"acct": long_df["acct"].unique()})

    base = (
        long_df[["acct", "direction"]]
        .copy()
        .assign(total=1)
        .groupby(["acct", "direction"])["total"]
        .sum()
    )

    subset = long_df[long_df[column].isin(top_values)]
    cat_counts = (
        subset.groupby(["acct", "direction", column])
        .size()
        .unstack(column, fill_value=0)
        .reindex(base.index, fill_value=0)
    )

    ratios = cat_counts.div(base, axis=0).fillna(0)
    ratios["other"] = (1 - ratios.sum(axis=1)).clip(lower=0)
    ratios = ratios.unstack("direction", fill_value=0)
    ratios.columns = [f"{prefix}_{str(cat)}_{direction}_ratio" for cat, direction in ratios.columns]
    ratios = ratios.reset_index()
    return ratios


def build_features(df_txn: pd.DataFrame, max_txn_day: int | None = None) -> pd.DataFrame:
    if max_txn_day is not None and max_txn_day >= 0:
        df = df_txn[df_txn["txn_date"] <= max_txn_day].copy()
        print(f"[Feature] Using {len(df):,} transactions with txn_date <= {max_txn_day}.")
    else:
        df = df_txn.copy()
    df["txn_time_sec"] = _time_to_seconds(df["txn_time"])
    df["txn_amt"] = df["txn_amt"].astype("float32")
    df["txn_date"] = df["txn_date"].astype("int16")
    df["is_self_txn"] = df["is_self_txn"].fillna("UNK")
    df["currency_type"] = df["currency_type"].fillna("UNK")
    df["channel_type"] = df["channel_type"].fillna("UNK")

    base_columns = [
        "acct",
        "acct_type",
        "counter_acct",
        "txn_amt",
        "txn_date",
        "txn_time_sec",
        "currency_type",
        "channel_type",
        "is_self_txn",
    ]
    send_df = df[
        [
            "from_acct",
            "from_acct_type",
            "to_acct",
            "txn_amt",
            "txn_date",
            "txn_time_sec",
            "currency_type",
            "channel_type",
            "is_self_txn",
        ]
    ].copy()
    send_df.columns = base_columns
    send_df["direction"] = "send"
    send_df["signed_amt"] = -send_df["txn_amt"]

    recv_df = df[
        [
            "to_acct",
            "to_acct_type",
            "from_acct",
            "txn_amt",
            "txn_date",
            "txn_time_sec",
            "currency_type",
            "channel_type",
            "is_self_txn",
        ]
    ].copy()
    recv_df.columns = base_columns
    recv_df["direction"] = "recv"
    recv_df["signed_amt"] = recv_df["txn_amt"]

    long_df = pd.concat([send_df, recv_df], axis=0, ignore_index=True)
    del send_df, recv_df
    gc.collect()

    long_df["is_night"] = (
        (long_df["txn_time_sec"] < 6 * 3600) | (long_df["txn_time_sec"] >= 21 * 3600)
    ).astype("int8")
    long_df["is_self_flag"] = long_df["is_self_txn"].eq("Y").astype("int8")
    long_df["is_self_unknown"] = long_df["is_self_txn"].eq("UNK").astype("int8")
    long_df["is_esun"] = long_df["acct_type"].eq(1).astype("int8")
    long_df["log_amt"] = np.log1p(long_df["txn_amt"])

    # Core directional aggregations
    agg = (
        long_df.groupby(["acct", "direction"])
        .agg(
            txn_cnt=("txn_amt", "size"),
            txn_amt_sum=("txn_amt", "sum"),
            txn_amt_mean=("txn_amt", "mean"),
            txn_amt_std=("txn_amt", "std"),
            txn_amt_max=("txn_amt", "max"),
            txn_amt_min=("txn_amt", "min"),
            txn_amt_median=("txn_amt", "median"),
            log_amt_mean=("log_amt", "mean"),
            log_amt_std=("log_amt", "std"),
            active_days=("txn_date", "nunique"),
            first_day=("txn_date", "min"),
            last_day=("txn_date", "max"),
            mean_day=("txn_date", "mean"),
            night_ratio=("is_night", "mean"),
            self_ratio=("is_self_flag", "mean"),
            self_unknown_ratio=("is_self_unknown", "mean"),
            esun_ratio=("is_esun", "mean"),
            avg_time_sec=("txn_time_sec", "mean"),
            time_std_sec=("txn_time_sec", "std"),
        )
        .unstack("direction", fill_value=0)
    )
    agg.columns = [f"{col}_{direction}" for col, direction in agg.columns]
    agg = agg.reset_index()

    counter = (
        long_df.groupby(["acct", "direction"])["counter_acct"]
        .nunique()
        .unstack("direction", fill_value=0)
        .rename(columns=lambda c: f"uniq_counter_{c}")
        .reset_index()
    )

    top_currencies = df["currency_type"].value_counts().head(5).index.tolist()
    currency_ratio = _category_ratios(long_df, "currency_type", top_currencies, "currency")

    top_channels = df["channel_type"].value_counts().head(7).index.tolist()
    channel_ratio = _category_ratios(long_df, "channel_type", top_channels, "channel")

    overall = (
        long_df.groupby("acct")
        .agg(
            total_txn_cnt=("txn_amt", "size"),
            total_amt=("txn_amt", "sum"),
            net_flow=("signed_amt", "sum"),
            total_log_amt=("log_amt", "sum"),
            esun_ratio_total=("is_esun", "mean"),
            all_active_days=("txn_date", "nunique"),
            first_day_all=("txn_date", "min"),
            last_day_all=("txn_date", "max"),
        )
        .reset_index()
    )
    overall["activity_span_all"] = overall["last_day_all"] - overall["first_day_all"]
    overall["avg_amt_per_day"] = overall["total_amt"] / (overall["all_active_days"] + 1e-6)
    overall["net_flow_ratio"] = overall["net_flow"] / (overall["total_amt"].abs() + 1e-6)

    features = agg.merge(counter, on="acct", how="left")
    features = features.merge(currency_ratio, on="acct", how="left")
    features = features.merge(channel_ratio, on="acct", how="left")
    features = features.merge(overall, on="acct", how="left")

    # Derived ratios
    def safe_ratio(num, denom):
        return num / (denom + 1e-6)

    features["send_recv_amt_ratio"] = safe_ratio(
        features.get("txn_amt_sum_send", pd.Series(0, index=features.index)),
        features.get("txn_amt_sum_recv", pd.Series(0, index=features.index)),
    )
    features["send_recv_cnt_ratio"] = safe_ratio(
        features.get("txn_cnt_send", pd.Series(0, index=features.index)),
        features.get("txn_cnt_recv", pd.Series(0, index=features.index)),
    )
    features["recv_send_cnt_ratio"] = safe_ratio(
        features.get("txn_cnt_recv", pd.Series(0, index=features.index)),
        features.get("txn_cnt_send", pd.Series(0, index=features.index)),
    )
    features["uniq_counter_ratio"] = safe_ratio(
        features.get("uniq_counter_send", pd.Series(0, index=features.index)),
        features.get("uniq_counter_recv", pd.Series(0, index=features.index)),
    )
    features["active_days_ratio"] = safe_ratio(
        features.get("active_days_send", pd.Series(0, index=features.index)),
        features.get("active_days_recv", pd.Series(0, index=features.index)),
    )
    features["avg_amt_gap"] = (
        features.get("txn_amt_mean_recv", pd.Series(0, index=features.index))
        - features.get("txn_amt_mean_send", pd.Series(0, index=features.index))
    )

    features = features.sort_values("acct").reset_index(drop=True)
    numeric_cols = [col for col in features.columns if col != "acct"]
    features[numeric_cols] = features[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0).astype("float32")

    print(f"[Feature] Built {len(features):,} account profiles with {len(numeric_cols)} features.")
    return features


# --------------------------------------------------------------------------------------
# Dataset preparation and modeling
# --------------------------------------------------------------------------------------

def prepare_datasets(
    features: pd.DataFrame,
    df_alert: pd.DataFrame,
    df_predict: pd.DataFrame,
    esun_threshold: float,
    obs_day: int,
    future_window: int,
    min_last_day: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    features = features.copy()
    predict_accounts = set(df_predict["acct"])

    if obs_day is None or obs_day < 0:
        raise ValueError("obs_day must be provided (>=0) for time-based split.")

    future_mask = df_alert["event_date"] > obs_day
    if future_window is not None and future_window > 0:
        future_mask &= df_alert["event_date"] <= obs_day + future_window
    future_alerts = df_alert[future_mask]
    label_set = set(future_alerts["acct"])

    past_alerts = df_alert[df_alert["event_date"] <= obs_day]
    excluded_accounts = set(past_alerts["acct"])

    if future_window is not None and future_window > 0:
        future_desc = f"{obs_day + future_window}"
    else:
        future_desc = "end"
    print(
        f"[Dataset] Future alerts within window ({obs_day}, {future_desc}]: "
        f"{len(label_set):,}. Dropping {len(excluded_accounts):,} earlier alerts."
    )

    features["label"] = features["acct"].isin(label_set).astype("int8")
    feature_cols = [col for col in features.columns if col not in ("acct", "label")]

    base_mask = ~features["acct"].isin(predict_accounts | excluded_accounts)

    if esun_threshold and esun_threshold > 0:
        esun_series = features.get("esun_ratio_total", pd.Series(1.0, index=features.index))
        base_mask &= esun_series >= esun_threshold

    if min_last_day is not None and min_last_day >= 0 and "last_day_all" in features.columns:
        base_mask &= features["last_day_all"] >= min_last_day

    train_df = features[base_mask].reset_index(drop=True)
    if train_df["label"].sum() == 0:
        print("[Dataset] Warning: no positives after filtering, reverting to using all non-predict accounts.")
        train_df = features[~features["acct"].isin(predict_accounts)].reset_index(drop=True)

    test_df = df_predict[["acct"]].merge(features.drop(columns=["label"]), on="acct", how="left")
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    print(
        f"[Dataset] Training accounts: {len(train_df):,} "
        f"(positive: {train_df['label'].sum():,}), Test accounts: {len(test_df):,}"
    )
    return train_df, test_df, feature_cols


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (probs >= thr).astype(int)
        score = f1_score(y_true, pred)
        if score > best_f1:
            best_f1 = score
            best_thr = thr
    return best_thr, best_f1


def balance_training(df: pd.DataFrame, neg_multiplier: float, seed: int) -> pd.DataFrame:
    if neg_multiplier is None or neg_multiplier <= 0:
        return df
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]
    if pos_df.empty:
        return df
    target_neg = min(len(neg_df), int(np.ceil(len(pos_df) * neg_multiplier)))
    neg_sample = neg_df.sample(target_neg, random_state=seed, replace=False)
    balanced = pd.concat([pos_df, neg_sample], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"[Dataset] Balanced training accounts: {len(balanced):,} "
        f"(pos {len(pos_df):,} / neg {target_neg:,})"
    )
    return balanced


def threshold_for_rate(probs: np.ndarray, rate: float) -> float:
    if rate <= 0:
        return 1.0
    rate = float(min(max(rate, 0.0), 1.0))
    k = max(int(round(rate * len(probs))), 1)
    sorted_probs = np.sort(probs)
    if k >= len(sorted_probs):
        return float(sorted_probs[0] - 1e-6)
    idx = max(len(sorted_probs) - k, 0)
    return float(sorted_probs[idx])


@dataclass
class TrainingResult:
    oof_preds: np.ndarray
    test_probs: np.ndarray
    global_threshold: float
    global_f1: float
    fold_metrics: List[Dict[str, float]]
    rate_threshold: float | None = None
    rate_f1: float | None = None


def train_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: int,
    seed: int,
    target_positive_rate: float | None = None,
) -> TrainingResult:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(X), dtype=np.float32)
    test_pred_accum = np.zeros(len(X_test), dtype=np.float32)
    fold_infos: List[Dict[str, float]] = []
    fold_thresholds: List[float] = []

    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.int32)
    X_test_np = X_test.values.astype(np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_np, y_np), 1):
        X_tr, X_val = X_np[tr_idx], X_np[val_idx]
        y_tr, y_val = y_np[tr_idx], y_np[val_idx]

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=2000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=6,
            loss_function="Logloss",
            eval_metric="F1",
            random_state=seed + fold,
            od_type="Iter",
            od_wait=60,
            auto_class_weights="Balanced",
            bootstrap_type="Bernoulli",
            subsample=0.8,
            grow_policy="Lossguide",
            task_type="CPU",
            verbose=False,
            allow_writing_files=False,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        val_cat = cat_model.predict_proba(X_val)[:, 1]
        test_cat = cat_model.predict_proba(X_test_np)[:, 1]

        # XGBoost
        pos = max(y_tr.sum(), 1)
        neg = len(y_tr) - pos
        scale_pos_weight = neg / pos
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=6,
            reg_lambda=2.0,
            reg_alpha=0.0,
            gamma=0.1,
            n_estimators=1000,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=seed + fold,
            n_jobs=4,
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_xgb = xgb_model.predict_proba(X_val)[:, 1]
        test_xgb = xgb_model.predict_proba(X_test_np)[:, 1]

        val_blend = 0.55 * val_cat + 0.45 * val_xgb
        test_blend = 0.55 * test_cat + 0.45 * test_xgb

        thr, fold_f1 = find_best_threshold(y_val, val_blend)
        fold_thresholds.append(thr)
        oof[val_idx] = val_blend
        test_pred_accum += test_blend

        fold_infos.append({"fold": fold, "threshold": thr, "f1": fold_f1})
        print(f"[Fold {fold}] F1={fold_f1:.5f}, threshold={thr:.3f}")

    test_pred_accum /= folds
    global_thr, global_f1 = find_best_threshold(y_np, oof)
    print(f"[CV] OOF F1={global_f1:.5f}, best threshold={global_thr:.3f}")

    rate_thr = None
    rate_f1 = None
    if target_positive_rate is not None and target_positive_rate >= 0:
        rate_thr = threshold_for_rate(oof, target_positive_rate)
        preds_rate = (oof >= rate_thr).astype(int)
        rate_f1 = f1_score(y_np, preds_rate)
        oof_rate = preds_rate.mean()
        print(
            f"[Threshold] Target positive rate={target_positive_rate:.3f}, "
            f"achieved OOF rate={oof_rate:.5f}, threshold={rate_thr:.3f}, F1={rate_f1:.5f}"
        )

    return TrainingResult(
        oof_preds=oof,
        test_probs=test_pred_accum,
        global_threshold=global_thr,
        global_f1=global_f1,
        fold_metrics=fold_infos,
        rate_threshold=rate_thr,
        rate_f1=rate_f1,
    )


# --------------------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()
    df_txn, df_alert, df_predict = load_raw_data(args.data_dir)
    features = build_features(df_txn, max_txn_day=args.obs_day)
    del df_txn
    gc.collect()

    train_df, test_df, feature_cols = prepare_datasets(
        features,
        df_alert,
        df_predict,
        esun_threshold=args.esun_threshold,
        obs_day=args.obs_day,
        future_window=args.future_window,
        min_last_day=args.min_train_last_day,
    )
    train_df = balance_training(train_df, args.neg_multiplier, args.seed)
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]

    target_rate = args.target_positive_rate if (args.target_positive_rate is not None and args.target_positive_rate >= 0) else None
    result = train_and_predict(
        X_train,
        y_train,
        X_test,
        folds=args.folds,
        seed=args.seed,
        target_positive_rate=target_rate,
    )

    submission = df_predict[["acct"]].copy()
    if target_rate is not None and target_rate >= 0:
        final_threshold = threshold_for_rate(result.test_probs, target_rate)
        threshold_source = f"rate {target_rate:.3f}"
    else:
        final_threshold = result.global_threshold
        threshold_source = "best-F1"
    submission["label"] = (result.test_probs >= final_threshold).astype(int)
    pos_count = submission["label"].sum()
    print(
        f"[Output] Positive predictions: {pos_count} / {len(submission)} ({pos_count / len(submission):.3%}) "
        f"using {threshold_source} threshold {final_threshold:.3f}"
    )
    submission.to_csv(args.output, index=False)
    print(f"[Output] Saved predictions to {args.output}")

    if args.probs_output:
        prob_df = df_predict[["acct"]].copy()
        prob_df["prob"] = result.test_probs
        prob_df.to_csv(args.probs_output, index=False)
        print(f"[Output] Saved blended probabilities to {args.probs_output}")


if __name__ == "__main__":
    main()
