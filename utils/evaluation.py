import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from utils.model_loader import get_fraud_probability, predict_row
from utils.preprocessing import align_features


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auprc": float(average_precision_score(y_true, y_proba)),
    }


def evaluate_model_on_test(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    X_aligned = align_features(X_test, model)
    if hasattr(model, "layers"):
        proba = np.asarray(model.predict(X_aligned.values.astype(np.float32), verbose=0)).ravel()
        y_pred = (proba >= 0.5).astype(int)
        y_proba = proba
    else:
        y_pred = model.predict(X_aligned)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_aligned)
            y_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            y_proba = y_pred.astype(float)
    metrics = compute_metrics(y_test, y_pred, y_proba)
    return {**metrics, "y_pred": y_pred, "y_proba": y_proba}


def benchmark_latency(model, X_sample: pd.DataFrame, n_iter: int = 100) -> float:
    row = X_sample.iloc[[0]]
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        if hasattr(model, "layers"):
            model.predict(row.values.astype(np.float32), verbose=0)
        else:
            model.predict(align_features(row, model))
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def build_metrics_dataframe(
    loaded_models: dict[str, dict],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []
    for entry_id, info in loaded_models.items():
        model = info["model"]
        result = evaluate_model_on_test(model, X_test, y_test)
        latency = benchmark_latency(model, X_test)
        rows.append(
            {
                "id": entry_id,
                "model": info["display"],
                "sampling": info["sampling"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "auprc": result["auprc"],
                "latency_ms": latency,
            }
        )
    return pd.DataFrame(rows)


def smote_delta_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    deltas = []
    for model_name in metrics_df["model"].unique():
        subset = metrics_df[metrics_df["model"] == model_name]
        no_smote = subset[subset["sampling"] == "No SMOTE"]
        smote = subset[subset["sampling"] == "SMOTE"]
        if no_smote.empty or smote.empty:
            continue
        no_row = no_smote.iloc[0]
        sm_row = smote.iloc[0]
        deltas.append(
            {
                "model": model_name,
                "delta_precision": sm_row["precision"] - no_row["precision"],
                "delta_recall": sm_row["recall"] - no_row["recall"],
                "delta_f1": sm_row["f1"] - no_row["f1"],
                "delta_auprc": sm_row["auprc"] - no_row["auprc"],
            }
        )
    return pd.DataFrame(deltas)


def pr_curve_data(y_true, y_proba) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return precision, recall
