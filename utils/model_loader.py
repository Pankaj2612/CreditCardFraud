from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.registry import MODEL_ENTRIES, entry_path_exists
from utils.preprocessing import align_features


def _is_valid_model(model) -> bool:
    return hasattr(model, "predict") or hasattr(model, "predict_proba") or hasattr(
        model, "layers"
    )


def _load_artifact(path: str):
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix in {".h5", ".keras"}:
        from keras.models import load_model

        return load_model(path)
    return joblib.load(path)


def load_all_models() -> dict[str, dict]:
    loaded = {}
    for entry in MODEL_ENTRIES:
        if not entry_path_exists(entry):
            continue
        try:
            model = _load_artifact(entry["path"])
            if not _is_valid_model(model):
                continue
            loaded[entry["id"]] = {**entry, "model": model}
        except Exception:
            continue
    return loaded


def get_fraud_probability(model, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return float(proba[0, 0])
        return float(proba[0, 1])
    if hasattr(model, "predict"):
        pred = model.predict(X, verbose=0) if hasattr(model, "layers") else model.predict(X)
        return float(np.asarray(pred).ravel()[0])
    raise ValueError("Model does not support prediction")


def predict_row(model, features: pd.DataFrame) -> tuple[int, float]:
    X = align_features(features, model)
    if hasattr(model, "layers"):
        arr = X.values.astype(np.float32)
        proba = float(np.asarray(model.predict(arr, verbose=0)).ravel()[0])
        label = int(proba >= 0.5)
        return label, proba

    proba = get_fraud_probability(model, X)
    label = int(proba >= 0.5)
    return label, proba


def predict_all_models(
    loaded_models: dict[str, dict],
    features: pd.DataFrame,
) -> list[dict]:
    import time

    results = []
    for entry_id, info in loaded_models.items():
        model = info["model"]
        start = time.perf_counter()
        label, proba = predict_row(model, features)
        latency_ms = (time.perf_counter() - start) * 1000
        results.append(
            {
                "id": entry_id,
                "display": info["display"],
                "sampling": info["sampling"],
                "label": label,
                "fraud_probability": proba,
                "latency_ms": latency_ms,
            }
        )
    return results
