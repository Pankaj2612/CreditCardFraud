from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.registry import CSV_PATH, SCALER_PATH

RANDOM_STATE = 42
TEST_SIZE = 0.2


def fit_scaler_on_raw(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[["Amount", "Time"]].values)
    return scaler


def load_scaler() -> StandardScaler:
    path = Path(SCALER_PATH)
    if path.is_file():
        return joblib.load(path)
    df = pd.read_csv(CSV_PATH)
    scaler = fit_scaler_on_raw(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return scaler


def save_scaler(scaler: StandardScaler) -> None:
    path = Path(SCALER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def transform_raw_df(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    scaled = scaler.transform(out[["Amount", "Time"]].values)
    out["scaled_amount"] = scaled[:, 0]
    out["scaled_time"] = scaled[:, 1]
    out.drop(["Amount", "Time"], axis=1, inplace=True)
    return out


def load_raw_dataset() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def load_and_preprocess() -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    raw = load_raw_dataset()
    scaler = load_scaler()
    processed = transform_raw_df(raw, scaler)
    X = processed.drop("Class", axis=1)
    y = processed["Class"]
    return processed, y, scaler


def get_train_test_split(
    processed: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if processed is None or y is None:
        processed, y, _ = load_and_preprocess()
    X = processed.drop("Class", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    if hasattr(model, "feature_names_in_"):
        return X.reindex(columns=model.feature_names_in_, fill_value=0)
    return X


def prepare_row_features(
    raw_row: pd.Series,
    scaler: StandardScaler,
) -> pd.DataFrame:
    row_df = raw_row.to_frame().T
    if "Class" in row_df.columns:
        row_df = row_df.drop(columns=["Class"])
    if "scaled_amount" in row_df.columns and "Amount" not in row_df.columns:
        features = row_df.drop(columns=["Class"], errors="ignore")
        return features

    scaled = scaler.transform(row_df[["Amount", "Time"]].values)
    features = row_df.drop(columns=["Amount", "Time"], errors="ignore")
    features = features.copy()
    features["scaled_amount"] = scaled[0, 0]
    features["scaled_time"] = scaled[0, 1]
    if "Class" in features.columns:
        features = features.drop(columns=["Class"])
    return features


def get_raw_row_by_index(index: int) -> pd.Series:
    return load_raw_dataset().iloc[index]


def get_processed_row_by_index(
    index: int,
    processed: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    if processed is None:
        processed, _, _ = load_and_preprocess()
    row = processed.iloc[index]
    raw = load_raw_dataset().iloc[index]
    return row, raw
