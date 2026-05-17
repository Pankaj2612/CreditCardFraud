from pathlib import Path

MODEL_ENTRIES = [
    {
        "id": "rf_no_smote",
        "display": "Random Forest",
        "sampling": "No SMOTE",
        "path": "models/random_forest_model_without_smote.pkl",
    },
    {
        "id": "rf_smote",
        "display": "Random Forest",
        "sampling": "SMOTE",
        "path": "models/random_forest_model_with_smote.pkl",
    },
    {
        "id": "xgb_no_smote",
        "display": "XGBoost",
        "sampling": "No SMOTE",
        "path": "models/xgboost_model_without_smote.pkl",
    },
    {
        "id": "xgb_smote",
        "display": "XGBoost",
        "sampling": "SMOTE",
        "path": "models/xgboost_model_with_smote.pkl",
    },
    # {
    #     "id": "mlp_no_smote",
    #     "display": "MLP",
    #     "sampling": "No SMOTE",
    #     "path": "models/mlp_model_without_smote.pkl",
    # },
    # {
    #     "id": "mlp_smote",
    #     "display": "MLP",
    #     "sampling": "SMOTE",
    #     "path": "models/mlp_model_with_smote.pkl",
    # },
]

SCALER_PATH = "models/scaler.pkl"
CSV_PATH = "https://github.com/Pankaj2612/CreditCardFraud/releases/download/tech/creditcard.csv"


def entry_path_exists(entry: dict) -> bool:
    return Path(entry["path"]).is_file()
