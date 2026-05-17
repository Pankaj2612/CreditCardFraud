import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

from utils.preprocessing import (
    align_features,
    fit_scaler_on_raw,
    get_train_test_split,
    load_and_preprocess,
    load_raw_dataset,
    save_scaler,
)

df = load_raw_dataset()
print(df.head())

scaler = fit_scaler_on_raw(df)
save_scaler(scaler)

processed, y, _ = load_and_preprocess()
print(processed.head())

X_train, X_test, y_train, y_test = get_train_test_split(processed, y)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(y_train_smote.value_counts())

rf_smote = joblib.load("models/random_forest_model_with_smote.pkl")
rf_no_smote = joblib.load("models/random_forest_model_without_smote.pkl")

X_test_aligned = align_features(X_test, rf_no_smote)

print("Random Forest with SMOTE:")
print(classification_report(y_test, rf_smote.predict(X_test_aligned)))
print("Random Forest without SMOTE:")
print(classification_report(y_test, rf_no_smote.predict(X_test_aligned)))
