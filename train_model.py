import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

DATASET_FILE = "dataset.csv"

MODEL_FILE = "yoga_pose_model_v3.pkl"
LABEL_FILE = "label_mapping_v3.pkl"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_FILE)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print(f"Total samples: {len(df)}")
print(f"Total features: {len(df.columns) - 1}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# FEATURES AND LABEL
# ============================================================

X = df.drop(
    "label",
    axis=1
)

y = df["label"]


# ============================================================
# LABEL ENCODING
# ============================================================

label_mapping = {
    "plank": 0,
    "tree": 1,
    "warrior_ii": 2,
    "unknown": 3
}

y = y.map(
    label_mapping
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=2,
    random_state=42,
    class_weight="balanced"
)


rf_model.fit(
    X_train,
    y_train
)


rf_predictions = rf_model.predict(
    X_test
)


rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)


print(
    f"\nRandom Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)


print("\nRandom Forest Report:")


print(
    classification_report(
        y_test,
        rf_predictions,
        labels=[0, 1, 2, 3],
        target_names=[
            "plank",
            "tree",
            "warrior_ii",
            "unknown"
        ],
        zero_division=0
    )
)


# ============================================================
# XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)


xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="multi:softmax",
    num_class=4,

    eval_metric="mlogloss",

    random_state=42
)


xgb_model.fit(
    X_train,
    y_train
)


xgb_predictions = xgb_model.predict(
    X_test
)


xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)


print(
    f"\nXGBoost Accuracy: "
    f"{xgb_accuracy * 100:.2f}%"
)


print("\nXGBoost Report:")


print(
    classification_report(
        y_test,
        xgb_predictions,
        labels=[0, 1, 2, 3],
        target_names=[
            "plank",
            "tree",
            "warrior_ii",
            "unknown"
        ],
        zero_division=0
    )
)


# ============================================================
# RANDOM FOREST CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST CONFUSION MATRIX")
print("=" * 60)


print(
    confusion_matrix(
        y_test,
        rf_predictions,
        labels=[0, 1, 2, 3]
    )
)


# ============================================================
# XGBOOST CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST CONFUSION MATRIX")
print("=" * 60)


print(
    confusion_matrix(
        y_test,
        xgb_predictions,
        labels=[0, 1, 2, 3]
    )
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)


print(
    f"Random Forest : "
    f"{rf_accuracy * 100:.2f}%"
)


print(
    f"XGBoost       : "
    f"{xgb_accuracy * 100:.2f}%"
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

if xgb_accuracy >= rf_accuracy:

    best_model = xgb_model

    best_model_name = "XGBoost"

else:

    best_model = rf_model

    best_model_name = "Random Forest"


print(
    f"\nBest Model: "
    f"{best_model_name}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


feature_importance = pd.DataFrame({

    "feature": X.columns,

    "importance":
        best_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    feature_importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_FILE
)


joblib.dump(
    label_mapping,
    LABEL_FILE
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)


print(
    f"Model saved as: "
    f"{MODEL_FILE}"
)


print(
    f"Label mapping saved as: "
    f"{LABEL_FILE}"
)