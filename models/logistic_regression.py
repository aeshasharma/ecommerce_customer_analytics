import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.project_paths import (
    DATA_FILE,
    MODEL_DIR,
    OUTPUT_DIR
)


print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)


# ==========================================
# FEATURES
# ==========================================

features = [
    "unit_price",
    "quantity",
    "discount_percent",
    "pages_viewed",
    "time_on_site_sec",
    "added_to_cart",
    "cart_abandoned",
    "rating"
]


X = df[features]

y = df["purchased"].astype(int)


# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


# ==========================================
# PIPELINE
# ==========================================

model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )

])


# ==========================================
# TRAIN
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# PREDICT
# ==========================================

y_pred = model.predict(
    X_test
)


# ==========================================
# PROBABILITY
# ==========================================

probability = (
    model
    .predict_proba(X_test)[:, 1]
)


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\nModel Results")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# CROSS VALIDATION
# ==========================================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)


print(
    "\n5-Fold CV Accuracy:",
    round(
        cv_scores.mean(),
        4
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_DIR /
    "logistic_regression.pkl"
)


# ==========================================
# SAVE PREDICTIONS
# ==========================================

results = pd.DataFrame({

    "actual_purchase":
        y_test.values,

    "predicted_purchase":
        y_pred,

    "purchase_probability":
        probability

})


results.to_csv(
    OUTPUT_DIR /
    "purchase_predictions.csv",
    index=False
)


print(
    "\nLogistic Regression model saved!"
)