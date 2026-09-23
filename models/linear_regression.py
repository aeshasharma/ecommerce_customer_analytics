import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.project_paths import (
    DATA_FILE,
    MODEL_DIR,
    OUTPUT_DIR
)


print("=" * 60)
print("LINEAR REGRESSION")
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
    "added_to_cart"
]


X = df[features]

y = df["revenue"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)


# ==========================================
# MODEL
# ==========================================

model = LinearRegression()


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
# EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\nModel Results")

print(
    "MAE:",
    round(mae, 2)
)

print(
    "MSE:",
    round(mse, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "R2 Score:",
    round(r2, 4)
)


# ==========================================
# CROSS VALIDATION
# ==========================================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)


print(
    "\n5-Fold Cross Validation:"
)

print(cv_scores)


print(
    "Average CV R2:",
    round(
        cv_scores.mean(),
        4
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

MODEL_DIR.mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_DIR /
    "linear_regression.pkl"
)


# ==========================================
# SAVE TEST PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({

    "actual_revenue":
        y_test.values,

    "predicted_revenue":
        y_pred

})


OUTPUT_DIR.mkdir(
    exist_ok=True
)


prediction_df.to_csv(
    OUTPUT_DIR /
    "linear_regression_predictions.csv",
    index=False
)


print(
    "\nLinear Regression model saved!"
)