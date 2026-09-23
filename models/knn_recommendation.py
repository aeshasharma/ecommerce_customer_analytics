import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from src.project_paths import (
    DATA_FILE,
    MODEL_DIR,
    OUTPUT_DIR
)


print("=" * 60)
print("KNN PRODUCT RECOMMENDATION")
print("=" * 60)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)


# ==========================================
# PRODUCT LEVEL DATA
# ==========================================

product_data = df.groupby(
    [
        "product_id",
        "product_category"
    ]
).agg({

    "unit_price": "mean",

    "rating": "mean",

    "pages_viewed": "mean",

    "time_on_site_sec": "mean",

    "added_to_cart": "mean",

    "purchased": "mean",

    "quantity": "mean"

}).reset_index()


print(
    "\nNumber of products:",
    len(product_data)
)


# ==========================================
# FEATURES
# ==========================================

features = [
    "unit_price",
    "rating",
    "pages_viewed",
    "time_on_site_sec",
    "added_to_cart",
    "purchased",
    "quantity"
]


X = product_data[
    features
]


# ==========================================
# SCALE
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ==========================================
# KNN
# ==========================================

knn = NearestNeighbors(
    n_neighbors=6,
    metric="cosine"
)


knn.fit(
    X_scaled
)


# ==========================================
# SELECT PRODUCT
# ==========================================

selected_product = (
    product_data[
        "product_id"
    ].iloc[0]
)


selected_index = (
    product_data.index[
        product_data["product_id"]
        == selected_product
    ][0]
)


# ==========================================
# FIND SIMILAR PRODUCTS
# ==========================================

distances, indices = (
    knn.kneighbors(
        [
            X_scaled[
                selected_index
            ]
        ]
    )
)


recommendations = (
    product_data.iloc[
        indices[0]
    ].copy()
)


recommendations[
    "similarity_distance"
] = distances[0]


# Remove selected product
recommendations = (
    recommendations[
        recommendations[
            "product_id"
        ] != selected_product
    ]
)


# ==========================================
# DISPLAY
# ==========================================

print(
    "\nSelected Product:",
    selected_product
)


print(
    "\nRecommended Products:"
)


print(
    recommendations[
        [
            "product_id",
            "product_category",
            "unit_price",
            "rating",
            "similarity_distance"
        ]
    ]
)


# ==========================================
# SAVE
# ==========================================

OUTPUT_DIR.mkdir(
    exist_ok=True
)


recommendations.to_csv(
    OUTPUT_DIR /
    "recommendations.csv",
    index=False
)


joblib.dump(
    knn,
    MODEL_DIR /
    "knn_model.pkl"
)


joblib.dump(
    scaler,
    MODEL_DIR /
    "knn_scaler.pkl"
)


print(
    "\nKNN recommendation system saved!"
)