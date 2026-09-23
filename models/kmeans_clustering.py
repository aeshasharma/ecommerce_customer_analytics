import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.project_paths import (
    DATA_FILE,
    MODEL_DIR,
    OUTPUT_DIR
)


print("=" * 60)
print("K-MEANS CUSTOMER SEGMENTATION")
print("=" * 60)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)


# ==========================================
# 2. DATE CONVERSION
# ==========================================

df["visit_date"] = pd.to_datetime(
    df["visit_date"],
    dayfirst=True,
    errors="coerce"
)


analysis_date = df[
    "visit_date"
].max()


# ==========================================
# 3. CUSTOMER RFM
# ==========================================

customer_data = df.groupby(
    "customer_id"
).agg(

    recency=(
        "visit_date",
        lambda x:
        (
            analysis_date - x.max()
        ).days
    ),

    frequency=(
        "purchased",
        "sum"
    ),

    monetary=(
        "revenue",
        "sum"
    ),

    total_quantity=(
        "quantity",
        "sum"
    ),

    average_rating=(
        "rating",
        "mean"
    ),

    average_pages_viewed=(
        "pages_viewed",
        "mean"
    ),

    average_time_on_site=(
        "time_on_site_sec",
        "mean"
    ),

    total_cart_additions=(
        "added_to_cart",
        "sum"
    )

).reset_index()


# ==========================================
# 4. CLEAN RFM
# ==========================================

customer_data[
    "recency"
] = customer_data[
    "recency"
].fillna(0)


customer_data[
    "frequency"
] = customer_data[
    "frequency"
].clip(
    lower=0
)


customer_data[
    "monetary"
] = customer_data[
    "monetary"
].clip(
    lower=0
)


# ==========================================
# 5. RFM FEATURES
# ==========================================

rfm_features = [
    "recency",
    "frequency",
    "monetary"
]


X = customer_data[
    rfm_features
].copy()


# ==========================================
# 6. SCALE
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ==========================================
# 7. K-MEANS
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=20
)


customer_data[
    "cluster"
] = kmeans.fit_predict(
    X_scaled
)


# ==========================================
# 8. CLUSTER SUMMARY
# ==========================================

cluster_summary = (
    customer_data
    .groupby("cluster")
    [
        [
            "recency",
            "frequency",
            "monetary"
        ]
    ]
    .mean()
)


print(
    "\nCluster Summary:"
)

print(
    cluster_summary
)


# ==========================================
# 9. CREATE STANDARDIZED CENTROIDS
# ==========================================

centroids = pd.DataFrame(

    kmeans.cluster_centers_,

    columns=[
        "recency_z",
        "frequency_z",
        "monetary_z"
    ]

)


centroids[
    "value_score"
] = (
    centroids["frequency_z"]
    +
    centroids["monetary_z"]
    -
    centroids["recency_z"]
)


# ==========================================
# 10. IDENTIFY VIP
# ==========================================

vip_cluster = (
    centroids[
        "value_score"
    ].idxmax()
)


# ==========================================
# 11. IDENTIFY LOST
# ==========================================

remaining = [
    cluster
    for cluster in centroids.index
    if cluster != vip_cluster
]


lost_cluster = (
    centroids.loc[
        remaining,
        "recency_z"
    ].idxmax()
)


remaining = [
    cluster
    for cluster in remaining
    if cluster != lost_cluster
]


# ==========================================
# 12. IDENTIFY AT RISK
# ==========================================

at_risk_cluster = (
    centroids.loc[
        remaining,
        "recency_z"
    ].idxmax()
)


remaining = [
    cluster
    for cluster in remaining
    if cluster != at_risk_cluster
]


# ==========================================
# 13. IDENTIFY NEW CUSTOMER
# ==========================================

# New customers:
# low recency + low frequency + low monetary

new_customer_score = (
    centroids.loc[
        remaining,
        "recency_z"
    ]
    -
    centroids.loc[
        remaining,
        "frequency_z"
    ]
    -
    centroids.loc[
        remaining,
        "monetary_z"
    ]
)


new_customer_cluster = (
    new_customer_score.idxmin()
)


remaining = [
    cluster
    for cluster in remaining
    if cluster != new_customer_cluster
]


# ==========================================
# 14. REMAINING = REGULAR
# ==========================================

regular_cluster = remaining[0]


# ==========================================
# 15. SEGMENT MAPPING
# ==========================================

segment_mapping = {

    vip_cluster:
        "VIP",

    regular_cluster:
        "Regular",

    new_customer_cluster:
        "New Customer",

    lost_cluster:
        "Lost",

    at_risk_cluster:
        "At Risk"

}


customer_data[
    "segment"
] = customer_data[
    "cluster"
].map(
    segment_mapping
)


# ==========================================
# 16. ACTIVITY SCORE
# ==========================================

customer_data[
    "activity_score"
] = (

    customer_data[
        "frequency"
    ].clip(lower=0) * 2

    +

    customer_data[
        "monetary"
    ].clip(lower=0) / 100

)


# ==========================================
# 17. SEGMENT SUMMARY
# ==========================================

segment_summary = (
    customer_data
    .groupby("segment")
    .agg({

        "customer_id":
            "count",

        "recency":
            "mean",

        "frequency":
            "mean",

        "monetary":
            "mean"

    })
    .reset_index()
)


print(
    "\nSegment Summary:"
)

print(
    segment_summary
)


# ==========================================
# 18. SAVE CUSTOMER SEGMENTS
# ==========================================

OUTPUT_DIR.mkdir(
    exist_ok=True
)


customer_data.to_csv(
    OUTPUT_DIR /
    "customer_segments.csv",
    index=False
)


# ==========================================
# 19. SAVE MODELS
# ==========================================

MODEL_DIR.mkdir(
    exist_ok=True
)


joblib.dump(
    kmeans,
    MODEL_DIR /
    "kmeans_model.pkl"
)


joblib.dump(
    scaler,
    MODEL_DIR /
    "kmeans_scaler.pkl"
)


print(
    "\nCustomer segmentation completed!"
)


print(
    "\nSegment counts:"
)


print(
    customer_data[
        "segment"
    ].value_counts()
)


print(
    "\nSaved:"
)

print(
    OUTPUT_DIR /
    "customer_segments.csv"
)