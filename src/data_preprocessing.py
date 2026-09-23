import pandas as pd
import numpy as np

from src.project_paths import (
    DATA_FILE,
    CLEANED_DATA_FILE
)


print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_FILE)

print("\nOriginal shape:")
print(df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print(df.shape)


# ==========================================
# 3. CONVERT DATE
# ==========================================

df["visit_date"] = pd.to_datetime(
    df["visit_date"],
    dayfirst=True,
    errors="coerce"
)


# ==========================================
# 4. NUMERICAL MISSING VALUES
# ==========================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns


for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


# ==========================================
# 5. CATEGORICAL MISSING VALUES
# ==========================================

categorical_columns = df.select_dtypes(
    include="object"
).columns


for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()

        if len(mode_value) > 0:

            df[column] = df[column].fillna(
                mode_value[0]
            )

        else:

            df[column] = df[column].fillna(
                "Unknown"
            )


# ==========================================
# 6. FIX IMPORTANT NUMERIC COLUMNS
# ==========================================

numeric_features = [
    "unit_price",
    "quantity",
    "discount_percent",
    "discount_amount",
    "revenue",
    "pages_viewed",
    "time_on_site_sec",
    "added_to_cart",
    "purchased",
    "cart_abandoned",
    "rating",
    "review_helpful_votes",
    "revenue_normalized"
]


for column in numeric_features:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ==========================================
# 7. FILL NUMERIC VALUES AGAIN
# ==========================================

for column in numeric_features:

    if column in df.columns:

        df[column] = df[column].fillna(
            df[column].median()
        )


# ==========================================
# 8. CREATE CALCULATED REVENUE
# ==========================================

df["calculated_revenue"] = (
    df["unit_price"]
    * df["quantity"]
    * (
        1 -
        df["discount_percent"] / 100
    )
)


# ==========================================
# 9. REMOVE REVIEW TEXT
# ==========================================

if "review_text" in df.columns:

    df = df.drop(
        columns=["review_text"]
    )


# ==========================================
# 10. CREATE PURCHASE VALUE
# ==========================================

df["purchase_value"] = df[
    "revenue"
].clip(
    lower=0
)


# ==========================================
# 11. SAVE CLEAN DATA
# ==========================================

df.to_csv(
    CLEANED_DATA_FILE,
    index=False
)


print("\nPreprocessing completed!")

print("\nFinal shape:")
print(df.shape)


print("\nRemaining missing values:")
print(
    df.isnull().sum().sum()
)


print("\nCleaned dataset saved at:")
print(CLEANED_DATA_FILE)