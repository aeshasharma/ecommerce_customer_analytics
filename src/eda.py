import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data(file_name="customer_analytics.csv"):
    file_path = f"data/{file_name}"
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    except Exception as e:
        raise ValueError(f"Unable to load dataset: {e}")

# =========================================================
# HELPER: ROBUST PURCHASE MAPPING
# =========================================================
def map_purchase(val):
    if pd.isna(val):
        return "Unknown"
    val_str = str(val).strip().lower()
    if val_str in ["1", "1.0", "true", "yes", "y", "purchased"]:
        return "Purchased"
    elif val_str in ["0", "0.0", "false", "no", "n", "not purchased"]:
        return "Not Purchased"
    return str(val)

# =========================================================
# 2. PURCHASE DISTRIBUTION (Fixed for your dataset)
# =========================================================
def purchase_distribution(df):
    if "purchased" not in df.columns:
        return None

    # Map values and print unique check to console to verify
    purchase_data = df["purchased"].apply(map_purchase)
    print("Value counts for debugging:\n", purchase_data.value_counts())

    data = purchase_data.value_counts().reset_index()
    data.columns = ["Status", "Count"]

    fig = px.pie(
        data,
        names="Status",
        values="Count",
        hole=0.35,
        title="Purchase vs Non-Purchase Sessions"
    )
    return fig

# =========================================================
# 3. AVERAGE CUSTOMER BEHAVIOR (Updated column names)
# =========================================================
def average_customer_behavior(df):
    result = {}
    # Aapke dataset ke exact column names yahan match karaye gaye hain:
    cols = ["pages_viewed", "time_on_site_sec", "discount_percent", "revenue", "rating"]
    for col in cols:
        if col in df.columns:
            result[f"average_{col}"] = float(df[col].mean())
    return result

# =========================================================
# EXAMPLE EXECUTION
# =========================================================
if __name__ == "__main__":
    # Apni file load karein
    df = load_data("customer_analytics.csv")
    
    # Check karein ki data mein kitne Purchased aur Not Purchased hain
    fig = purchase_distribution(df)
    if fig:
        print("Chart generated successfully!")