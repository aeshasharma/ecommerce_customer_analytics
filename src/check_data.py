import pandas as pd

from src.project_paths import DATA_FILE


print("=" * 60)
print("E-COMMERCE DATASET CHECK")
print("=" * 60)


# Load dataset
df = pd.read_csv(DATA_FILE)


print("\nDataset loaded successfully!")


print("\nShape:")
print(df.shape)


print("\nNumber of rows:")
print(df.shape[0])


print("\nNumber of columns:")
print(df.shape[1])


print("\nColumn names:")
for column in df.columns:
    print("-", column)


print("\nFirst 5 rows:")
print(df.head())


print("\nData types:")
print(df.dtypes)


print("\nMissing values:")
print(df.isnull().sum())


print("\nDuplicate rows:")
print(df.duplicated().sum())


print("\nDataset information:")
df.info()