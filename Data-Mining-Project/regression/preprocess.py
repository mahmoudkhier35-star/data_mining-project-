import pandas as pd
import numpy as np

df = pd.read_csv("used_car_dataset.csv")

df = df.drop_duplicates().reset_index(drop=True)

df["kmDriven"] = pd.to_numeric(
    df["kmDriven"].astype(str).str.replace(r"[^\d.]", "", regex=True),
    errors="coerce"
)

df["AskPrice"] = pd.to_numeric(
    df["AskPrice"].astype(str).str.replace(r"[^\d.]", "", regex=True),
    errors="coerce"
)

num_cols = ["Year", "Age", "kmDriven", "AskPrice"]
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=["object", "string"]).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"Outliers detected in {col}: {outliers_count} (preserved)")

df.to_csv("cleaned_used_car_dataset.csv", index=False)
print("Done! Saved cleaned_used_car_dataset.csv with shape:", df.shape)