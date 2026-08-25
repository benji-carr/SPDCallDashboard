import requests
import pandas as pd

ENDPOINT = "https://data.seattle.gov/resource/tazs-3rd5.json"

params = {
    "$limit": 10000,
    "$order": ":id",
}

response = requests.get(ENDPOINT, params=params, timeout=30)
response.raise_for_status()

records = response.json()
df = pd.DataFrame(records)

print("DF SHAPE: ")
print(df.shape)
print("\n")
print("DF COLUMNS: ")
print(df.columns.tolist())
print("\n")
print("DF HEAD: ")
print(df.head())
print("\n")
print("DF DATA TYPES: ")
print(df.dtypes)
print("\n")
print("DF NULL VALUE COUNTS: ")
print(df.isna().mean().sort_values(ascending=False).head(20))

print("DF SHAPE: ")
print(df['offense_date'].head(5))
print("\n")

print("ROW GRAIN CHECK: ")
print(f"offense_id is unique: {df['offense_id'].is_unique}")
print(f"Head of offense_id frequencies: \n{df['offense_id'].value_counts().head(5)}")
print(f"report_number is unique: {df['report_number'].is_unique}")
print(f"Head of report_number frequencies: \n{df['report_number'].value_counts().head(5)}")

df["offense_date"] = pd.to_datetime(df["offense_date"], errors="coerce")
df["report_date_time"] = pd.to_datetime(df["report_date_time"], errors="coerce")

print(df["offense_date"].isna().mean(), df["report_date_time"].isna().mean())

lag_days = (
    df["report_date_time"].dt.normalize()
    - df["offense_date"].dt.normalize()
).dt.days

print(lag_days.describe())
print(lag_days.quantile([0.5, 0.9, 0.95, 0.99]))