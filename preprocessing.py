
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

# Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove customer ID
df = df.drop("customerID", axis=1)

# Feature Engineering
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New", "Medium", "Long", "Loyal"]
)

df["AverageMonthlySpend"] = (
    df["TotalCharges"] / df["tenure"].replace(0, 1)
)

service_columns = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]

df["ServiceCount"] = (df[service_columns] == "Yes").sum(axis=1)

# Separate target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=df.select_dtypes(include="str").columns,
    drop_first=True,
    dtype=int
)

# Scale numerical features
numerical_columns = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "AverageMonthlySpend", "ServiceCount"
]

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Display results
print("Processed Dataset:")
print(df.head())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nChurn Values:")
print(df["Churn"].value_counts())
