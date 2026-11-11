
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

# 1. Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# 2. Clean data
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 3. Feature Engineering
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

# 4. Remove customer ID
df = df.drop("customerID", axis=1)

# 5. Separate input and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# 6. Identify numerical and categorical columns
numeric_features = X.select_dtypes(include="number").columns.tolist()

categorical_features = (
    X.select_dtypes(include="str").columns.tolist()
    + ["TenureGroup"]
)

# 7. Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Define three models and parameter grids
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"classifier__C": [0.1, 1, 10]}
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, None]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1]
        }
    )
}

# 10. Train and tune models
results = []
best_score = -1
best_model = None

for name, (model, parameters) in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=5,
        scoring="average_precision",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Predict on test data
    predictions = grid_search.predict(X_test)
    probabilities = grid_search.predict_proba(X_test)[:, 1]

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print("\nModel:", name)
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Average Precision:", round(pr_auc, 4))

    results.append((name, pr_auc))

    # Select model using Average Precision
    if pr_auc > best_score:
        best_score = pr_auc
        best_model = grid_search.best_estimator_

# 11. Save the best model
joblib.dump(best_model, "models/best_churn_model.joblib")

print("\nModel Comparison:")
for name, score in results:
    print(name, ":", round(score, 4))

print("\nBest Model:", max(results, key=lambda x: x[1])[0])
print("Best model saved successfully!")
