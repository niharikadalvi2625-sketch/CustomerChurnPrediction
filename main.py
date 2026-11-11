
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/best_churn_model.joblib")

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using Machine Learning",
    version="1.0"
)

# Customer input structure
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }

# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    # Feature Engineering
    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Medium", "Long", "Loyal"]
    )

    data["AverageMonthlySpend"] = (
        data["TotalCharges"] / data["tenure"].replace(0, 1)
    )

    service_columns = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    data["ServiceCount"] = (
        data[service_columns] == "Yes"
    ).sum(axis=1)

    # Predict churn
    probability = model.predict_proba(data)[0][1]
    prediction = int(probability >= 0.5)

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction,
        "churn_status": "Yes" if prediction == 1 else "No"
    }
