# Customer Churn Prediction Pipeline

## 1. Project Overview

The Customer Churn Prediction Pipeline is a Machine Learning project that predicts whether a customer is likely to leave a subscription-based service.

The project uses customer information such as tenure, contract type, monthly charges and services subscribed to identify customers who may churn.

The trained model is deployed using FastAPI, allowing users to submit customer details and receive churn predictions through a REST API.

## 2. Objectives

- Load and understand customer churn data.
- Perform data cleaning and preprocessing.
- Create new features to improve model performance.
- Train and compare multiple Machine Learning models.
- Perform hyperparameter tuning.
- Select and save the best-performing model.
- Deploy the model using FastAPI.
- Test predictions through an API endpoint.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- FastAPI
- Uvicorn
- Joblib

## 4. Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains 7,043 customer records and 21 original columns, including:

- Customer tenure
- Monthly charges
- Total charges
- Contract type
- Internet services
- Payment method
- Churn status

## 5. Data Preprocessing

The following preprocessing steps were performed:

1. Handled missing values in TotalCharges.
2. Removed the customerID column.
3. Converted the Churn column into numerical values.
4. Applied one-hot encoding to categorical features.
5. Standardized numerical features.
6. Created additional features such as TenureGroup, AverageMonthlySpend and ServiceCount.

## 6. Machine Learning Models

Three Machine Learning algorithms were trained and compared:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

GridSearchCV was used for hyperparameter tuning with 5-fold cross-validation.

Average Precision was used as the primary model-selection metric.

### Model Comparison

| Model | Average Precision |
|---|---:|
| Logistic Regression | 0.631 |
| Random Forest | 0.637 |
| Gradient Boosting | 0.658 |

Gradient Boosting achieved the highest Average Precision score and was selected as the final model.

## 7. Model Deployment

The trained model was saved using Joblib in the `models` folder.

FastAPI was used to create a REST API with the following endpoints:

- GET `/` – Checks whether the API is running.
- POST `/predict` – Accepts customer details and predicts churn.

## 8. How to Run the Project

### Step 1: Clone or download the project

Open the project folder in VS Code.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Train the model

```bash
python train.py
```

### Step 6: Start the API server

```bash
python -m uvicorn main:app --reload
```

### Step 7: Open API documentation

Visit:

http://127.0.0.1:8000/docs

Use the POST `/predict` endpoint to submit customer details and receive a prediction.

## 9. Sample Prediction Output

```json
{
  "churn_probability": 0.5821,
  "churn_prediction": 1,
  "churn_status": "Yes"
}
```

The result indicates an estimated 58.21% probability of customer churn.

## 10. Conclusion

This project demonstrates an end-to-end Machine Learning pipeline, starting from data preprocessing and model training to deployment through a REST API.

It helps identify customers who may be at risk of leaving a subscription-based service and can support businesses in planning customer retention strategies.