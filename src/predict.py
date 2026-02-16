import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Run train_model.py first.")

pipeline = joblib.load(MODEL_PATH)


def predict_customer(data_dict):
    df = pd.DataFrame([data_dict])
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][1]
    return pred, prob


if __name__ == "__main__":
    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80,
        "TotalCharges": 400
    }

    pred, prob = predict_customer(sample_customer)

    print("Prediction:", "Churn" if pred == 1 else "No Churn")
    print("Probability:", f"{prob*100:.1f}%")
