from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="PD Logistic Model API v2")

# Load trained model
model = joblib.load("pd_model.pkl")

# Print model info at startup (IMPORTANT)
print("=== MODEL LOADED ===")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("====================")

# Input schema (must match training order exactly)
class PDInput(BaseModel):
    Age: int = Field(..., ge=18, le=80)
    CreditScore: int = Field(..., ge=300, le=900)
    LoanTerm: int = Field(..., gt=0)
    InterestRate: float = Field(..., gt=0)
    Emp_Risk: int = Field(..., ge=0, le=1)
    FOIR: float = Field(..., ge=0, le=1)

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/v2/predict")
def predict(data: PDInput):
    try:
        # Feature order MUST match training
        X = np.array([[ 
            data.Age,
            data.CreditScore,
            data.LoanTerm,
            data.InterestRate,
            data.Emp_Risk,
            data.FOIR
        ]])

        pd_value = model.predict_proba(X)[0][1]

        return {
            "raw_probability": float(pd_value),
            "pd_percent": float(pd_value) * 100,
            "risk_band": risk_band(pd_value)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def risk_band(pd):
    if pd < 0.03:
        return "LOW"
    elif pd < 0.15:
        return "MEDIUM"
    elif pd < 0.25:
        return "HIGH"
    else:
        return "VERY_HIGH"
print("=== MODEL INFO ===")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Number of features expected:", model.coef_.shape[1])
print("===================")
