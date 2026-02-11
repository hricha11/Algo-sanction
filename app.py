from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="PD Logistic Model API vFinal")

# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("pd_model.pkl")

print("=== MODEL LOADED ===")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Number of features expected:", model.coef_.shape[1])
print("====================")

EXPECTED_FEATURES = 6

if model.coef_.shape[1] != EXPECTED_FEATURES:
    raise Exception("Model feature mismatch! Check training feature order.")


FEATURE_ORDER = [
    "Age",
    "CreditScore",
    "LoanTerm",
    "InterestRate",
    "Emp_Risk",
    "FOIR"
]

# ==============================
# EMPLOYMENT RISK ENCODING
# ==============================

def encode_emp_risk(emp_type: str) -> int:
    """
    Must match training logic exactly.
    0 = Low Risk
    1 = High Risk
    """
    mapping = {
        "Salaried": 0,
        "Self Employed": 1,
        "Unemployed": 2
    }

    return mapping.get(emp_type, 1)


# ==============================
# SALESFORCE TRANSFORMATION
# ==============================

def transform_salesforce_record(record: dict):
    """
    Converts raw Salesforce fields into model-ready numeric format.
    Confirm percent scaling matches training.
    """

    return {
        "Age": record["Age__c"],
        "CreditScore": record["Cibil_Score__c"],
        "LoanTerm": record["Tenure_required_months__c"],
        "InterestRate": record["ROI__c"],   
        "Emp_Risk": encode_emp_risk(record["Employment_Type__c"]),
        "FOIR": record["FOIR__c"]/100          
    }


# ==============================
# INPUT SCHEMA (MANUAL NUMERIC INPUT)
# ==============================

class PDInput(BaseModel):
    Age: int = Field(..., ge=18, le=80)
    CreditScore: int = Field(..., ge=300, le=900)
    LoanTerm: int = Field(..., gt=0)
    InterestRate: float = Field(..., gt=0)
    Emp_Risk: int = Field(..., ge=0, le=1)
    FOIR: float = Field(..., ge=0, le=1)


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def health():
    return {"status": "running"}


# ==============================
# PREDICT (NUMERIC DIRECT INPUT)
# ==============================

@app.post("/predict")
def predict(data: PDInput):
    try:
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


# ==============================
# PREDICT (RAW SALESFORCE RECORD)
# ==============================

@app.post("/predict-salesforce")
def predict_salesforce(record: dict):
    try:
        transformed = transform_salesforce_record(record)

        X = np.array([[ 
            transformed["Age"],
            transformed["CreditScore"],
            transformed["LoanTerm"],
            transformed["InterestRate"],
            transformed["Emp_Risk"],
            transformed["FOIR"]
        ]])

        pd_value = model.predict_proba(X)[0][1]

        return {
            "raw_probability": float(pd_value),
            "pd_percent": float(pd_value) * 100,
            "risk_band": risk_band(pd_value)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# RISK BANDING
# ==============================

def risk_band(pd: float):

    if pd < 0.05:
        return "LOW"
    elif pd < 0.20:
        return "MEDIUM"
    elif pd < 0.35:
        return "HIGH"
    else:
        return "VERY_HIGH"
