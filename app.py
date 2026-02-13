from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="PD Logistic Model API vFinal")


model = joblib.load("pd_model.pkl")

print("=== MODEL LOADED ===")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Number of features expected:", model.coef_.shape[1])
print("====================")

EXPECTED_FEATURES = 5

if model.coef_.shape[1] != EXPECTED_FEATURES:
    raise Exception("Model feature mismatch!")

FEATURE_ORDER = [
    "Age",
    "CreditScore",
    "InterestRate",
    "Emp_Risk",
    "FOIR"
]


def encode_emp_risk(emp_type: str) -> int:
    """
    Must match training logic exactly.
    0 = Salaried
    1 = Self Employed
    2 = Unemployed
    """
    mapping = {
        "Salaried": 0,
        "Self Employed": 1,
        "Unemployed": 2
    }

    return mapping.get(emp_type, 1)


def transform_salesforce_record(record: dict):
    """
    Converts raw Salesforce fields into model-ready numeric format.
    Ensure FOIR scale matches training (0–1 or 0–100).
    """

    return {
        "Age": record["Age__c"],
        "CreditScore": record["Cibil_Score__c"],
        "InterestRate": record["ROI__c"],
        "Emp_Risk": encode_emp_risk(record["Employment_Type__c"]),
        "FOIR": record["FOIR__c"]
    }


class PDInput(BaseModel):
    Age: int = Field(..., ge=18, le=80)
    CreditScore: int = Field(..., ge=300, le=900)
    InterestRate: float = Field(..., gt=0)
    Emp_Risk: int = Field(..., ge=0, le=2)
    FOIR: float = Field(..., ge=0, le=100)


@app.get("/")
def health():
    return {"status": "running"}



def risk_band(pd: float):
    if pd < 0.05:
        return "LOW"
    elif pd < 0.20:
        return "MEDIUM"
    elif pd < 0.35:
        return "HIGH"
    else:
        return "VERY_HIGH"

@app.post("/predict")
def predict(data: PDInput):
    try:
        X = np.array([[
            data.Age,
            data.CreditScore,
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


@app.post("/predict-salesforce")
def predict_salesforce(record: dict):
    try:
        # Check required fields exist 
        required_fields = [
            "Age__c",
            "Cibil_Score__c",
            "ROI__c",
            "Employment_Type__c",
            "FOIR__c"
        ]

        field_labels = {
            "Age__c": "Age",
            "Cibil_Score__c": "Credit Score",
            "ROI__c": "Interest Rate",
            "Employment_Type__c": "Employment Type",
            "FOIR__c": "FOIR"
        }

        for field in required_fields:
            if field not in record or record[field] in [None, ""]:
                raise HTTPException(
                    status_code=400,
                    detail=f"{field_labels[field]} is required."
                )


        # Transform 
        transformed = transform_salesforce_record(record)

        # Convert safely to float
        try:
            X = np.array([[ 
                float(transformed["Age"]),
                float(transformed["CreditScore"]),
                float(transformed["InterestRate"]),
                float(transformed["Emp_Risk"]),
                float(transformed["FOIR"])
            ]])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid numeric format in input fields."
            )

        # Check for NaN explicitly 
        if np.isnan(X).any():
            raise HTTPException(
                status_code=400,
                detail="Input contains invalid or missing numeric values."
            )

        # Predict
        pd_value = model.predict_proba(X)[0][1]

        return {
            "raw_probability": float(pd_value),
            "pd_percent": float(pd_value) * 100,
            "risk_band": risk_band(pd_value)
        }

    except HTTPException as e:
        raise e

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal risk engine error."
        )
