from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Optional

app = FastAPI()

# Allow the React frontend (running on a different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("sepsis_model.pkl")
feature_names = ["heart_rate", "temp", "resp_rate", "wbc", "systolic_bp"]
RISK_THRESHOLD = 0.6       # risk score above this counts as "elevated"
PERSISTENCE_COUNT = 2      # must stay elevated for this many consecutive readings to alert

# --- PYDANTIC MODELS (Defines the exact JSON structure for the frontend) ---
class Vitals(BaseModel):
    heart_rate: float
    temp: float
    resp_rate: float
    wbc: float
    systolic_bp: float

class FactorImpact(BaseModel):
    factor: str
    impact: float
    direction: str

class TrajectoryPoint(BaseModel):
    time_point: int
    time_horizon: str
    risk_score: float
    alert: bool
    consecutive_elevated_readings: int
    top_factors: List[FactorImpact]

class TrajectoryResponse(BaseModel):
    trajectory: List[TrajectoryPoint]
    note: str

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Hello, this is your Python backend!"}

@app.post("/predict")
def predict_risk(vitals: Vitals):
    input_data = pd.DataFrame([{
        "heart_rate": vitals.heart_rate,
        "temp": vitals.temp,
        "resp_rate": vitals.resp_rate,
        "wbc": vitals.wbc,
        "systolic_bp": vitals.systolic_bp
    }])

    risk_prob = model.predict_proba(input_data)[0][1]

    coefficients = model.feature_importances_
    contributions = {}
    for name, coef in zip(feature_names, coefficients):
        contributions[name] = round(float(coef * input_data[name].values[0]), 2)

    sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_factors = [
        {"factor": name, "impact": value, "direction": "increases risk" if value > 0 else "decreases risk"}
        for name, value in sorted_contributions
    ]

    return {
        "risk_score": round(float(risk_prob), 4),
        "top_factors": top_factors,
        "note": "Prediction from trained XGBoost model (PhysioNet 2019 dataset)"
    }

@app.post("/predict_trajectory", response_model=TrajectoryResponse)
def predict_trajectory(vitals_sequence: List[Vitals]):
    trajectory_output = []
    consecutive_high = 0

    for i, vitals in enumerate(vitals_sequence):
        input_data = pd.DataFrame([{
            "heart_rate": vitals.heart_rate,
            "temp": vitals.temp,
            "resp_rate": vitals.resp_rate,
            "wbc": vitals.wbc,
            "systolic_bp": vitals.systolic_bp
        }])
        
        # 1. Calculate risk probability
        risk_prob = float(model.predict_proba(input_data)[0][1])
        
        # 2. Calculate feature importances for this specific time point
        coefficients = model.feature_importances_
        contributions = {}
        for name, coef in zip(feature_names, coefficients):
            contributions[name] = round(float(coef * input_data[name].values[0]), 2)

        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_factors = [
            {"factor": name, "impact": value, "direction": "increases risk" if value > 0 else "decreases risk"}
            for name, value in sorted_contributions
        ]

        # 3. Alert Fatigue Logic
        if risk_prob >= RISK_THRESHOLD:
            consecutive_high += 1
        else:
            consecutive_high = 0

        alert_triggered = consecutive_high >= PERSISTENCE_COUNT

        # 4. Append to trajectory array
        trajectory_output.append({
            "time_point": i + 1,
            "time_horizon": "6h ahead",
            "risk_score": round(risk_prob, 4),
            "alert": alert_triggered,
            "consecutive_elevated_readings": consecutive_high,
            "top_factors": top_factors
        })

    return {
        "trajectory": trajectory_output,
        "note": "Prediction from trained XGBoost model (PhysioNet 2019 dataset)"
    }