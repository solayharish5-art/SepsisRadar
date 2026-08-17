from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List
app = FastAPI()

# Allow the React frontend (running on a different port) to call this API.
# allow_origins=["*"] is fine for a hackathon demo; a production app would
# restrict this to the specific frontend URL instead.
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

def add_alert_logic(trajectory):
    consecutive_high = 0
    result = []

    for point in trajectory:
        if point["risk_score"] >= RISK_THRESHOLD:
            consecutive_high += 1
        else:
            consecutive_high = 0

        alert_triggered = consecutive_high >= PERSISTENCE_COUNT

        result.append({
            **point,
            "alert": alert_triggered,
            "consecutive_elevated_readings": consecutive_high
        })

    return result

class Vitals(BaseModel):
    heart_rate: float
    temp: float
    resp_rate: float
    wbc: float
    systolic_bp: float

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

    coefficients = model.coef_[0]
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
        "note": "Prediction from trained logistic regression model"
    }

@app.post("/predict_trajectory")
def predict_trajectory(vitals_sequence: List[Vitals]):
    trajectory = []
    for i, vitals in enumerate(vitals_sequence):
        input_data = pd.DataFrame([{
            "heart_rate": vitals.heart_rate,
            "temp": vitals.temp,
            "resp_rate": vitals.resp_rate,
            "wbc": vitals.wbc,
            "systolic_bp": vitals.systolic_bp
        }])
        risk_prob = model.predict_proba(input_data)[0][1]
        trajectory.append({
            "time_point": i + 1,
            "risk_score": round(float(risk_prob), 4)
        })

    return {"trajectory": add_alert_logic(trajectory)}
