import joblib as jb
import pandas as pd
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

app = FastAPI()
model = jb.load("recovery_model.pkl")

class RecoveryRequest(BaseModel):
    sleep_hours: float
    resting_heart_rate: int
    previous_workout_intensity: int
    muscle_soreness: int
    water_intake_liters: float


router = APIRouter(prefix="/Recovery",tags=["Recovery Prediction"])

@app.post("/predict_recovery")

def predict_recovery_score(data: RecoveryRequest):
    input_data = pd.DataFrame([{
        "sleep_hours": data.sleep_hours,
        "resting_heart_rate": data.resting_heart_rate,
        "previous_workout_intensity": data.previous_workout_intensity,
        "muscle_soreness": data.muscle_soreness,
        "water_intake_liters": data.water_intake_liters
    }])

    prediction = model.predict(input_data)
    return {"predicted_recovery_score": round(float(prediction[0]), 2)}
app.include_router(router)