from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import List
from starlette.responses import JSONResponse

app = FastAPI(title="Hand Gesture Recognition API")

model = joblib.load("best model files/rf_model.pkl")
label_encoder = joblib.load("best model files/label_encoder.pkl")

columns = [f"{axis}{i}" for i in range(1, 22) for axis in ["x", "y", "z"]]

class LandmarkRequest(BaseModel):
    landmarks: List[List[float]]  # List of 21 landmarks, each with [x, y, z]

@app.get("/")
def root():
    return {"message": "Hand Gesture Recognition API is running"}

@app.post("/predict")
def predict_landmark(request: LandmarkRequest):
    landmarks = request.landmarks

    if len(landmarks) != 21:
        raise HTTPException(status_code=400, detail="Exactly 21 landmarks are required.")

    try:
        # Normalize
        wrist = np.array(landmarks[0][:2])
        middle_tip = np.array(landmarks[12][:2])
        scale = np.linalg.norm(wrist - middle_tip)
        scale = 1 if scale == 0 else scale
        normalized = [( (x - wrist[0])/scale, (y - wrist[1])/scale, z/scale ) for x, y, z in landmarks]

        # Prepare input
        flat = np.array(normalized).flatten().reshape(1, -1)
        df_input = pd.DataFrame(flat, columns=columns)

        # Predict
        prediction = model.predict(df_input)[0]
        label = label_encoder.inverse_transform([prediction])[0]
        return JSONResponse(content={"predicted_label": label})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
