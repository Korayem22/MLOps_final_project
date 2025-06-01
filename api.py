from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from starlette.responses import JSONResponse
import time
import numpy as np
import pandas as pd
import joblib
from typing import List

app = FastAPI(title="Hand Gesture Recognition API")
Instrumentator().instrument(app).expose(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or set specific origin ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Custom metrics
inference_latency = Histogram("inference_latency_seconds", "Latency for model predictions")
invalid_input_counter = Counter("invalid_landmark_shape_total", "Count of invalid input shapes")

# Load model and encoder
model = joblib.load("best model files/rf_model.pkl")
label_encoder = joblib.load("best model files/label_encoder.pkl")

# Feature columns
columns = [f"{axis}{i}" for i in range(1, 22) for axis in ["x", "y", "z"]]

# Request schema
class LandmarkRequest(BaseModel):
    landmarks: List[List[float]]

@app.get("/")
def root():
    return {"message": "Hand Gesture Recognition API is running"}

@app.post("/predict")
@inference_latency.time()
def predict_landmark(request: LandmarkRequest):
    landmarks = request.landmarks
    #print("Received request JSON:", landmarks)

    # Input validation
    if len(landmarks) != 21 or any(len(point) != 3 for point in landmarks):
        invalid_input_counter.inc()
        raise HTTPException(status_code=400, detail="Exactly 21 landmarks with (x, y, z) each are required.")

    try:
        # Normalize input
        wrist = np.array(landmarks[0][:2])
        middle_tip = np.array(landmarks[12][:2])
        scale = np.linalg.norm(wrist - middle_tip) or 1
        normalized = [( (x - wrist[0]) / scale, (y - wrist[1]) / scale, z / scale ) for x, y, z in landmarks]

        # Prepare for prediction
        flat = np.array(normalized).flatten().reshape(1, -1)
        df_input = pd.DataFrame(flat, columns=columns)

        # Predict and return result
        prediction = model.predict(df_input)[0]
        label = label_encoder.inverse_transform([prediction])[0]
        return JSONResponse(content={"predicted_label": label})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))