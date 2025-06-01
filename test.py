from fastapi.testclient import TestClient
from api import app
import pytest

client = TestClient(app)

# Test root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hand Gesture Recognition API is running"}

# Valid landmarks sample (21 points)
valid_landmarks = {
    "landmarks": [[0.1, 0.2, 0.0]] * 21  # dummy normalized data
}

# Test /predict with valid input
def test_predict_valid():
    response = client.post("/predict", json=valid_landmarks)
    assert response.status_code == 200
    assert "predicted_label" in response.json()

# Test /predict with missing landmarks
def test_predict_missing_points():
    invalid_data = {"landmarks": [[0.1, 0.2, 0.0]] * 20}  # only 20 points
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Exactly 21 landmarks with (x, y, z) each are required."

# Test /predict with malformed input
def test_predict_malformed():
    malformed_data = {"invalid_key": []}
    response = client.post("/predict", json=malformed_data)
    assert response.status_code == 422
