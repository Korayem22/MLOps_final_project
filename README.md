# Hand Gesture Recognition API

A FastAPI-based service that classifies hand gestures using MediaPipe landmarks and a trained Random Forest model.

## Features

- Accepts 21 (x, y, z) hand landmarks as input
- Predicts hand gesture label
- Includes real-time inference support
- Exposed via REST API (FastAPI)
- Tested with pytest
- Dockerized for portability

## Project Structure

├── api.py # FastAPI app  
├── test.py # Unit tests  
├── requirements.txt # Dependencies  
├── Dockerfile # Build and run the app  
├── best model files/ # Contains rf_model.pkl and label_encoder.pkl

## Example Input

{
"landmarks": [[0.1, 0.2, 0.0], ..., [1.1, 1.2, 0.0]] // 21 total
}

## Run with Docker

docker build -t hand-gesture-api .
docker run -p 8000:8000 hand-gesture-api

## API Docs

Once running, open: http://127.0.0.1:8000/docs

## Tests

pytest test.py

Tests run automatically during Docker image build.

## Monitoring Metrics

We track the following metrics in our production setup:

- **Model-related**: `inference_latency_seconds`
  - Reason: Helps detect model slowness, essential for real-time gesture response.
- **Data-related**: `invalid_landmark_shape_total`
  - Reason: Detects malformed or incomplete MediaPipe inputs.
- **Server-related**: `http_request_total` and `request_time_total`
  - Reason: Monitors API usage volume and request time trends.
