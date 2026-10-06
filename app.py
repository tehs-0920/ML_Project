from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Load the final Random Forest model package
PROJECT_DIR = Path(__file__).resolve().parent
model_package = joblib.load(PROJECT_DIR / "final_crop_recommendation_model.pkl")

model = model_package["model"]
label_encoder = model_package["label_encoder"]
features = model_package["features"]


# Create SHAP explainer
explainer = shap.TreeExplainer(model)


# Create FastAPI application
app = FastAPI(
    title="Intelligent Crop Recommendation System",
    description="Crop recommendation using Random Forest with SHAP explainability",
    version="1.1"
)


# Allow the frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# Home page
@app.get("/")
def home():
    return FileResponse(PROJECT_DIR / "index.html")


# Crop prediction endpoint
@app.post("/predict")
def predict_crop(request: CropInput):
    # Create input DataFrame
    input_data = pd.DataFrame([request.model_dump()])

    # Keep the exact feature order used during training
    input_data = input_data[features]

    # Predict crop class
    predicted_class = model.predict(input_data)[0]

    # Convert class number to crop name
    predicted_crop = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Calculate confidence
    confidence = probabilities[predicted_class] * 100

    # Calculate SHAP values
    shap_values = explainer.shap_values(input_data)

    # SHAP output for this model:
    # samples × features × classes
    predicted_shap_values = shap_values[0, :, predicted_class]

    # Create feature contribution list
    explanations = []

    for feature, value in zip(features, predicted_shap_values):
        explanations.append({
            "feature": feature,
            "contribution": round(float(value), 6)
        })

    # Sort by absolute contribution
    explanations.sort(
        key=lambda x: abs(x["contribution"]),
        reverse=True
    )

    return {
        "recommended_crop": predicted_crop,
        "confidence": round(float(confidence), 2),
        "shap_explanation": explanations
    }
