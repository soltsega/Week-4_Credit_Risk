from fastapi import FastAPI, HTTPException, Request
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import joblib
import numpy as np
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk API")

# Small helper for simple unit tests
def greet(name: str) -> str:
    return f"Hello, {name}!"

MODEL_PATH = "credit_risk_model_v1.joblib"

class DummyModel:
    """Simple fallback model used when a real model cannot be loaded during development/tests."""
    def predict_proba(self, X):
        import numpy as np
        n = X.shape[0]
        # return constant probabilities (col0 = prob(class 0), col1 = prob(class 1))
        return np.vstack([np.full(n, 0.3), np.full(n, 0.7)]).T

# Default fallback model bundle (ensures predict is callable even if startup doesn't run)
model_bundle = {"model": DummyModel(), "feature_names": ["feature_0", "feature_1", "feature_2"]}
model = model_bundle["model"]
preprocessor = None
le = None

@app.on_event("startup")
def load_model():
    global model_bundle, model, preprocessor, le
    try:
        model_bundle = joblib.load(MODEL_PATH)
        model = model_bundle.get("model")
        preprocessor = model_bundle.get("preprocessor", None)
        le = model_bundle.get("label_encoder", None)
        logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as exc:
        # Fall back to a minimal dummy model for local development and testing
        logger.warning(f"Could not load model from {MODEL_PATH}: {exc}; using DummyModel instead")
        model_bundle = {"model": DummyModel(), "feature_names": ["feature_0", "feature_1", "feature_2"]}
        model = model_bundle["model"]
        preprocessor = None
        le = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = request.features
        logger.info(f"Received prediction request for features: {list(features.keys())}")
        # Order the features consistently; use feature_names saved in model bundle
        if "feature_names" in model_bundle:
            ordered = [features.get(fn, 0.0) for fn in model_bundle["feature_names"]]
            X = np.array(ordered).reshape(1, -1)
        else:
            # Infer ordering from dict
            X = np.array(list(features.values())).reshape(1, -1)
        if preprocessor is not None:
            X = preprocessor.transform(X)
        prob = float(model.predict_proba(X)[0][1])
        label = int(prob >= 0.5)  # threshold 0.5 (document this)
        logger.info(f"Prediction: prob={prob:.4f}, label={label}")
        return {"probability": prob, "risk_label": label}
    except Exception as e:
        logger.exception("Error handling prediction request")
        raise HTTPException(status_code=400, detail=str(e))