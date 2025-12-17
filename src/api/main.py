from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import os
import os
from pathlib import Path

# Update this path to point to your actual model file
MODEL_PATH = os.path.join(Path(__file__).parent.parent.parent, 'models', 'credit_risk_model_v1.joblib')

# Then in your load_model function:
model_data = joblib.load(MODEL_PATH)

# Initialize FastAPI
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk probability",
    version="1.0.0"
)

# Model and preprocessor will be loaded at startup
model = None
preprocessor = None
label_encoder = None

class PredictionInput(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: int
    FraudResult: int

class PredictionOutput(BaseModel):
    transaction_id: str
    risk_probability: float
    risk_category: str

@app.on_event("startup")
def load_model():
    """Load the model and related artifacts at startup"""
    global model, preprocessor, label_encoder
    try:
        # Update this path to your actual model file
        model_data = joblib.load('models/credit_risk_model_v1.joblib')
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        label_encoder = model_data['label_encoder']
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction on a single transaction"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame for processing
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict_proba(input_df)
        risk_probability = float(prediction[0][1])  # Probability of being high risk
        risk_category = "High Risk" if risk_probability > 0.5 else "Low Risk"
        
        return {
            "transaction_id": input_dict["TransactionId"],
            "risk_probability": risk_probability,
            "risk_category": risk_category
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)