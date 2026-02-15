#!/usr/bin/env python3
"""
Enhanced Production API - Credit Risk Project
========================================

Production-ready API with monitoring, security, and advanced endpoints.
Focus on reliability and business requirements.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import logging
import time
from datetime import datetime
import json
from contextlib import asynccontextmanager
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global variables for model and data
model_bundle = None
model_metrics = {
    'total_predictions': 0,
    'error_count': 0,
    'avg_response_time': 0.0,
    'last_prediction_time': None,
    'model_load_time': None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    global model_bundle, model_metrics
    
    try:
        logger.info("Loading model...")
        model_bundle = joblib.load('random_forest_weighted_balanced.joblib')
        model_metrics['model_load_time'] = datetime.now()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Assessment API",
    description="Production-ready credit risk assessment service",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TransactionFeatures(BaseModel):
    """Transaction features for prediction."""
    Amount: float = Field(..., description="Transaction amount")
    Value: float = Field(..., description="Transaction value")
    PricingStrategy: int = Field(..., ge=0, le=4, description="Pricing strategy (0-4)")
    FraudResult: int = Field(..., ge=0, le=1, description="Fraud detection result")
    CountryCode: int = Field(..., ge=0, le=999, description="Country code")
    ProviderId: str = Field(..., description="Transaction provider")
    ProductCategory: str = Field(..., description="Product category")
    ChannelId: str = Field(..., description="Transaction channel")

class PredictionRequest(BaseModel):
    """Request for single prediction."""
    transaction: TransactionFeatures
    request_id: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    transactions: List[TransactionFeatures]
    request_id: Optional[str] = None

class PredictionResponse(BaseModel):
    """Prediction response."""
    request_id: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    risk_factors: List[str]
    timestamp: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    request_id: str
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_predictions: int
    error_rate: float
    avg_response_time_ms: float

class MetricsResponse(BaseModel):
    """Metrics response."""
    total_predictions: int
    error_count: int
    error_rate: float
    avg_response_time_ms: float
    last_prediction: Optional[str]
    model_load_time: str

# Utility functions
def get_request_id(request_id: Optional[str] = None) -> str:
    """Generate or return request ID."""
    return request_id or f"req_{int(time.time())}_{np.random.randint(1000, 9999)}"

def prepare_features(transaction: TransactionFeatures) -> pd.DataFrame:
    """Prepare features for model prediction."""
    # Create DataFrame
    data = {
        'Amount': [transaction.Amount],
        'Value': [transaction.Value],
        'PricingStrategy': [transaction.PricingStrategy],
        'FraudResult': [transaction.FraudResult],
        'CountryCode': [transaction.CountryCode],
        'ProviderId': [transaction.ProviderId],
        'ProductCategory': [transaction.ProductCategory],
        'ChannelId': [transaction.ChannelId]
    }
    df = pd.DataFrame(data)
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['ProviderId', 'ProductCategory', 'ChannelId'])
    
    # Ensure all required features exist
    feature_names = model_bundle['feature_names']
    for feature in feature_names:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Select only required features in correct order
    X = df_encoded[feature_names]
    
    return X

def make_prediction(X: pd.DataFrame) -> tuple:
    """Make prediction and return results."""
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    class_names = model_bundle['class_names']
    
    # Scale if needed
    if scaler is not None:
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
    else:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
    
    # Get class name and confidence
    predicted_class = class_names[prediction]
    confidence = probabilities[prediction]
    
    # Create probability breakdown
    prob_breakdown = {}
    for i, class_name in enumerate(class_names):
        prob_breakdown[class_name] = probabilities[i]
    
    # Get risk factors (top features)
    importances = model.feature_importances_
    feature_names = model_bundle['feature_names']
    top_indices = np.argsort(importances)[-3:][::-1]
    risk_factors = [feature_names[i] for i in top_indices]
    
    return predicted_class, confidence, prob_breakdown, risk_factors

def update_metrics(processing_time: float, error: bool = False):
    """Update performance metrics."""
    global model_metrics
    
    model_metrics['total_predictions'] += 1
    if error:
        model_metrics['error_count'] += 1
    
    # Update average response time
    total_time = model_metrics['avg_response_time'] * (model_metrics['total_predictions'] - 1)
    model_metrics['avg_response_time'] = (total_time + processing_time) / model_metrics['total_predictions']
    model_metrics['last_prediction_time'] = datetime.now()

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - (model_metrics['model_load_time'].timestamp() if model_metrics['model_load_time'] else time.time())
    error_rate = model_metrics['error_count'] / max(model_metrics['total_predictions'], 1)
    
    return HealthResponse(
        status="healthy" if model_bundle else "unhealthy",
        model_loaded=model_bundle is not None,
        uptime_seconds=uptime,
        total_predictions=model_metrics['total_predictions'],
        error_rate=error_rate,
        avg_response_time_ms=model_metrics['avg_response_time'] * 1000
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics."""
    error_rate = model_metrics['error_count'] / max(model_metrics['total_predictions'], 1)
    
    return MetricsResponse(
        total_predictions=model_metrics['total_predictions'],
        error_count=model_metrics['error_count'],
        error_rate=error_rate,
        avg_response_time_ms=model_metrics['avg_response_time'] * 1000,
        last_prediction=model_metrics['last_prediction_time'].isoformat() if model_metrics['last_prediction_time'] else None,
        model_load_time=model_metrics['model_load_time'].isoformat() if model_metrics['model_load_time'] else None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    start_time = time.time()
    request_id = get_request_id(request.request_id)
    
    try:
        # Prepare features
        X = prepare_features(request.transaction)
        
        # Make prediction
        predicted_class, confidence, probabilities, risk_factors = make_prediction(X)
        
        processing_time = (time.time() - start_time) * 1000
        update_metrics(processing_time / 1000)
        
        return PredictionResponse(
            request_id=request_id,
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            risk_factors=risk_factors,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        update_metrics(processing_time / 1000, error=True)
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    start_time = time.time()
    request_id = get_request_id(request.request_id)
    
    try:
        predictions = []
        
        for i, transaction in enumerate(request.transactions):
            # Prepare features
            X = prepare_features(transaction)
            
            # Make prediction
            predicted_class, confidence, probabilities, risk_factors = make_prediction(X)
            
            prediction = PredictionResponse(
                request_id=f"{request_id}_{i}",
                prediction=predicted_class,
                confidence=confidence,
                probabilities=probabilities,
                risk_factors=risk_factors,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=0  # Individual timing not tracked for batch
            )
            predictions.append(prediction)
        
        processing_time = (time.time() - start_time) * 1000
        update_metrics(processing_time / 1000)
        
        return BatchPredictionResponse(
            request_id=request_id,
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        update_metrics(processing_time / 1000, error=True)
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model_bundle['model']).__name__,
        "feature_count": len(model_bundle['feature_names']),
        "class_names": model_bundle['class_names'],
        "model_version": "2.0.0",
        "load_time": model_metrics['model_load_time'].isoformat() if model_metrics['model_load_time'] else None
    }

@app.get("/features")
async def get_features():
    """Get available features."""
    if not model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "features": model_bundle['feature_names'],
        "feature_count": len(model_bundle['feature_names'])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
