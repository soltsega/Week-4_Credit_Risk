# tests/test_model.py
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# Test data setup
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 2, 100)  # Binary classification
    return X, y

def test_model_loading():
    """Test that the saved model can be loaded correctly"""
    try:
        model_data = joblib.load('credit_risk_model_v1.joblib')
        assert 'model' in model_data
        assert 'preprocessor' in model_data
        assert 'label_encoder' in model_data
    except FileNotFoundError:
        pytest.fail("Model file not found")
    except Exception as e:
        pytest.fail(f"Error loading model: {str(e)}")

def test_model_prediction_shape(sample_data):
    """Test that model predictions have the correct shape"""
    X, y = sample_data
    try:
        model_data = joblib.load('credit_risk_model_v1.joblib')
        model = model_data['model']
        predictions = model.predict(X)
        assert predictions.shape[0] == X.shape[0]
    except Exception as e:
        pytest.fail(f"Prediction test failed: {str(e)}")

def test_preprocessing_steps():
    """Test that the preprocessor transforms data correctly"""
    try:
        model_data = joblib.load('credit_risk_model_v1.joblib')
        preprocessor = model_data['preprocessor']
        
        # Test with sample data
        sample_data = np.random.rand(10, 5)  # 10 samples, 5 features
        transformed = preprocessor.transform(sample_data)
        
        # Check if transformation maintains sample count
        assert transformed.shape[0] == sample_data.shape[0]
    except Exception as e:
        pytest.fail(f"Preprocessing test failed: {str(e)}")

def test_label_encoder():
    """Test the label encoder functionality"""
    try:
        model_data = joblib.load('credit_risk_model_v1.joblib')
        le = model_data['label_encoder']
        
        # Test inverse transform
        test_labels = [0, 1, 2]  # Assuming 3 classes
        inverse = le.inverse_transform(test_labels)
        assert len(inverse) == len(test_labels)
    except Exception as e:
        pytest.fail(f"Label encoder test failed: {str(e)}")

def test_prediction_probabilities(sample_data):
    """Test that prediction probabilities sum to ~1"""
    X, _ = sample_data
    try:
        model_data = joblib.load('credit_risk_model_v1.joblib')
        model = model_data['model']
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            assert np.allclose(probs.sum(axis=1), 1.0, rtol=1e-3)
    except Exception as e:
        pytest.fail(f"Probability test failed: {str(e)}")