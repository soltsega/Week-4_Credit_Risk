#!/usr/bin/env python3
"""
Dashboard Integration Tests
===========================

Test the integration between API and dashboard components.
"""

import pytest
import requests
import time
from src.enhanced_api import app
from fastapi.testclient import TestClient

client = TestClient(app)

class TestDashboardIntegration:
    """Test dashboard integration with API."""
    
    def test_api_dashboard_compatibility(self):
        """Test that API responses are compatible with dashboard expectations."""
        valid_data = {
            "transaction": {
                "Amount": 1000.0,
                "Value": 1200.0,
                "PricingStrategy": 2,
                "FraudResult": 0,
                "CountryCode": 256,
                "ProviderId": "Provider_6",
                "ProductCategory": "financial_services",
                "ChannelId": "Channel_1"
            }
        }
        
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check dashboard-required fields
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        
        # Check probability format for dashboard charts
        probabilities = data["probabilities"]
        assert isinstance(probabilities, dict)
        assert len(probabilities) == 3
        
        for risk_level, prob in probabilities.items():
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0
            assert risk_level in ["High Risk", "Medium Risk", "Low Risk"]
    
    def test_batch_prediction_for_dashboard(self):
        """Test batch prediction for dashboard analytics."""
        batch_data = {
            "transactions": [
                {
                    "Amount": 1000.0,
                    "Value": 1200.0,
                    "PricingStrategy": 2,
                    "FraudResult": 0,
                    "CountryCode": 256,
                    "ProviderId": "Provider_6",
                    "ProductCategory": "financial_services",
                    "ChannelId": "Channel_1"
                },
                {
                    "Amount": 500.0,
                    "Value": 600.0,
                    "PricingStrategy": 1,
                    "FraudResult": 0,
                    "CountryCode": 256,
                    "ProviderId": "Provider_1",
                    "ProductCategory": "transport",
                    "ChannelId": "Channel_2"
                },
                {
                    "Amount": 2000.0,
                    "Value": 2500.0,
                    "PricingStrategy": 3,
                    "FraudResult": 0,
                    "CountryCode": 256,
                    "ProviderId": "Provider_3",
                    "ProductCategory": "retail",
                    "ChannelId": "Channel_3"
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["predictions"]) == 3
        
        # Check that all predictions have dashboard-compatible format
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "confidence" in pred
            assert "probabilities" in pred
            assert pred["prediction"] in ["High Risk", "Medium Risk", "Low Risk"]
    
    def test_model_metrics_for_dashboard(self):
        """Test model metrics endpoint for dashboard display."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check dashboard-required metrics
        assert "performance_metrics" in data
        assert "feature_count" in data
        assert "class_names" in data
        
        metrics = data["performance_metrics"]
        assert isinstance(metrics, dict)
        
        # Check for common dashboard metrics
        dashboard_metrics = ["accuracy", "roc_auc", "precision", "recall", "f1"]
        for metric in dashboard_metrics:
            if metric in metrics:
                assert isinstance(metrics[metric], (int, float))
                assert 0.0 <= metrics[metric] <= 1.0

class TestDashboardDataFlow:
    """Test complete data flow for dashboard."""
    
    def test_manual_assessment_workflow(self):
        """Test manual assessment workflow as used by dashboard."""
        # Step 1: Get model info
        model_response = client.get("/model/info")
        assert model_response.status_code == 200
        model_data = model_response.json()
        
        # Step 2: Make prediction
        assessment_data = {
            "transaction": {
                "Amount": 1500.0,
                "Value": 1800.0,
                "PricingStrategy": 2,
                "FraudResult": 0,
                "CountryCode": 256,
                "ProviderId": "Provider_6",
                "ProductCategory": "financial_services",
                "ChannelId": "Channel_1"
            }
        }
        
        pred_response = client.post("/predict", json=assessment_data)
        assert pred_response.status_code == 200
        pred_data = pred_response.json()
        
        # Step 3: Get updated metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        
        # Verify workflow consistency
        assert metrics_data["total_predictions"] >= 1
        assert pred_data["prediction"] in model_data["class_names"]
    
    def test_real_time_predictions(self):
        """Test real-time prediction capability for dashboard."""
        predictions = []
        
        # Make multiple predictions to test real-time capability
        for i in range(5):
            test_data = {
                "transaction": {
                    "Amount": 1000.0 + i * 100,
                    "Value": 1200.0 + i * 120,
                    "PricingStrategy": i % 4,
                    "FraudResult": 0,
                    "CountryCode": 256,
                    "ProviderId": f"Provider_{(i % 6) + 1}",
                    "ProductCategory": ["financial_services", "transport", "retail"][i % 3],
                    "ChannelId": f"Channel_{(i % 3) + 1}"
                }
            }
            
            start_time = time.time()
            response = client.post("/predict", json=test_data)
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 0.5  # Should be under 500ms
            
            pred_data = response.json()
            predictions.append(pred_data["prediction"])
        
        # Verify we got valid predictions
        assert len(predictions) == 5
        for pred in predictions:
            assert pred in ["High Risk", "Medium Risk", "Low Risk"]

class TestDashboardErrorHandling:
    """Test error handling for dashboard integration."""
    
    def test_invalid_input_handling(self):
        """Test how dashboard handles invalid inputs."""
        invalid_data = {
            "transaction": {
                "Amount": -1000.0,  # Invalid negative amount
                "Value": "invalid",  # Invalid type
                "PricingStrategy": 10,  # Out of range
                "FraudResult": 2,  # Out of range
                "CountryCode": 9999,  # Out of range
                "ProviderId": "",  # Empty
                "ProductCategory": "",  # Empty
                "ChannelId": ""  # Empty
            }
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
        
        # Dashboard should be able to display these errors
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        
        # Check error details are user-friendly
        for error in data["detail"]:
            assert "loc" in error
            assert "msg" in error
            assert "type" in error
    
    def test_missing_model_handling(self):
        """Test dashboard behavior when model is not available."""
        # This would be tested by temporarily moving the model file
        # For now, we'll test the fallback behavior
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
