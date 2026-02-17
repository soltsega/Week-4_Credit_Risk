#!/usr/bin/env python3
"""
Enhanced API Tests - Production Ready Testing
===========================================

Comprehensive test suite for the enhanced credit risk API.
Tests all endpoints, security, performance, and error handling.
"""

import pytest
import json
import time
from fastapi.testclient import TestClient
from src.enhanced_api import app

client = TestClient(app)

class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "total_predictions" in data
        assert "error_rate" in data
        assert "avg_response_time_ms" in data
        
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], float)
        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["error_rate"], float)
        assert isinstance(data["avg_response_time_ms"], float)

class TestPredictionEndpoints:
    """Test prediction endpoints."""
    
    def test_single_prediction_valid(self):
        """Test single prediction with valid data."""
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
        assert "request_id" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "risk_factors" in data
        assert "timestamp" in data
        assert "processing_time_ms" in data
        
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["prediction"] in ["High Risk", "Medium Risk", "Low Risk"]
        assert len(data["probabilities"]) == 3
        assert isinstance(data["risk_factors"], list)
        assert isinstance(data["processing_time_ms"], float)
        assert data["processing_time_ms"] < 1000  # Should be under 1 second
    
    def test_single_prediction_invalid_data(self):
        """Test single prediction with invalid data."""
        invalid_data = {
            "transaction": {
                "Amount": -1000.0,  # Negative amount
                "Value": "invalid",  # Invalid type
                "PricingStrategy": 10,  # Out of range
                "FraudResult": 2,  # Out of range
                "CountryCode": 9999,  # Out of range
                "ProviderId": "",  # Empty string
                "ProductCategory": "",  # Empty string
                "ChannelId": ""  # Empty string
            }
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0
    
    def test_batch_prediction_valid(self):
        """Test batch prediction with valid data."""
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
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "request_id" in data
        assert "predictions" in data
        assert "total_processed" in data
        assert "processing_time_ms" in data
        assert "timestamp" in data
        
        assert len(data["predictions"]) == 2
        assert data["total_processed"] == 2
        assert isinstance(data["processing_time_ms"], float)
        
        # Check each prediction has required fields
        for pred in data["predictions"]:
            assert "request_id" in pred
            assert "prediction" in pred
            assert "confidence" in pred
            assert "probabilities" in pred
            assert "risk_factors" in pred
            assert "timestamp" in pred
            assert "processing_time_ms" in pred

class TestModelEndpoints:
    """Test model information endpoints."""
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_name" in data
        assert "model_type" in data
        assert "feature_count" in data
        assert "class_names" in data
        assert "model_version" in data
        assert "training_date" in data
        assert "performance_metrics" in data
        
        assert isinstance(data["feature_count"], int)
        assert isinstance(data["class_names"], list)
        assert len(data["class_names"]) == 3
        assert "High Risk" in data["class_names"]
        assert "Medium Risk" in data["class_names"]
        assert "Low Risk" in data["class_names"]
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "error_count" in data
        assert "error_rate" in data
        assert "avg_response_time_ms" in data
        assert "last_prediction" in data
        assert "model_load_time" in data
        
        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["error_count"], int)
        assert isinstance(data["error_rate"], float)
        assert isinstance(data["avg_response_time_ms"], float)

class TestMonitoringEndpoints:
    """Test monitoring endpoints."""
    
    def test_monitoring_log(self):
        """Test monitoring log endpoint."""
        log_data = {
            "level": "info",
            "message": "Test log message",
            "component": "test",
            "user_id": "test_user"
        }
        
        response = client.post("/monitoring/log", json=log_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "logged"
    
    def test_monitoring_status(self):
        """Test monitoring status endpoint."""
        response = client.get("/monitoring/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "system_status" in data
        assert "model_status" in data
        assert "database_status" in data
        assert "last_check" in data
        assert "uptime_seconds" in data

class TestSecurityEndpoints:
    """Test security endpoints."""
    
    def test_auth_validate_valid(self):
        """Test authentication validation with valid token."""
        auth_data = {
            "token": "Bearer valid_token_123"
        }
        
        response = client.post("/auth/validate", json=auth_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "valid" in data
        assert "user_id" in data
        assert "expires_at" in data
    
    def test_auth_validate_invalid(self):
        """Test authentication validation with invalid token."""
        auth_data = {
            "token": "invalid_token"
        }
        
        response = client.post("/auth/validate", json=auth_data)
        assert response.status_code == 401
        
        data = response.json()
        assert "detail" in data
        assert "Invalid token" in data["detail"]

class TestPerformance:
    """Test performance requirements."""
    
    def test_response_time_under_200ms(self):
        """Test that response times are under 200ms."""
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
        
        start_time = time.time()
        response = client.post("/predict", json=valid_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 200, f"Response time {response_time_ms}ms exceeds 200ms limit"
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
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
        
        results = queue.Queue()
        
        def make_request():
            response = client.post("/predict", json=valid_data)
            results.put(response.status_code)
        
        # Start 10 concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all responses were successful
        success_count = 0
        while not results.empty():
            status = results.get()
            if status == 200:
                success_count += 1
        
        assert success_count >= 8, f"Only {success_count}/10 requests succeeded"

class TestErrorHandling:
    """Test error handling capabilities."""
    
    def test_missing_transaction_field(self):
        """Test error handling for missing transaction field."""
        invalid_data = {
            "amount": 1000.0,  # Wrong field name
            "value": 1200.0
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        assert "Field required" in str(data["detail"])
    
    def test_empty_request_body(self):
        """Test error handling for empty request body."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        assert "Field required" in str(data["detail"])
    
    def test_invalid_json(self):
        """Test error handling for invalid JSON."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
