from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)
resp = client.post('/predict', json={'features': {'feature_0': 0.1, 'feature_1':1.2, 'feature_2': -0.3}})
print('status', resp.status_code)
print(resp.json())
