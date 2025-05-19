import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Churn Prediction API"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_predict():
    sample_input = {
        "CreditScore": 650,
        "Age": 35,
        "Tenure": 2,
        "Balance": 80000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 45000,
        "Geography_France": 1,
        "Geography_Germany": 0,
        "Gender_Male": 1
    }
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
