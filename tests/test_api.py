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
      "Geography": "France",
      "Gender": "Female",
      "Age": 30,
      "Tenure": 5,
      "Balance": 100000.0,
      "NumOfProducts": 2,
      "HasCrCard": 1,
      "IsActiveMember": 5,
      "EstimatedSalary": 50000.0
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()

