import os
import mlflow
import mlflow.sklearn

MODEL_NAME = "SVM prediction"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def load_model():
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    return model
