import mlflow
import mlflow.sklearn

MODEL_NAME = "SVM prediction"

mlflow.set_tracking_uri("http://localhost:5000")

def load_model():
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    return model
