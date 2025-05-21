import os
import mlflow
import mlflow.sklearn
import joblib
import tempfile

MODEL_NAME = "SVM prediction"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def load_model():
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

    client = mlflow.tracking.MlflowClient()
    model_info = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    run_id = model_info.run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        preproc_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="preprocessor/preprocessor_svm.pkl",
            dst_path=tmp_dir
        )
        preprocessor = joblib.load(preproc_path)

    return model, preprocessor

