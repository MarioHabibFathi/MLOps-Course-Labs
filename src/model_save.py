# -*- coding: utf-8 -*-
"""
Created on Mon May 26 22:12:31 2025

@author: mario
"""

# Once, outside the container, run this to save your current model:
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.sklearn.load_model("models:/SVM prediction/Production")
joblib.dump(model, "model.pkl")

client = mlflow.tracking.MlflowClient()
run_id = client.get_latest_versions("SVM prediction", ["Production"])[0].run_id
preproc_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="preprocessor/preprocessor_svm.pkl"
)
import shutil
shutil.copy(preproc_path, "preprocessor_svm.pkl")
