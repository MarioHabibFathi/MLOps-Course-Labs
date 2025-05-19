from fastapi import FastAPI, Request
from src.model import load_model
from src.predict import predict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = load_model()

@app.get("/")
def home():
    logger.info("Home endpoint hit")
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def health():
    logger.info("Health check called")
    return {"status": "OK"}

@app.post("/predict")
async def make_prediction(request: Request):
    data = await request.json()
    logger.info(f"Predict called with: {data}")
    result = predict(model, data)
    return {"prediction": result}
