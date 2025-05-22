from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from prometheus_fastapi_instrumentator import Instrumentator
from src.model import load_model
from src.predict import predict
from monitoring.metrics import start_metrics_collection, INFERENCE_TIME_HISTOGRAM, REQUEST_COUNT, REQUEST_ERRORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model, preprocessor = load_model()
Instrumentator().instrument(app).expose(app)

def get_input_columns(preprocessor):
    if hasattr(preprocessor, 'get_feature_names_out'):
        return preprocessor.get_feature_names_out().tolist()
    else:
        return []

@app.on_event("startup")
def startup_event():
    start_metrics_collection()
    
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
    REQUEST_COUNT.inc()
    data = await request.json()
    logger.info(f"Predict called with: {data}")
    result = predict(model, preprocessor, data)
    return {"prediction": result}

    REQUEST_COUNT.inc()
    try:
        with INFERENCE_TIME_HISTOGRAM.time():
            data = await request.json()
            logger.info(f"Predict called with: {data}")
            result = predict(model, preprocessor, await request.json())
        return {"prediction": result}
    except Exception as e:
        REQUEST_ERRORS.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not make prediction")


@app.get("/columns")
def get_columns():
    return {"columns": get_input_columns(preprocessor)}

@app.get("/form", response_class=HTMLResponse)
async def form_page():
    columns = get_input_columns(preprocessor)
    form_fields = ""
    for col in columns:
        form_fields += f'<tr><td>{col}:</td><td><input type="text" name="{col}"></td></tr>'

    html = f"""
    <html>
    <head><title>Prediction Form</title></head>
    <body>
      <h2>Enter values for prediction</h2>
      <form id="prediction-form">
        <table>
          {form_fields}
        </table>
        <button type="submit">Predict</button>
      </form>

      <script>
        document.getElementById("prediction-form").onsubmit = async function(e) {{
          e.preventDefault();
          const formData = new FormData(e.target);
          const json = Object.fromEntries(formData.entries());

          const response = await fetch("/predict", {{
            method: "POST",
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify(json)
          }});

          const result = await response.json();
          alert("Prediction: " + result.prediction + "\\nInference time: " + result.inference_time_sec + "s");
        }};
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
