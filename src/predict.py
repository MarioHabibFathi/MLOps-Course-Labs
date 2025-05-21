import pandas as pd
import logging

logger = logging.getLogger(__name__)

def predict(model, preprocessor, input_data: dict):
    """
    Predict churn based on input data using pre-fitted model and preprocessor.

    Args:
        model: Trained scikit-learn model
        preprocessor: Fitted ColumnTransformer
        input_data: Raw input features (dict)

    Returns:
        int: Prediction (0 or 1)
    """
    df = pd.DataFrame([input_data])
    logger.info(f"Input data: {input_data}")

    try:
        # Transform input using preprocessor
        df_transformed = preprocessor.transform(df)

        # Predict
        prediction = model.predict(df_transformed)[0]
        logger.info(f"Prediction successful: {prediction}")
        return int(prediction)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise ValueError(f"Prediction error: {str(e)}")

