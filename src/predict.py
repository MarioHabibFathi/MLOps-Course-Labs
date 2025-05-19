import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
import numpy as np

logger = logging.getLogger(__name__)

def predict(model, input_data: dict):
    """
    Predict with proper dummy column handling that matches training.
    
    Args:
        model: Trained scikit-learn model
        input_data: Raw input features
        
    Returns:
        int: Prediction (0 or 1)
    """
    # 1. Convert to DataFrame
    df = pd.DataFrame([input_data])
    logger.info(f"Input data: {input_data}")

    # 2. Define features EXACTLY as in training
    num_cols = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    cat_cols = ["Geography", "Gender"]
    
    # 3. Manually create expected dummy columns (from training)
    expected_categories = {
        'Geography': ['France', 'Germany', 'Spain'],  # Replace with your actual categories
        'Gender': ['Female', 'Male']  # Replace with your actual categories
    }
    
    # 4. Create dummy columns manually to ensure consistency
    for col in cat_cols:
        for category in expected_categories[col]:
            dummy_col = f"{col}_{category}"
            df[dummy_col] = (df[col] == category).astype(int)
    
    # Keep only first n-1 categories (to match drop='first')
    df.drop(columns=[f"Geography_{expected_categories['Geography'][0]}", 
                    f"Gender_{expected_categories['Gender'][0]}"], 
            inplace=True)
    df.drop(columns=cat_cols, inplace=True)  # Remove original categorical columns
    
    # 5. Scale numerical features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # 6. Reorder columns to match training (critical!)
    # Replace this with your actual feature order from training
    expected_features = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_Germany', 'Geography_Spain', 'Gender_Male'
    ]
    df = df[expected_features]
    
    try:
        # 7. Predict
        prediction = model.predict(df.values)[0]
        logger.info(f"Prediction successful: {prediction}")
        return int(prediction)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise ValueError(f"Prediction error: {str(e)}")

# import pandas as pd
# import logging
# from sklearn.utils import resample
# from sklearn.model_selection import train_test_split
# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# logger = logging.getLogger(__name__)

# def rebalance(data: pd.DataFrame) -> pd.DataFrame:
#     churn_0 = data[data["Exited"] == 0]
#     churn_1 = data[data["Exited"] == 1]
#     if len(churn_0) > len(churn_1):
#         maj, min_ = churn_0, churn_1
#     else:
#         maj, min_ = churn_1, churn_0
#     down = resample(maj, n_samples=len(min_), replace=False, random_state=1234)
#     return pd.concat([down, min_])

# def preprocess(df: pd.DataFrame):
#     """
#     Recreate exactly your training transformer on raw df,
#     returning only the fitted ColumnTransformer.
#     """
#     filter_feat = [
#         "CreditScore","Geography","Gender","Age","Tenure",
#         "Balance","NumOfProducts","HasCrCard","IsActiveMember",
#         "EstimatedSalary","Exited"
#     ]
#     cat_cols = ["Geography", "Gender"]
#     num_cols = [
#         "CreditScore","Age","Tenure","Balance",
#         "NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"
#     ]
#     data = df.loc[:, filter_feat]
#     data_bal = rebalance(data)
#     X = data_bal.drop("Exited", axis=1)

#     col_transf = make_column_transformer(
#         (StandardScaler(), num_cols),
#         (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
#         remainder="passthrough",
#     )
#     # **Fit only** on ALL available data
#     col_transf.fit(X)
#     return col_transf

# def predict(model, input_data: dict):
#     """
#     1) Wrap input_data into a DataFrame.
#     2) Add dummy 'Exited' so preprocess() can run.
#     3) Extract the transformer, apply it to raw df.
#     4) Call model.predict on the transformed array.
#     """
#     # 1) raw
#     df = pd.DataFrame([input_data])
#     logger.info(f"Received input for prediction: {input_data}")

#     # 2) dummy
#     df_dummy = df.copy()
#     df_dummy["Exited"] = 0

#     # 3) transform
#     transformer = preprocess(df_dummy)
#     X_trans = transformer.transform(df)

#     # 4) predict
#     pred = model.predict(X_trans)[0]
#     logger.info(f"Prediction result: {pred}")
#     return int(pred)


# import pandas as pd
# import logging

# logger = logging.getLogger(__name__)

# def predict(model, input_data: dict):
#     df = pd.DataFrame([input_data])
#     logger.info(f"Received input for prediction: {input_data}")
#     prediction = model.predict(df)[0]
#     logger.info(f"Prediction result: {prediction}")
#     return int(prediction)


# # def preprocess(df):
# #     """
# #     Preprocess and split data into training and test sets.

# #     Args:
# #         df (pd.DataFrame): DataFrame with features and target variables

# #     Returns:
# #         ColumnTransformer: ColumnTransformer with scalers and encoders
# #         pd.DataFrame: training set with transformed features
# #         pd.DataFrame: test set with transformed features
# #         pd.Series: training set target
# #         pd.Series: test set target
# #     """
# #     filter_feat = [
# #         "CreditScore",
# #         "Geography",
# #         "Gender",
# #         "Age",
# #         "Tenure",
# #         "Balance",
# #         "NumOfProducts",
# #         "HasCrCard",
# #         "IsActiveMember",
# #         "EstimatedSalary",
# #         "Exited",
# #     ]
# #     cat_cols = ["Geography", "Gender"]
# #     num_cols = [
# #         "CreditScore",
# #         "Age",
# #         "Tenure",
# #         "Balance",
# #         "NumOfProducts",
# #         "HasCrCard",
# #         "IsActiveMember",
# #         "EstimatedSalary",
# #     ]
# #     data = df.loc[:, filter_feat]
# #     data_bal = rebalance(data=data)
# #     X = data_bal.drop("Exited", axis=1)
# #     y = data_bal["Exited"]

# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.3, random_state=1912
# #     )
# #     col_transf = make_column_transformer(
# #         (StandardScaler(), num_cols), 
# #         (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
# #         remainder="passthrough",
# #     )

# #     X_train = col_transf.fit_transform(X_train)
# #     X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

# #     X_test = col_transf.transform(X_test)
# #     X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

# #     # Log the transformer as an artifact

# #     return col_transf, X_train, X_test, y_train, y_test
