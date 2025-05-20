"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow
import mlflow.sklearn
import time

### Import MLflow

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train, model_type="logistic"):
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "svm":
        model = SVC(probability=True)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model




def main(model_type="logistic"):
    start_time = time.time()

    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    ### Set the experiment name
    mlflow.set_experiment("bank-customer-churn")

    ### Start a new run and leave all the main function code as part of the experiment

    df = pd.read_csv("data/Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    ### Log the max_iter parameter

    model = train(X_train, y_train, model_type)

    end_time = time.time()
    training_time = end_time - start_time
    
    
    if model_type == "logistic":
        mlflow.log_param("max_iter", 1000)
    elif model_type == "decision_tree":
        mlflow.log_param("max_depth", model.get_depth())
    elif model_type == "svm":
        mlflow.log_param("kernel", model.kernel)
    elif model_type == "random_forest":
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
    elif model_type == "knn":
        mlflow.log_param("n_neighbors", model.n_neighbors)


    mlflow.log_metric("training_time_sec", training_time)

    
    y_pred = model.predict(X_test)

    ### Log metrics after calculating them
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_param("model_name", model_type)
    mlflow.set_tag("model_name", model_type)


    ### Log tag
    mlflow.set_tag("developer", f"Model {model_type}")


    mlflow.sklearn.log_model(model, "model", input_example=X_train.head())

    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    
    # Log the image as an artifact in MLflow
    fig_path = f"outputs/conf_matrix {model_type}.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)

    plt.show()


if __name__ == "__main__":
    main(model_type='knn')
