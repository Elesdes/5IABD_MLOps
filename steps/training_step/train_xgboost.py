import numpy
import pandas as pd
import xgboost as xgb
import mlflow
from zenml import step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step


@step(enable_cache=False)
def train_xgboost(
    x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[xgb.XGBClassifier, numpy.ndarray]:
    """
    The training step of the XGBoost
    :param x_train: The dataframe input for the prediction
    :param y_train: The dataframe input to achieve
    :param x_test: The dataframe input to test the accuracy of the model
    :return:
    The XGBoost model and the predicted data from the test dataframe
    """
    model_XGB = xgb.XGBClassifier()
    # mlflow.sklearn.autolog()
    model_XGB.fit(x_train, y_train)
    y_predicted = model_XGB.predict(x_test)
    # mlflow.sklearn.log_model(model_XGB)
    # mlflow_register_model_step(model=model_XGB, name="XGB-model")
    return model_XGB, y_predicted
