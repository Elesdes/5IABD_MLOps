import numpy
import pandas as pd
import xgboost as xgb
from zenml import step


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
    model_XGB.fit(x_train, y_train)
    y_predicted = model_XGB.predict(x_test)
    return model_XGB, y_predicted
