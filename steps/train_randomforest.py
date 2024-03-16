import mlflow
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from zenml import step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step


@step()
def train_randomforest(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[RandomForestClassifier, numpy.ndarray]:
    """
    The training step of the Random Forest
    :param x_train: The dataframe input for the prediction
    :param y_train: The dataframe input to achieve
    :param x_test: The dataframe input to test the accuracy of the model
    :return:
    The Random Forest model and the predicted data from the test dataframe
    """
    model_randomforest = RandomForestClassifier(n_estimators=100, random_state=1)
    model_randomforest.fit(x_train, y_train)
    y_predicted = model_randomforest.predict(x_test)
    # mlflow.sklearn.log_model(model_randomforest)
    # mlflow_register_model_step(model=model_randomforest, name="tensorflow-mnist-model")
    return model_randomforest, y_predicted
