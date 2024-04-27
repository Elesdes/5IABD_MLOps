import mlflow
from mlflow import MlflowClient
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from zenml import step
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


def fetch_logged_data(run_id: str) -> tuple[dict, dict, dict, list]:
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


@step(experiment_tracker=experiment_tracker.name)
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
    mlflow.sklearn.autolog()
    run = mlflow.active_run()
    model_randomforest.fit(x_train, y_train)
    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    y_predicted = model_randomforest.predict(x_test)
    mlflow.log_param(key="Key", value=params)
    mlflow.log_metric(key="Training score", value=metrics["training_accuracy_score"])
    return model_randomforest, y_predicted
