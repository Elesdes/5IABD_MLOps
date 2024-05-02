import mlflow
from mlflow import MlflowClient
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


def fetch_logged_data(run_id: str) -> tuple[dict, dict, dict, list]:
    """
    Fetches logged information from an MLflow run.

    This function uses the MLflowClient to retrieve various data associated with a specific run
    identified by its run ID. The data retrieved includes:

    - Parameters: A dictionary of key-value pairs representing parameters used during the model training.
    - Metrics: A dictionary of key-value pairs representing various metrics recorded during the run.
    - Tags: A dictionary of additional information tagged to the run, excluding those starting with "mlflow.".
    - Artifacts: A list of paths pointing to artifacts stored in the run, particularly focusing on the model artifacts.

    Args:
        run_id (str): The ID of the run to retrieve data from.

    Returns:
        tuple[dict, dict, dict, list]: A tuple containing dictionaries for parameters, metrics, tags,
        and a list of artifact paths.
    """
    client = MlflowClient()

    data = client.get_run(run_id).data

    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}

    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]

    return data.params, data.metrics, tags, artifacts


@step(experiment_tracker=experiment_tracker.name)
def train_randomforest(
    x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame
) -> numpy.ndarray:
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
    params, _, _, _ = fetch_logged_data(run.info.run_id)

    model_randomforest.fit(x_train, y_train)

    y_predicted = model_randomforest.predict(x_test)

    mlflow.log_param(key="Key", value=params)

    return y_predicted
