from zenml import pipeline
from steps.loader_step.load_data import load_data
from steps.training_step.data_preprocessing import data_preprocessing
from steps.training_step.train_custom_model import train_custom_model
from steps.training_step.train_randomforest_gcp import train_randomforest
from steps.training_step.train_xgboost import train_xgboost
from steps.training_step.accuracy_score_global import accuracy_score_global


@pipeline(enable_cache=False, name="GCP Train Pipeline")
def gcp_train_pipeline():
    """
    GCP Training Pipeline for data preparation, model training, and evaluation.

    This function implements a complete machine learning workflow, running on cloud.
    It consists of the following steps:

    1. Data Loading: Loads a dataset using the `load_data` function.
    2. Data Preprocessing: Splits the dataset into training and testing sets and returns necessary variables using `data_preprocessing`.
    3. Model Training:
        - Trains an XGBoost model with `train_xgboost`.
        - Trains a Random Forest model with `train_randomforest`.
        - Trains a custom model with a specified batch size and epochs.
    4. Model Evaluation: Evaluates the models by computing and comparing their accuracies using `accuracy_score_global`.

    The function completes the workflow from data preparation to model evaluation, running entirely on cloud.

    Returns:
        None
    """
    batch_size = 32
    epochs = 50
    df = load_data()

    x_train, x_test, y_train, y_test, x = data_preprocessing(df)

    y_xgboost_predicted = train_xgboost(x_train, y_train, x_test)
    y_randomforest_predicted = train_randomforest(x_train, y_train, x_test)
    train_custom_model(x_train, y_train, x, batch_size, epochs)

    accuracy_score_global(y_xgboost_predicted, y_randomforest_predicted, y_test)
