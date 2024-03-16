from zenml import pipeline
from steps.train_randomforest import train_randomforest
from steps.train_custom_model import train_custom_model
from steps.train_xgboost import train_xgboost
from steps.load_data import load_data
from steps.plot_data import plot_data
from steps.data_preprocessing import data_preprocessing
from steps.accuracy_score_global import accuracy_score_global
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step


@pipeline(enable_cache=True)
def train_pipeline():
    batch_size = 32
    epochs = 50
    df, raw = load_data()
    plot_data(raw)
    x_train, x_test, y_train, y_test, x = data_preprocessing(df)
    model_XGB, y_xgboost_predicted = train_xgboost(x_train, y_train, x_test)
    model_randomforest, y_randomforest_predicted = train_randomforest(x_train, y_train, x_test)
    train_custom_model(x_train, y_train, x, batch_size, epochs)
    accuracy_xgb, accuracy_randomforest = accuracy_score_global(y_xgboost_predicted, y_randomforest_predicted, y_test)
    # mlflow_register_model_step(model=model_XGB, name="model_XGB", metadata=accuracy_xgb)
    # mlflow_register_model_step(model=model_randomforest, name="model_randomforest", metadata=accuracy_randomforest)
