import zenml
from zenml import pipeline
# from zenml.config import DockerSettings
from zenml.integrations.discord.steps.discord_alerter_ask_step import discord_alerter_ask_step
from steps.loader_step.load_data import load_data
from steps.training_step.data_preprocessing import data_preprocessing
from steps.training_step.plot_data import plot_data
from steps.training_step.train_custom_model import train_custom_model
from steps.training_step.train_randomforest import train_randomforest
from steps.training_step.train_xgboost import train_xgboost
from steps.training_step.accuracy_score_global import accuracy_score_global

#docker_settings = DockerSettings(
#    dockerfile="deepchecks-zenml.Dockerfile",
#    build_options={
#        "buildargs": {
#            "ZENML_VERSION": f"{zenml.__version__}"
#        },
#    },
#)


@pipeline(enable_cache=False, name="Train_pipeline")#, settings={"docker": docker_settings})
def train_pipeline():
    batch_size = 32
    epochs = 50
    df, raw = load_data()
    plot_data(raw)
    x_train, x_test, y_train, y_test, x = data_preprocessing(df)
    model_XGB, y_xgboost_predicted = train_xgboost(x_train, y_train, x_test)
    model_randomforest, y_randomforest_predicted = train_randomforest(x_train, y_train, x_test)
    train_custom_model(x_train, y_train, x, batch_size, epochs)
    accuracy_xgb, accuracy_randomforest, message = accuracy_score_global(y_xgboost_predicted, y_randomforest_predicted, y_test)
    approved = discord_alerter_ask_step(message)
    # mlflow_register_model_step(model=model_XGB, name="model_XGB", metadata=accuracy_xgb)
    # mlflow_register_model_step(model=model_randomforest, name="model_randomforest", metadata=accuracy_randomforest)
