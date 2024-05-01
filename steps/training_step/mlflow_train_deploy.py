from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_registry_deployer_step


@pipeline
def mlflow_registry_deploy_pipeline():
    deployed_model = mlflow_model_registry_deployer_step(
        registry_model_name="tensorflow-mnist-model",
        registry_model_version="1",  # Either specify a model version
        # or use the model stage if you have set it in the MLflow registry:
        # registered_model_stage="Staging"
    )
