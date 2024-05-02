from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_registry_deployer_step


@pipeline
def mlflow_registry_deploy_pipeline() -> None:
    """
    A pipeline function to deploy a model registered in MLflow's model registry.

    This function uses the MLflow model registry deployer step from ZenML's integrations.
    It retrieves a model specified by its name and version from MLflow's model registry
    and deploys it for usage in downstream tasks.

    The deployment targets the model identified by:
        - registry_model_name: The name of the model as registered in MLflow.
        - registry_model_version: The version of the model to deploy.

    The deployed model can then be used for inference or other operations.

    Returns:
        deployed_model: The deployed model object.
    """
    deployed_model = mlflow_model_registry_deployer_step(
        registry_model_name="tensorflow-mnist-model", registry_model_version="1"
    )
