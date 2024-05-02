import subprocess

command = "zenml stack set "
subprocess.run(command + "DiscordStack", shell=True)

from typing import Literal
from pipelines.training_pipeline.local_train_pipeline import local_train_pipeline
from pipelines.training_pipeline.gcp_train_pipeline import gcp_train_pipeline
import click


@click.command()
@click.option("--pipeline", prompt="Choose your pipeline", required=True)
def main(pipeline: Literal["local", "gcp"]):
    """
    Selects and runs a specific training pipeline based on user input.

    This function provides an interface for choosing and executing one of two training pipelines:
    a local pipeline or a GCP-based pipeline. The choice of pipeline is made interactively via a
    command-line prompt.

    - "local": Uses the "DiscordStack" for local training, and triggers the `local_train_pipeline`.
    - "gcp": Uses the "MLOpsIcarus" stack for training on Google Cloud Platform (GCP),
      and triggers the `gcp_train_pipeline`.

    Args:
        pipeline (Literal["local", "gcp"]): Specifies the pipeline to be executed:
            - "local" for local training
            - "gcp" for GCP-based training

    The function then sets the corresponding stack for the chosen pipeline, executes the appropriate
    pipeline function, and initiates training.
    """
    command = "zenml stack set "
    match pipeline:
        case "local":
            command += "DiscordStack"
            subprocess.run(command, shell=True)
            local_train_pipeline()
        case "gcp":
            command += "MLOpsIcarus"
            subprocess.run(command, shell=True)
            gcp_train_pipeline()


if __name__ == "__main__":
    main()
