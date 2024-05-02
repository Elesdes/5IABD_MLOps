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
