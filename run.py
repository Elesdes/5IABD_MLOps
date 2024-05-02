from pipelines.training_pipeline.train_pipeline import train_pipeline
# from pipelines.training_pipeline.train_pipeline_gcp import train_pipeline

# Il faut dans ZenML : Orchestrator | Artifact Store | model deployer | model registry | Experiment tracker
# Bonus : Data Validators | Alerters | Step Operators | Annotators
# Grand Bonus : Test unitaire et de performances

# ZenML s'attend à avoir deux dossiers. L'un avec steps l'autre avec pipeline. Chaque steps a son fichier.
# Un __init__ est attendu dans chaque dossier.

# Regarder du côté de Manage Artifact, on peut passer des metadonnées tel que la précision

# Compte par défaut de ZenML
# user : default
# mdp :


if __name__ == "__main__":
    train_pipeline()
