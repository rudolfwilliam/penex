"""This script is used to tune the hyperparameters of the model using the optuna library."""

import os

assert 'STUDY_NAME' in os.environ, "STUDY_NAME environment variable not set"
STUDY_NAME = os.environ['STUDY_NAME']
print("STUDY_NAME:", STUDY_NAME)
assert 'STORAGE_URL' in os.environ, "STORAGE_URL environment variable not set"
STORAGE_URL = os.environ['STORAGE_URL']
print("STORAGE_URL:", STORAGE_URL)

from project_files.vision.scripts.tune_hparams import tune_vision_model
from project_files.vision.pathMNIST.models.classifiers import create_pathMNIST_classifier
from project_files.vision.pathMNIST.scripts.base import load_pathMNIST
from project_files.vision.pathMNIST.paths import CONFIG_DIR


WANDB_PROJECT_ROOT = "pathMNIST_"
N_TRIALS = 50
OVERRIDABLE_PARAMS = ['training_params.max_epochs']


def main():
    tune_vision_model(
        config_dir=CONFIG_DIR,
        create_classifier_func=create_pathMNIST_classifier,
        load_data_func=load_pathMNIST,
        wandb_project_root=WANDB_PROJECT_ROOT,
        study_name=STUDY_NAME,
        storage_url=STORAGE_URL,
        n_trials=N_TRIALS,
        overridable_params=OVERRIDABLE_PARAMS
    )

if __name__ == "__main__":
    main()
