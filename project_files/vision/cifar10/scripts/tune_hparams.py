"""This script is used to tune the hyperparameters of the model using the optuna library."""

import os

assert 'STUDY_NAME' in os.environ, "STUDY_NAME environment variable not set"
STUDY_NAME = os.environ['STUDY_NAME']
print("STUDY_NAME:", STUDY_NAME)
assert 'STORAGE_URL' in os.environ, "STORAGE_URL environment variable not set"
STORAGE_URL = os.environ['STORAGE_URL']
print("STORAGE_URL:", STORAGE_URL)

from project_files.vision.scripts.tune_hparams import tune_vision_model
from project_files.vision.cifar10.models.classifiers import create_cifar10_classifier
from project_files.vision.cifar10.scripts.base import load_cifar10
from project_files.vision.cifar10.paths import CONFIG_DIR


WANDB_PROJECT_ROOT = "cifar10_"
N_TRIALS = 50
OVERRIDABLE_PARAMS = ['training_params.max_epochs']


def main():
    tune_vision_model(
        config_dir=CONFIG_DIR,
        create_classifier_func=create_cifar10_classifier,
        load_data_func=load_cifar10,
        wandb_project_root=WANDB_PROJECT_ROOT,
        study_name=STUDY_NAME,
        storage_url=STORAGE_URL,
        n_trials=N_TRIALS,
        overridable_params=OVERRIDABLE_PARAMS
    )

if __name__ == "__main__":
    main()
    