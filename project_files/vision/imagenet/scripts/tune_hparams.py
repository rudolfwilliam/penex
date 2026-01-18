"""This script is used to tune the hyperparameters of the model using the optuna library."""

import os
from functools import partial

import optuna
from optuna.samplers import TPESampler

from project_files.utils import set_seed, load_json
from project_files.vision.scripts.tune_hparams import objective
from project_files.vision.base import parse_args
from project_files.vision.base import SEARCH_SPACES
from project_files.vision.base import update_cfg_from_args
from project_files.vision.imagenet.models.classifiers import create_imagenet_classifier
from project_files.vision.imagenet.scripts.base import load_imagenet
from project_files.vision.imagenet.paths import CONFIG_DIR


WANDB_PROJECT_ROOT = "imagenet_"
N_TRIALS = 50
OVERRIDABLE_PARAMS = ['training_params.max_epochs']


if __name__ == "__main__":
    args = parse_args()
    cfg = load_json(os.path.join(CONFIG_DIR, args.cfg_name))

    set_seed(cfg["training_params"]["seed"])
    sampler = TPESampler(seed=cfg["training_params"]["seed"])

    # Load imagenet dataset
    train_dataset, _ = load_imagenet(label_noise=args.label_noise)
    # Add the models parameters to the config
    cfg["model_params"]["optimizer"] = args.optimizer
    # Update cfg with command-line arguments if provided
    update_cfg_from_args(cfg, args, OVERRIDABLE_PARAMS)
    # Note: parameters that are not set in the search space dir 
    # will be set to the default values in the config file

    # create objective function
    imagenet_objective = partial(
                            objective,
                            train_dataset=train_dataset,
                            wandb_project_name=WANDB_PROJECT_ROOT + args.name,
                            search_spaces=SEARCH_SPACES,
                            create_classifier_func=create_imagenet_classifier,
                            args=args,
                            cfg=cfg
                            )
    
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(imagenet_objective, n_trials=N_TRIALS)
