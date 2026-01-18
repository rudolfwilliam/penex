import os
import json
import logging
import argparse

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from project_files.utils import load_json, set_seed
from project_files.vision.scripts.utils import bootstrap_evaluate

from project_files.vision.cifar10.models.classifiers import create_cifar10_classifier
from project_files.vision.cifar10.scripts.base import load_cifar10
from project_files.vision.cifar100.models.classifiers import create_cifar100_classifier
from project_files.vision.cifar100.scripts.base import load_cifar100
from project_files.vision.pathMNIST.models.classifiers import create_pathMNIST_classifier
from project_files.vision.pathMNIST.scripts.base import load_pathMNIST

logger = logging.getLogger(__name__)

CHECKPOINT_DIRS = {
    #"CIFAR10" : "project_files/vision/cifar10/checkpoints",
    #"Noisy CIFAR10" : "project_files/vision/cifar10_noise_01/checkpoints",
    "PathMNIST" : "project_files/vision/pathMNIST/checkpoints",
    #"CIFAR100" : "project_files/vision/cifar100/checkpoints"
}

RESULT_DIRS = {
    "CIFAR10" : "project_files/vision/cifar10/logs",
    "Noisy CIFAR10" : "project_files/vision/cifar10_noise_01/logs",
    "CIFAR100" : "project_files/vision/cifar100/logs",
    "PathMNIST" : "project_files/vision/pathMNIST/logs",
}

CONFIG_DIRS = {
    "CIFAR10" : "project_files/vision/cifar10/configs",
    "Noisy CIFAR10" : "project_files/vision/cifar10/configs",
    "CIFAR100" : "project_files/vision/cifar100/configs",
    "PathMNIST" : "project_files/vision/pathMNIST/configs",
}

CONFIG_NAMES = {
    "ce" : "train_ce_dummy_adam.json",
    "smoothing" : "train_ce_dummy_adam_smoothing.json",
    "entropy" : "train_ce_entropy-penalty_adam.json",
    "penex" : "train_exp-loss_logsumexp-penalty_adam.json",
    "focal" : "train_focal-loss_dummy_adam.json",
}

METRIC = "eval_loss"

MODEL_ARGS = {
    "ce" : {
        "constraint_handler" : "dummy",
        "loss_func" : "ce"
    },
    "smoothing" : {
        "constraint_handler" : "dummy",
        "loss_func" : "ce"
    },
    "entropy" : {
        "constraint_handler" : "entropy-penalty",
        "loss_func" : "ce"
    },
    "penex" : {
        "constraint_handler" : "logsumexp-penalty",
        "loss_func" : "exp-loss"
    },
    "focal" : {
        "constraint_handler" : "dummy",
        "loss_func" : "ce"
    },
}

CREATE_MODEL_FUNCS = {
    "CIFAR10" : create_cifar10_classifier,
    "Noisy CIFAR10" : create_cifar10_classifier,
    "CIFAR100" : create_cifar100_classifier,
    "PathMNIST" : create_pathMNIST_classifier,
}

LOAD_DATA_FUNCS = {
    "CIFAR10" : load_cifar10,
    "Noisy CIFAR10" : load_cifar10,
    "CIFAR100" : load_cifar100,
    "PathMNIST" : load_pathMNIST,
}


def main(seed):

    set_seed(seed)

    for dataset_name, checkpoint_dir in CHECKPOINT_DIRS.items():
        for dir in os.listdir(checkpoint_dir):
            if dir not in MODEL_ARGS.keys():
                continue
            cfg = load_json(os.path.join(CONFIG_DIRS[dataset_name], CONFIG_NAMES[dir]))
            model = CREATE_MODEL_FUNCS[dataset_name](
                constraint_handler=MODEL_ARGS[dir]["constraint_handler"], 
                loss_func=MODEL_ARGS[dir]["loss_func"], 
                model_params=cfg["model_params"], 
                constraint_params=cfg["constraint_params"]
            )
            # load best checkpoint
            model.load_state_dict(
                torch.load(os.path.join(CHECKPOINT_DIRS[dataset_name], dir, "best-checkpoint.ckpt"))["state_dict"]
                )
            # test it
            _, dataset_test = LOAD_DATA_FUNCS[dataset_name]()
            data_loader = DataLoader(
                            dataset_test, 
                            batch_size=cfg["training_params"]["batch_size"]
                            )
            trainer = Trainer()
            metrics = trainer.test(model=model, dataloaders=data_loader)[0]
            bootstrapped_metrics = bootstrap_evaluate(
                                        dataset=dataset_test, 
                                        trainer=trainer,
                                        model=model,
                                        batch_size=cfg["training_params"]["batch_size"]
                                    ) # evaluate with bootstrapping
            log_dir = os.path.join(RESULT_DIRS[dataset_name], dir)
            with open(os.path.join(log_dir, "test_results_val_opt.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            with open(os.path.join(log_dir, "bootstrap_results_val_opt.json"), "w") as f:
                json.dump(bootstrapped_metrics, f, indent=4)
        logger.info("Finished dataset " + dataset_name)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run with given parameters")
    parser.add_argument('--seed', type=int, default=0, help="Random seed. Defaults to 0.")
    args = parser.parse_args()

    main(args.seed)
