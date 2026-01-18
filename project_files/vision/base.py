import argparse
import os
from pathlib import Path

import torch

from project_files.vision.models.classifiers import CONSTRAINT_HANDLERS


def parse_args():
    parser = argparse.ArgumentParser(description="Run the evaluation module with given parameters")
    parser.add_argument('--config', type=str, default=os.path.join("CONFIG_DIR", "eval.json"), 
                        help='Path to configuration JSON file')
    parser.add_argument('--constraint_handler', type=str, default="sumexp-penalty", 
                        help='Constraint handler to use to enforce the constraint', 
                        choices=CONSTRAINT_HANDLERS.keys())
    parser.add_argument('--loss_func', type=str, default="exp-loss", help='name of the method', choices=["exp-loss", "ce", "focal-loss"])
    parser.add_argument('--name_extension', type=str, default=None, help='name of the method')
    parser.add_argument('--cfg_extension', type=str, default=None, help='extension to append to the config')
    parser.add_argument('--optimizer', type=str, default="adam", help='name of the optimizer', choices=["adam", "lbfgs"])
    parser.add_argument('--cfg_name', type=str, default=None, help='name of the config file to choose. "Trumps" the cfg_extension argument.')
    parser.add_argument('--vis_dir', type=str, default=None, help='directory name to save visualizations')
    parser.add_argument('--log_name', type=str, default=None, help='directory name to save logs')
    parser.add_argument('--ckpt_name', type=str, default=None, help='directory name to save checkpoints')
    parser.add_argument('--log_dir', type=str, default=None, help='logging directory to use')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='checkpointing directory to use')
    parser.add_argument('--label_noise', type=float, default=None, help='Label noise to add to the dataset') 
    parser.add_argument('--ckpt_path', type=str, default=None, help='Model checkpoint path to initialize the classifier (optional)')
    parser.add_argument('--training_params.max_epochs', type=int, default=None, help='Maximum number of epochs to train the model')
    parser.add_argument('--seed', type=int, default=None, help="Random seed. Defaults to 0.")
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes in the dataset.')
    parser.add_argument('--train_set_size', type=train_set_size_type, default="full", help="Size of the training set. Must be >0 and smaller than the full training set size.")
    args = parser.parse_args()

    name_plain = "train_" + args.loss_func + "_" + args.constraint_handler + "_" + args.optimizer
    args.name = name_plain + (("_" + args.name_extension) if args.name_extension else "") # modify run name
    if args.cfg_name is None:
        args.cfg_name = name_plain + (("_" + args.cfg_extension) if args.cfg_extension else "") + ".json" # modify the config name

    return args


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', 'no', '0'}:
        return False
    elif value.lower() in {'true', 't', 'yes', '1'}:
        return True
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def train_set_size_type(value):
    if value == "full":
        return value
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("Train set size must be > 0 or 'full'")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError("Train set size must be an integer or 'full'")

def update_cfg_from_args(cfg, args, overrideable_params):
    """Update cfg with command-line arguments if provided"""
    for param in overrideable_params:
        value = getattr(args, param)
        if value is not None:
            # Split the parameter by dots to handle nested structures
            keys = param.split('.')
            nested_cfg = cfg
            for key in keys[:-1]:
                nested_cfg = nested_cfg.setdefault(key, {})
            nested_cfg[keys[-1]] = value

    # Add the model optimizer to the config
    cfg["model_params"]["optimizer"] = args.optimizer


def setup_data(args, model, log_dir, ckpt_dir, vis_dir):
    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'], strict=False) 
    
    if args.log_name is None:
        log_dir = os.path.join(log_dir, "standard")
    else:
        log_dir = os.path.join(log_dir, args.log_name)

    if args.ckpt_name is None:
        ckpt_dir = os.path.join(ckpt_dir, "standard")
    else:
        ckpt_dir = os.path.join(ckpt_dir, args.ckpt_name)
    
    if args.log_dir is not None: # this feature is used for the label noise experiment,
        log_dir = args.log_dir   # overrides the log_dir argument
    
    if args.ckpt_dir is not None: # this feature is used for the label noise experiment
        ckpt_dir = args.ckpt_dir  # overrides the ckpt_dir argument
    
    if args.vis_dir is None:
        vis_dir = os.path.join(vis_dir, "standard")
    else:
        vis_dir = args.vis_dir
    
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    return log_dir, vis_dir, ckpt_dir


# search spaces for hyperparameters of different methods
SEARCH_SPACES = {
    "train_exp-loss_sumexp-penalty_adam" : {
        "model_params": {
        #    "lr" : [1e-5, 1e-2],
            "sensitivity" : [0.0, 1.0]
        },
        "constraint_params": {}
    },
    "train_exp-loss_sumexp-penalty_adam_noise_01" : {
        "model_params": {
        #    "lr" : [1e-5, 1e-2],
            "sensitivity" : [0.0, 1.0]
        },
        "constraint_params": {}
    },
    "train_exp-loss_logsumexp-penalty_adam" : {
        "model_params": {
        #    "lr" : [1e-5, 1e-2],
            "sensitivity" : [0.0, 1.0]
        },
        "constraint_params": {}
    },
    "train_exp-loss_logsumexp-penalty_adam_noise_01" : {
        "model_params": {
        #    "lr" : [1e-5, 1e-2],
            "sensitivity" : [0.0, 1.0]
        },
        "constraint_params": {}
    },
    "train_exp-loss_augmented-lagrangian_adam" : {
        "model_params": {},
        "constraint_params": {
            "rho" : [0.0, 1e3],
            "nu" : [1.0, 1e3]
        }
    },
    "train_exp-loss_squared-penalty_adam" : {
        "model_params": {},
        "constraint_params": {
            "rho" : [0.0, 1e3]
        }
    },
    "train_ce_entropy-penalty_adam" : {
        "model_params": {
        #    "lr" : [1e-5, 1e-2],
        },
        "constraint_params": {
            "rho" : [0.0, 10.0]
        }
    },
    "train_ce_entropy-penalty_adam_noise_01" : {
        "model_params": {
        #    "lr" : [1e-5, 1e-2],
        },
        "constraint_params": {
            "rho" : [0.0, 10.0]
        }
    },
    "train_ce_dummy_adam_smoothing" : {
        "model_params": {
        #    "lr" : [1e-5, 1e-2],
            "label_smoothing" : [0.0, 1.0]
        },
        "constraint_params": {}
    },
    "train_ce_dummy_adam_smoothing_noise_01" : {
        "model_params": {
        #    "lr" : [1e-5, 1e-2],
            "label_smoothing" : [0.0, 1.0]
        },
        "constraint_params": {}
    },
    "train_focal-loss_dummy_adam" : {
        "model_params": {
            "gamma" : [0.0, 5.0]
        },
        "constraint_params": {}
    },
    "train_focal-loss_dummy_adam_noise_01" : {
        "model_params": {
            "gamma" : [0.0, 5.0]
        },
        "constraint_params": {}
    }
}
