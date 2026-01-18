import os

from project_files.vision.cifar100.scripts.base import load_cifar100
from project_files.vision.cifar100.paths import LOG_DIR, CONFIG_DIR, VIS_DIR, CKPT_DIR
from project_files.vision.cifar100.models.classifiers import create_cifar100_classifier
from project_files.vision.scripts.train import train_vision_model

# Common setup
if 'WANDB_API_KEY' not in os.environ:
    raise ValueError("WANDB_API_KEY environment variable not set")

os.environ["WANDB_DIR"] = os.path.join(LOG_DIR, "wandb")

if 'WANDB_PROJECT_NAME' in os.environ:
    WANDB_PROJECT_NAME = os.environ['WANDB_PROJECT_NAME']
else:
    WANDB_PROJECT_NAME = "cifar100"

OVERRIDABLE_PARAMS = ['name_extension', 'training_params.max_epochs']

def run_cifar100_training(create_model_func, **kwargs):
    """Shared training function for CIFAR-100."""
    print("WANDB_API_KEY:", os.environ.get('WANDB_API_KEY'))
    print("WANDB_PROJECT_NAME:", WANDB_PROJECT_NAME)
    
    train_vision_model(
        config_dir=CONFIG_DIR,
        log_dir=LOG_DIR,
        vis_dir=VIS_DIR,
        ckpt_dir=CKPT_DIR,
        wandb_project_name=WANDB_PROJECT_NAME,
        overridable_params=OVERRIDABLE_PARAMS,
        load_data_func=load_cifar100,
        create_model_func=create_model_func,
        test_flag=True
    )

if __name__ == "__main__":
    run_cifar100_training(create_cifar100_classifier)
