import os

from project_files.vision.pathMNIST.scripts.base import load_pathMNIST
from project_files.vision.pathMNIST.models.classifiers import create_pathMNIST_classifier
from project_files.vision.pathMNIST.paths import LOG_DIR, CONFIG_DIR, VIS_DIR, CKPT_DIR
from project_files.vision.scripts.train import train_vision_model


# Ensure the WANDB_API_KEY is set in the environment
if 'WANDB_API_KEY' not in os.environ:
    raise ValueError("WANDB_API_KEY environment variable not set")
print("WANDB_API_KEY:", os.environ.get('WANDB_API_KEY'))
if 'WANDB_PROJECT_NAME' in os.environ: # allow for custom WANDB_PROJECT_NAME
    WANDB_PROJECT_NAME = os.environ['WANDB_PROJECT_NAME']
else:
    WANDB_PROJECT_NAME = "pathMNIST"
print("WANDB_PROJECT_NAME:", os.environ.get('WANDB_PROJECT_NAME'))
# Set WANDB DIR
os.environ["WANDB_DIR"] = os.path.join(LOG_DIR, "wandb")

OVERRIDABLE_PARAMS = ['name_extension', 'training_params.max_epochs']


if __name__ == "__main__":
    train_vision_model(
        config_dir=CONFIG_DIR,
        log_dir=LOG_DIR,
        vis_dir=VIS_DIR,
        ckpt_dir=CKPT_DIR,
        wandb_project_name=WANDB_PROJECT_NAME,
        overridable_params=OVERRIDABLE_PARAMS,
        load_data_func=load_pathMNIST,
        create_model_func=create_pathMNIST_classifier,
        test_flag=True
    )
    