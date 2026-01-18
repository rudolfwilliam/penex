import os

from project_files.vision.imagenet.scripts.base import load_imagenet
from project_files.vision.imagenet.models.classifiers import create_imagenet_classifier
from project_files.vision.imagenet.paths import LOG_DIR, CONFIG_DIR, VIS_DIR
from project_files.vision.scripts.train import train_vision_model


# obtain CKPT_DIR from environment variable
assert 'CKPT_DIR' in os.environ, "CKPT_DIR environment variable not set"
CKPT_DIR = os.environ['CKPT_DIR']

print("CKPT_DIR:", CKPT_DIR)

# Ensure the WANDB_API_KEY is set in the environment
if 'WANDB_API_KEY' not in os.environ:
    raise ValueError("WANDB_API_KEY environment variable not set")
print("WANDB_API_KEY:", os.environ.get('WANDB_API_KEY'))
# Set WANDB DIR
os.environ["WANDB_DIR"] = os.path.join(LOG_DIR, "wandb")

if 'WANDB_PROJECT_NAME' in os.environ: # allow for custom WANDB_PROJECT_NAME
    WANDB_PROJECT_NAME = os.environ['WANDB_PROJECT_NAME']
else:
    WANDB_PROJECT_NAME = "imagenet"
print("WANDB_PROJECT_NAME:", WANDB_PROJECT_NAME)

OVERRIDABLE_PARAMS = ['name_extension', 'training_params.max_epochs']


if __name__ == "__main__":
    train_vision_model(
        config_dir=CONFIG_DIR,
        log_dir=LOG_DIR,
        vis_dir=VIS_DIR,
        ckpt_dir=CKPT_DIR,
        wandb_project_name=WANDB_PROJECT_NAME,
        overridable_params=OVERRIDABLE_PARAMS,
        load_data_func=load_imagenet,
        create_model_func=create_imagenet_classifier,
        test_flag=True
    )
    