import os

DIR = os.path.relpath(os.path.dirname(__file__), ".")
SCRIPTS_DIR = os.path.join(DIR, 'scripts')
CONFIG_DIR = os.path.join(DIR, 'configs')
LOG_DIR = os.path.join(DIR, 'logs')
VIS_DIR = os.path.join(DIR, 'visualizations')
CKPT_DIR = os.path.join(DIR, 'checkpoints')
