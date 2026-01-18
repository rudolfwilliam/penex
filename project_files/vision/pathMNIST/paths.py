import os


DIR = os.path.relpath(os.path.dirname(__file__), ".")
SCRIPTS_DIR = os.path.join(DIR, 'scripts')
PLOTTING_DIR = os.path.join(SCRIPTS_DIR, 'plotting')
CONFIG_DIR = os.path.join(DIR, 'configs')
CKPT_DIR = os.path.join(DIR, 'checkpoints')
VIS_DIR = os.path.join(DIR, 'visualizations')
LOG_DIR = os.path.join(DIR, 'logs')
