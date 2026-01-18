import argparse

from project_files.vision.pathMNIST.paths import PLOTTING_DIR
from project_files.vision.pathMNIST.scripts.plotting.base import (
                                                            RHOS,
                                                            GAMMAS,
                                                            EPSILONS,
                                                            ALPHAS,
                                                            NAME2RHO,
                                                            NAME2GAMMA,
                                                            NAME2EPSILON,
                                                            NAME2ALPHA
                                                            )
from project_files.scripts.plotting.plot_calibration_analysis import plot_calibration_analysis

WANDB_PROJECT_NAME = "pathMNIST_robustness"

NUM_EPOCHS = 100

FAILED_STATES = ["failed", "error", "crashed", "killed", "running"]

Y_RANGE = [-0.15, -0.02]
X_RANGE = [-0.02, 0.12]

Y_TICKS_ARGS = [Y_RANGE[0] + 0.01, Y_RANGE[1], 0.02]

FILE_NAME = "calibration_curves_pathMNIST"


def main(reload_data=False):

    args = {
        "plotting_dir" : PLOTTING_DIR,
        "wandb_project_name" : WANDB_PROJECT_NAME,
        "failed_states" : FAILED_STATES,
        "num_epochs" : NUM_EPOCHS,
        "y_range" : Y_RANGE,
        "x_range" : X_RANGE,
        "y_ticks_args" : Y_TICKS_ARGS,
        "rhos" : RHOS,
        "name2rho" : NAME2RHO,
        "gammas" : GAMMAS,
        "name2gamma" : NAME2GAMMA,
        "epsilons" : EPSILONS,
        "name2epsilon" : NAME2EPSILON,
        "alphas" : ALPHAS,
        "name2alpha" : NAME2ALPHA,
        "file_name" : FILE_NAME,
        "reload_data" : reload_data
    }

    plot_calibration_analysis(
        **args
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation module with given parameters")
    parser.add_argument('--reload_data', action='store_true',
                        help='reload the data from wandb instead of using what is stored in tmp')
    args = parser.parse_args()

    main(args.reload_data)
