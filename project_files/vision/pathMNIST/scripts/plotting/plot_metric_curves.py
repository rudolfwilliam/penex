import argparse

from project_files.scripts.plotting.plot_metric_curves import plot_metric_curves
from project_files.vision.pathMNIST.scripts.plotting.base import METHODS

NUM_EPOCHS = 200

WANDB_PROJECT_NAMES = {
    "standard" : "pathMNIST"
}

METRICS = {
    "eval_accuracy" : "ACC",
    "eval_loss" : "-CE",
    "eval_ece" : "-ECE",
    "eval_brier_score" : "-BRIER"
}

LINEWIDTHS = {
    "train_ce_dummy_adam" : 1,
    "train_ce_entropy-penalty_adam" : 1,
    "train_ce_dummy_adam_smoothing" : 1,
    "train_ce_entropy-penalty_adam" : 1,
    "train_focal-loss_dummy_adam" : 1,
    "train_exp-loss_sumexp-penalty_adam" : 2.5
    }

ALPHAS = {
    "train_ce_dummy_adam" : 0.7,
    "train_ce_entropy-penalty_adam" : 0.7,
    "train_ce_dummy_adam_smoothing" : 0.7,
    "train_ce_entropy-penalty_adam" : 0.7,
    "train_focal-loss_dummy_adam" : 0.7,
    "train_exp-loss_sumexp-penalty_adam" : 1
}

ZORDERS = {
    "train_ce_dummy_adam" : 1,
    "train_ce_entropy-penalty_adam" : 1,
    "train_ce_dummy_adam_smoothing" : 1,
    "train_ce_entropy-penalty_adam" : 1,
    "train_focal-loss_dummy_adam" : 1,
    "train_exp-loss_sumexp-penalty_adam" : 2
    }

INVERT = {
    "eval_accuracy" : False,
    "eval_loss" : True,
    "eval_brier_score" : True,
    "eval_ece" : True
    }

def main(baselines):
    plot_metric_curves(
        wandb_project_name=WANDB_PROJECT_NAMES[baselines],
        baselines=baselines,
        num_epochs=NUM_EPOCHS,
        baselines_dict={
            "standard" : METHODS
        },
        metrics=METRICS,
        invert=INVERT,
        zorders=ZORDERS,
        alphas=ALPHAS,
        line_widths=LINEWIDTHS,
        project_name="pathMNIST"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=str, choices=WANDB_PROJECT_NAMES.keys(), default="standard", help="which baselines to plot")
    args = parser.parse_args()

    main(args.baselines)
