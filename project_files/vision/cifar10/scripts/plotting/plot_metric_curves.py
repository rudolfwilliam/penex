import argparse

from project_files.vision.cifar10.scripts.plotting.base import METHODS, ABLATIONS_GOOD, ABLATIONS_POOR
from project_files.scripts.plotting.plot_metric_curves import plot_metric_curves


NUM_EPOCHS = 200


WANDB_PROJECT_NAMES = {
    "standard" : "cifar10",
    "ablations_poor" : "cifar10_ablations",
    "ablations_good" : "cifar10_ablations"
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
    "train_exp-loss_sumexp-penalty_adam" : 3
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


def main(baselines, wandb_project_name):
    plot_metric_curves(
        wandb_project_name=WANDB_PROJECT_NAMES[baselines] if wandb_project_name == "cifar10" else wandb_project_name,
        baselines=baselines,
        num_epochs=NUM_EPOCHS,
        baselines_dict={
            "standard" : METHODS,
            "ablations_poor" : ABLATIONS_POOR,
            "ablations_good" : ABLATIONS_GOOD
        },
        metrics=METRICS,
        invert=INVERT,
        line_widths=LINEWIDTHS,
        alphas=ALPHAS,
        zorders=ZORDERS,
        project_name=wandb_project_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=str, choices=WANDB_PROJECT_NAMES.keys(), default="standard", help="which baselines to plot")
    parser.add_argument("--wandb_project_name", type=str, default="cifar10", help="wandb project name")
    args = parser.parse_args()

    main(args.baselines, args.wandb_project_name)
