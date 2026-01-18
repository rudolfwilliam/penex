import argparse

from project_files.scripts.plotting.plot_metric_curves import plot_metric_curves
from project_files.vision.cifar100.scripts.plotting.base import METHODS, SCALINGS


NUM_EPOCHS = 200

WANDB_PROJECT_NAMES = {
    "standard" : "cifar100",
    "standard_long" : "cifar100_long"
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
    "train_ce_dummy_adam" : 1,
    "train_ce_entropy-penalty_adam" : 1,
    "train_ce_dummy_adam_smoothing" : 1,
    "train_ce_entropy-penalty_adam" : 1,
    "train_focal-loss_dummy_adam" : 1,
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
        num_epochs=NUM_EPOCHS if baselines != "standard_long" else 800,
        baselines_dict={
            "standard" : METHODS,
            "standard_long" : METHODS,
            "scaling" : SCALINGS
        },
        metrics=METRICS,
        invert=INVERT,
        line_widths=LINEWIDTHS,
        alphas=ALPHAS,
        zorders=ZORDERS,
        project_name="cifar100"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=str, choices=["standard", "standard_long", "scaling"], default="standard", help="which baselines to plot")
    args = parser.parse_args()

    main(args.baselines)
