import argparse

from project_files.scripts.plotting.plot_metric_curves import plot_metric_curves
from project_files.nlp.bbc_news.scripts.plotting.base import METHODS

NUM_EPOCHS = 200

WANDB_PROJECT_NAMES = {
    "standard" : "bbc_news"
}

METRICS = {
    "eval/accuracy" : "ACC",
    "eval/loss" : "-CE",
    "eval/ece" : "-ECE",
    "eval/brier_score" : "-BRIER"
    }

INVERT = {
    "eval/accuracy" : False,
    "eval/loss" : True,
    "eval/ece" : True,
    "eval/brier_score" : True
}

LINEWIDTHS = {
    "ce" : 1,
    "entropy" : 1,
    "smoothing" : 1,
    "focal" : 1,
    "penex" : 2.5
    }

ALPHAS = {
    "ce" : 0.7,
    "entropy" : 0.7,
    "smoothing" : 0.7,
    "focal" : 0.7,
    "penex" : 1
}

ZORDERS = {
    "ce" : 1,
    "entropy" : 1,
    "smoothing" : 1,
    "focal" : 1,
    "penex" : 2
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
        alphas=ALPHAS,
        zorders=ZORDERS,
        line_widths=LINEWIDTHS,
        project_name="bbc_news"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=str, choices=WANDB_PROJECT_NAMES.keys(), default="standard", help="which baselines to plot")
    args = parser.parse_args()

    main(args.baselines)
