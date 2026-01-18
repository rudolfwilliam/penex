from project_files.vision.pathMNIST.scripts.plotting.base import SENSITIVITY_ANALYSIS
from project_files.scripts.plotting.plot_metric_curves import plot_metric_curves


NUM_EPOCHS = 200


METRICS = {
    "eval_accuracy" : "ACC",
    "eval_loss" : "-CE",
    "eval_ece" : "-ECE",
    "eval_brier_score" : "-BRIER"
}

LINEWIDTHS = {
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_02" : 0.7,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_04" : 0.7,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_08" : 0.7,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_16" : 0.7
    }

ALPHAS = {
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_02" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_04" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_08" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_16" : 1
}

ZORDERS = {
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_02" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_04" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_08" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_16" : 1
    }

INVERT = {
    "eval_accuracy" : False,
    "eval_loss" : True,
    "eval_brier_score" : True,
    "eval_ece" : True
    }


def main():
    plot_metric_curves(
        wandb_project_name="pathMNIST_robustness",
        baselines="sensitivity analysis",
        num_epochs=NUM_EPOCHS,
        baselines_dict={
            "sensitivity analysis" : SENSITIVITY_ANALYSIS
        },
        metrics=METRICS,
        invert=INVERT,
        line_widths=LINEWIDTHS,
        alphas=ALPHAS,
        zorders=ZORDERS,
        project_name="pathMNIST_sensitivity_analysis"
    )

if __name__ == "__main__":
    main()
