from project_files.vision.pathMNIST.scripts.plotting.base import SENSITIVITY_ANALYSIS
from project_files.scripts.plotting.plot_metric_curves import plot_metric_curves


NUM_EPOCHS = 200


METRICS = {
    "grad_norm_epoch" : "Gradient Norm",
    "eval_accuracy" : "ACC",
}

LINEWIDTHS = {
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_02" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_04" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_08" : 1,
    "train_exp-loss_logsumexp-penalty_adam_sensitivity_16" : 1
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
    "grad_norm_epoch" : False,
    "eval_accuracy" : False,
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
        project_name="pathMNIST_sensitivity_analysis_grad_norm"
    )

if __name__ == "__main__":
    main()
