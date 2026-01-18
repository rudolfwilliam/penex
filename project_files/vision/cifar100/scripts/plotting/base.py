METHODS = {
    "train_ce_dummy_adam" : "cross entropy",
    "train_ce_entropy-penalty_adam" : "entropy penalty",
    "train_ce_dummy_adam_smoothing" : "label smoothing",
    "train_ce_entropy-penalty_adam" : "confidence penalty",
    "train_focal-loss_dummy_adam" : "focal loss",
    "train_exp-loss_sumexp-penalty_adam" : "PENEX"
}

SCALINGS = {
    "train_exp-loss_sumexp-penalty_adam" : "no scaling",
    "train_ce_dummy_adam_smoothing" : "label smoothing",
    "train_exp-loss_sumexp-penalty_adam_linear-scheduling" : "linear scaling",
    "train_exp-loss_sumexp-penalty_adam_exponential-scheduling" : "exponential scaling",
}

PARAMETER_ANALYSIS = {
    "train_exp-loss_sumexp-penalty_adam_sensitivity_00" : "PENEX ($\\alpha=10^{-5}$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_02" : "PENEX ($\\alpha=0.2$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_04" : "PENEX ($\\alpha=0.4$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_08" : "PENEX ($\\alpha=0.8$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_16" : "PENEX ($\\alpha=1.6$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_32" : "PENEX ($\\alpha=3.2$)",
}
