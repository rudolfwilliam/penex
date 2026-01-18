METHODS = {
    "train_ce_dummy_adam" : "cross entropy",
    "train_ce_entropy-penalty_adam" : "entropy penalty",
    "train_ce_dummy_adam_smoothing" : "label smoothing",
    "train_ce_entropy-penalty_adam" : "confidence penalty",
    "train_focal-loss_dummy_adam" : "focal loss",
    "train_exp-loss_sumexp-penalty_adam" : "PENEX"
}

SENSITIVITY_COMPARISONS = {
    "train_exp-loss_sumexp-penalty_adam_sensitivity_01" : "PENEX ($\\alpha=0.1$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_05" : "PENEX ($\\alpha=0.5$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_09" : "PENEX ($\\alpha=0.9$)",
    "train_ce_dummy_adam_smoothing_pen_01" : "smoothing ($\\alpha=0.1$)",
    "train_ce_dummy_adam_smoothing_pen_05" : "smoothing ($\\alpha=0.5$)",
    "train_ce_dummy_adam_smoothing_pen_09" : "smoothing ($\\alpha=0.9$)"
}

PARAMETER_ANALYSIS = {
    "train_exp-loss_sumexp-penalty_adam_sensitivity_00" : "PENEX ($\\alpha=10^{-5}$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_02" : "PENEX ($\\alpha=0.2$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_04" : "PENEX ($\\alpha=0.4$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_08" : "PENEX ($\\alpha=0.8$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_16" : "PENEX ($\\alpha=1.6$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_32" : "PENEX ($\\alpha=3.2$)",
}

ABLATIONS_POOR = {
    "train_exp-loss_hard_adam" : "CONEX w. hard constraint",
    "train_exp-loss_dummy_adam" : "EX",
    "train_exp-loss_sumexp-penalty_adam" : "PENEX" # doesn't perform poorly, but for comparison
}

ABLATIONS_GOOD = { # better, but still not as good as PENEX hehe
    "train_exp-loss_augmented-lagrangian_adam" : "CONEX augmented Lagrangian",
    "train_exp-loss_squared-penalty_adam" : "CONEX squared penalty",  
    "train_exp-loss_sumexp-penalty_adam" : "PENEX"  
}

ALPHAS = [0.00001, 0.1, 0.2, 0.4, 0.8, 1.6]
NAME2ALPHA = {
     "00" : "0.00001",
     "01" : "0.100",
     "02" : "0.200",
     "04" : "0.400",
     "08" : "0.800",
     "16" : "1.600",
     "32" : "3.200",
}

EPSILONS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
NAME2EPSILON = {
    "00" : "0.00",
    "05" : "0.05",
    "10" : "0.10",
    "15" : "0.15",
    "20" : "0.20",
    "25" : "0.25",
}

RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NAME2RHO = {
    "00" : "0.0",
    "01" : "0.1",
    "02" : "0.2",
    "03" : "0.3",
    "04" : "0.4",
    "05" : "0.5",
}

GAMMAS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
NAME2GAMMA = {
    "00" : "0.0",
    "01" : "1.0",
    "02" : "2.0",
    "03" : "3.0",
    "04" : "4.0",
    "05" : "5.0",
}
