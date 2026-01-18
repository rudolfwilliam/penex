METHODS = {
    "train_ce_dummy_adam" : "cross entropy",
    "train_ce_entropy-penalty_adam" : "entropy penalty",
    "train_ce_dummy_adam_smoothing" : "label smoothing",
    "train_ce_entropy-penalty_adam" : "confidence penalty",
    "train_focal-loss_dummy_adam" : "focal loss",
    "train_exp-loss_sumexp-penalty_adam" : "PENEX"
}

ABLATIONS_GOOD = { # better, but still not as good as PENEX hehe
    "train_exp-loss_augmented-lagrangian_adam" : "CONEX augmented Lagrangian",
    "train_exp-loss_squared-penalty_adam" : "CONEX squared penalty",  
    "train_exp-loss_sumexp-penalty_adam" : "PENEX"  
}

SENSITIVITY_ANALYSIS = {
    "train_exp-loss_sumexp-penalty_adam_sensitivity_02" : "PENEX ($\\alpha=0.2$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_04" : "PENEX ($\\alpha=0.4$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_08" : "PENEX ($\\alpha=0.8$)",
    "train_exp-loss_sumexp-penalty_adam_sensitivity_16" : "PENEX ($\\alpha=1.6$)",
}

ALPHAS = [0.001, 0.1, 0.2, 0.4, 0.8, 1.6]
NAME2ALPHA = {
     "00" : "0.001",
     "01" : "0.100",
     "02" : "0.200",
     "04" : "0.400",
     "08" : "0.800",
     "16" : "1.600",
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
