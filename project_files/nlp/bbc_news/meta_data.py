NUM_TRAINING_EXAMPLES = 200
MODEL_ID = "roberta-base"

ID2LABEL = {
        0 : "tech",
        1 : "business",
        2 : "sport",
        3 : "entertainment",
        4 : "politics"
    }

LABEL2ID = {label: id for id, label in ID2LABEL.items()}
