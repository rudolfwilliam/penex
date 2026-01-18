import os
from functools import partial

from datasets import load_dataset, ClassLabel
from transformers import RobertaTokenizer
from optuna.samplers import TPESampler
import optuna
import wandb

from project_files.utils import set_seed
from project_files.nlp.bbc_news.scripts.train import train
from project_files.nlp.bbc_news.utils import (
                                            compute_metrics, 
                                            preprocess
                                            )
from project_files.nlp.bbc_news.base import get_args, log_GPU_info, SEARCH_SPACES
from project_files.nlp.bbc_news.meta_data import (
                                                NUM_TRAINING_EXAMPLES,
                                                ID2LABEL,
                                                MODEL_ID
                                                )

N_TRIALS = 50
N_EPOCHS = 200

PARAM_LIST = ["model_params", "hf_params"]



def objective(
              trial, 
              name,
              train_dataset,
              val_dataset,
              search_spaces,
              loss_func,
              project_name,
              cfg
              ):
    
    log_GPU_info()
    
    set_seed(cfg["hf_params"]["seed"])
    
    for params in PARAM_LIST:
        for key, value in search_spaces[name][params].items():
            cfg[params][key] = suggest_param(trial, key, value[0], value[1])
    run_name = "nr_" + str(trial.number) + "_" + \
                            "_".join([key + "_" + str(round(trial.params[key], ndigits=5)) for key in trial.params.keys()])
    cfg["hf_params"]["run_name"] = run_name
    cfg["hf_params"]["lr_scheduler_type"] = "constant"

    wandb.init(
        project=project_name,
        name=run_name
    )

    try:
        _, _, criterion = train(
                            cfg=cfg,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            loss_func=loss_func,
                            compute_metrics=compute_metrics,
                            checkpoint=None
                            )
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Gracefully mark the trial as failed
        return float("inf")
    
    wandb.finish() # wandb needs to be handled explicitly for multiple runs

    return criterion


def suggest_param(trial, key, value0, value1):
    if isinstance(value0, int) and isinstance(value1, int):
        return trial.suggest_int(key, value0, value1)
    elif isinstance(value0, float) and isinstance(value1, float):
        return trial.suggest_float(key, value0, value1)
    else:
        raise ValueError("Value types must be the same")

def main(loss_func, cfg):

    project_name = "bbcnews_" + loss_func
    os.environ["WANDB_PROJECT"] = project_name

    dataset = load_dataset("SetFit/bbc-news")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_ID)   
    # Apply tokenization to the dataset
    preprocess_ = partial(preprocess, tokenizer=tokenizer)
    encoded_dataset = dataset.map(preprocess_, batched=True)
    # Remove unnecessary columns
    encoded_dataset = encoded_dataset.remove_columns(["text", "label_text"])

    class_label = ClassLabel(names=list(ID2LABEL.values()))

    # Cast the 'label' column to ClassLabel. 
    casted_dataset = encoded_dataset.cast_column("label", class_label)
    split_dataset = casted_dataset["train"].train_test_split(
                                                            test_size=0.2, 
                                                            seed=cfg["hf_params"]["seed"], 
                                                            stratify_by_column="label"
                                                            )
    # Get train and validation splits
    train_dataset = split_dataset["train"].select(range(NUM_TRAINING_EXAMPLES))
    val_dataset = split_dataset["test"]

    # Initialize optimizer
    sampler = TPESampler(seed=cfg["hf_params"]["seed"])
    objective_ = partial(
                        objective,
                        name=loss_func,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        search_spaces=SEARCH_SPACES,
                        loss_func=loss_func,
                        project_name=project_name,
                        cfg=cfg 
                        )

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective_, n_trials=N_TRIALS)


if __name__ == "__main__":
    args, cfg = get_args()
    cfg["hf_params"]["num_train_epochs"] = N_EPOCHS
    main(args.loss_func, cfg)
    