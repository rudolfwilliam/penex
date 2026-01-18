import os
import json
import logging
import argparse

from pytorch_lightning import Trainer
from transformers import (
            RobertaTokenizer, 
            RobertaForSequenceClassification, 
            Trainer, 
            TrainingArguments, 
            HfArgumentParser,
            DataCollatorWithPadding
            )

from project_files.utils import load_json, set_seed
from project_files.nlp.models import PENEXRobertaForSequenceClassification
from project_files.nlp.trainers import (
                                PENEXClassificationTrainer,
                                FocalClassificationTrainer,
                                EntropyClassificationTrainer
                                )

from project_files.nlp.bbc_news.meta_data import MODEL_ID, ID2LABEL
from project_files.nlp.bbc_news.utils import compute_metrics, bootstrap_evaluate
from project_files.nlp.bbc_news.base import load_data

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "project_files/nlp/bbc_news/checkpoints"
RESULT_DIR = "project_files/nlp/bbc_news/logs"
CONFIG_DIR = "project_files/nlp/bbc_news/configs"

CONFIG_NAMES = {
    "ce" : "train_ce_dummy_adam.json",
    "smoothing" : "train_ce_dummy_adam_smoothing.json",
    "entropy" : "train_ce_entropy-penalty_adam.json",
    "penex" : "train_exp-loss_logsumexp-penalty_adam.json",
    "focal" : "train_focal-loss_dummy_adam.json",
}

METRIC = "eval_loss"

MODEL_ARGS = {
    "ce" : {
        "constraint_handler" : "dummy",
        "loss_func" : "ce"
    },
    "smoothing" : {
        "constraint_handler" : "dummy",
        "loss_func" : "ce"
    },
    "entropy" : {
        "constraint_handler" : "entropy-penalty",
        "loss_func" : "ce"
    },
    "penex" : {
        "constraint_handler" : "logsumexp-penalty",
        "loss_func" : "exp-loss"
    },
    "focal" : {
        "constraint_handler" : "dummy",
        "loss_func" : "ce"
    },
}

NAME2LOSS = {
    "penex" : "penex", 
    "ce" : "ce", 
    "smoothing" : "ce_smoothing", 
    "entropy" : "ce_entropy", 
    "focal" : "focal-loss"
}

def main(seed):

    set_seed(seed)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_ID)
    train_dataset, val_dataset, test_dataset = load_data(
            seed,
            tokenizer,
            1, # does not matter since we only test
    )

    for dir in os.listdir(CHECKPOINT_DIR):
        config_path = os.path.join(CONFIG_DIR, NAME2LOSS[dir]) + ".json"
        cfg = load_json(config_path)

        # Load model from checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, dir)
        # there should be two checkpoints in the directory, take the one with smaller epoch number (must be the best one)
        ckpt_folder = [f for f in os.listdir(ckpt_path) if "checkpoint" in f]
        assert len(ckpt_folder) == 1, f"More than one checkpoint found in {ckpt_path}"
        best_ckpt = os.path.join(ckpt_path, ckpt_folder[0])
        if dir == "penex":
            model = PENEXRobertaForSequenceClassification.from_pretrained(
                best_ckpt,
                sensitivity=cfg["model_params"]["sensitivity"]
            )
        else:
            model = RobertaForSequenceClassification.from_pretrained(
                best_ckpt,
                num_labels=len(ID2LABEL)
            )
        
        # Dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        parser = HfArgumentParser(TrainingArguments)
        args, = parser.parse_dict(args=cfg["hf_params"])
        
        trainer_args = {
            "model" : model,
            "args" : args,
            "train_dataset" : train_dataset,
            "eval_dataset" : val_dataset,
            "tokenizer" : tokenizer,
            "data_collator" : data_collator,
            "compute_metrics" : compute_metrics
        }
        if dir in ["ce", "smoothing"]:
            trainer = Trainer(
                **trainer_args
            )
        elif dir == "focal":
            trainer = FocalClassificationTrainer(
                gamma=cfg["model_params"]["gamma"],
                **trainer_args
            )
        elif dir == "entropy":
            trainer = EntropyClassificationTrainer(
                rho=cfg["model_params"]["rho"],
                **trainer_args
            )
        else: # penex
            trainer = PENEXClassificationTrainer(
                sensitivity=cfg["model_params"]["sensitivity"],
                **trainer_args
            )

        bootstrapped_metrics = bootstrap_evaluate(
                                                dataset=test_dataset, 
                                                trainer=trainer
                                                )
        trainer.eval_dataset = test_dataset
        metrics = trainer.evaluate()

        log_dir = os.path.join(RESULT_DIR, dir)
        with open(os.path.join(log_dir, "test_results_val_opt.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(log_dir, "bootstrap_results_val_opt.json"), "w") as f:
            json.dump(bootstrapped_metrics, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run with given parameters")
    parser.add_argument('--seed', type=int, default=0, help="Random seed. Defaults to 0.")
    args = parser.parse_args()
    main(args.seed)
