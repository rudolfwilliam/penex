import os
import sys
import logging
import json

from transformers import (
            RobertaTokenizer, 
            RobertaForSequenceClassification, 
            Trainer, 
            TrainingArguments, 
            HfArgumentParser,
            DataCollatorWithPadding
            )
import transformers

from project_files.utils import set_seed
from project_files.nlp.models import PENEXRobertaForSequenceClassification
from project_files.nlp.trainers import (
                                PENEXClassificationTrainer,
                                FocalClassificationTrainer,
                                EntropyClassificationTrainer
                                )
from project_files.nlp.bbc_news.utils import (
                                            compute_metrics, 
                                            bootstrap_evaluate
                                            )
from project_files.nlp.bbc_news.base import get_args, log_GPU_info
from project_files.nlp.bbc_news.meta_data import (
                                                MODEL_ID, 
                                                NUM_TRAINING_EXAMPLES, 
                                                ID2LABEL
                                                )
from project_files.nlp.bbc_news.base import load_data

if not 'WANDB_PROJECT' in os.environ: # allow for custom WANDB_PROJECT
    os.environ["WANDB_PROJECT"] = "bbc_news"
print("WANDB_PROJECT:", os.environ.get('WANDB_PROJECT'))

# set tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main(cfg, loss_func):

    log_GPU_info()

    set_seed(cfg["hf_params"]["seed"])

    parser = HfArgumentParser(TrainingArguments)
    args, = parser.parse_dict(args=cfg["hf_params"])

    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f", distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_ID)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(ID2LABEL),
    )
    if loss_func == "penex":
        checkpoint = model.state_dict()
        def rename_keys(state_dict):
            new_state_dict = {}
            for old_key, value in state_dict.items():
                if old_key.startswith("classifier.out_proj.weight"):
                    # take into account wrapper
                    new_key = old_key.replace("classifier.out_proj.weight", "classifier.out_proj.lm_head.weight")
                elif old_key.startswith("classifier.out_proj.bias"):
                    # take inlog_dirto account wrapper
                    new_key = old_key.replace("classifier.out_proj.bias", "classifier.out_proj.lm_head.bias")
                else:
                    new_key = old_key
                new_state_dict[new_key] = value
            return new_state_dict
        checkpoint_renamed = rename_keys(checkpoint)
        model = PENEXRobertaForSequenceClassification(
                            model.config, 
                            sensitivity=cfg["model_params"]["sensitivity"]
                            )
        missing, unexpected = model.load_state_dict(checkpoint_renamed, strict=False)
        if len(missing) > 0:
            logger.warning(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            logger.warning(f"Unexpected keys: {unexpected}")

    train_dataset, val_dataset, test_dataset = load_data(
            seed=cfg["hf_params"]["seed"],
            tokenizer=tokenizer,
            num_training_examples=NUM_TRAINING_EXAMPLES
    )

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        trainer, train_result, _ = train( # args need to be parsed again for modularity reasons
                                    cfg=cfg,
                                    train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    loss_func=loss_func,
                                    compute_metrics=compute_metrics,
                                    checkpoint=checkpoint
                                    )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Test ***")
        bootstrapped_metrics = bootstrap_evaluate(dataset=test_dataset, trainer=trainer)
        trainer.eval_dataset = test_dataset
        metrics = trainer.evaluate()
        
        log_dir = cfg["hf_params"]["logging_dir"]
        # create directory if it does not exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, "test_results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(log_dir, "bootstrap_results.json"), "w") as f:
            json.dump(bootstrapped_metrics, f, indent=4)


def train( # function needs to be separated for hyper parameter tuning
        cfg,
        train_dataset,
        val_dataset,
        loss_func,
        compute_metrics,
        checkpoint
        ):
    
    set_seed(cfg["hf_params"]["seed"]) # set seed again for compatibility between train and tune

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_ID)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(ID2LABEL),
    )
    if loss_func == "penex":
        pretrained = model.state_dict()
        def rename_keys(state_dict):
            new_state_dict = {}
            for old_key, value in state_dict.items():
                if old_key.startswith("classifier.out_proj.weight"):
                    # take into account wrapper
                    new_key = old_key.replace("classifier.out_proj.weight", "classifier.out_proj.lm_head.weight")
                elif old_key.startswith("classifier.out_proj.bias"):
                    # take into account wrapper
                    new_key = old_key.replace("classifier.out_proj.bias", "classifier.out_proj.lm_head.bias")
                else:
                    new_key = old_key
                new_state_dict[new_key] = value
            return new_state_dict
        pretrained_renamed = rename_keys(pretrained)
        model = PENEXRobertaForSequenceClassification(
                            model.config, 
                            sensitivity=cfg["model_params"]["sensitivity"]
                            )
        missing, unexpected = model.load_state_dict(pretrained_renamed, strict=False)
        if len(missing) > 0:
            logger.warning(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            logger.warning(f"Unexpected keys: {unexpected}")
    
    # Dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    parser = HfArgumentParser(TrainingArguments)
    args, = parser.parse_dict(args=cfg["hf_params"])
    # Create Trainer
    trainer_args = {
        "model" : model,
        "args" : args,
        "train_dataset" : train_dataset,
        "eval_dataset" : val_dataset,
        "tokenizer" : tokenizer,
        "data_collator" : data_collator,
        "compute_metrics" : compute_metrics
    }
    if loss_func in ["ce", "ce_smoothing"]:
        trainer = Trainer(
            **trainer_args
        )
    elif loss_func == "focal-loss":
        trainer = FocalClassificationTrainer(
            gamma=cfg["model_params"]["gamma"],
            **trainer_args
        )
    elif loss_func == "ce_entropy":
        trainer = EntropyClassificationTrainer(
            rho=cfg["model_params"]["rho"],
            **trainer_args
        )
    else: # penex
        trainer = PENEXClassificationTrainer(
            sensitivity=cfg["model_params"]["sensitivity"],
            **trainer_args
        )
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    eval_results = trainer.evaluate()

    return trainer, train_result, eval_results["eval_loss"]


if __name__ == "__main__":
    args, cfg = get_args()
    main(cfg=cfg, loss_func=args.loss_func)
