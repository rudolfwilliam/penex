import torch
from torch import nn
import evaluate
from torchmetrics.classification import MulticlassCalibrationError

from project_files.utils import transform_y

ACCURACY = evaluate.load("accuracy")


def bootstrap_evaluate(dataset, trainer, num_it=100):
    bootstrap_metrics = {
        "eval_accuracy" : [],
        "eval_brier_score" : [],
        "eval_ece" : [],
        "eval_loss" :[]
    }
    size = len(dataset)
    idxs = torch.arange(size)
    for i in range(num_it):
        # subsample dataset idxs
        sampled_idxs = idxs[torch.randint(0, len(idxs), (size,))]
        trainer.eval_dataset = dataset.select(sampled_idxs)
        metrics = trainer.evaluate()
        for key in bootstrap_metrics.keys():
            bootstrap_metrics[key].append(metrics[key])
    return bootstrap_metrics


running_metrics = {
    "accuracy" : 0.0,
    "brier_score" : 0.0,
    "ece" : 0.0,
    "num_samples" : 0
}

def compute_metrics(eval_pred, compute_result): # I hate this hugging face implementation... just saying...
    global running_metrics
    logits, labels = eval_pred
    predictions = torch.argmax(logits, axis=-1)
    batch_size = predictions.shape[0]
    running_metrics["num_samples"] += batch_size
    running_metrics["accuracy"] += batch_size * ACCURACY.compute(predictions=predictions, references=labels)["accuracy"]
    # probability that is assigned to the correct label
    probs = nn.Softmax(dim=-1)(logits)
    labels_tilde = transform_y(labels, num_classes=logits.shape[-1])
    running_metrics["brier_score"] += (probs - labels_tilde).pow(2).sum(-1).sum() # I don't trust the hugging face implementation of brier
    running_metrics["ece"] += batch_size * MulticlassCalibrationError(num_classes=logits.shape[1])(logits, labels)
    if compute_result:
        for k in running_metrics.keys():
            running_metrics[k] /= running_metrics["num_samples"]
        final_result = running_metrics.copy()
        # reset running metrics
        running_metrics = {
            "accuracy" : 0.0,
            "brier_score" : 0.0,
            "ece" : 0.0,
            "num_samples" : 0
        }
        return final_result
    else:
        return {}


def preprocess(batch, tokenizer):
    return tokenizer(
        batch["text"], 
        padding = False,      # We'll dynamically pad later
        truncation = True, 
        max_length = 512
    )
