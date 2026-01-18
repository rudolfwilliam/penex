import torch
from torch.utils.data import Subset, DataLoader


def bootstrap_evaluate(dataset, trainer, model, batch_size, num_it=100):
    bootstrap_metrics = {
        "eval_accuracy" : [],
        "eval_brier_score" : [],
        "eval_ece" : [],
        "eval_loss" :[]
    }
    size = len(dataset)
    idxs = torch.arange(size)
    for _ in range(num_it):
        # subsample dataset idxs
        sampled_idxs = idxs[torch.randint(0, len(idxs), (size,))]
        subset = Subset(dataset, sampled_idxs)
        subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        metrics = trainer.test(dataloaders=subset_loader, model=model)[0]
        for key in bootstrap_metrics.keys():
            bootstrap_metrics[key].append(metrics[key])
    return bootstrap_metrics
