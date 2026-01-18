import time

import torch
from torch.nn import CrossEntropyLoss

from penex.losses import PENEX, FocalLoss
from project_files.utils import set_seed

METHODS = ["CE", "SMOOTHING", "ENTROPY", "PENEX", "FOCAL"]
N = 100000

def main():

    set_seed(0)

    data = [torch.randn(64, 10) for _ in range(N)]  # Simulated data
    labels = [torch.randint(0, 10, (64,)) for _ in range(N)]  # Simulated labels
    for method in METHODS:
        times = []
        if method == "CE":
            loss_fn = CrossEntropyLoss(reduction="mean")
            for i in range(N):
                start_time = time.time()
                loss_fn(data[i], labels[i])
                end_time = time.time()
                times.append(end_time - start_time)
        elif method == "SMOOTHING":
            loss_fn = CrossEntropyLoss(label_smoothing=0.1, reduction="mean")
            for i in range(N):
                start_time = time.time()
                torch.nn.functional.cross_entropy(data[i], labels[i], label_smoothing=0.1)
                end_time = time.time()
                times.append(end_time - start_time)
        elif method == "ENTROPY":
            for i in range(N):
                start_time = time.time()
                torch.nn.functional.cross_entropy(data[i], labels[i]) - 0.1*(torch.distributions.Categorical(logits=data[i]).entropy()).mean()
                end_time = time.time()
                times.append(end_time - start_time)
        elif method == "PENEX":
            loss_fn = PENEX(reduction="mean")
            for i in range(N):
                start_time = time.time()
                loss_fn(data[i], labels[i])
                end_time = time.time()
                times.append(end_time - start_time)
        elif method == "FOCAL":
            loss_fn = FocalLoss(gamma=2.0, size_average=True)
            for i in range(N):
                start_time = time.time()
                # Simulated Focal loss computation
                loss_fn(data[i], labels[i])
                end_time = time.time()
                times.append(end_time - start_time)
        print(f"Method: {method}, Average Time: {sum(times)/len(times):.6f} seconds, std: {torch.std(torch.tensor(times)):.6f} seconds")


if __name__ == "__main__":
    main()
