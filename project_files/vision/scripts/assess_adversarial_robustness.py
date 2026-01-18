import os
import argparse

import math
import torch
from torch import nn
import numpy as np
from robustbench.eval import benchmark

from project_files.utils import load_json, set_seed
from project_files.vision.cifar10.models.classifiers import create_cifar10_classifier
from project_files.vision.cifar10.scripts.base import load_cifar10
from project_files.vision.cifar100.models.classifiers import create_cifar100_classifier
from project_files.vision.cifar100.scripts.base import load_cifar100


CHECKPOINT_DIRS = {
    "CIFAR10" : "project_files/vision/cifar10/checkpoints",
    "CIFAR100" : "project_files/vision/cifar100/checkpoints",
    #"Noisy CIFAR10" : "project_files/vision/cifar10_noise_01/checkpoints",
    #"PathMNIST" : "project_files/vision/pathMNIST/checkpoints",
    #"CIFAR100" : "project_files/vision/cifar100/checkpoints"
}

RESULT_DIRS = {
    "CIFAR10" : "project_files/vision/cifar10/logs",
    "CIFAR100" : "project_files/vision/cifar100/logs",
}

CONFIG_DIRS = {
    "CIFAR10" : "project_files/vision/cifar10/configs",
    "CIFAR100" : "project_files/vision/cifar100/configs",
}

CONFIG_NAMES = {
    "ce" : "train_ce_dummy_adam.json",
    "smoothing" : "train_ce_dummy_adam_smoothing.json",
    "entropy" : "train_ce_entropy-penalty_adam.json",
    "penex" : "train_exp-loss_sumexp-penalty_adam.json",
    "focal" : "train_focal-loss_dummy_adam.json",
    "ce_ViT" : "train_ce_dummy_adam_ViT.json",
    "smoothing_ViT" : "train_ce_dummy_adam_smoothing_ViT.json",
    "entropy_ViT" : "train_ce_entropy-penalty_adam_ViT.json",
    "penex_ViT" : "train_exp-loss_sumexp-penalty_adam_ViT.json",
    "focal_ViT" : "train_focal-loss_dummy_adam_ViT.json",
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

CREATE_MODEL_FUNCS = {
    "CIFAR10" : create_cifar10_classifier,
    "CIFAR100" : create_cifar100_classifier,
}

LOAD_DATA_FUNCS = {
    "CIFAR10" : load_cifar10,
    "CIFAR100" : load_cifar100,
}

DATA_MEANS = {
    "CIFAR10" : (0.5, 0.5, 0.5),
    "CIFAR100" : (0.5, 0.5, 0.5),
}

DATA_STDS = {
    "CIFAR10" : (0.5, 0.5, 0.5),
    "CIFAR100" : (0.5, 0.5, 0.5),
}

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
    def forward(self, x):
        return (x - self.mean) / self.std


def main(dataset='CIFAR10', method='ce', architecture='convnet', deep_supervision=False):

    set_seed(0)  # Set seed for reproducibility

    appx = "" if architecture == "convnet" else "_ViT"

    cfg = load_json(os.path.join(CONFIG_DIRS[dataset], CONFIG_NAMES[method + appx]))
    base_model = CREATE_MODEL_FUNCS[dataset](
        constraint_handler=MODEL_ARGS[method]["constraint_handler"], 
        loss_func=MODEL_ARGS[method]["loss_func"], 
        model_params=cfg["model_params"], 
        constraint_params=cfg["constraint_params"],
        architecture=architecture,
        deep_supervision=deep_supervision
    )
    # load best checkpoint
    def safe_load_state_dict(model, ckpt_path):
        print("Resolving checkpoint:", os.path.abspath(ckpt_path), flush=True)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        # Quick IO sanity
        with open(ckpt_path, "rb") as f:
            f.read(1024)
        print("Checkpoint readable; loading to CPU…", flush=True)

        # Load on CPU to avoid CUDA init during deserialization
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")  # add weights_only=True if available
        except Exception as e:
            raise RuntimeError(f"torch.load failed: {e}")

        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"State dict loaded. Missing: {missing}, Unexpected: {unexpected}", flush=True)

        return model

    base_model = safe_load_state_dict(
        model=base_model, 
        ckpt_path=os.path.join(CHECKPOINT_DIRS[dataset], method + appx, "best-checkpoint.ckpt")
    )

    device = pick_device()
    print("device that is used:", device, flush=True)
    
    model = nn.Sequential(Normalize(DATA_MEANS[dataset], DATA_STDS[dataset]), base_model)
    print("Wrapper created")

    # Ensure the device is usable
    if device.type == 'cuda':
        torch.cuda.current_device()
        # Force runtime init now, not hidden in model.to(...)
        torch.zeros(1, device="cuda").sum().item()
        torch.cuda.synchronize()

    model.eval()
    model.to(device)

    print(f"Evaluating {method} on {dataset} with architecture {architecture}", flush=True)

    def run_sweep(model, eps_list, norm='Linf'):
        ra = []
        for eps in eps_list:
            res = benchmark(model, dataset=dataset.lower(), threat_model=norm,
                            eps=eps, n_examples=10000, batch_size=256,
                            device=device)
            p = res[1]
            ra.append(p)
            print(f"epsilon={eps:.5f} → RA={p*100:.2f}%")

        # Trapezoidal AUC over the discrete grid
        x = np.array([float(e) for e in eps_list])
        y = np.array(ra)
        auc = np.trapz(y, x) / (x.max() - x.min())  # normalized to [0,1] range of ε
        print(f"AUC(epsilon in [{x.min():.5f},{x.max():.5f}]) = {auc:.4f}")

        return ra, auc

    # L∞ small-ε sweep (helps your story vs. label smoothing)
    run_sweep(model=model, eps_list=[1/255, 2/255, 4/255], norm='Linf')
    # L2 moderate budgets
    run_sweep(model=model, eps_list=[0.25, 0.5], norm='L2')


def pick_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            torch.zeros(1, device="cuda").sum().item()
            torch.cuda.synchronize()
            return torch.device("cuda")
        except Exception as e:
            print(f"[warn] CUDA not usable: {e}")
    return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assess adversarial robustness on CIFAR-10")
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=list(LOAD_DATA_FUNCS.keys()), help='Dataset to use')
    parser.add_argument('--method', type=str, default='ce', choices=list(MODEL_ARGS.keys()), help='Method to use for training')
    parser.add_argument('--architecture', type=str, default="convnet", help='Directory to load checkpoints from')
    parser.add_argument('--deep_supervision', action='store_true', help='Use deep supervision in the model architecture')
    args = parser.parse_args()

    main(
        dataset=args.dataset, 
        method=args.method, 
        architecture=args.architecture,
        deep_supervision=args.deep_supervision
        )
