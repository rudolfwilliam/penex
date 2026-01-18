"""Plot histograms of the error distributions for each method."""

import argparse
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import scienceplots

from project_files.vision.cifar10.scripts.plotting.base import METHODS

EPOCH = 199


def main(log_dir):

    # convert string to Path
    log_dir = Path(log_dir)

    plt.style.use(["science"])
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']

    # load errors for each method, represented by a directory in the log_dir
    errors = {}
    for dir in log_dir.iterdir():
        if dir.is_dir():
            try:
                errors[dir.name] = torch.load(os.path.join(dir, f"errors_epoch_{EPOCH}.pt"))
                plt.hist(errors[dir.name], bins=30, alpha=0.5, label=dir.name, density=True)
            except:
                pass
    plt.legend(frameon=True,
                fancybox=True)
    plt.title("Error Distributions")
    plt.ylabel("Density")
    plt.xlabel("$1 - P(y|x)$")
    plt.savefig(os.path.join("plots", "error_hists.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing error distributions")
    args = parser.parse_args()
    main(args.log_dir)
