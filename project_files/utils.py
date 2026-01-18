"""Some helper functions."""

import os
import argparse
import itertools

from json import load
import torch


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def flatten_list(list):
     return sum(list, [])


def get_config(config_dir, default):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of the config file to choose", type=str, default=default)
    args = argParser.parse_args()
    config = load(open(config_dir + args.name + ".json", "r"))
    return config


def load_json(json_file):
    import json
    # ensure the file exists
    assert os.path.exists(json_file), f"File not found: {json_file}"
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config


def set_seed(seed):
    from transformers import set_seed
    from pytorch_lightning import seed_everything
    seed_everything(seed)
    set_seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def transform_y(y, num_classes):
    y = one_hot_encode(y, num_classes)
    return y


def compute_weights(y, logits, transform_y_flag=False, epsilon=1.0):
    if transform_y_flag:
        y = transform_y(y, num_classes=logits.shape[1]).to(logits.device)
    w = (-epsilon * y * logits).sum(-1).exp().clone().detach()
    return w


def one_hot_encode(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()


def plot_weights(vis_dir, weights, current_epoch, x_label, max_val=3, bin_width=0.1):
    import matplotlib
    import matplotlib.pyplot as plt
    import scienceplots
    import numpy as np
    matplotlib.use('Agg')
    plt.style.use(['science', 'bright'])
    assert vis_dir is not None, "Please provide a directory to save the weights visualization"
    plt.xlim(0, max_val)
    fixed_range = (0, max_val)
    bins = np.arange(fixed_range[0], fixed_range[1] + bin_width, bin_width)
    plt.hist(weights, bins=bins, range=fixed_range, density=True)
    # put some things on the axes
    plt.xlabel(x_label)
    plt.ylabel('Density')
    # save each epoch's weights
    plt.savefig(vis_dir + "/weight_estimates_" + str(current_epoch) + ".pdf")
    # close the plot
    plt.close()


def plot_errors(vis_dir, errors, current_epoch, x_label, max_val=1, bin_width=0.1):
    import matplotlib.pyplot as plt
    #import scienceplots
    import numpy as np
    #matplotlib.use('Agg')
    #plt.style.use(['science', 'bright'])
    assert vis_dir is not None, "Please provide a directory to save the errors visualization"
    plt.xlim(0, max_val)
    fixed_range = (0, max_val)
    bins = np.arange(fixed_range[0], fixed_range[1] + bin_width, bin_width)
    plt.hist(errors, bins=bins, range=fixed_range, density=True)
    # put some things on the axes
    plt.xlabel(x_label)
    plt.ylabel('Density')
    # save each epoch's errors
    plt.savefig(vis_dir + "/errors_" + str(current_epoch) + ".pdf")
    # close the plot
    plt.close()


def plot_confusion_matrix(vis_dir, current_epoch, cm, normalize=False, title='Confusion matrix'):
    import matplotlib.pyplot as plt
    #import scienceplots
    import numpy as np
    #matplotlib.use('Agg')
    #plt.style.use(['science', 'bright'])
    assert vis_dir is not None, "Please provide a directory to save the errors visualization"
    plt.figure(figsize=(8, 6))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm.numpy().astype('float'), row_sums, where=row_sums != 0)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, rotation=45)
    plt.yticks(tick_marks)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(vis_dir + "/confusion_matrix_" + str(current_epoch) + ".pdf")
    # close the plot
    plt.close()


def get_num_devices():
    """Detect number of devices for HTCondor cluster."""
    
    # HTCondor GPU allocation
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_devices and cuda_devices != '-1':
            # Count comma-separated device IDs
            return len(cuda_devices.split(','))
    
    # HTCondor sets this for GPU jobs
    if '_CONDOR_SLOT_GPU_COUNT' in os.environ:
        return int(os.environ['_CONDOR_SLOT_GPU_COUNT'])
    
    # Alternative HTCondor GPU variable
    if 'GPU_COUNT' in os.environ:
        return int(os.environ['GPU_COUNT'])
    
    # Check PyTorch's direct GPU count
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"PyTorch detected {num_gpus} GPUs")
        return num_gpus
    
    # Fallback to CPU
    print("No GPUs detected, using CPU")
    return 1
 