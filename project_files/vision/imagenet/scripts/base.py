import os
import random
import torch
import numpy as np
from torchvision import datasets, transforms

# get data dir from environment variable or set to default
assert "DATA_DIR" in os.environ, "DATA_DIR environment variable not set"
DATA_DIR = os.environ["DATA_DIR"]


def load_imagenet(
    label_noise=None, 
    seed=0, 
    size="full",
    num_classes=None,
    keep_classes=None
    ):

    assert size == "full", "Only 'full' size is supported for ImageNet."
    assert keep_classes is None, "keep_classes is not supported for ImageNet."
    assert label_noise is None, "label_noise is not supported for ImageNet."
    assert num_classes in (None, 1000), "num_classes must be None or 1000 for ImageNet."

    # Set seeds for reproducible augmentations
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Training transforms with minimal, research-appropriate augmentation
    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Geometric augmentation
        transforms.RandomHorizontalFlip(p=0.5),               # Simple flip
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    # Validation transforms (no augmentation)
    val_tfm = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    train_dataset = datasets.ImageFolder(f'{DATA_DIR}/train', transform=train_tfm)
    test_dataset   = datasets.ImageFolder(f'{DATA_DIR}/val',   transform=val_tfm)
    

    return train_dataset, test_dataset
