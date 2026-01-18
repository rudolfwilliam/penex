import random

from torchvision import datasets, transforms


def _filter_classes(dataset, keep):
    """
    Keep only the classes listed in `keep` (iterable of ints)
    and re-index the targets accordingly.
    """
    keep = sorted(set(keep))
    idx = [i for i, t in enumerate(dataset.targets) if t in keep]

    # Slice the underlying arrays / lists
    dataset.data    = dataset.data[idx]                     # Numpy array (32,32,3)
    dataset.targets = [dataset.targets[i] for i in idx]     # List of ints

    mapping = {old : new for new, old in enumerate(keep)}
    dataset.targets = [mapping[t] for t in dataset.targets]

    return dataset


def load_cifar100(
    label_noise=None, 
    seed=0, 
    size="full",
    num_classes=None,
    keep_classes=None
    ):

    if num_classes is None:
        num_classes = 100

    assert size == "full", "Size must be 'full' for CIFAR-100."

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Mean and std for CIFAR-100
    ])

    # Load the dataset
    train_dataset = datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
        )

    test_dataset = datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Decide which classes to keep
    if keep_classes is None:
        keep_classes = list(range(num_classes))
    else:
        num_classes = len(keep_classes)

    # --- keep only those classes
    train_dataset = _filter_classes(train_dataset, keep_classes)
    test_dataset  = _filter_classes(test_dataset,  keep_classes)

    if label_noise is not None:
        assert 0 <= label_noise <= 1, "Label noise must be in [0, 1]"
        local_rng = random.Random(seed)
        new_targets = []
        for label in train_dataset.targets:
            if local_rng.random() < label_noise:
                new_targets.append(local_rng.randint(0, num_classes - 1))
            else:
                new_targets.append(label)
        train_dataset.targets = new_targets

    return train_dataset, test_dataset
