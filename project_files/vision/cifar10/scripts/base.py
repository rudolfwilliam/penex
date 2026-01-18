import random

from torchvision import datasets, transforms


def load_cifar10(
        label_noise=None, 
        seed=0, 
        size="full",
        num_classes=None,
        keep_classes=None
        ):

    assert size == "full", "Size must be 'full' for CIFAR-10."
    assert num_classes == None, "Number of classes must be 10 for CIFAR-10."
    assert keep_classes is None, "Keep classes is not supported for CIFAR-10."

    NUM_CLASSES = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Mean and std for CIFAR-10
    ])

    # Load the dataset
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
        )

    if label_noise is not None:
        assert 0 <= label_noise <= 1, "Label noise must be in [0, 1]"
        local_rng = random.Random(seed)
        new_targets = []
        for label in train_dataset.targets:
            if local_rng.random() < label_noise:
                new_targets.append(local_rng.randint(0, NUM_CLASSES - 1))
            else:
                new_targets.append(label)
        train_dataset.targets = new_targets

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset


#SEARCH_SPACES["train_exp-loss_sumexp-penalty_adam"]["model_params"]["sensitivity"] = [0.0, 0.3]

__all__ = ['load_cifar10']
