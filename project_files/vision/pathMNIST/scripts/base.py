from torch.utils.data import ConcatDataset, Subset
from project_files.vision.base import SEARCH_SPACES
import torchvision.transforms as transforms

from medmnist import PathMNIST


def load_pathMNIST(
        label_noise=None, 
        seed=0, 
        size=10000,
        num_classes=None,
        keep_classes=None
        ):
    assert label_noise is None, "Label noise is not supported for pathMNIST."
    assert num_classes == None, "Number of classes must be None for pathMNIST."
    assert keep_classes is None, "Keep classes is not supported for pathMNIST."

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    common_args = {
        "download": True,
        "size": 64,
        "transform": transform,
        "target_transform": lambda y: y.squeeze()
    }

    train_dataset = PathMNIST(split="train", **common_args)
    val_dataset   = PathMNIST(split="val", **common_args)
    test_dataset  = PathMNIST(split="test", **common_args)

    # merge train and val, because the validation set is created in training function already.
    merged_train = ConcatDataset([train_dataset, val_dataset])
    if size != "full":
        assert size > 0, "Size must be greater than 0."
        assert size <= len(train_dataset), f"Size must be less than or equal to {len(train_dataset)}."
        merged_train = Subset(merged_train, list(range(size)))    

    return merged_train, test_dataset


SEARCH_SPACES["train_exp-loss_sumexp-penalty_adam"]["model_params"]["sensitivity"] = [0.0, 0.3]

__all__ = ['load_pathMNIST', 'SEARCH_SPACES']
