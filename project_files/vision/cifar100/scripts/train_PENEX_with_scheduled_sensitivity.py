from project_files.vision.cifar100.scripts.train import run_cifar100_training
from project_files.vision.cifar100.models.classifiers import create_adaptive_penex_cifar100_classifier


if __name__ == "__main__":
    run_cifar100_training(create_adaptive_penex_cifar100_classifier)
