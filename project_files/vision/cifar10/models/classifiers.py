from functools import partial

from project_files.vision.models.classifiers import Classifier

from project_files.vision.models.classifiers import create_generic_classifier
from project_files.vision.cifar10.models.forward_modules import Cifar10HardConstraintForwardModule, Cifar10SoftConstraintForwardModule


class Cifar10Classifier(Classifier):
    """Base class for classifiers that only specifies the architecture and validation, no training details."""

    _soft_constraint_module_cls = Cifar10SoftConstraintForwardModule
    _hard_constraint_module_cls = Cifar10HardConstraintForwardModule
    _default_num_classes = 10

    def __init__(self, constraint_handler, num_classes=None, **kwargs):
        super().__init__(
            constraint_handler=constraint_handler,
            soft_constraint_forward_module_cls=self._soft_constraint_module_cls,
            hard_constraint_forward_module_cls=self._hard_constraint_module_cls,
            num_classes=num_classes or self._default_num_classes,
            **kwargs
        )


create_cifar10_classifier = partial(
    create_generic_classifier,
    base_class=Cifar10Classifier
)
