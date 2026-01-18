from functools import partial

from project_files.vision.models.classifiers import Classifier, AdaptivePENEXClassifier
from project_files.vision.models.classifiers import create_generic_classifier
from project_files.vision.cifar100.models.forward_modules import Cifar100HardConstraintForwardModule, Cifar100SoftConstraintForwardModule


def create_cifar100_classifier_base(base_class):
    """Factory function to create CIFAR-100 classifier classes."""
    
    class Cifar100ClassifierImpl(base_class):
        """CIFAR-100 classifier implementation."""
        
        _soft_constraint_module_cls = Cifar100SoftConstraintForwardModule
        _hard_constraint_module_cls = Cifar100HardConstraintForwardModule
        _default_num_classes = 100

        def __init__(self, constraint_handler, num_classes=None, **kwargs):
            super().__init__(
                constraint_handler=constraint_handler,
                soft_constraint_forward_module_cls=self._soft_constraint_module_cls,
                hard_constraint_forward_module_cls=self._hard_constraint_module_cls,
                num_classes=num_classes or self._default_num_classes,
                **kwargs
            )
    
    return Cifar100ClassifierImpl


# Create the specific classes
Cifar100Classifier = create_cifar100_classifier_base(Classifier)
Cifar100AdaptivePENEXClassifier = create_cifar100_classifier_base(AdaptivePENEXClassifier)

# Create the factory functions
create_cifar100_classifier = partial(
    create_generic_classifier, 
    base_class=Cifar100Classifier,
)

create_adaptive_penex_cifar100_classifier = partial(
    create_generic_classifier, 
    base_class=Cifar100AdaptivePENEXClassifier
)
