from functools import partial
from typing import override
import math
import torch

from project_files.vision.models.classifiers import Classifier
from project_files.vision.models.classifiers import create_generic_classifier
from project_files.vision.imagenet.models.forward_modules import ImagenetHardConstraintForwardModule, ImagenetSoftConstraintForwardModule


class ImagenetClassifier(Classifier):
    """
    Base class for classifiers that only specifies the architecture and validation, no training details.
    For ImageNet, we always use learning rate scheduling.
    """

    _soft_constraint_module_cls = ImagenetSoftConstraintForwardModule
    _hard_constraint_module_cls = ImagenetHardConstraintForwardModule
    _default_num_classes = 1000

    def __init__(self, constraint_handler, num_classes=None, **kwargs):
        super().__init__(
            constraint_handler=constraint_handler,
            soft_constraint_forward_module_cls=self._soft_constraint_module_cls,
            hard_constraint_forward_module_cls=self._hard_constraint_module_cls,
            num_classes=num_classes or self._default_num_classes,
            **kwargs
        )
    
    @override
    def training_step(self, batch, batch_idx):
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_step=True, on_epoch=False)
        return super().training_step(batch, batch_idx)
    
    @override
    def configure_optimizers(self):
        assert self.warmup_epochs is not None, "warmup_epochs must be specified for ImagenetClassifier"        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Calculate total steps
        total_epochs = self.max_epochs
        effective_batch_size = self.batch_size * self.num_devices
        dataset_size = self.dataset_size
        steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
        total_steps = total_epochs * steps_per_epoch
        warmup_epochs = self.warmup_epochs
        warmup_steps = warmup_epochs * steps_per_epoch
        
        # Define LR schedule function (linear warmup + cosine annealing)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing from 1 to 0
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",    # Update every step
                "frequency": 1,        # Every step
                "name": "learning_rate"
            }
        }


create_imagenet_classifier = partial(
    create_generic_classifier, 
    base_class=ImagenetClassifier
)
