from abc import ABC
import math

import torch
from torch import nn
from torchmetrics.classification import MulticlassCalibrationError
import pytorch_lightning as pl

from project_files.utils import transform_y
from penex.losses import exp_loss, FocalLoss
from penex.constraint_handlers import (
                                        AugmentedLagrangian,
                                        SquaredPenalty,
                                        LogSumExpPenalty,
                                        SumExpPenalty,
                                        LinPenalty,
                                        DummyHandler,
                                        EntropyPenalty
                                        )


CONSTRAINT_HANDLERS = {
    "logsumexp-penalty": LogSumExpPenalty,
    "sumexp-penalty": SumExpPenalty,
    "lin-penalty": LinPenalty,
    "augmented-lagrangian": AugmentedLagrangian,
    "squared-penalty": SquaredPenalty,
    "dummy": DummyHandler,
    "hard": DummyHandler, # for hard constraint, is handled differently
    "entropy-penalty": EntropyPenalty
}


class Classifier(pl.LightningModule, ABC):
    """Base class for classifiers that only specifies the architecture and validation, no training details."""
    def __init__(self, 
                 *,
                 constraint_handler,
                 soft_constraint_forward_module_cls,
                 hard_constraint_forward_module_cls,
                 optimizer,
                 num_classes,
                 architecture="convnet",
                 deep_supervision=False,
                 max_epochs=None,
                 batch_size=None,
                 dataset_size=None,
                 num_devices=None,
                 warmup_epochs=None,
                 lr=1e-3,
                 weight_decay=0.0,
                 sensitivity=0.0,
                 label_smoothing=0.0,
                 inv_temp=1.0,
                 huberization=None,
                 gamma=0.0
                 ):
        super().__init__()
        self.lr = lr
        self.optimizer_str = optimizer # just the string, not the actual optimizer
        self.sensitivity = sensitivity
        self.label_smoothing = label_smoothing
        self.inv_temp = inv_temp # will be overwritten for exp-loss
        self.gamma = gamma
        self.huberization = huberization
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.num_devices = num_devices
        self.warmup_epochs = warmup_epochs

        self.errors = [] # Stores validation errors
        self.weight_norm = None
        self.confusion_matrix = None

        if self.optimizer_str == "adam":
            self.optimizer_cls = torch.optim.Adam
        elif self.optimizer_str == "lbfgs":
            self.optimizer_cls = torch.optim.LBFGS
        else:
            raise ValueError("Optimizer not recognized.")
        
        args = {
                "n_out": num_classes, 
                "architecture": architecture,
                "deep_supervision": deep_supervision
            }
        # deal with constraint
        self.constraint_handler = constraint_handler
        if isinstance(constraint_handler, DummyHandler):
            if constraint_handler.constraint == "hard":
                self.forward_module = hard_constraint_forward_module_cls(**args)
            else:
                self.forward_module = soft_constraint_forward_module_cls(**args)
        else:
            self.forward_module = soft_constraint_forward_module_cls(**args)

    def forward(self, x):
        return self.forward_module(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if len(logits.shape) > 2 and self.loss_func != "exp-loss":
            # if multi-layer output, apply across all layers
            batch_size, num_layers, num_classes = logits.shape
            # Reshape for vectorized computation: (batch_size * num_layers, num_classes)
            logits_flat = logits.view(-1, num_classes)
            # Expand y to match: (batch_size * num_layers,)
            y_expanded = y.unsqueeze(1).expand(-1, num_layers).contiguous().view(-1)
            # Compute loss with reduction='none' to get per-sample losses
            if self.loss_func == "focal-loss":
                losses_flat = FocalLoss(gamma=self.gamma, reduction="none")(logits_flat, y_expanded)
            else:
                losses_flat = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, reduction='none')(logits_flat, y_expanded)
            # Reshape back to (batch_size, num_layers) and average over layers, then over batch
            losses_reshaped = losses_flat.view(batch_size, num_layers)
            loss = losses_reshaped.mean()
        elif len(logits.shape) == 2 and self.loss_func != "exp-loss":
            if self.loss_func == "focal-loss":
                loss = FocalLoss(gamma=self.gamma)(logits, y)
            else:
                loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(logits, y)
        elif self.loss_func == "exp-loss":
            y_tilde = transform_y(y, num_classes=logits.shape[-1])
            if len(logits.shape) > 2:
                # if multi-layer output, apply across all layers
                batch_size, num_layers, num_classes = logits.shape
                # Reshape for vectorized computation: (batch_size * num_layers, num_classes)
                logits_flat = logits.view(-1, num_classes)
                # Expand y_tilde to match: (batch_size * num_layers, num_classes)
                y_tilde_expanded = y_tilde.unsqueeze(1).expand(-1, num_layers, -1).contiguous().view(-1, num_classes)
                # Compute loss with reduction='none' to get per-sample losses
                losses_flat = exp_loss(
                    y_tilde_expanded, 
                    logits_flat, 
                    sensitivity=self.sensitivity,
                    reduction='none',
                    huberization=self.huberization
                    )
                # Reshape back to (batch_size, num_layers) and average over layers, then over batch
                losses_reshaped = losses_flat.view(batch_size, num_layers)
                loss = losses_reshaped.mean()
            else:
                loss = exp_loss(
                    y_tilde,
                    logits,
                    sensitivity=self.sensitivity,
                    huberization=self.huberization
                    )
        # deal with constraint. Only required for exp-loss
        loss = self.constraint_handler.compute_loss(model=self, loss=loss, logits=logits)
        total_norm = self.compute_gradient_norm()
        # cross entropy loss (always needs to be computed for logging. Only taken from the very last layer if multi-layer output)
        ce_loss = nn.CrossEntropyLoss()(logits[:, -1, :], y) if len(logits.shape) > 2 else nn.CrossEntropyLoss()(logits, y)
        self.log("train_loss", ce_loss, on_epoch=True)
        self.log("grad_norm", total_norm, on_epoch=True)
        if type(self.constraint_handler) == SumExpPenalty:
            self.log("rho", self.constraint_handler.rho, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.loss_func == "exp-loss":
            self.inv_temp = 1 + self.sensitivity
        logits = self(x)*self.inv_temp # temperature scaling
        if len(logits.shape) > 2:
            # if multi-layer output, take the last layer
            logits = logits[:, -1, :]
        y_tilde = transform_y(y, num_classes=logits.shape[1])
        loss = exp_loss(
            y_tilde, 
            logits/self.inv_temp, 
            sensitivity=self.sensitivity,
            huberization=self.huberization
            )
        # probability that is assigned to the correct label
        probs = nn.Softmax(dim=-1)(logits)
        probs_correct = probs[torch.arange(probs.shape[0]), y]
        self.errors += list([1 - prob.item() for prob in probs_correct.cpu()])
        # cross entropy loss
        ce_loss = nn.CrossEntropyLoss()(logits, y)
        # accuracy for classification
        predicted_labels = torch.argmax(logits, dim=1)
        # add to confusion matrix
        if self.confusion_matrix is None:
            self.confusion_matrix = torch.zeros(logits.shape[1], logits.shape[1])
        for i, j in zip(y, predicted_labels):
            self.confusion_matrix[i, j] += 1
        acc = (predicted_labels == y).sum().item() / y.shape[0]
        brier_score = (probs - y_tilde).pow(2).sum(-1).mean()
        ece = MulticlassCalibrationError(num_classes=logits.shape[1])(logits, y)
        # margins
        true_class_scores = logits[torch.arange(logits.shape[0]), y]
        logits_clone = logits.clone()
        logits_clone[torch.arange(logits.shape[0]), y] = -float('inf')  # Exclude true class
        second_highest_scores, _ = logits_clone.max(dim=1)
        margin = (true_class_scores - second_highest_scores).mean()

        self.log("eval_accuracy", acc, on_epoch=True)
        self.log("eval_brier_score", brier_score, on_epoch=True)
        self.log("eval_exp_loss", loss, on_epoch=True)
        self.log("eval_loss", ce_loss, on_epoch=True)
        self.log("eval_ece", ece, on_epoch=True)
        self.log("eval_margins", margin, on_epoch=True)

        return -acc
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def compute_gradient_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)  # L2 norm
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def configure_optimizers(self):
        return self.optimizer_cls(
                                self.parameters(), 
                                lr=self.lr, 
                                weight_decay=self.weight_decay
                                )

    def optimizer_step(self, optimizer, lr_scheduler, optimizer_idx, closure, **kwargs):
        if hasattr(self.constraint_handler, "optimizer_step"):
            # Delegate to the wrapper's optimizer_step if it exists
            self.constraint_handler.optimizer_step(
                optimizer, lr_scheduler, optimizer_idx, closure, **kwargs
            )
        else:
            # Default optimizer step
            super().optimizer_step(optimizer, lr_scheduler, optimizer_idx, closure, **kwargs)


class AdaptivePENEXClassifier(Classifier):
    """PENEX Classifier with scaled sensitivity factor."""
    def __init__(self, epsilon=1e-6, scaling="linear", **kwargs):
        super().__init__(**kwargs)
        assert scaling in ["linear", "exponential"], "scaling must be 'linear' or 'exponential'."
        self.scaling = scaling
        self.sensitivity = self.sensitivity
        self.epsilon = epsilon

        total_epochs = self.max_epochs
        effective_batch_size = self.batch_size * self.num_devices
        dataset_size = self.dataset_size
        steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
        self.total_steps = total_epochs * steps_per_epoch

    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        assert self.loss_func == "exp-loss", "AdaptivePENEXClassifier only supports exp-loss."
        # Update the sensitivity factor after each training step
        current_step = self.trainer.global_step
        progress = min(current_step / self.total_steps, 1.0)

        if self.scaling == "linear":
            # Linear increase from epsilon to sensitivityax
            self.sensitivity = progress * self.sensitivity
        elif self.scaling == "exponential":
            # Exponential increase from epsilon to sensitivityax
            self.sensitivity = self.epsilon * (self.sensitivity / self.epsilon) ** progress

        self.sensitivity = max(self.sensitivity, self.epsilon)
        
        # Log the current sensitivity factor (every 100 steps to avoid spam)
        if current_step % 100 == 0:
            self.log("sensitivity", self.sensitivity, on_step=True)


# Factory function to create a subclass of BaseClassifier
def create_classifier(
                    *,
                    constraint_handler, 
                    loss_func,
                    model_params, 
                    constraint_params,
                    classifier_cls,
                    architecture="convnet",
                    deep_supervision=False,
                    num_classes=None,
                    max_epochs=None, # important for scheduling
                    batch_size=None,
                    dataset_size=None,
                    num_devices=None,
                    warmup_epochs=None
                    ):
    if constraint_handler == "hard":
        handler = DummyHandler(constraint="hard", **constraint_params)
    elif loss_func in ["exp-loss", "ce", "focal-loss"]:
        assert constraint_handler in CONSTRAINT_HANDLERS.keys(), f"constraint_handler {constraint_handler} not recognized."
        handler = CONSTRAINT_HANDLERS[constraint_handler](**constraint_params)
    else:
        raise ValueError(f"loss_func {loss_func} not recognized.")
    
    return classifier_cls(
                constraint_handler=handler, 
                loss_func=loss_func,
                num_classes=num_classes,
                max_epochs=max_epochs,
                dataset_size=dataset_size,
                batch_size=batch_size,
                num_devices=num_devices,
                warmup_epochs=warmup_epochs,
                architecture=architecture,
                deep_supervision=deep_supervision,
                **model_params
                )
    

def create_generic_classifier(
    base_class, 
    **kwargs
    ):

    ClassifierCls = create_classifier_cls(base_class)
    
    return create_classifier(
        classifier_cls=ClassifierCls,
        **kwargs
    )


def create_classifier_cls(base_class):
    class ExperimentClassifier(base_class):
        def __init__(self, loss_func, **kwargs):
            super().__init__(**kwargs)
            self.loss_func = loss_func
    
    return ExperimentClassifier
