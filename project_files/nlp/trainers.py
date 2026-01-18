from transformers import Trainer
import torch

from penex.losses import PENEX, FocalLoss


class PENEXTrainerMixin(Trainer):
    def __init__(
            self, 
            sensitivity=0.3, 
            huberize=None, 
            ema: float = 0.1, 
            rho_min: float = 1e-10, 
            rho_max: float = 100.0,
            penalty_mode="running",
            **kwargs
            ):
        super().__init__(**kwargs)
        self.sensitivity = sensitivity
        self.huberize = huberize
        self.ema = ema
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.loss_fct = PENEX(
            reduction="mean", 
            sensitivity=sensitivity, 
            penalty_mode=penalty_mode, 
            penalty_min=rho_min, 
            penalty_max=rho_max, 
            ema=ema, 
            ignore_index=-100
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if model.training:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

class PENEXClassificationTrainer(PENEXTrainerMixin):
    pass

class FocalClassificationTrainer(Trainer):
    def __init__(self, gamma=0.3, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if model.training:
            loss_fct = FocalLoss(gamma=self.gamma)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)


class EntropyClassificationTrainer(Trainer):
    def __init__(self, rho=1.0, **kwargs):
        super().__init__(**kwargs)
        self.rho = rho

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if model.training:
            loss = outputs["loss"]
            loss -= self.rho * (torch.distributions.Categorical(logits=logits).entropy()).mean()
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        