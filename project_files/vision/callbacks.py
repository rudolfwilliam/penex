import torch
import pytorch_lightning as pl

from project_files.utils import plot_errors, plot_confusion_matrix


class SaveErrorDistributionCallback(pl.Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        super().__init__()
    
    def on_validation_epoch_start(self, trainer, model):
        model.errors = [] # overwrite errors
    
    def on_validation_epoch_end(self, trainer, model):
        errors = torch.tensor(model.errors)
        torch.save(errors, f"{self.log_dir}/errors_epoch_{trainer.current_epoch}.pt") # save errors


class LogErrorQuantileCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    
    def on_validation_epoch_start(self, trainer, model):
        model.errors = [] # overwrite errors
    
    def on_validation_epoch_end(self, trainer, model):
        errors = torch.tensor(model.errors)
        model.log(
                "error_quantile", 
                errors.quantile(0.9).item(), 
                on_step=False,
                on_epoch=True,
                logger=True,
                )


class EMAErrorQuantileCallback(pl.Callback):
    def __init__(self, alpha=0.1):
        """
        Exponential moving average of the quantile.
        """
        super().__init__()
        self.alpha = alpha
        self.ema = None

    def on_validation_epoch_end(self, trainer, model):
        # skip if in sanity check, otherwise we get a sub-optimal initialization of the EMA
        if trainer.sanity_checking: 
            return
        current_val_metric = trainer.callback_metrics.get("error_quantile")

        if self.ema is None:
            self.ema = current_val_metric
        else:
            self.ema = self.alpha * current_val_metric + (1 - self.alpha) * self.ema

        model.log(
                "error_quantile_ema", 
                self.ema, 
                on_step=False,
                on_epoch=True,
                logger=True
                )


class VisualizeErrorDistributionCallback(pl.Callback):
    def __init__(self, vis_dir):
        self.vis_dir = vis_dir
        super().__init__()
    
    def on_validation_epoch_start(self, trainer, model):
        model.errors = [] # overwrite errors
        model.confusion_matrix = None
    
    def on_validation_epoch_end(self, trainer, model):
        errors = torch.tensor(model.errors)
        plot_errors(vis_dir=self.vis_dir, errors=errors, current_epoch=trainer.current_epoch, x_label="$1 - p(y|x)$", bin_width=0.05)
        plot_confusion_matrix(vis_dir=self.vis_dir, 
                                current_epoch=trainer.current_epoch,
                                cm=model.confusion_matrix,
                                normalize=True)
