from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def exp_loss(y, 
             logits, 
             sensitivity=1.0, 
             transform_y_flag=False, 
             reduction="mean", # "mean", "sum", "none"
             huberization: Optional[float] = None
             ):
    if transform_y_flag:
        pass
    else:
        # detransform y
        y = y.argmax(dim=1)
    
    # scalar product between y_tilde and logits
    loss = (-sensitivity * torch.gather(logits, 1, y.unsqueeze(1))).exp().squeeze(1)
    
    # Apply Huberization if specified
    if huberization is not None:
        loss = _apply_huberization(loss, huberization, sensitivity)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def _apply_huberization(loss: torch.Tensor, threshold: float, sensitivity: float) -> torch.Tensor:
    """
    Apply Huberization to the exponential loss.
    
    For losses above the threshold, we switch to a linear approximation to prevent
    explosive growth while maintaining gradient continuity.
    
    Args:
        loss: The exponential loss values
        threshold: Huberization threshold 
        sensitivity: The sensitivity factor used in the original loss
        
    Returns:
        Huberized loss values
    """
    # For small losses (< threshold): use original exponential loss
    # For large losses (>= threshold): use linear continuation
    
    # Convert scalars to tensors on the same device/dtype as loss
    threshold_tensor = torch.tensor(threshold, dtype=loss.dtype, device=loss.device)
    sensitivity_tensor = torch.tensor(sensitivity, dtype=loss.dtype, device=loss.device)
    
    # The linear part: slope * (original_logit - threshold_logit) + threshold
    # where slope is the derivative of exp loss at the threshold point
    original_logits = -(1/sensitivity_tensor) * torch.log(loss/sensitivity_tensor)
    threshold_logit = -(1/sensitivity_tensor) * torch.log(threshold_tensor/sensitivity_tensor)
    
    # Derivative of exp(loss) at threshold = sensitivity * threshold
    slope = -sensitivity_tensor * threshold_tensor
    linear_part = slope * (original_logits - threshold_logit) + threshold_tensor
    
    return torch.where(loss < threshold_tensor, loss, linear_part)


class PENEX(nn.Module):
    """PENEX loss function as a PyTorch class."""
    MODES = ["auto", "running"]
    def __init__(self,
                 reduction: str = "mean",
                 sensitivity: float = 0.1,
                 penalty_mode: str = "running",
                 penalty: Optional[float] = None,
                 penalty_min: float = 1e-10,
                 penalty_max: float = 100.0,
                 ema: float = 0.1,
                 huberization: Optional[float] = None,
                 ignore_index: int | None = -100) -> None:
        super(PENEX, self).__init__()
        self.reduction = reduction
        self.sensitivity = sensitivity
        self.penalty_mode = penalty_mode
        self.penalty_min = penalty_min
        self.penalty_max = penalty_max
        self.ema = ema
        self.huberization = huberization
        self.ignore_index = ignore_index

        if penalty_mode not in self.MODES and penalty is None:
            raise ValueError(f"Invalid penalty_mode: {penalty_mode}. Choose from {self.MODES} or provide a float penalty value.")
        
        # Register penalty as a buffer so it's part of the model state
        if penalty is not None:
            self.register_buffer('penalty', torch.tensor(penalty, dtype=torch.float32))
        else:
            self.register_buffer('penalty', None)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # check if target is one-hot encoded
        if target.ndim == 2 and target.shape[1] > 1:
            target = target.argmax(dim=1)
        
        # filter out the ignore_index (if any)
        if self.ignore_index is not None:
            input = input[target != self.ignore_index]
            target = target[target != self.ignore_index]

        sumexp = torch.exp(torch.logsumexp(input, dim=-1))
        sumexp_mean = sumexp.mean()
        exp_loss_values = exp_loss(
                target,
                input,
                self.sensitivity,
                transform_y_flag=True,
                reduction="none",
                huberization=self.huberization
                )
        loss_mean = exp_loss_values.mean()

        # Handle penalty computation
        if self.penalty_mode in ["auto", "running"]:
            # Compute penalty estimate (detached from gradient graph)
            with torch.no_grad():
                penalty_est = self.sensitivity * (loss_mean.detach() / (sumexp_mean.detach() + 1e-12))
                penalty_est = penalty_est.clamp(self.penalty_min, self.penalty_max).to(input.dtype)
                
                if self.penalty_mode == "auto":
                    # For auto mode, just use the computed penalty (don't store it)
                    current_penalty = penalty_est
                elif self.penalty_mode == "running":
                    # For running mode, update the stored penalty
                    if self.penalty is None:
                        self.penalty = penalty_est.clone()
                        current_penalty = penalty_est
                    else:
                        # Update using EMA
                        self.penalty.mul_(1 - self.ema).add_(penalty_est, alpha=self.ema)
                        current_penalty = self.penalty.clone()
        else:
            # Use fixed penalty
            current_penalty = self.penalty
            if current_penalty is None:
                raise ValueError("Penalty must be provided for non-auto/running modes")
        
        # Compute final loss: exp_loss + penalty * sumexp
        total_loss = exp_loss_values + current_penalty * sumexp

        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        elif self.reduction == "none":
            return total_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Choose from 'mean', 'sum', or 'none'.")


class FocalLoss(nn.Module):
    """Mostly copied from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py"""
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
