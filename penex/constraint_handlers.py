"""Classes that take a loss function and deal with constraints in order to prevent logits from diverging."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ConstraintHandler(nn.Module, ABC):
    """Base class for constraint handlers."""
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        """Optional forward pass logic. Subclasses may override this if needed."""
        return None

    @abstractmethod
    def compute_loss(self, model, loss, logits, w=None):
        pass


class DummyHandler(ConstraintHandler):
    """Handler that does not do anything. However, it saves whether constraint is hard or soft."""
    def __init__(self, constraint="soft"):
        self.constraint = constraint
        super().__init__()
    
    def compute_loss(self, model, loss, logits):
        return loss


class LogSumExpPenalty(ConstraintHandler):
    """Handler for the Penalized Loss method."""
    def __init__(self, rho=1.0):
        super().__init__()
        self.rho = rho
    
    def compute_loss(self, model, loss, logits):
        penalty = (self.rho*(torch.logsumexp(logits, dim=-1))).mean()
        return loss + penalty


class SumExpPenalty(ConstraintHandler):
    """Handler for the Penalized Loss method."""
    def __init__(self, huberize=None, ema: float = 0.1, rho_min: float = 1e-6, rho_max: float = 100.0):
        super().__init__()
        self.rho = None
        self.ema = ema
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.huberize = huberize
    
    def _sumexp(self, logits: torch.Tensor) -> torch.Tensor:
        """Return E[ sum_j φ(f_j) ], where φ is exp or Huberized exp."""
        if self.huberize is None:
            # exact Σ_j exp(f_j) computed stably
            return torch.exp(torch.logsumexp(logits, dim=-1)).mean()

        # Huberized exponential: exp(z) for z<=z0, exp(z0)*(z - z0 + 1) for z>z0
        z = logits
        z0 = torch.as_tensor(float(self.huberize), dtype=z.dtype, device=z.device)
        exp_z0 = torch.exp(z0)
        linear_tail = exp_z0 * (z - z0 + 1.0)
        huber_exp = torch.where(z <= z0, torch.exp(z), linear_tail)

        return huber_exp.sum(dim=-1).mean()
    
    def compute_loss(self, model, loss, logits):
        sensitivity = float(getattr(model, "sensitivity"))
        sumexp = self._sumexp(logits)

        with torch.no_grad():
            rho_est = sensitivity * (loss.detach()/(sumexp.detach() + 1e-12))
            rho_est = rho_est.to(logits.dtype).to(logits.device).clamp(self.rho_min, self.rho_max)
            if self.rho is None:
                self.rho = rho_est
            else:
                self.rho = (1 - self.ema) * self.rho + self.ema * rho_est

        penalty = self.rho*sumexp

        return loss + penalty


class LinPenalty(ConstraintHandler):
    """Alternative linear handler for the Penalized Loss method."""
    def __init__(self, rho=2.0):
        super().__init__()
        self.rho = rho
    
    def compute_loss(self, model, loss, logits):
        penalty = self.rho*(logits.mean(-1)).mean()
        return loss + penalty
    

class EntropyPenalty(ConstraintHandler):
    """Alternative linear handler for the Penalized Loss method."""
    def __init__(self, rho=2.0):
        super().__init__()
        self.rho = rho
    
    def compute_loss(self, model, loss, logits):
        penalty = -self.rho*(torch.distributions.Categorical(logits=logits).entropy()).mean()
        return loss + penalty


class SquaredPenalty(ConstraintHandler):
    # simply add squared penalty term to the loss
    def __init__(self, rho=1e2):
        super().__init__()
        self.rho = rho
    
    def compute_loss(self, model, loss, logits):
        h_x = (logits.sum(-1).pow(2)).mean()
        penalty = (self.rho / 2) * h_x.pow(2) # we square twice
        return loss + penalty
    

class OptimizationConstraintHandler(ConstraintHandler, ABC):
    """Base class for constraint handlers that require optimizer steps."""
    @abstractmethod
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        pass


class AugmentedLagrangian(OptimizationConstraintHandler):
    def __init__(self, rho=1e3, nu=1e3):
        super().__init__()
        self.rho = rho  # Penalty parameter
        self.nu = nu  # Update parameter for lambda
        # Initialize Lagrange multiplier lambda
        self.register_buffer('lambda_', torch.zeros(1, requires_grad=False))
        self.register_buffer('h_x', torch.zeros(1, requires_grad=False))
    
    def compute_loss(self, model, loss, logits):
        # Compute augmented Lagrangian estimate
        h_x = (logits.sum(-1).pow(2)).mean()
        aug_lagrangian = loss + self.lambda_ * h_x + (self.rho / 2) * h_x.pow(2)
        # Save detached h_x for use in optimizer_step
        self.h_x = h_x.detach()
        return aug_lagrangian
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Perform the closure (compute loss and backward)
        optimizer_closure()
        # Perform the Adam optimizer step w.r.t. primal variables
        optimizer.step()
        with torch.no_grad():
            # Perform the update step w.r.t. dual variables
            self.lambda_ += (self.rho/self.nu) * self.h_x
