import torch
from torch import nn

from project_files.vision.models.convnet import ConvNet
from project_files.vision.models.vit import ViT
from project_files.vision.forward_modules import ForwardModule


class Cifar10ForwardModule(ForwardModule):
    def __init__(self, n_out, architecture="convnet", deep_supervision=False):
        super().__init__(n_out=n_out, deep_supervision=deep_supervision)
        if architecture == "convnet":
            self.architecture = ConvNet(n_out=n_out, deep_supervision=deep_supervision)
        elif architecture == "ViT":
            self.architecture = ViT(
                num_classes = n_out,
                image_size = 32,
                patch_size = 4,
                dim = 128,
                depth = 6,
                heads = 4,
                mlp_dim = 512,      # 4x dim
                pool = 'cls',
                channels = 3,
                dim_head = 32,      # dim / heads
                dropout = 0.1,
                emb_dropout = 0.1,
                deep_supervision=deep_supervision
            )
        else:
            raise ValueError(f"Architecture {architecture} not recognized.")
    
    def forward(self, x):
        return self.architecture.forward(x)
    

class Cifar10HardConstraintForwardModule(Cifar10ForwardModule):
    """Classifier that has the constraint built-in as a hard constraint."""
    def __init__(self, n_out, architecture="convnet", deep_supervision=False):
        if n_out is None:
            n_out = 10
        super().__init__(
                        n_out=n_out-1, # One class less to account for the constraint
                        architecture=architecture, 
                        deep_supervision=deep_supervision
                        )
    
    def forward(self, x):
        out = super().forward(x)  # Get the output from the base forward module
        logit_sum = out.sum(dim=1, keepdim=True)  # Sum across classes
        final = -logit_sum  # Constraint: sum of logits should be zero
        out = torch.cat((out, final), dim=1)  # Append the constraint logit
        return out
    

class Cifar10SoftConstraintForwardModule(Cifar10ForwardModule):
    """Classifier where the constraint needs to be taken care of 
    through a separate constraint handler."""
    def __init__(self, n_out, architecture="convnet", deep_supervision=False):
        if n_out is None:
            n_out = 10
        super().__init__(
                    n_out=n_out, # All 10 classes are present
                    architecture=architecture, 
                    deep_supervision=deep_supervision
                    )
    
    def forward(self, x):
        out = super().forward(x)
        return out
    