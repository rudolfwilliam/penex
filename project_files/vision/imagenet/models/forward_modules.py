import torch
from torch import nn

from project_files.vision.models.convnet import ConvNet
from project_files.vision.models.vit import ViT
from project_files.vision.forward_modules import ForwardModule


class ImagenetForwardModule(ForwardModule):
    def __init__(self, n_out, architecture="ViT", deep_supervision=False):
        super().__init__(n_out=n_out, deep_supervision=deep_supervision)
        assert architecture == "ViT", "Only ViT architecture is supported for ImageNet."
        self.architecture = ViT(
            num_classes = n_out,
            image_size = 224,
            patch_size = 16,
            dim = 768,
            depth = 12,
            heads = 12,
            mlp_dim = 3072,
            pool = 'cls',
            channels = 3,
            dim_head = 64,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        # debugging with a smaller model
        """self.architecture = ViT(
            num_classes = n_out,
            image_size = 224,
            patch_size = 16,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            pool = 'cls',
            channels = 3,
            dim_head = 64,
            dropout = 0.1,
            emb_dropout = 0.1,
            deep_supervision = deep_supervision
        )"""
    
    def forward(self, x):
        return self.architecture.forward(x)
    

class ImagenetHardConstraintForwardModule(ImagenetForwardModule):
    """Classifier that has the constraint built-in as a hard constraint."""
    def __init__(self, n_out, architecture="convnet", deep_supervision=False):
        if n_out is None:
            n_out = 1000
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
    

class ImagenetSoftConstraintForwardModule(ImagenetForwardModule):
    """Classifier where the constraint needs to be taken care of 
    through a separate constraint handler."""
    def __init__(self, n_out, architecture="convnet", deep_supervision=False):
        if n_out is None:
            n_out = 1000
        super().__init__(
                    n_out=n_out, # All 10 classes are present
                    architecture=architecture,
                    deep_supervision=deep_supervision
                    )
    
    def forward(self, x):
        out = super().forward(x)
        return out
    