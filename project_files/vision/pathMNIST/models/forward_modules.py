import torch
from torch import nn

from project_files.vision.forward_modules import ForwardModule


class PathMNISTForwardModule(ForwardModule):
    def __init__(self, n_out, architecture="convnet", deep_supervision=False):
        assert architecture == "convnet", "Only 'convnet' architecture is supported for PathMNIST."
        super().__init__(n_out=n_out, deep_supervision=False)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # 32x32x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 32x32x64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # 32x32x128

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by 2

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, n_out)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)

        x = self.relu(self.conv2(x))
        x = self.max_pool(x)

        x = self.relu(self.conv3(x))
        x = self.max_pool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    

class PathMNISTHardConstraintForwardModule(PathMNISTForwardModule):
    """Classifier that has the constraint built-in as a hard constraint."""
    def __init__(self, n_out, architecture="convnet", deep_supervision=False):
        if n_out is None:
            n_out = 9
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
    

class PathMNISTSoftConstraintForwardModule(PathMNISTForwardModule):
    def __init__(self, n_out, architecture="convnet", deep_supervision=False):
        if n_out is None:
            n_out = 9
        super().__init__(
                    n_out=n_out, # All 9 classes are present
                    architecture=architecture, 
                    deep_supervision=deep_supervision
                    )
    
    def forward(self, x):
        out = super().forward(x)
        return out
    