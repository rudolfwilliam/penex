from torch import nn
import torch

from project_files.vision.forward_modules import ForwardModule


class ConvNet(ForwardModule):
    def __init__(self, n_out, deep_supervision):
        super().__init__(n_out=n_out, deep_supervision=deep_supervision)
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # 32x32x32
        self.fc_conv1 = nn.Linear(32 * 32 * 32, n_out)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) # 64x16x16
        self.fc_conv2 = nn.Linear(64*16*16, n_out)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # 32x32x128
        self.fc_conv3 = nn.Linear(32 * 32 * 128, n_out)
        # Pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by 2
        # Activation function
        self.relu = nn.ReLU()
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)
        # Fully connected layers
        # After three conv + pool layers: 32 -> 16 -> 8 -> 4 (spatial dimensions)
        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        self.fc2 = nn.Linear(256, n_out)
    
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        out_conv1 = self.fc_conv1(x.view(x.size(0), -1))
        x = self.relu(x)
        x = self.max_pool(x)
        # Second convolutional block
        x = self.conv2(x)
        out_conv2 = self.fc_conv2(x.view(x.size(0), -1))
        x = self.relu(x)
        x = self.max_pool(x)
        # Third convolutional block
        x = self.relu(self.conv3(x))
        x = self.max_pool(x)
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        # Apply dropout
        x = self.dropout(x)
        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
        out_final = self.fc2(x)
        outs = torch.stack((out_conv1, out_conv2, out_final), dim=1)
        
        return outs if self.deep_supervision else out_final # shape: (batch_size, 3, n_out)
    