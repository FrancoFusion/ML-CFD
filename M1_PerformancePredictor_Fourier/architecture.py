import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

class HeatChannelNet(nn.Module):
    def __init__(self):
        super(HeatChannelNet, self).__init__()
        
        # Hyperparameters for fourier stuff
        self.n_modes = (20, 20)          # Fourier modes
        self.hidden_channels = 128       # Hidden channels
        self.num_fno_blocks = 2          # FNO blocks
        
        # Initial convolution to project input to hidden dimensions (so called lifting)
        self.fc0 = nn.Conv2d(3, self.hidden_channels, 1)
        
        # FNO Blocks loop
        self.fno_blocks = nn.ModuleList([
            FNO(
                n_modes=self.n_modes,
                hidden_channels=self.hidden_channels,
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels
            ) for _ in range(self.num_fno_blocks)
        ])
        
        # Batch norm after FNO blocks
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        
        # Pressure Drop Branch
        self.pd_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                # [B, C, 1, 1]
            nn.Flatten(),                           # [B, C]
            nn.Linear(self.hidden_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.ReLU()                                # Positive pressure drop
        )
        
        # Temperature Branch
        self.temp_branch = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()                             # Normalize to [0, 1] meaning [298, maxT]
        )
        
    def forward(self, heat_source, channel_geometry, inlet_velocity):
        # Concatenate inputs along the channel dimension
        x = torch.cat((heat_source, channel_geometry, inlet_velocity), dim=1)  # [B, 3, 101, 215]
        
        # Initial projection
        x = self.fc0(x)
        
        # Pass through FNO blocks
        for fno in self.fno_blocks:
            x = fno(x)
        
        # Batch normalization and activation
        x = self.bn(x)
        x = F.relu(x)
        
        # Pressure drop prediction
        pd = self.pd_branch(x)  # [B, 1]
        
        # Temperature prediction
        temp = self.temp_branch(x)  # [B, 1, 101, 215]
        
        return pd, temp
    
"""
# Testing the updated model
if __name__ == "__main__":
    model = HeatChannelNet()
    model.eval()

    # Create dummy input data with the correct dimensions
    batch_size = 1
    heat_source_sample = torch.ones(batch_size, 1, 101, 215)               # Constant heat source
    channel_geometry_sample = torch.randint(0, 2, (batch_size, 1, 101, 215)).float()  # Binary geometry
    inlet_velocity_sample = torch.ones(batch_size, 1, 101, 215)            # Constant inlet velocity

    # Forward pass
    with torch.no_grad():
        pressure_drop_pred, temperature_pred = model(heat_source_sample, channel_geometry_sample, inlet_velocity_sample)

    # Print output shapes and values
    print('Pressure drop prediction shape:', pressure_drop_pred.shape)      # Should be [1, 1]
    print('Temperature prediction shape:', temperature_pred.shape)          # Should be [1, 1, 101, 215]
"""