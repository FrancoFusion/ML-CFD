import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

class HeatChannelNet(nn.Module):
    def __init__(self):
        super(HeatChannelNet, self).__init__()
        
        # Hyperparameters
        self.n_modes = (20, 20)
        self.hidden_channels = 128
        self.num_fno_blocks = 2
        
        # Lifting
        self.fc0 = nn.Conv2d(3, self.hidden_channels, 1)
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNO(
                n_modes=self.n_modes,
                hidden_channels=self.hidden_channels,
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
            ) for _ in range(self.num_fno_blocks)
        ])
        
        # Batch normalization after FNO blocks
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        
        # Pressure Drop Branch
        self.pd_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.ReLU()                                # Pressure drop is positive
        )
        
        # Temperature Branch
        self.temp_branch = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()                             # Normalize to [0, 1]
        )
        
    def forward(self, heat_source, channel_geometry, inlet_velocity):
        # Concatenate inputs along the channel dimension
        x = torch.cat((heat_source, channel_geometry, inlet_velocity), dim=1)
        
        # Lifting
        x = self.fc0(x)
        
        # Pass through FNO blocks
        for fno in self.fno_blocks:
            x = fno(x)
        
        # Batch norm and activation
        x = self.bn(x)
        x = F.relu(x)
        
        # Pressure drop prediction
        pd = self.pd_branch(x)
        
        # Temperature prediction
        temp = self.temp_branch(x)
        
        return pd, temp
    
"""
if __name__ == "__main__":
    model = HeatChannelNet()
    model.eval()

    batch_size = 1
    heat_source_sample = torch.ones(batch_size, 1, 101, 181)               # Constant heat source
    channel_geometry_sample = torch.randint(0, 2, (batch_size, 1, 101, 181)).float()  # Binary geometry
    inlet_velocity_sample = torch.ones(batch_size, 1, 101, 181)            # Constant inlet velocity

    with torch.no_grad():
        pressure_drop_pred, temperature_pred = model(heat_source_sample, channel_geometry_sample, inlet_velocity_sample)

    print('Pressure drop prediction shape:', pressure_drop_pred.shape)
    print('Temperature prediction shape:', temperature_pred.shape)
"""