import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class PerformancePredictorNet(nn.Module):
    def __init__(self):
        super(PerformancePredictorNet, self).__init__()

        # Initial convolutional layers
        self.conv1 = nn.Conv3d(2, 8, kernel_size=5, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=5, padding=1)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=5, padding=1)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=5, padding=1)

        # Global average pooling for scalar output (pressure_drop)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected layers for scalar output
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        # Deconvolutional layers for temperature output
        self.deconv1 = nn.ConvTranspose3d(64, 32, kernel_size=5, padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 16, kernel_size=5, padding=1)
        self.deconv3 = nn.ConvTranspose3d(16, 8, kernel_size=5, padding=1)
        self.deconv4 = nn.ConvTranspose3d(8, 1, kernel_size=5, padding=1)

    def forward(self, heat_source, channel_geometry):        
        # Concatenate along channel dimension
        x = torch.cat([heat_source, channel_geometry], dim=1)  # Shape: [batch_size, 2, D, H, W]

        # Shared convolutional layers
        x = F.relu(self.conv1(x))   # [batch_size, 8, D, H, W]
        x = F.relu(self.conv2(x))   # [batch_size, 16, D, H, W]
        x = F.relu(self.conv3(x))   # [batch_size, 32, D, H, W]
        x = F.relu(self.conv4(x))   # [batch_size, 64, D, H, W]

        # Branch for pressure_drop (scalar output)
        x_pool = self.global_pool(x)        # [batch_size, 64, 1, 1, 1]
        x_flat = x_pool.view(-1, 64)        # Flatten to [batch_size, 64]
        x_fc = F.relu(self.fc1(x_flat))     # [batch_size, 32]
        pressure_drop = self.fc2(x_fc)      # [batch_size, 1]

        # Branch for temperature output (3d tensor)
        x_deconv = F.relu(self.deconv1(x))          # [batch_size, 32, D, H, W]
        x_deconv = F.relu(self.deconv2(x_deconv))   # [batch_size, 16, D, H, W]
        x_deconv = F.relu(self.deconv3(x_deconv))   # [batch_size, 8, D, H, W]
        temperature = self.deconv4(x_deconv)        # [batch_size, 1, D, H, W]

        return pressure_drop, temperature

"""
if __name__ == '__main__':
    net = PerformancePredictorNet()

    # Load test data
    test_data = torch.load('Data/M1_test_data.pt')

    # Define batch size
    batch_size = 4

    # Prepare lists to collect inputs
    heat_sources = []
    channel_geometries = []

    # Collect samples
    for i in range(batch_size):
        sample = test_data[i]

        # Prepare heat_source
        heat_source = sample['heat_source']  # Shape: [D, H, W]
        heat_source = heat_source.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]
        heat_sources.append(heat_source)

        # Prepare channel_geometry
        channel_geometry = sample['channel_geometry']  # Shape: [D, H, W]
        channel_geometry = channel_geometry.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]
        channel_geometries.append(channel_geometry)

    # Stack inputs along the batch dimension
    heat_source_batch = torch.cat(heat_sources, dim=0)  # Shape: [batch_size, 1, D, H, W]
    channel_geometry_batch = torch.cat(channel_geometries, dim=0)  # Shape: [batch_size, 1, D, H, W]

    # Forward pass
    pressure_drop_batch, temperature_batch = net(heat_source_batch, channel_geometry_batch)

    # Output shapes
    print('pressure_drop shape:', pressure_drop_batch.shape)  # Expected: [batch_size, 1]
    print('temperature shape:', temperature_batch.shape)      # Expected: [batch_size, 1, D, H, W])

    

    # Print model summary
    summary(net)
"""