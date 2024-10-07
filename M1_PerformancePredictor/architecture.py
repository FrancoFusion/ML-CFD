import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatChannelNet(nn.Module):
    def __init__(self):
        super(HeatChannelNet, self).__init__()
        """ Shared convolutional layers """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(80)

        """ Pressure Drop Branch """
        self.fc1 = nn.Linear(80 * 50 * 107, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)

        """ Temperature Branch """
        self.convT1 = nn.ConvTranspose2d(80, 64, kernel_size=2, stride=2, output_padding=1)
        self.convT2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convT3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, heat_source, channel_geometry, inlet_velocity):
        """ Input concatenation """
        x = torch.cat((heat_source, channel_geometry, inlet_velocity), dim=1)
        
        """ Shared convolutional layers """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x)))

        """ Pressure Drop Branch """
        x_flat = x.view(x.size(0), -1)
        pd = F.relu(self.fc1(x_flat))
        pd = F.relu(self.fc2(pd))
        pd = self.fc3(pd)
        pd = torch.clamp(pd, min=0, max=2000)  # Ensuring pressure drop is between 0 and 2000

        """ Temperature Branch """
        temp = F.relu(self.convT1(x))
        temp = F.relu(self.convT2(temp))
        temp = F.relu(self.convT3(temp))
        temp = self.conv_final(temp)
        temp = torch.clamp(temp, min=298, max=600)  # Ensuring temperature is between 0 and 600

        # Set corners to 0
        temp[:, :, 0:44, 0:17] = 0          # Top-left corner
        temp[:, :, 57:101, 0:17] = 0        # Bottom-left corner
        temp[:, :, 0:44, 198:215] = 0       # Top-right corner
        temp[:, :, 57:101, 198:215] = 0     # Bottom-right corner

        return pd, temp

"""
if __name__ == "__main__":
    model = HeatChannelNet()
    model.eval()

    # Create dummy input data with the correct dimensions
    batch_size = 1
    heat_source_sample = torch.randn(batch_size, 1, 101, 215)
    channel_geometry_sample = torch.randn(batch_size, 1, 101, 215)
    inlet_velocity_sample = torch.randn(batch_size, 1, 101, 215)

    # Forward pass
    pressure_drop_pred, temperature_pred = model(heat_source_sample, channel_geometry_sample, inlet_velocity_sample)
    
    # Print output shapes and values
    print('Pressure drop prediction shape:', pressure_drop_pred.shape)  # Must be torch.Size([1, 1])
    print('Temperature prediction shape:', temperature_pred.shape)      # Must be torch.Size([1, 1, 101, 215])
"""