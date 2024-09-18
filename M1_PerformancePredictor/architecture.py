import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatChannelNet(nn.Module):
    def __init__(self):
        super(HeatChannelNet, self).__init__()

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)  # Output: (16, 50, 50)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: (32, 50, 50)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, 25, 25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: (64, 25, 25)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: (128, 25, 25)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (128, 12, 12)

        # Pressure Drop Branch (Fully Connected Layers)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)  # Scalar output

        # Temperature Branch (Convolutional and Upsampling Layers)
        self.conv_temp1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)  # Output: (64, 25, 25)
        self.conv_temp2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)   # Output: (32, 50, 50)
        self.conv_temp3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)    # Output: (1, 50, 50)

    def forward(self, heat_source, channel_geometry):
        # Concatenate inputs along the channel dimension
        x = torch.cat((heat_source, channel_geometry), dim=1)  # Shape: (batch_size, 2, 50, 50)

        # Shared convolutional layers
        x = F.relu(self.conv1(x))    # Shape: (batch_size, 16, 50, 50)
        x = F.relu(self.conv2(x))    # Shape: (batch_size, 32, 50, 50)
        x = self.pool1(x)            # Shape: (batch_size, 32, 25, 25)
        x = F.relu(self.conv3(x))    # Shape: (batch_size, 64, 25, 25)
        x = F.relu(self.conv4(x))    # Shape: (batch_size, 128, 25, 25)
        x = self.pool2(x)            # Shape: (batch_size, 128, 12, 12)

        # Pressure Drop Branch
        x_flat = x.view(x.size(0), -1)        # Flatten: (batch_size, 128*12*12)
        pd = F.relu(self.fc1(x_flat))         # Fully connected layers
        pd = F.relu(self.fc2(pd))
        pd = self.fc3(pd)                     # Output: (batch_size, 1)

        # Temperature Branch
        temp = F.interpolate(x, size=(25, 25), mode='bilinear', align_corners=False)  # Upsample to (25, 25)
        temp = F.relu(self.conv_temp1(temp))   # Shape: (batch_size, 64, 25, 25)
        temp = F.interpolate(temp, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to (50, 50)
        temp = F.relu(self.conv_temp2(temp))   # Shape: (batch_size, 32, 50, 50)
        temp = self.conv_temp3(temp)           # Output: (batch_size, 1, 50, 50)

        return pd, temp

if __name__ == "__main__":
    model = HeatChannelNet()

    # One sample
    test_data = torch.load('Data/M1_testing_data.pt')
    heat_source_sample = test_data['heat_source'][0].unsqueeze(0).unsqueeze(0)              # Shape: (1, 1, 50, 50)
    channel_geometry_sample = test_data['channel_geometry'][0].unsqueeze(0).unsqueeze(0)    # Shape: (1, 1, 50, 50)

    # Forward pass
    pressure_drop_pred, temperature_pred = model(heat_source_sample, channel_geometry_sample)

    # Print output shapes
    print('Pressure drop prediction shape:', pressure_drop_pred.shape)
    print('Temperature prediction shape:', temperature_pred.shape)
