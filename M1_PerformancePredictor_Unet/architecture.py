import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatChannelNet(nn.Module):
    def __init__(self):
        super(HeatChannelNet, self).__init__()
        """ Encoder """
        # Level 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Level 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Level 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Level 4 (Bottleneck)
        self.conv7 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        """ Pressure Drop Branch """
        self.fc1 = nn.Linear(512 * 13 * 27, 2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 1)

        """ Decoder """
        # Level 4
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.bn_up3 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)

        # Level 3
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn_up2 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(128)

        # Level 2
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=(1,1))
        self.bn_up1 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)

        # Final Convolution
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, heat_source, channel_geometry, inlet_velocity):
        x = torch.cat((heat_source, channel_geometry, inlet_velocity), dim=1)

        """ Encoder """
        # Level 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1_pooled = self.pool1(x1)

        # Level 2
        x2 = F.relu(self.bn3(self.conv3(x1_pooled)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        x2_pooled = self.pool2(x2)

        # Level 3
        x3 = F.relu(self.bn5(self.conv5(x2_pooled)))
        x3 = F.relu(self.bn6(self.conv6(x3)))
        x3_pooled = self.pool3(x3)

        # Level 4 (Bottleneck)
        x4 = F.relu(self.bn7(self.conv7(x3_pooled)))
        x4 = F.relu(self.bn8(self.conv8(x4)))

        """ Pressure Drop Branch """
        x_flat = x4.view(x4.size(0), -1)
        pd = F.relu(self.fc1(x_flat))
        pd = self.dropout1(pd)
        pd = F.relu(self.fc2(pd))
        pd = self.dropout2(pd)
        pd = self.fc3(pd)
        pd = torch.relu(pd)

        """ Decoder """
        # Level 4
        x = F.relu(self.bn_up3(self.upconv3(x4)))

        x3_padded = F.pad(x3, (0, 0, 0, 1))

        x = torch.cat([x, x3_padded], dim=1)
        x = F.relu(self.bn9(self.conv9(x)))

        # Level 3
        x = F.relu(self.bn_up2(self.upconv2(x)))

        x2_padded = F.pad(x2, (0, 1, 0, 2))

        x = torch.cat([x, x2_padded], dim=1)
        x = F.relu(self.bn10(self.conv10(x)))

        # Level 2
        x = F.relu(self.bn_up1(self.upconv1(x)))

        x1_padded = F.pad(x1, (1, 1, 2, 2))

        x = torch.cat([x, x1_padded], dim=1)
        x = F.relu(self.bn11(self.conv11(x)))

        # Final Convolution
        temp = self.conv_final(x)
        temp = F.interpolate(temp, size=(101, 215), mode='bilinear', align_corners=False)
        temp = torch.sigmoid(temp)

        return pd, temp

"""
if __name__ == "__main__":
    model = HeatChannelNet()
    model.eval()

    batch_size = 1
    heat_source_sample = torch.randn(batch_size, 1, 101, 215)
    channel_geometry_sample = torch.randn(batch_size, 1, 101, 215)
    inlet_velocity_sample = torch.randn(batch_size, 1, 101, 215)

    pressure_drop_pred, temperature_pred = model(heat_source_sample, channel_geometry_sample, inlet_velocity_sample)

    print('Pressure drop prediction shape:', pressure_drop_pred.shape)  # Should be [1, 1]
    print('Temperature prediction shape:', temperature_pred.shape)      # Should be [1, 1, 101, 215])
"""