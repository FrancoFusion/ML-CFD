import torch.nn as nn
import torch.nn.functional as F

class CoolingNetwork(nn.Module):
    def __init__(self):
        super(CoolingNetwork, self).__init__()
        # Parameters
        self.input_channels = 2               # Number of input channels (cooling channel, heat source)
        self.matrix_width = 50                # Width of the input/output matrix
        self.matrix_height = 50               # Height of the input/output matrix
        
        self.conv1_filters = 16                            # Number of filters in Conv1
        self.conv2_filters = 32                            # Number of filters in Conv2
        self.conv3_filters = 64                            # Number of filters in Conv3
        self.kernel_size = 3                               # Kernel size for all convolutional layers
        self.fc1_neurons = 2048                            # Number of neurons in fully connected layer 1
        self.fc2_neurons = 1024                            # Number of neurons in fully connected layer 2
        self.output_temp_neurons = self.matrix_width * self.matrix_height  # Output neurons for the temperature map
        self.output_pressure_neuron = 1                    # Single neuron for the pressure drop output
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, self.conv1_filters, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv2d(self.conv1_filters, self.conv2_filters, kernel_size=self.kernel_size)
        self.conv3 = nn.Conv2d(self.conv2_filters, self.conv3_filters, kernel_size=self.kernel_size)
        
        # Fully connected layers to produce temperature map and pressure drop
        self.fc1 = nn.Linear(self.conv3_filters * self.matrix_width * self.matrix_height, self.fc1_neurons)
        self.fc2 = nn.Linear(self.fc1_neurons, self.fc2_neurons)

        self.fc3_temp = nn.Linear(self.fc2_neurons, self.output_temp_neurons)           # Output1: Temperature map
        self.fc3_pressure = nn.Linear(self.fc2_neurons, self.output_pressure_neuron)    # Output2: Pressure drop
    
    def forward(self, x):
        # Apply convolutions with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten feature maps for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Split the outputs
        temperature_map = self.fc3_temp(x)
        temperature_map = temperature_map.view(-1, self.matrix_width, self.matrix_height)
        pressure_drop = self.fc3_pressure(x)
        
        return temperature_map, pressure_drop
