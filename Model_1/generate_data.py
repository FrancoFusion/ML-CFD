"""
Temporary random data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class RandomCoolingDataset(Dataset):
    def __init__(self, num_samples=1000, size=(50, 50)):
        self.num_samples = num_samples
        self.size = size

        self.cooling_channels = []
        self.heating_sources = []
        self.true_temps = []
        self.true_pressures = []

        for _ in range(self.num_samples):
            cooling_channel = self.generate_cooling_channel()   # Random cooling channel geometry
            heating_source = np.random.rand(*self.size)  # Random heating source
            true_temp = np.random.rand(*self.size)        # Random ground truth temperature
            true_pressure = np.random.rand(1)            # Random pressure drop

            self.cooling_channels.append(cooling_channel)
            self.heating_sources.append(heating_source)
            self.true_temps.append(true_temp)
            self.true_pressures.append(true_pressure)

        # Lists to tensors
        self.cooling_channels = torch.tensor(self.cooling_channels, dtype=torch.float32)
        self.heating_sources = torch.tensor(self.heating_sources, dtype=torch.float32)
        self.true_temps = torch.tensor(self.true_temps, dtype=torch.float32)
        self.true_pressures = torch.tensor(self.true_pressures, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.cooling_channels[idx],
            self.heating_sources[idx],
            self.true_temps[idx],
            self.true_pressures[idx]
        )

    def generate_cooling_channel(self):
        """Cooling channel geometry design."""
        cooling_channel = np.zeros(self.size)
        
        input_pos = self.get_random_boundary_position()
        output_pos = self.get_random_boundary_position()

        while input_pos == output_pos:
            output_pos = self.get_random_boundary_position()

        cooling_channel[input_pos] = -1  # Input point
        cooling_channel[output_pos] = -2  # Output point

        # Fill rest of matrix with random values where the channel does not exist
        cooling_channel += np.random.rand(*self.size) * (cooling_channel == 0)
        return cooling_channel

    def get_random_boundary_position(self):
        """Select a random position on the boundary of the matrix."""
        boundary_choices = []

        # Top row (excluding corners, handled below)
        boundary_choices.extend([(0, i) for i in range(1, self.size[1] - 1)])

        # Bottom row (excluding corners)
        boundary_choices.extend([(self.size[0] - 1, i) for i in range(1, self.size[1] - 1)])

        # Left column
        boundary_choices.extend([(i, 0) for i in range(1, self.size[0] - 1)])

        # Right column
        boundary_choices.extend([(i, self.size[1] - 1) for i in range(1, self.size[0] - 1)])

        # Corners
        boundary_choices.extend([
            (0, 0),                               # Top-left corner
            (0, self.size[1] - 1),                # Top-right corner
            (self.size[0] - 1, 0),                # Bottom-left corner
            (self.size[0] - 1, self.size[1] - 1)  # Bottom-right corner
        ])

        return boundary_choices[np.random.randint(len(boundary_choices))]

def save_datasets(train_samples=200, test_samples=40, size=(50, 50)):
    train_dataset = RandomCoolingDataset(num_samples=train_samples, size=size)
    test_dataset = RandomCoolingDataset(num_samples=test_samples, size=size)

    # Save as .pt files
    
    torch.save({
        'cooling_channels': train_dataset.cooling_channels,
        'heating_sources': train_dataset.heating_sources,
        'true_temps': train_dataset.true_temps,
        'true_pressures': train_dataset.true_pressures
    }, 'train_data.pt')

    torch.save({
        'cooling_channels': test_dataset.cooling_channels,
        'heating_sources': test_dataset.heating_sources,
        'true_temps': test_dataset.true_temps,
        'true_pressures': test_dataset.true_pressures
    }, 'test_data.pt')

    print("Train and test datasets saved!")


save_datasets(train_samples=200, test_samples=40)
