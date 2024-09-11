# main.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from network import CoolingNetwork
from loss import CustomLoss
from train_test import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load saved datasets
def load_data(file_path):
    data = torch.load(file_path)
    cooling_channels = data['cooling_channels']
    heating_sources = data['heating_sources']
    true_temps = data['true_temps']
    true_pressures = data['true_pressures']
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(cooling_channels, heating_sources, true_temps, true_pressures)
    return dataset

# Load the datasets
train_dataset = load_data('train_data.pt')
test_dataset = load_data('test_data.pt')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(train_loader)
    test_loss = test(test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")
