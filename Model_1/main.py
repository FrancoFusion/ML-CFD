import torch
from torch.utils.data import TensorDataset, DataLoader
from train_test import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    data = torch.load(file_path)
    cooling_channels = data['cooling_channels']
    heating_sources = data['heating_sources']
    true_temps = data['true_temps']
    true_pressures = data['true_pressures']
    
    dataset = TensorDataset(cooling_channels, heating_sources, true_temps, true_pressures)
    return dataset

# Load datasets
train_dataset = load_data('train_data.pt')
test_dataset = load_data('test_data.pt')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Train/test loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(train_loader)
    test_loss = test(test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {train_loss}, Test loss: {test_loss}")
