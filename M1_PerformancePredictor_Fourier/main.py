import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from architecture import HeatChannelNet
from loss_fcn import PerformanceCustomLoss
import time
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Training parameters
LEARNING_RATE = 0.00005
NUM_EPOCHS = 150
BATCH_SIZE = 16

# Initialize model, loss function, optimizer
model_net = HeatChannelNet().to(device)
loss_fcn = PerformanceCustomLoss(alpha=1, beta=1)
optimizer = optim.Adam(model_net.parameters(), lr=LEARNING_RATE)

# Custom Dataset class
class HeatChannelDataset(Dataset):
    def __init__(self, data):
        self.heat_source = data['heat_source']
        self.channel_geometry = data['channel_geometry']
        self.inlet_velocity = data['inlet_velocity']
        self.pressure_drop = data['pressure_drop']
        self.temperature = data['temperature']

    def __len__(self):
        return len(self.pressure_drop)

    def __getitem__(self, idx):
        return {
            'heat_source': self.heat_source[idx].unsqueeze(0),
            'channel_geometry': self.channel_geometry[idx].unsqueeze(0),
            'inlet_velocity': self.inlet_velocity[idx].unsqueeze(0),
            'pressure_drop': self.pressure_drop[idx],
            'temperature': self.temperature[idx]
        }

# Load data
train_data = torch.load('M1_training_data.pt')
test_data = torch.load('M1_testing_data.pt')

# Create Dataset and DataLoader
train_dataset = HeatChannelDataset(train_data)
test_dataset = HeatChannelDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop
def train(model_net, train_loader, loss_fcn, optimizer):
    model_net.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        heat_source = batch['heat_source'].to(device)
        channel_geometry = batch['channel_geometry'].to(device)
        inlet_velocity = batch['inlet_velocity'].to(device)
        pressure_true = batch['pressure_drop'].to(device)
        temperature_true = batch['temperature'].to(device)

        optimizer.zero_grad()

        # Forward pass with all three inputs
        pressure_pred, temperature_pred = model_net(heat_source, channel_geometry, inlet_velocity)
        loss = loss_fcn(pressure_pred, temperature_pred, pressure_true, temperature_true)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss

# Test loop
def test(model_net, test_loader, loss_fcn):
    model_net.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            heat_source = batch['heat_source'].to(device)
            channel_geometry = batch['channel_geometry'].to(device)
            inlet_velocity = batch['inlet_velocity'].to(device)
            pressure_true = batch['pressure_drop'].to(device)
            temperature_true = batch['temperature'].to(device)

            # Forward pass with all three inputs
            pressure_pred, temperature_pred = model_net(heat_source, channel_geometry, inlet_velocity)

            loss = loss_fcn(pressure_pred, temperature_pred, pressure_true, temperature_true)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss

# Training
start_time = time.time()
train_losses = []
test_losses = []

for epoch in range(NUM_EPOCHS):
    train_loss = train(model_net, train_loader, loss_fcn, optimizer)
    test_loss = test(model_net, test_loader, loss_fcn)
    print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
    train_losses.append(train_loss)
    test_losses.append(test_loss)

end_time = time.time()
total_training_time = end_time - start_time

with open('losses.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Epoch', 'Train loss', 'Test loss'])
    for epoch in range(NUM_EPOCHS):
        csvwriter.writerow([epoch + 1, train_losses[epoch], test_losses[epoch]])

with open('training_time.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Total training time (seconds)'])
    csvwriter.writerow([total_training_time])

torch.save(model_net.state_dict(), 'M1_performance_predictor.pth')
