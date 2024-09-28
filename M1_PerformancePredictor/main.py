import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from architecture import HeatChannelNet
from loss_fcn import PerformanceCustomLoss


# Training parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 2
BATCH_SIZE = 8

model_net = HeatChannelNet()
loss_fcn = PerformanceCustomLoss(alpha=0.5, beta=0.5)
optimizer = optim.Adam(model_net.parameters(), lr=LEARNING_RATE)


# Custom Dataset class
class HeatChannelDataset(Dataset):
    def __init__(self, data):
        self.heat_source = data['heat_source']
        self.channel_geometry = data['channel_geometry']
        self.pressure_drop = data['pressure_drop']
        self.temperature = data['temperature']

    def __len__(self):
        return len(self.pressure_drop)

    def __getitem__(self, idx):
        return {'heat_source': self.heat_source[idx].unsqueeze(0),
                'channel_geometry': self.channel_geometry[idx].unsqueeze(0),
                'pressure_drop': self.pressure_drop[idx].unsqueeze(0),
                'temperature': self.temperature[idx].unsqueeze(0)}


# Load data
train_data = torch.load('Data/random_data/M1_training_data.pt')
test_data = torch.load('Data/random_data/M1_testing_data.pt')

# Create Dataset and DataLoader
train_dataset = HeatChannelDataset(train_data)
test_dataset = HeatChannelDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Training loop
def train(model_net, train_loader, loss_fcn, optimizer):
    model_net.train()  # Set the model_net to training mode
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        # Get data
        heat_source = batch['heat_source']
        channel_geometry = batch['channel_geometry']
        pressure_true = batch['pressure_drop']
        temperature_true = batch['temperature']

        optimizer.zero_grad()

        pressure_pred, temperature_pred = model_net(heat_source, channel_geometry)
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
            heat_source = batch['heat_source']
            channel_geometry = batch['channel_geometry']
            pressure_true = batch['pressure_drop']
            temperature_true = batch['temperature']

            pressure_pred, temperature_pred = model_net(heat_source, channel_geometry)

            loss = loss_fcn(pressure_pred, temperature_pred, pressure_true, temperature_true)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss


for epoch in range(NUM_EPOCHS):
    train_loss = train(model_net, train_loader, loss_fcn, optimizer)

    test_loss = test(model_net, test_loader, loss_fcn)

    print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
