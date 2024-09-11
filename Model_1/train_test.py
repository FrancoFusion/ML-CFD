import torch
from network import CoolingNetwork
from loss import CustomLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CoolingNetwork().to(device)
loss_fn = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(train_data_loader):
    model.train()
    running_loss = 0.0
    
    for batch in train_data_loader:
        # Move inputs and outputs to the device
        cooling_channel, heating_source, true_temp, true_pressure = [t.to(device) for t in batch]
        
        # Concatenate the two input channels
        inputs = torch.cat([cooling_channel.unsqueeze(1), heating_source.unsqueeze(1)], dim=1)
        
        optimizer.zero_grad()
        
        pred_temp, pred_pressure = model(inputs)
        
        loss = loss_fn(pred_temp, pred_pressure, true_temp, true_pressure)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    return running_loss / len(train_data_loader)

def test(test_data_loader):
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_data_loader:
            cooling_channel, heating_source, true_temp, true_pressure = [t.to(device) for t in batch]
            inputs = torch.cat([cooling_channel.unsqueeze(1), heating_source.unsqueeze(1)], dim=1)
            pred_temp, pred_pressure = model(inputs)
            
            loss = loss_fn(pred_temp, pred_pressure, true_temp, true_pressure)
            test_loss += loss.item()
    
    return test_loss / len(test_data_loader)
