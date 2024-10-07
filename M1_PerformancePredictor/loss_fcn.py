## L1 approach:
import torch
import torch.nn as nn

class PerformanceCustomLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(PerformanceCustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pressure_loss_fn = nn.L1Loss()
    
    def forward(self, pressure_pred, temperature_pred, pressure_true, temperature_true):
        pressure_pred = pressure_pred.squeeze(1) 
        temperature_pred = temperature_pred.squeeze(1) 

        pressure_loss = self.pressure_loss_fn(pressure_pred, pressure_true)

        temperature_diff = torch.abs(temperature_pred - temperature_true)
        temperature_loss_per_sample = temperature_diff.sum(dim=[1, 2])

        temperature_loss = temperature_loss_per_sample.mean()
        total_loss = pressure_loss + temperature_loss
        return total_loss




""" MSE APPROACH
import torch
import torch.nn as nn

class PerformanceCustomLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(PerformanceCustomLoss, self).__init__()

    def forward(self, pressure_pred, temperature_pred, pressure_true, temperature_true):
        # Adjust shapes to match
        pressure_pred = pressure_pred.squeeze(1)  # Shape: [batch_size]
        temperature_pred = temperature_pred.squeeze(1)  # Shape: [batch_size, 101, 215]

        # Ensure pressure_true has the correct shape
        if pressure_true.dim() == 1:
            pressure_true = pressure_true  # Shape: [batch_size]
        else:
            pressure_true = pressure_true.squeeze()  # Shape: [batch_size]

        # Compute MSE between pressure predictions and true values
        pressure_loss = nn.MSELoss()(pressure_pred, pressure_true)

        # Compute element-wise squared differences for temperature
        temperature_diff = temperature_pred - temperature_true  # Shape: [batch_size, 101, 215]
        temperature_squared_diff = temperature_diff ** 2  # Element-wise squared differences

        # Sum over the temperature matrix dimensions (1 and 2)
        temperature_loss_per_sample = temperature_squared_diff.sum(dim=[1, 2])  # Shape: [batch_size]

        # Compute the mean temperature loss over the batch
        temperature_loss = temperature_loss_per_sample.mean()

        # Total loss is the sum of pressure loss and temperature loss
        total_loss = pressure_loss + temperature_loss

        return total_loss
"""