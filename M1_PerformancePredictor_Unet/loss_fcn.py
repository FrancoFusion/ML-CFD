import torch
import torch.nn as nn

class PerformanceCustomLoss(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(PerformanceCustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, pressure_pred, temperature_pred, pressure_true, temperature_true):
        pressure_pred = pressure_pred.squeeze(1)
        temperature_pred = temperature_pred.squeeze(1)

        pressure_loss = self.mse_loss(pressure_pred, pressure_true)

        temperature_loss = self.mse_loss(temperature_pred, temperature_true)

        total_loss = self.alpha * pressure_loss + self.beta * temperature_loss
        return total_loss
    

""" or:
import torch
import torch.nn as nn

class PerformanceCustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(PerformanceCustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
    
    def total_variation_loss(self, x):
        # Compute differences between adjacent pixels in height and width
        tv_h = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
        return tv_h + tv_w

    def forward(self, pressure_pred, temperature_pred, pressure_true, temperature_true):
        # Squeeze the channel dimension for pressure and temperature
        pressure_pred = pressure_pred.squeeze(1)  # Shape: [batch_size]
        temperature_pred = temperature_pred.squeeze(1)  # Shape: [batch_size, H, W]
        
        # Compute MSE loss for pressure
        pressure_loss = self.mse_loss(pressure_pred, pressure_true)
        
        # Compute MSE loss for temperature
        temperature_loss = self.mse_loss(temperature_pred, temperature_true)
        
        # Compute Total Variation loss for temperature_pred to enforce smoothness
        tv_loss = self.total_variation_loss(temperature_pred)
        
        # Combine all losses with respective weights
        total_loss = self.alpha * pressure_loss + self.beta * temperature_loss + 0.1 * tv_loss
        
        return total_loss

"""
