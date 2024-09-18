import torch.nn as nn

class PerformanceCustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(PerformanceCustomLoss, self).__init__()
        self.alpha = alpha  # Weight for pressure drop loss
        self.beta = beta    # Weight for temperature loss
        self.pressure_loss_fn = nn.MSELoss()
        self.temperature_loss_fn = nn.MSELoss()

    def forward(self, pressure_pred, temperature_pred, pressure_true, temperature_true):
        # Pressure drop loss
        pressure_loss = self.pressure_loss_fn(pressure_pred, pressure_true)
        
        # Temperature loss
        temperature_loss = self.temperature_loss_fn(temperature_pred, temperature_true)
        
        # Combine losses
        total_loss = self.alpha * pressure_loss + self.beta * temperature_loss
        return total_loss
