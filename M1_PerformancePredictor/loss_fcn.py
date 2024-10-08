import torch
import torch.nn as nn

class PerformanceCustomLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(PerformanceCustomLoss, self).__init__()

    def forward(self, pressure_pred, temperature_pred, pressure_true, temperature_true):
        pressure_pred = pressure_pred.squeeze(1)
        temperature_pred = temperature_pred.squeeze(1)

        pressure_loss = nn.MSELoss()(pressure_pred, pressure_true)

        temperature_diff = temperature_pred - temperature_true
        temperature_squared_diff = temperature_diff ** 2
        temperature_loss_per_sample = temperature_squared_diff.sum(dim=[1, 2])

        temperature_loss = temperature_loss_per_sample.mean()
        total_loss = pressure_loss + temperature_loss
        return total_loss