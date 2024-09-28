import torch
import torch.nn as nn

class PerformanceCustomLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(PerformanceCustomLoss, self).__init__()
        self.alpha = alpha  # Weight for pressure drop loss
        self.beta = beta    # Weight for temperature loss

    def forward(self, pressure_pred, temperature_pred, pressure_true, temperature_true):
        epsilon = 1e-8

        # Compute pressure loss: Lp = |p'-p| / (|p'|+|p|)
        Lp = (torch.abs(pressure_pred - pressure_true) / 
              (torch.abs(pressure_pred) + torch.abs(pressure_true) + epsilon)).squeeze(1)

        # Compute temperature loss using frobenius norm: LT = ||T'-T||_F / (||T'||_F+||T||_F)
        batch_size = temperature_pred.size(0)
        T_pred_flat = temperature_pred.view(batch_size, -1)
        T_true_flat = temperature_true.view(batch_size, -1)
        LT = torch.norm(T_pred_flat - T_true_flat, p='fro', dim=1) / (
             torch.norm(T_pred_flat, p='fro', dim=1) + torch.norm(T_true_flat, p='fro', dim=1) + epsilon)

        # L = alpha*Lp + beta*LT, mean for batch
        loss = (self.alpha * Lp + self.beta * LT).mean()

        return loss
