import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_temp, pred_pressure, true_temp, true_pressure):
        temp_loss = self.mse_loss(pred_temp, true_temp)
        
        pressure_loss = self.mse_loss(pred_pressure, true_pressure)
        
        total_loss = temp_loss + pressure_loss
        
        return total_loss
