import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from architecture import HeatChannelNet

model_net = HeatChannelNet()
model_net.load_state_dict(torch.load('M1_performance_predictor.pth'))
model_net.eval()

test_data = torch.load('Data/M1_testing_data.pt')

sample_index = 0
heat_source_sample = test_data['heat_source'][sample_index].unsqueeze(0).unsqueeze(0)
channel_geometry_sample = test_data['channel_geometry'][sample_index].unsqueeze(0).unsqueeze(0)
inlet_velocity_sample = test_data['inlet_velocity'][sample_index].unsqueeze(0).unsqueeze(0)

pressure_drop_true = test_data['pressure_drop'][sample_index]
temperature_true = test_data['temperature'][sample_index]

with torch.no_grad():
    pressure_drop_pred, temperature_pred = model_net(
        heat_source_sample, channel_geometry_sample, inlet_velocity_sample
    )

print('Predicted pressure drop:', pressure_drop_pred.item())
print('True pressure drop:', pressure_drop_true.item())

temperature_pred_np = temperature_pred.squeeze().cpu().numpy()
temperature_true_np = temperature_true.cpu().numpy()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(temperature_true_np, cmap='hot', interpolation='nearest', vmin=260, vmax=370)
plt.title('True temperature')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(temperature_pred_np, cmap='hot', interpolation='nearest', vmin=260, vmax=370)
plt.title('Predicted temperature')
plt.colorbar()

plt.tight_layout()
plt.show()
