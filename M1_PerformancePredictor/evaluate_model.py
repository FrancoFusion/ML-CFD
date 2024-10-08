import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from architecture import HeatChannelNet
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

color_stops = [
    (0.0,   (28/255, 29/255, 138/255)),
    (0.1355, (17/255, 67/255, 130/255)),
    (0.29,  (39/255, 118/255, 196/255)),
    (0.41,  (106/255, 186/255, 210/255)),
    (0.45,  (170/255, 213/255, 108/255)),
    (0.5,   (252/255, 253/255, 88/255)),
    (0.76,  (255/255, 128/255, 65/255)),
    (0.9,   (240/255, 60/255, 37/255)),
    (1.0,   (136/255, 46/255, 16/255))
]
cmap_name = 'custom_blue_yellow_red'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, color_stops)
custom_cmap.set_bad(color='black')
test_data = torch.load('Data/M1_testing_data.pt')
sample_index = 2

model_net = HeatChannelNet()
model_net.load_state_dict(torch.load('M1_performance_predictor.pth'))
model_net.eval()

heat_source_sample = test_data['heat_source'][sample_index].unsqueeze(0).unsqueeze(0)
channel_geometry_sample = test_data['channel_geometry'][sample_index].unsqueeze(0).unsqueeze(0)
inlet_velocity_sample = test_data['inlet_velocity'][sample_index].unsqueeze(0).unsqueeze(0)
pressure_drop_true = test_data['pressure_drop'][sample_index]
temperature_true = test_data['temperature'][sample_index]

with torch.no_grad():
    pressure_drop_pred, temperature_pred = model_net(heat_source_sample, channel_geometry_sample, inlet_velocity_sample)

print('Predicted pressure drop:', pressure_drop_pred.item())
print('True pressure drop:', pressure_drop_true.item())

temperature_pred_np = temperature_pred.squeeze().cpu().numpy()
temperature_true_np = temperature_true.cpu().numpy()
temperature_pred_masked = np.ma.masked_where(temperature_pred_np == 0, temperature_pred_np)
temperature_true_masked = np.ma.masked_where(temperature_true_np == 0, temperature_true_np)
combined_temps = np.concatenate((temperature_true_masked.compressed(), temperature_pred_masked.compressed()))
global_min = combined_temps.min()
global_max = combined_temps.max()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
im1 = plt.imshow(temperature_true_masked, cmap=custom_cmap, interpolation='nearest', vmin=global_min)
plt.title('True Temperature')
plt.colorbar(im1, fraction=0.046, pad=0.04)
plt.subplot(1, 2, 2)
im2 = plt.imshow(temperature_pred_masked, cmap=custom_cmap, interpolation='nearest', vmin=global_min)
plt.title('Predicted Temperature')
plt.colorbar(im2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 6))
channel_geometry_np = channel_geometry_sample.squeeze().cpu().numpy()
plt.imshow(channel_geometry_np, interpolation='nearest')
plt.title('Channel Geometry')
plt.colorbar()
plt.show()
