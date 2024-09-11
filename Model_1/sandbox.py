import torch
import matplotlib.pyplot as plt

data = torch.load('test_data.pt')

cooling_channels = data['cooling_channels']
heating_sources = data['heating_sources']
true_temps = data['true_temps']
true_pressures = data['true_pressures']

print(f"Cooling channels shape: {cooling_channels.shape}")
print(f"Heating sources shape: {heating_sources.shape}")
print(f"True temperatures shape: {true_temps.shape}")
print(f"True pressures shape: {true_pressures.shape}")

index_to_plot = 27
cooling_channel_matrix = cooling_channels[index_to_plot].numpy()

plt.imshow(cooling_channel_matrix, cmap='hot', interpolation='nearest')
plt.show()
