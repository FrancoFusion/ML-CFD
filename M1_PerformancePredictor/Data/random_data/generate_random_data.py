import torch

num_train_samples = 80
num_test_samples = 20
matrix_size = (50, 50)

# Train data
heat_source_train = torch.randn(num_train_samples, *matrix_size)
channel_geometry_train = torch.randn(num_train_samples, *matrix_size)
pressure_drop_train = torch.randn(num_train_samples)
temperature_train = torch.randn(num_train_samples, *matrix_size)

# Test data
heat_source_test = torch.randn(num_test_samples, *matrix_size)
channel_geometry_test = torch.randn(num_test_samples, *matrix_size)
pressure_drop_test = torch.randn(num_test_samples)
temperature_test = torch.randn(num_test_samples, *matrix_size)

# Save train data
torch.save({
    'heat_source': heat_source_train,
    'channel_geometry': channel_geometry_train,
    'pressure_drop': pressure_drop_train,
    'temperature': temperature_train
}, 'Data/M1_training_data.pt')

# Save test data
torch.save({
    'heat_source': heat_source_test,
    'channel_geometry': channel_geometry_test,
    'pressure_drop': pressure_drop_test,
    'temperature': temperature_test
}, 'Data/M1_testing_data.pt')
