import torch

def generate_sample():
    # Define dimensions
    D, H, W = 30, 30, 30  # Depth, Height, Width

    # Random input tensors with matching dimensions
    heat_source = torch.randn(D, H, W)
    channel_geometry = torch.randn(D, H, W)

    # Random outputs
    pressure_drop = torch.randn(1)
    temperature = torch.randn(D, H, W)

    # Package sample into dictionary
    sample = {
        'heat_source': heat_source,
        'channel_geometry': channel_geometry,
        'pressure_drop': pressure_drop,
        'temperature': temperature
    }

    return sample

def generate_dataset(num_samples):
    return [generate_sample() for _ in range(num_samples)]

if __name__ == '__main__':
    train_samples = 80
    test_samples = 20

    train_data = generate_dataset(train_samples)
    test_data = generate_dataset(test_samples)

    torch.save(train_data, 'Data/M1_train_data.pt')
    torch.save(test_data, 'Data/M1_test_data.pt')
