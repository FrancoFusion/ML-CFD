import torch

def generate_sample():
    
    # Random input
    heat_source = torch.randn(30, 30)
    channel_geometry = torch.randn(30, 30, 30)

    # Random output
    pressure_drop = torch.randn(1)
    temperature = torch.randn(30, 30, 30)

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