import torch

train_data = torch.load('Data/M1_training_data.pt')
test_data = torch.load('Data/M1_testing_data.pt')

def print_dataset_info(data, dataset_name):
    print(f"\n{dataset_name}")
    print("-" * len(dataset_name))
    print(f"Keys in the dataset: {list(data.keys())}")

    num_samples = data['heat_source'].shape[0]
    print(f"Number of samples: {num_samples}\n")

    for key, tensor in data.items():
        print(f"Variable: '{key}'")
        print(f"  Shape: {tensor.shape}")
        print(f"  Data type: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print()

print_dataset_info(train_data, "Training data information")
print_dataset_info(test_data, "Testing data information")
