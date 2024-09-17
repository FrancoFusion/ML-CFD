import torch

def inspect_dataset(file_path):
    data = torch.load(file_path)

    num_samples = len(data)
    print(f'\n\n\nNumber of samples in {file_path}: {num_samples}\n')

    sample = data[0]
    print('Fields in each sample:')
    for key in sample.keys():
        print(f' - {key}')

    print('\nTensor sizes in the first sample:')
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f'{key}: {tuple(value.size())}')
        else:
            print(f'{key}: Not a tensor, type is {type(value)}')

if __name__ == '__main__':
    inspect_dataset('Data/M1_train_data.pt')
    inspect_dataset('Data/M1_test_data.pt')
