import os

os.chdir('Data/400Samples')

filenames = os.listdir('.')

sample_dict = {}
for filename in filenames:
    if not filename.endswith('.csv'):
        continue
    parts = filename.rstrip('.csv').split('_')
    if len(parts) != 5:
        print(f"Filename {filename} does not have the expected format.")
        continue
    g_number = parts[1]  # e.g., 'g1'
    num1 = parts[3]      # e.g., '200'
    num2 = parts[4]      # e.g., '0.1'
    sample_key = (g_number, num1, num2)
    if sample_key not in sample_dict:
        sample_dict[sample_key] = []
    sample_dict[sample_key].append(filename)

sample_keys = sorted(sample_dict.keys())
sample_number = 1

tmp_mapping = {}
for sample_key in sample_keys:
    sample_files = sample_dict[sample_key]
    new_number = f"{sample_number:04d}"
    for filename in sample_files:
        parts = filename.rstrip('.csv').split('_')
        parts[0] = new_number
        new_filename = '_'.join(parts) + '.csv'
        tmp_filename = 'tmp_' + filename
        tmp_mapping[filename] = (tmp_filename, new_filename)
    sample_number += 1

for old_name, (tmp_name, _) in tmp_mapping.items():
    print(f"Renaming {old_name} to {tmp_name}")
    os.rename(old_name, tmp_name)

for _, (tmp_name, new_name) in tmp_mapping.items():
    print(f"Renaming {tmp_name} to {new_name}")
    os.rename(tmp_name, new_name)
