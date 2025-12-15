import h5py
import pandas as pd

# Open HDF5 file and count items
f = h5py.File('/workspaces/HeartBeatIrregularity/data/mel_128.h5', 'r')

# Count all datasets
count = 0
def count_datasets(name, obj):
    global count
    if isinstance(obj, h5py.Dataset):
        count += 1

f.visititems(count_datasets)
print(f'Total features in HDF5: {count}')

# Read label CSV
df = pd.read_csv('/workspaces/HeartBeatIrregularity/data/label.csv')
print(f'Total samples in CSV: {len(df)}')

# Check which files are missing
print('\nChecking for missing files...')
missing = []
for idx, filename in enumerate(df['filename'].values):
    try:
        data = f[filename]
    except KeyError:
        missing.append(filename)
        if len(missing) <= 5:
            print(f'Missing: {filename}')

print(f'\nTotal missing: {len(missing)}')

f.close()
