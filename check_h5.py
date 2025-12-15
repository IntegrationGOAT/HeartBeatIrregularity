import h5py
import pandas as pd

# Open HDF5 file
f = h5py.File('/workspaces/HeartBeatIrregularity/data/mel_128.h5', 'r')

# Read label CSV
df = pd.read_csv('/workspaces/HeartBeatIrregularity/data/label.csv')
filename = df['filename'].iloc[0]

print(f'Looking for: {filename}')
print(f'Top level keys: {list(f.keys())}')

# Try to access the file
print('\nAttempting to access...')
try:
    data = f[filename][()]
    print(f'Success! Shape: {data.shape}')
except KeyError as e:
    print(f'KeyError: {e}')
    print('\nLet me explore the structure...')
    
    def print_structure(name, obj):
        print(name)
    
    print('\nFull HDF5 structure:')
    f.visititems(lambda name, obj: print(f'  {name}: {type(obj)}') if len(name.split('/')) <= 3 else None)

f.close()
