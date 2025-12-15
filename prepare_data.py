"""
Script to download and prepare PhysioNet 2016 dataset
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
from utils.util import read_audio
from utils.audio_feature_extractor import LogMelExtractor

def download_dataset():
    """Download PhysioNet 2016 Challenge dataset"""
    print("Downloading PhysioNet 2016 Challenge dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the dataset using wget
    dataset_url = "https://physionet.org/static/published-projects/challenge-2016/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0.zip"
    
    import subprocess
    
    # Download
    zip_file = "data/physionet2016.zip"
    if not os.path.exists(zip_file):
        print(f"Downloading from {dataset_url}")
        subprocess.run(["wget", "-O", zip_file, dataset_url], check=True)
    
    # Extract
    print("Extracting dwataset...")
    subprocess.run(["unzip", "-q", "-o", zip_file, "-d", "data/"], check=True)
    print("Dataset downloaded and extracted!")
    
    return True

def create_label_csv():
    """Create label.csv from the dataset"""
    print("\nCreating label.csv...")
    
    # Path to the extracted dataset
    dataset_path = Path("data/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0")
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please ensure the dataset is downloaded and extracted correctly.")
        return False
    
    # Collect all .wav files and their labels
    data = []
    
    # The dataset is organized in folders (training-a, training-b, etc.)
    for subset_dir in dataset_path.glob("training-*"):
        print(f"Processing {subset_dir.name}...")
        
        # Read the REFERENCE.csv file for this subset
        ref_file = subset_dir / "REFERENCE.csv"
        if ref_file.exists():
            ref_df = pd.read_csv(ref_file, header=None, names=['filename', 'label'])
            
            for _, row in ref_df.iterrows():
                wav_file = subset_dir / f"{row['filename']}.wav"
                if wav_file.exists():
                    # Convert label: -1 (normal) -> 1, 1 (abnormal) -> 0
                    # OR: -1 (normal) -> 0, 1 (abnormal) -> 1
                    # Check the original repo to see which convention they use
                    label_bin = 0 if row['label'] == -1 else 1
                    data.append({
                        'filename': str(wav_file),
                        'label': label_bin
                    })
    
    # Create DataFrame and save
    if data:
        df = pd.DataFrame(data)
        df.to_csv('data/label.csv', index=False)
        print(f"Created label.csv with {len(df)} samples")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        return True
    else:
        print("No data found!")
        return False

def extract_features():
    """Extract Log-Mel features from audio files"""
    print("\nExtracting Log-Mel features...")
    
    if not os.path.exists('data/label.csv'):
        print("label.csv not found! Please run create_label_csv() first.")
        return False
    
    df = pd.read_csv('data/label.csv')
    
    print(f"Processing {len(df)} audio files...")
    
    with h5py.File('data/mel_128.h5', 'w') as store:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            filename = row['filename']
            
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                continue
            
            try:
                # Read audio with bandpass filter
                audio, fs = read_audio(filename, filter=True)
                
                # Extract Log-Mel Spectrogram
                feature = LogMelExtractor(audio, fs, mel_bins=128, log=True, snv=False)
                
                # Store in HDF5 file using filename as key
                store[filename] = feature
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    print("Feature extraction complete! Saved to data/mel_128.h5")
    return True

def main():
    """Main function to prepare the dataset"""
    print("=" * 60)
    print("PhysioNet 2016 Dataset Preparation")
    print("=" * 60)
    
    # Step 1: Download dataset
    if not os.path.exists('data/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0'):
        download_dataset()
    else:
        print("Dataset already exists, skipping download...")
    
    # Step 2: Create label CSV
    if not os.path.exists('data/label.csv'):
        create_label_csv()
    else:
        print("label.csv already exists, skipping...")
        df = pd.read_csv('data/label.csv')
        print(f"Found {len(df)} samples")
    
    # Step 3: Extract features
    if not os.path.exists('data/mel_128.h5'):
        extract_features()
    else:
        print("mel_128.h5 already exists, skipping...")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python train.py -c config/config_crnn.json")
    print("  python train_fold_validation.py -c config/config_crnn.json")

if __name__ == "__main__":
    main()
