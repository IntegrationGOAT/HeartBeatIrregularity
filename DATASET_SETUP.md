# PhysioNet 2016 Dataset Setup Instructions

## Overview
This project uses the [PhysioNet/CinC Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/) dataset for heart sound classification.

## Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# Download and prepare the dataset (will take ~10-15 minutes)
python prepare_data.py
```

### Option 2: Manual Setup

#### Step 1: Download the Dataset
Download the dataset from PhysioNet:
```bash
cd data
wget https://physionet.org/static/published-projects/challenge-2016/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0.zip
unzip classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0.zip
```

Or visit: https://physionet.org/content/challenge-2016/1.0.0/

#### Step 2: Create Label CSV
Run the label creation step:
```python
python -c "from prepare_data import create_label_csv; create_label_csv()"
```

Or create `data/label.csv` manually with columns: `filename,label`
- filename: path to .wav file
- label: 0 (normal) or 1 (abnormal)

#### Step 3: Extract Audio Features
Extract Log-Mel spectrograms:
```python
python -c "from prepare_data import extract_features; extract_features()"
```

This will create `data/mel_128.h5` containing 128-bin Log-Mel spectrograms.

## Testing with Example Data

If you just want to test the code without the full dataset, you can create a minimal dataset:

```bash
python create_test_dataset.py
```

This will create a small test dataset using the example .wav files.

## After Setup

Once you have:
- ✅ `data/label.csv` 
- ✅ `data/mel_128.h5`

You can train models:

```bash
# Single fold training
python train.py -c config/config_crnn.json

# 10-fold cross validation
python train_fold_validation.py -c config/config_crnn.json
```

## Dataset Statistics

- Total recordings: ~3,240
- Duration: 5 seconds to 120 seconds
- Classes: Normal vs Abnormal
- Sample rate: varies (typically 2000 Hz)

## Troubleshooting

### Large Download
The dataset is ~1GB. If download is slow:
1. Download manually from browser
2. Place in `data/` directory
3. Run extraction steps only

### Memory Issues
If feature extraction runs out of memory:
- Process files in batches
- Reduce mel_bins in config
- Use a subset of data for testing
