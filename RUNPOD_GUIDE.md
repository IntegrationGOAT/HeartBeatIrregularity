# RunPod Training Setup Guide

## Quick Start (3 Steps)

### Step 1: Install RunPod SDK
```bash
pip install runpod
```

### Step 2: Create and Setup Pod

**Option A: Using the Python Script (Automated)**
```bash
python runpod_setup.py
```

**Option B: Manual Setup via RunPod Website**
1. Go to https://www.runpod.io/console/pods
2. Click "Deploy"
3. Select GPU (RTX 4090 recommended for cost/performance)
4. Choose PyTorch template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
5. Set volume size: 50GB
6. Expose ports: 8888 (Jupyter), 6006 (TensorBoard)
7. Click "Deploy"

### Step 3: Setup Your Code on RunPod

Once your pod is running, connect via SSH or web terminal:

```bash
# 1. Clone your repository
git clone https://github.com/Tanziruz/HeartBeatIrregularity.git
cd HeartBeatIrregularity

# 2. Install dependencies
pip install -r requirements.txt
pip install protobuf==3.20.3

# 3. Upload your data
# From your LOCAL machine, run this:
scp -P <PORT> data/mel_128.h5 root@<POD_IP>:~/HeartBeatIrregularity/data/

# Or regenerate features on the pod:
python prepare_data.py

# 4. Run training
python train_fold_validation.py -c config/config_crnn.json
```

## Uploading Large Data Files

If your `data/mel_128.h5` is large (542MB), here are options:

### Option 1: SCP (Recommended for <1GB)
```bash
# Find your pod's SSH info in RunPod console
scp -P <SSH_PORT> data/mel_128.h5 root@<POD_HOST>:~/HeartBeatIrregularity/data/
```

### Option 2: Upload to Cloud Storage First
```bash
# Upload to Google Drive, Dropbox, or S3, then on the pod:
wget "<SHARED_LINK>" -O data/mel_128.h5
```

### Option 3: Regenerate on Pod (Best for reproducibility)
```bash
# Just run prepare_data.py on the pod - it will download and process everything
python prepare_data.py
```

## Monitoring Training

### Via Terminal
```bash
# Watch the training output in real-time
python train_fold_validation.py -c config/config_crnn.json
```

### Via TensorBoard (Web UI)
```bash
# On the pod, run:
tensorboard --logdir=saved/log --host=0.0.0.0 --port=6006

# Then access via the RunPod port forwarding URL (shown in console)
```

## Cost Optimization Tips

1. **Use Community Cloud** - Cheaper but can be interrupted
2. **Stop pod when not training** - Only pay for running time
3. **RTX 4090** - Best cost/performance (~$0.34/hr)
4. **Use spot instances** - Up to 70% cheaper

## Troubleshooting

### "No module named X"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Reduce batch size in config/config_crnn.json:
```json
"data_loader": {
    "args": {
        "batch_size": 16  // Reduce from 32
    }
}
```

### Data not found
Make sure data files are in the correct location:
```bash
ls -lh data/mel_128.h5
ls -lh data/label.csv
```

## Getting Help

Check pod logs in RunPod console if training fails.
