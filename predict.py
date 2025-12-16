import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import pandas as pd
from pathlib import Path, PosixPath, WindowsPath
import librosa
import soundfile
from scipy.signal import butter, lfilter
import pathlib
import pickle

EPS = 1E-8

# ============================================================================
# Inlined utility functions
# ============================================================================

def butter_bandpass(lowcut, hightcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = hightcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def butter_bandpass_filter(data, lowcut, hightcut, fs, order=5):
    b, a = butter_bandpass(lowcut, hightcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_audio(path, target_fs=None, filter=False, lowcut=25, hightcut=400):
    (audio, fs) = soundfile.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    if filter:
        # apply bandpass filter
        audio = butter_bandpass_filter(audio, lowcut, hightcut, fs)
    return audio, fs

def LogMelExtractor(data, sr, mel_bins=128, hoplen=15, winlen=25, log=True, snv=False):
    # define parameters
    MEL_ARGS = {
        'n_mels': mel_bins,
        'sr': sr,
        'hop_length': int(sr * hoplen / 1000),
        'win_length': int(sr * winlen / 1000)
    }
    mel_spectrogram = librosa.feature.melspectrogram(y=data, **MEL_ARGS)
    if log:
        mel_spectrogram = np.log(mel_spectrogram + EPS)
    if snv:
        mel_spectrogram = standard_normal_variate(mel_spectrogram)
    
    return mel_spectrogram

# ============================================================================
# Inlined model classes
# ============================================================================

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='max', activation='relu'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if activation == 'relu':
            x = F.relu_(self.bn2(self.conv2(x)))
        elif activation == 'sigmoid':
            x = torch.sigmoid(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class VGG_11(nn.Module):
    def __init__(self, num_classes, in_channel):
        super(VGG_11, self).__init__()
        self.in_channel = in_channel
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = ConvBlock(in_channels=in_channel, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc_final = nn.Linear(512, num_classes)
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc_final)

    def forward(self, input):
        # (batch_size, 3, mel_bins, time_stamps)
        B, mel_bins, num_frames = input.size()
        x = input.view(B, self.in_channel, -1, num_frames)
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        # (samples_num, channel, mel_bins, time_stamps)
        x = self.conv1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        
        output = F.max_pool2d(x, kernel_size=x.shape[2:])
        output = output.view(output.shape[0:2])
        output = F.log_softmax(self.fc_final(output), dim=-1)
        return output

# ============================================================================
# Predictor class
# ============================================================================

def standard_normal_variate(data):
    """Normalize data using standard normal variate"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


class HeartSoundPredictor:
    def __init__(self, checkpoint_path, config=None, device=None):
        """
        Initialize the predictor with a trained model checkpoint
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pth file)
            config: Optional config dict with model parameters
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Fix for loading checkpoints saved on Linux/Mac on Windows
        pathlib.PosixPath = pathlib.WindowsPath
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Default config for VGG_11 (can be overridden)
        if config is None:
            config = {
                'num_classes': 2,
                'in_channel': 3,  # mel + delta + delta2
                'duration': 5,     # seconds
                'delta': True,
                'norm': True,
                'mel_bins': 128
            }
        
        self.config = config
        
        # Initialize model
        self.model = VGG_11(
            num_classes=config['num_classes'],
            in_channel=config['in_channel']
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Using device: {self.device}")
        
        # Class labels
        self.class_labels = ['Normal', 'Abnormal']
    
    def extract_features(self, audio_path):
        """
        Extract mel-spectrogram features from an audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed feature array
        """
        # Read audio with filtering
        audio, sr = read_audio(audio_path, filter=True)
        
        # Extract mel-spectrogram
        feature = LogMelExtractor(audio, sr, mel_bins=self.config['mel_bins'], log=True, snv=False)
        
        # Apply normalization if required
        if self.config['norm']:
            feature = standard_normal_variate(feature)
        
        # Add deltas if required
        if self.config['delta']:
            delta = librosa.feature.delta(feature)
            delta_2 = librosa.feature.delta(delta)
            feature = np.concatenate((feature, delta, delta_2), axis=0)
        
        # Pad or crop to fixed duration
        cycle_len = int(self.config['duration'] * 1000 / 15)  # hop_length = 15ms
        mel_bins, num_frames = feature.shape
        
        if num_frames >= cycle_len:
            # Use center crop for inference
            start_ind = int((num_frames - cycle_len) / 2)
            feature_pad = feature[:, start_ind:start_ind + cycle_len]
        elif num_frames < cycle_len:
            # Pad with wrapping
            feature_pad = np.pad(feature, ((0, 0), (0, cycle_len - num_frames)), mode='wrap')
        else:
            feature_pad = feature
        
        return feature_pad
    
    def predict_file(self, audio_path):
        """
        Make prediction on a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        feature = self.extract_features(audio_path)
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(feature).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(feature_tensor)
            probabilities = torch.exp(output)  # Convert log probabilities to probabilities
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'file': Path(audio_path).name,
            'predicted_class': predicted_class,
            'predicted_label': self.class_labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Normal': probabilities[0][0].item(),
                'Abnormal': probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, audio_paths):
        """
        Make predictions on multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict_file(audio_path)
                results.append(result)
                print(f"✓ {result['file']}: {result['predicted_label']} (confidence: {result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ Error processing {audio_path}: {str(e)}")
                results.append({
                    'file': Path(audio_path).name,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_features(self, h5_path, filenames):
        """
        Make predictions from pre-extracted features in HDF5 file
        
        Args:
            h5_path: Path to HDF5 file with features
            filenames: List of filenames to predict
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        with h5py.File(h5_path, 'r') as h5_file:
            for filename in filenames:
                try:
                    # Load feature
                    feature = h5_file[filename][()]
                    
                    if self.config['norm']:
                        feature = standard_normal_variate(feature)
                    
                    if self.config['delta']:
                        delta = librosa.feature.delta(feature)
                        delta_2 = librosa.feature.delta(delta)
                        feature = np.concatenate((feature, delta, delta_2), axis=0)
                    
                    # Pad or crop
                    cycle_len = int(self.config['duration'] * 1000 / 15)
                    mel_bins, num_frames = feature.shape
                    
                    if num_frames >= cycle_len:
                        start_ind = int((num_frames - cycle_len) / 2)
                        feature_pad = feature[:, start_ind:start_ind + cycle_len]
                    elif num_frames < cycle_len:
                        feature_pad = np.pad(feature, ((0, 0), (0, cycle_len - num_frames)), mode='wrap')
                    else:
                        feature_pad = feature
                    
                    # Convert to tensor and predict
                    feature_tensor = torch.FloatTensor(feature_pad).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(feature_tensor)
                        probabilities = torch.exp(output)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    results.append({
                        'file': filename,
                        'predicted_class': predicted_class,
                        'predicted_label': self.class_labels[predicted_class],
                        'confidence': confidence,
                        'probabilities': {
                            'Normal': probabilities[0][0].item(),
                            'Abnormal': probabilities[0][1].item()
                        }
                    })
                    
                    print(f"✓ {filename}: {self.class_labels[predicted_class]} (confidence: {confidence:.2%})")
                    
                except Exception as e:
                    print(f"✗ Error processing {filename}: {str(e)}")
                    results.append({
                        'file': filename,
                        'error': str(e)
                    })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Heart Sound Classification - Prediction')
    parser.add_argument('--checkpoint', type=str, default='model_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to audio file, directory of audio files, or HDF5 feature file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save prediction results (CSV)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--in_channel', type=int, default=3,
                        help='Number of input channels (1: mel only, 3: mel+delta+delta2)')
    parser.add_argument('--duration', type=int, default=5,
                        help='Audio duration in seconds')
    parser.add_argument('--no-delta', action='store_true',
                        help='Disable delta features')
    parser.add_argument('--no-norm', action='store_true',
                        help='Disable normalization')
    
    args = parser.parse_args()
    
    # Prepare config
    config = {
        'num_classes': 2,
        'in_channel': args.in_channel,
        'duration': args.duration,
        'delta': not args.no_delta,
        'norm': not args.no_norm,
        'mel_bins': 128
    }
    
    # Initialize predictor
    predictor = HeartSoundPredictor(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device
    )
    
    # Determine input type and make predictions
    input_path = Path(args.input)
    
    if input_path.suffix == '.h5':
        # HDF5 feature file
        print(f"\nPredicting from HDF5 features: {input_path}")
        with h5py.File(input_path, 'r') as h5_file:
            filenames = list(h5_file.keys())
        print(f"Found {len(filenames)} files in HDF5")
        results = predictor.predict_from_features(str(input_path), filenames)
        
    elif input_path.is_dir():
        # Directory of audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
            audio_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"\nFound {len(audio_files)} audio files in {input_path}")
        results = predictor.predict_batch([str(f) for f in audio_files])
        
    elif input_path.is_file():
        # Single audio file
        print(f"\nPredicting on single file: {input_path}")
        result = predictor.predict_file(str(input_path))
        results = [result]
        print(f"\nPrediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities:")
        print(f"  Normal: {result['probabilities']['Normal']:.2%}")
        print(f"  Abnormal: {result['probabilities']['Abnormal']:.2%}")
    else:
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    # Save results if output path specified
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    successful = [r for r in results if 'error' not in r]
    if successful:
        normal_count = sum(1 for r in successful if r['predicted_class'] == 0)
        abnormal_count = sum(1 for r in successful if r['predicted_class'] == 1)
        avg_confidence = np.mean([r['confidence'] for r in successful])
        
        print(f"Total predictions: {len(successful)}")
        print(f"Normal: {normal_count} ({normal_count/len(successful):.1%})")
        print(f"Abnormal: {abnormal_count} ({abnormal_count/len(successful):.1%})")
        print(f"Average confidence: {avg_confidence:.2%}")
    
    failed = [r for r in results if 'error' in r]
    if failed:
        print(f"\nFailed predictions: {len(failed)}")


if __name__ == '__main__':
    main()
