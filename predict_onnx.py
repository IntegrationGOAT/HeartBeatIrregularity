"""
Make predictions using ONNX model
Optimized for deployment and cross-platform compatibility
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import soundfile
from scipy.signal import butter, lfilter

EPS = 1E-8

# ============================================================================
# Audio Processing Functions
# ============================================================================

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_audio(audio_path, filter=True):
    """Read and preprocess audio file"""
    data, sr = soundfile.read(audio_path)
    
    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Apply high-pass filter to remove DC offset
    if filter:
        data = butter_highpass_filter(data, cutoff=20, fs=sr, order=5)
    
    return data, sr


def LogMelExtractor(data, sr, mel_bins=128, log=True, snv=False):
    """Extract log mel-spectrogram features"""
    MEL_ARGS = {
        'n_mels': mel_bins,
        'n_fft': 1024,
        'hop_length': 512,
        'win_length': 1024,
        'window': 'hann',
        'center': True,
        'pad_mode': 'reflect',
        'power': 2.0,
        'fmin': 20.0,
        'fmax': sr // 2,
    }
    
    mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, **MEL_ARGS)
    
    if log:
        mel_spectrogram = np.log(mel_spectrogram + EPS)
    
    if snv:
        mel_spectrogram = standard_normal_variate(mel_spectrogram)
    
    return mel_spectrogram


def standard_normal_variate(data):
    """Normalize data using standard normal variate"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / (std + EPS)


# ============================================================================
# ONNX Predictor Class
# ============================================================================

class ONNXPredictor:
    def __init__(self, onnx_model_path, config=None):
        """
        Initialize ONNX predictor
        
        Args:
            onnx_model_path: Path to ONNX model file
            config: Configuration dict for preprocessing
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")
        
        # Load ONNX model
        self.session = ort.InferenceSession(str(onnx_model_path))
        
        # Get input and output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Default config
        if config is None:
            config = {
                'in_channel': 3,
                'duration': 5,
                'delta': True,
                'norm': True,
                'mel_bins': 128
            }
        
        self.config = config
        self.class_labels = ['Normal', 'Abnormal']
        
        print(f"ONNX model loaded: {onnx_model_path}")
        print(f"Input: {self.input_name}, Shape: {self.input_shape}")
        print(f"Output: {self.output_name}")
        print(f"Config: {self.config}")
    
    def extract_features(self, audio_path):
        """
        Extract features from audio file
        
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
        
        # Pad or crop to expected time dimension
        # Expected shape from model: (mel_bins * channels, time_frames)
        expected_time = self.input_shape[2]  # 216
        current_time = feature.shape[1]
        
        if current_time < expected_time:
            # Pad with zeros
            pad_width = expected_time - current_time
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        elif current_time > expected_time:
            # Crop to expected length
            feature = feature[:, :expected_time]
        
        return feature
    
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
        
        # Prepare input (add batch dimension)
        feature_array = np.expand_dims(feature, axis=0).astype(np.float32)
        
        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: feature_array})
        
        # Process output (log softmax output from model)
        log_probs = outputs[0][0]  # Shape: (num_classes,)
        probabilities = np.exp(log_probs)  # Convert log probabilities to probabilities
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return {
            'file': Path(audio_path).name,
            'predicted_class': int(predicted_class),
            'predicted_label': self.class_labels[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                self.class_labels[i]: float(probabilities[i]) 
                for i in range(len(self.class_labels))
            }
        }
    
    def predict_batch(self, audio_paths, output_csv=None):
        """
        Make predictions on multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            output_csv: Optional path to save results as CSV
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        print(f"\nPredicting on {len(audio_paths)} files...")
        for i, audio_path in enumerate(audio_paths, 1):
            try:
                result = self.predict_file(str(audio_path))
                results.append(result)
                print(f"[{i}/{len(audio_paths)}] {result['file']}: {result['predicted_label']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"[{i}/{len(audio_paths)}] Error processing {audio_path}: {e}")
                results.append({
                    'file': Path(audio_path).name,
                    'error': str(e)
                })
        
        # Save to CSV if requested
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict using ONNX model')
    parser.add_argument('--model', type=str, default='model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results')
    parser.add_argument('--in-channel', type=int, default=3,
                        help='Number of input channels (1=mel, 3=mel+delta+delta2)')
    parser.add_argument('--duration', type=int, default=5,
                        help='Audio duration in seconds')
    parser.add_argument('--no-delta', action='store_true',
                        help='Disable delta features')
    parser.add_argument('--no-norm', action='store_true',
                        help='Disable normalization')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'in_channel': args.in_channel,
        'duration': args.duration,
        'delta': not args.no_delta,
        'norm': not args.no_norm,
        'mel_bins': 128
    }
    
    # Initialize predictor
    predictor = ONNXPredictor(args.model, config)
    
    # Handle input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        print(f"\nPredicting on single file: {input_path}")
        result = predictor.predict_file(str(input_path))
        
        print("\n" + "=" * 60)
        print(f"Prediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        print("=" * 60)
        
    elif input_path.is_dir():
        # Directory of files
        audio_files = list(input_path.glob('*.wav'))
        if not audio_files:
            print(f"No .wav files found in {input_path}")
            return
        
        results = predictor.predict_batch(audio_files, args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            print(f"Total predictions: {len(valid_results)}")
            
            # Count by class
            for label in predictor.class_labels:
                count = sum(1 for r in valid_results if r['predicted_label'] == label)
                pct = 100 * count / len(valid_results)
                print(f"{label}: {count} ({pct:.1f}%)")
            
            # Average confidence
            avg_conf = np.mean([r['confidence'] for r in valid_results])
            print(f"Average confidence: {avg_conf:.2%}")
        
        # Errors
        errors = [r for r in results if 'error' in r]
        if errors:
            print(f"\nErrors: {len(errors)} files failed")
    else:
        print(f"Error: {input_path} not found")


if __name__ == '__main__':
    main()
