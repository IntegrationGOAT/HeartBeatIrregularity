"""
Convert PyTorch model to TensorFlow Lite format
PyTorch (.pth) -> ONNX (.onnx) -> TensorFlow (.pb) -> TFLite (.tflite)
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
import pathlib
from pathlib import Path

# Fix for loading checkpoints saved on Linux/Mac on Windows
pathlib.PosixPath = pathlib.WindowsPath

def convert_pytorch_to_tflite(checkpoint_path, output_path='model.tflite', input_shape=None):
    """
    Convert PyTorch model to TFLite
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth)
        output_path: Path for output TFLite model
        input_shape: Input tensor shape (batch, features, time_frames)
                    Default is (1, 384, 216) for 3-channel 128-bin mel-spectrogram
    """
    if input_shape is None:
        # Default: (batch, mel_bins*channels, time_frames) = (1, 128*3, 216)
        input_shape = (1, 384, 216)
    
    print(f"Loading PyTorch checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model architecture from checkpoint
    from model.model import VGG_11
    
    # Extract config - handle both dict and ConfigParser
    if 'config' in checkpoint:
        config_obj = checkpoint['config']
        # Check if it's a ConfigParser object (has a config property)
        if hasattr(config_obj, 'config'):
            # It's a ConfigParser object, access the underlying config dict
            config_dict = config_obj.config
            num_classes = config_dict['arch']['args']['num_classes']
            in_channel = config_dict['arch']['args']['in_channel']
        else:
            # It's a regular dict
            try:
                num_classes = config_obj['arch']['args']['num_classes']
                in_channel = config_obj['arch']['args']['in_channel']
            except:
                num_classes = config_obj.get('num_classes', 2)
                in_channel = config_obj.get('in_channel', 3)
    else:
        num_classes = 2
        in_channel = 3
    
    print(f"Model config: num_classes={num_classes}, in_channel={in_channel}")
    
    # Initialize model
    model = VGG_11(num_classes=num_classes, in_channel=in_channel)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"Model loaded. Input shape: {input_shape}")
    
    # Wrap model to make it ONNX-compatible (fix dynamic shapes)
    class ONNXCompatibleModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Forward through the model
            return self.model(x)
    
    onnx_model = ONNXCompatibleModel(model)
    onnx_model.eval()
    
    # Step 1: Export to ONNX
    dummy_input = torch.randn(input_shape)
    onnx_path = output_path.replace('.tflite', '.onnx')
    
    print(f"Converting to ONNX: {onnx_path}...")
    
    # Use tracing with concrete input
    with torch.no_grad():
        traced_model = torch.jit.trace(onnx_model, dummy_input)
    
    torch.onnx.export(
        traced_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print("✓ ONNX export complete")
    
    # Step 2: Convert ONNX to TensorFlow using onnx2tf
    try:
        import onnx
        print("Converting ONNX to TensorFlow using onnx2tf...")
        
        # Try using onnx2tf instead of onnx-tf
        try:
            import onnx2tf
            pb_path = output_path.replace('.tflite', '_saved_model')
            
            # Convert ONNX to TensorFlow SavedModel
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=pb_path,
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True
            )
            print(f"✓ TensorFlow SavedModel saved to {pb_path}")
            
        except ImportError:
            print("\n❌ onnx2tf not found. Trying alternative method...")
            # Try the old onnx-tf method
            from onnx_tf.backend import prepare
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            pb_path = output_path.replace('.tflite', '_saved_model')
            tf_rep.export_graph(pb_path)
            print(f"✓ TensorFlow model saved to {pb_path}")
        
    except ImportError as e:
        print("\n❌ Error: Missing required packages")
        print("Install with:")
        print("  pip install onnx2tf")
        print("\nAlternatively, the ONNX model has been created at:", onnx_path)
        print("You can convert it manually using online tools or:")
        print("  https://github.com/onnx/onnx-tensorflow")
        return False
    except Exception as e:
        print(f"\n❌ Error during ONNX to TensorFlow conversion: {e}")
        print("\nThe ONNX model is available at:", onnx_path)
        return False
    
    # Step 3: Convert TensorFlow to TFLite
    try:
        import tensorflow as tf
        
        print("Converting TensorFlow to TFLite...")
        
        # Load the TensorFlow model
        converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)
        
        # Optional optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ TFLite model saved to {output_path}")
        
        # Get model size
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during TFLite conversion: {e}")
        print("\nTrying alternative approach...")
        
        # Alternative: Direct conversion using concrete function
        try:
            import tensorflow as tf
            
            # Create a concrete function
            class TFModel(tf.Module):
                def __init__(self, onnx_path):
                    super().__init__()
                    # Load from saved model
                    self.model = tf.saved_model.load(pb_path)
                
                @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
                def __call__(self, x):
                    return self.model(x)
            
            tf_model = TFModel(onnx_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_model.__call__.get_concrete_function()])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✓ TFLite model saved to {output_path}")
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"Model size: {size_mb:.2f} MB")
            
            return True
            
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {e2}")
            return False


def verify_tflite_model(tflite_path, input_shape=(1, 3, 128, 216)):
    """Verify the TFLite model works"""
    try:
        import tensorflow as tf
        
        print(f"\nVerifying TFLite model: {tflite_path}")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Input details:")
        for detail in input_details:
            print(f"  - Shape: {detail['shape']}, Type: {detail['dtype']}")
        
        print("Output details:")
        for detail in output_details:
            print(f"  - Shape: {detail['shape']}, Type: {detail['dtype']}")
        
        # Test inference
        test_input = np.random.randn(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"\n✓ Model verification successful!")
        print(f"Test output shape: {output.shape}")
        print(f"Test output: {output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to TFLite')
    parser.add_argument('--checkpoint', type=str, default='model_best.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='model.tflite',
                        help='Output TFLite model path')
    parser.add_argument('--input-shape', type=str, default='1,384,216',
                        help='Input shape as comma-separated values (batch,features,time_frames). Default: 1,384,216 for 3-channel 128-bin mel-spectrogram')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the converted model')
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    print("=" * 60)
    print("PyTorch to TFLite Conversion")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Input shape: {input_shape}")
    print("=" * 60)
    
    # Convert
    success = convert_pytorch_to_tflite(args.checkpoint, args.output, input_shape)
    
    if success and args.verify:
        verify_tflite_model(args.output, input_shape)
    
    if success:
        print("\n✅ Conversion complete!")
        print(f"\nYou can now use {args.output} in your Android app")
        print(f"Test it with: python predict_tflite.py --model {args.output} --input your_audio.wav")
    else:
        print("\n❌ Conversion failed. Please check the errors above.")


if __name__ == '__main__':
    main()
