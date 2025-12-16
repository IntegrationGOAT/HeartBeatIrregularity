"""
Export PyTorch model to ONNX format for use with ONNX Runtime on Android
ONNX Runtime is better supported and easier to use than TFLite for this use case.
"""
import argparse
import torch
import pathlib
from pathlib import Path

# Fix for loading checkpoints saved on Linux/Mac on Windows
pathlib.PosixPath = pathlib.WindowsPath

def export_to_onnx(checkpoint_path, output_path='model.onnx', input_shape=None):
    """
    Export PyTorch model to ONNX format
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth)
        output_path: Path for output ONNX model
        input_shape: Input tensor shape (batch, features, time_frames)
    """
    if input_shape is None:
        # Default: (batch, mel_bins*channels, time_frames) = (1, 128*3, 216)
        input_shape = (1, 384, 216)
    
    print(f"Loading PyTorch checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get model architecture
    from model.model import VGG_11
    
    # Extract config
    if 'config' in checkpoint:
        config_obj = checkpoint['config']
        if hasattr(config_obj, 'config'):
            config_dict = config_obj.config
            num_classes = config_dict['arch']['args']['num_classes']
            in_channel = config_dict['arch']['args']['in_channel']
        else:
            try:
                num_classes = config_obj['arch']['args']['num_classes']
                in_channel = config_obj['arch']['args']['in_channel']
            except:
                num_classes = 2
                in_channel = 3
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
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Wrap model for ONNX compatibility
    class ONNXModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            return self.model(x)
    
    onnx_model = ONNXModel(model)
    onnx_model.eval()
    
    print(f"Exporting to ONNX: {output_path}...")
    
    # Trace and export
    with torch.no_grad():
        traced_model = torch.jit.trace(onnx_model, dummy_input)
    
    torch.onnx.export(
        traced_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    print("✓ ONNX export complete")
    
    # Get model size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    return True


def verify_onnx_model(onnx_path):
    """Verify the ONNX model"""
    try:
        import onnx
        import numpy as np
        
        print(f"\nVerifying ONNX model: {onnx_path}")
        
        # Load and check model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Try ONNX Runtime inference
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(onnx_path)
            
            # Get input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            print(f"\nInput: {input_info.name}")
            print(f"  Shape: {input_info.shape}")
            print(f"  Type: {input_info.type}")
            
            print(f"\nOutput: {output_info.name}")
            print(f"  Shape: {output_info.shape}")
            print(f"  Type: {output_info.type}")
            
            # Test inference
            test_input = np.random.randn(1, 384, 216).astype(np.float32)
            outputs = session.run(None, {input_info.name: test_input})
            
            print(f"\n✓ ONNX Runtime inference successful!")
            print(f"Output shape: {outputs[0].shape}")
            print(f"Sample output: {outputs[0]}")
            
        except ImportError:
            print("\nNote: Install onnxruntime to test inference:")
            print("  pip install onnxruntime")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, default='model_best.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Output ONNX model path')
    parser.add_argument('--input-shape', type=str, default='1,384,216',
                        help='Input shape (batch,features,time_frames)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the exported model')
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    print("=" * 60)
    print("PyTorch to ONNX Export")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Input shape: {input_shape}")
    print("=" * 60)
    
    # Export
    success = export_to_onnx(args.checkpoint, args.output, input_shape)
    
    if success and args.verify:
        verify_onnx_model(args.output)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Export complete!")
        print("=" * 60)
        print(f"\nONNX model saved to: {args.output}")
        print("\nFor Android deployment:")
        print("1. Use ONNX Runtime Mobile: https://onnxruntime.ai/docs/tutorials/mobile/")
        print("2. Add to your Android project:")
        print("   implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'")
        print("\nONNX Runtime is:")
        print("  • Faster and smaller than TensorFlow Lite for many models")
        print("  • Better PyTorch compatibility")
        print("  • Easier to integrate")
        print("\nTest your model:")
        print(f"  pip install onnxruntime")
        print(f"  python -c \"import onnxruntime; sess = onnxruntime.InferenceSession('{args.output}')\"")
    else:
        print("\n❌ Export failed")


if __name__ == '__main__':
    main()
