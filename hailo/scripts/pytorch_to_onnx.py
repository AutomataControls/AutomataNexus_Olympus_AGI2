#!/usr/bin/env python3
"""
PyTorch (.pt/.pth) to ONNX Conversion Script for OLYMPUS Models
AutomataNexus OLYMPUS AGI2 - Hailo-8 NPU Integration

Converts trained PyTorch model files (.pt/.pth) to ONNX format for Hailo optimization.
"""

import torch
import torch.onnx
import numpy as np
import argparse
import os
import sys
from pathlib import Path

def convert_model_to_onnx(model_path, output_path, model_name, input_shape=(1, 10, 30, 30)):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to .pt model file
        output_path: Output directory for ONNX model
        model_name: Name of the model (MINERVA, ATLAS, etc.)
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    print(f"Converting {model_name} to ONNX...")
    
    # Load the PyTorch model
    try:
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        print(f"✓ Loaded {model_name} model")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Create dummy input tensor
    dummy_input = torch.randn(input_shape)
    
    # Output path
    onnx_path = os.path.join(output_path, f"{model_name.lower()}.onnx")
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        print(f"✓ Exported to {onnx_path}")
        return True
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch (.pt/.pth) models to ONNX')
    parser.add_argument('--models_dir', required=True, help='Directory containing .pt/.pth model files')
    parser.add_argument('--output_dir', required=True, help='Output directory for ONNX models')
    parser.add_argument('--model_names', nargs='+', 
                       default=['MINERVA', 'ATLAS', 'IRIS', 'CHRONOS', 'PROMETHEUS'],
                       help='Model names to convert')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting PyTorch (.pt/.pth) to ONNX conversion for OLYMPUS models")
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    
    success_count = 0
    total_count = len(args.model_names)
    
    for model_name in args.model_names:
        # Try both .pt and .pth extensions
        for ext in ['.pt', '.pth']:
            model_file = f"{model_name.lower()}_best_model{ext}"
            model_path = os.path.join(args.models_dir, model_file)
            
            if os.path.exists(model_path):
                if convert_model_to_onnx(model_path, args.output_dir, model_name):
                    success_count += 1
                break
        else:
            print(f"✗ Model file not found: {model_name.lower()}_best_model.pt/.pth")
    
    print(f"\n📊 Conversion Summary: {success_count}/{total_count} models converted successfully")
    
    if success_count == total_count:
        print("🎉 All models converted! Ready for Hailo optimization.")
    else:
        print("⚠️  Some conversions failed. Check error messages above.")

if __name__ == "__main__":
    main()