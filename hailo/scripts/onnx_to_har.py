#!/usr/bin/env python3
"""
ONNX to HAR Conversion Script for OLYMPUS Models
AutomataNexus OLYMPUS AGI2 - Hailo-8 NPU Integration

Converts ONNX models to Hailo Archive (HAR) format using Hailo Dataflow Compiler (DFC) in WSL.
"""

import os
import subprocess
import argparse
import json
import sys
from pathlib import Path

def create_hn_config(model_name, onnx_path, output_dir):
    """
    Create Hailo Network configuration file for model optimization
    
    Args:
        model_name: Name of the model
        onnx_path: Path to ONNX model
        output_dir: Output directory for HAR file
    """
    config = {
        "model_name": model_name.lower(),
        "model_path": onnx_path,
        "output_dir": output_dir,
        "preprocessing": {
            "input_format": "NCHW",
            "normalization": "zscore",
            "resize_mode": "bilinear"
        },
        "optimization": {
            "batch_size": 1,
            "precision": "int8",
            "calibration_set_size": 64,
            "optimization_level": "O1"
        },
        "quantization": {
            "calibration_dataset": "custom",
            "percentile": 99.99
        }
    }
    
    config_path = os.path.join(output_dir, f"{model_name.lower()}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def convert_onnx_to_har(onnx_path, output_dir, model_name):
    """
    Convert ONNX model to HAR format using Hailo Dataflow Compiler
    
    Args:
        onnx_path: Path to ONNX model file
        output_dir: Output directory for HAR file
        model_name: Name of the model
    """
    print(f"Converting {model_name} ONNX to HAR...")
    
    if not os.path.exists(onnx_path):
        print(f"✗ ONNX file not found: {onnx_path}")
        return False
    
    # Create configuration file
    config_path = create_hn_config(model_name, onnx_path, output_dir)
    
    # Output HAR path
    har_path = os.path.join(output_dir, f"{model_name.lower()}.har")
    
    try:
        # Hailo Dataflow Compiler (DFC) command in WSL
        cmd = [
            "hailo", "parser",
            "onnx", onnx_path,
            "--output-dir", output_dir,
            "--name", model_name.lower(),
            "--batch-size", "1"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Execute conversion
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ Successfully created {har_path}")
            
            # Print compilation stats
            if "Compilation successful" in result.stdout:
                print("✓ Model compilation successful")
            
            return True
        else:
            print(f"✗ Compilation failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Compilation timed out (5 minutes)")
        return False
    except FileNotFoundError:
        print("✗ Hailo Dataflow Compiler not found. Please install Hailo Software Suite.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def validate_hailo_installation():
    """Check if Hailo DFC is available in WSL"""
    try:
        result = subprocess.run(["hailo", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Hailo Dataflow Compiler found: {result.stdout.strip()}")
            return True
        else:
            print("✗ Hailo DFC not properly installed in WSL")
            return False
    except FileNotFoundError:
        print("✗ Hailo DFC not found in PATH. Please install in WSL environment.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX models to HAR format')
    parser.add_argument('--onnx_dir', required=True, help='Directory containing ONNX models')
    parser.add_argument('--output_dir', required=True, help='Output directory for HAR models')
    parser.add_argument('--model_names', nargs='+',
                       default=['MINERVA', 'ATLAS', 'IRIS', 'CHRONOS', 'PROMETHEUS'],
                       help='Model names to convert')
    
    args = parser.parse_args()
    
    print("🚀 Starting ONNX to HAR conversion for OLYMPUS models")
    
    # Validate Hailo installation
    if not validate_hailo_installation():
        print("Please install Hailo DFC in WSL and ensure 'hailo' is in your PATH")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"ONNX directory: {args.onnx_dir}")
    print(f"Output directory: {args.output_dir}")
    
    success_count = 0
    total_count = len(args.model_names)
    
    for model_name in args.model_names:
        onnx_file = f"{model_name.lower()}.onnx"
        onnx_path = os.path.join(args.onnx_dir, onnx_file)
        
        if convert_onnx_to_har(onnx_path, args.output_dir, model_name):
            success_count += 1
        
        print()  # Add spacing between models
    
    print(f"📊 Conversion Summary: {success_count}/{total_count} models converted to HAR")
    
    if success_count == total_count:
        print("🎉 All models converted to HAR! Ready for HEF compilation.")
    else:
        print("⚠️  Some conversions failed. Check Hailo DFC installation in WSL.")

if __name__ == "__main__":
    main()