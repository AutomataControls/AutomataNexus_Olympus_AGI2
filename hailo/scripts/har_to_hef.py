#!/usr/bin/env python3
"""
HAR to HEF Conversion Script for OLYMPUS Models
AutomataNexus OLYMPUS AGI2 - Hailo-8 NPU Integration

Compiles Hailo Archive (HAR) models to Hailo Executable Format (HEF) for deployment.
"""

import os
import subprocess
import argparse
import json
import time
import sys
from pathlib import Path

def create_hef_config(model_name, har_path, output_dir):
    """
    Create HEF compilation configuration
    
    Args:
        model_name: Name of the model
        har_path: Path to HAR file
        output_dir: Output directory for HEF file
    """
    config = {
        "model_name": model_name.lower(),
        "har_path": har_path,
        "target_platform": "hailo8",
        "optimization": {
            "performance_mode": "high_performance",
            "memory_optimization": True,
            "batch_size": 1
        },
        "scheduling": {
            "multi_context": True,
            "context_switch_defs": "auto"
        }
    }
    
    config_path = os.path.join(output_dir, f"{model_name.lower()}_hef_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def compile_har_to_hef(har_path, output_dir, model_name):
    """
    Compile HAR model to HEF format for Hailo-8 deployment
    
    Args:
        har_path: Path to HAR model file
        output_dir: Output directory for HEF file
        model_name: Name of the model
    """
    print(f"Compiling {model_name} HAR to HEF...")
    
    if not os.path.exists(har_path):
        print(f"✗ HAR file not found: {har_path}")
        return False
    
    # Create configuration
    config_path = create_hef_config(model_name, har_path, output_dir)
    
    # Output HEF path
    hef_path = os.path.join(output_dir, f"{model_name.lower()}.hef")
    
    try:
        # Hailo Compiler command for HEF generation
        cmd = [
            "hailo", "compiler",
            "--har", har_path,
            "--calib-dataset-path", "/dev/null",
            "--ckpt-path", hef_path,
            "--hw-arch", "hailo8",
            "--performance-mode", "high_performance"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        start_time = time.time()
        
        # Execute compilation
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        compilation_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Successfully compiled {hef_path} in {compilation_time:.1f}s")
            
            # Print HEF statistics
            if os.path.exists(hef_path):
                file_size = os.path.getsize(hef_path) / (1024 * 1024)  # MB
                print(f"  HEF size: {file_size:.2f} MB")
            
            # Parse performance info from output
            if "Performance" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "FPS" in line or "Throughput" in line:
                        print(f"  {line.strip()}")
            
            return True
        else:
            print(f"✗ Compilation failed:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Compilation timed out (10 minutes)")
        return False
    except FileNotFoundError:
        print("✗ Hailo Compiler not found. Please install Hailo Software Suite.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def analyze_hef_performance(hef_path, model_name):
    """
    Analyze HEF performance characteristics
    
    Args:
        hef_path: Path to HEF file
        model_name: Model name
    """
    try:
        cmd = ["hailo", "analyze", "--hef", hef_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(f"📊 Performance Analysis for {model_name}:")
            
            # Parse key metrics
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['fps', 'latency', 'throughput', 'memory']):
                    print(f"  {line.strip()}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"⚠️  Could not analyze {model_name} performance: {e}")
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
    parser = argparse.ArgumentParser(description='Compile HAR models to HEF format')
    parser.add_argument('--har_dir', required=True, help='Directory containing HAR models')
    parser.add_argument('--output_dir', required=True, help='Output directory for HEF models')
    parser.add_argument('--model_names', nargs='+',
                       default=['MINERVA', 'ATLAS', 'IRIS', 'CHRONOS', 'PROMETHEUS'],
                       help='Model names to compile')
    parser.add_argument('--analyze', action='store_true', help='Analyze HEF performance after compilation')
    
    args = parser.parse_args()
    
    print("🚀 Starting HAR to HEF compilation for OLYMPUS models")
    
    # Validate Hailo installation
    if not validate_hailo_installation():
        print("Please install Hailo Software Suite and ensure 'hailo' is in your PATH")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"HAR directory: {args.har_dir}")
    print(f"Output directory: {args.output_dir}")
    
    success_count = 0
    total_count = len(args.model_names)
    
    for model_name in args.model_names:
        har_file = f"{model_name.lower()}.har"
        har_path = os.path.join(args.har_dir, har_file)
        
        if compile_har_to_hef(har_path, args.output_dir, model_name):
            success_count += 1
            
            if args.analyze:
                hef_path = os.path.join(args.output_dir, f"{model_name.lower()}.hef")
                analyze_hef_performance(hef_path, model_name)
        
        print()  # Add spacing between models
    
    print(f"📊 Compilation Summary: {success_count}/{total_count} models compiled to HEF")
    
    if success_count == total_count:
        print("🎉 All models compiled to HEF! Ready for Hailo-8 deployment.")
        print(f"📁 HEF models available in: {args.output_dir}")
    else:
        print("⚠️  Some compilations failed. Check Hailo Software Suite installation.")

if __name__ == "__main__":
    main()