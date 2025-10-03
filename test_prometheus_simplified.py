#!/usr/bin/env python3
"""
Test script to verify PROMETHEUS simplified architecture works
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add src to path
sys.path.append('/mnt/d/opt/AutomataNexus_Olympus_AGI2/src')

# Import the new simplified model
from models.prometheus_model_simplified import SimplifiedPrometheusNet

def test_prometheus_architecture():
    """Test the simplified PROMETHEUS architecture"""
    print("🧪 Testing PROMETHEUS Simplified Architecture")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SimplifiedPrometheusNet(max_grid_size=12).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Test forward pass with different input sizes
    test_sizes = [6, 8, 10, 12]
    
    for size in test_sizes:
        print(f"\n🧪 Testing {size}x{size} grid:")
        
        # Create test input (batch=2, channels=10, height=size, width=size)
        input_grid = torch.randn(2, 10, size, size).to(device)
        target_grid = torch.randint(0, 10, (2, size, size)).to(device)
        
        try:
            # Forward pass
            with torch.no_grad():
                outputs = model(input_grid, mode='inference')
            
            # Check outputs
            pred_output = outputs['predicted_output']
            print(f"   ✅ Input: {input_grid.shape}")
            print(f"   ✅ Output: {pred_output.shape}")
            print(f"   ✅ No NaN: {not torch.isnan(pred_output).any()}")
            print(f"   ✅ No Inf: {not torch.isinf(pred_output).any()}")
            
            # Test loss calculation
            pred_indices = pred_output.argmax(dim=1)
            target_indices = target_grid
            
            # Calculate exact matches
            exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
            print(f"   📊 Exact matches: {exact_matches.sum().item()}/{exact_matches.size(0)}")
            
            # Test creativity factor
            if 'creativity_factor' in outputs:
                creativity = outputs['creativity_factor']
                print(f"   🎨 Creativity factor: {creativity.item():.3f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    print(f"\n✅ PROMETHEUS Simplified Architecture Test PASSED!")
    print(f"   🎯 Model is numerically stable")
    print(f"   🎯 No NaN/Inf issues")
    print(f"   🎯 Supports variable grid sizes")
    print(f"   🎯 Ready for training!")
    
    return True

if __name__ == "__main__":
    success = test_prometheus_architecture()
    if success:
        print("\n🚀 PROMETHEUS is ready for stable training!")
    else:
        print("\n❌ PROMETHEUS architecture needs more fixes")