#!/usr/bin/env python3
"""Test script to verify MINERVA V2 collate function handles all grid sizes correctly"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

# Add project paths
sys.path.append('/mnt/d/opt/AutomataNexus_Olympus_AGI2')
sys.path.append('/mnt/d/opt/AutomataNexus_Olympus_AGI2/scripts/training')

# Import the collate function
from scripts.training.train_minerva_specialized2 import custom_collate_fn_v2

def test_collate_function():
    """Test the collate function with various grid sizes"""
    print("Testing MINERVA V2 custom_collate_fn_v2...")
    print("=" * 60)
    
    # Test different stages
    test_cases = [
        (0, 6, "Stage 0 (6x6)"),
        (1, 8, "Stage 1 (8x8)"),
        (3, 12, "Stage 3 (12x12)"),
        (7, 30, "Stage 7 (30x30)")
    ]
    
    for stage, target_size, stage_name in test_cases:
        print(f"\n📊 Testing {stage_name} - Target size: {target_size}x{target_size}")
        
        # Create a batch with various grid sizes
        test_batch = []
        grid_sizes = [3, 4, 5, 6, 8, 10, 15, 20, 30]  # Various sizes
        
        for i, size in enumerate(grid_sizes):
            # Create random grids
            input_grid = np.random.randint(0, 10, (size, size))
            output_grid = np.random.randint(0, 10, (size, size))
            
            # Test with numpy arrays (will test .copy() handling)
            if i % 2 == 0:
                # Test with flipped arrays (negative stride)
                input_grid = np.flip(input_grid, axis=0)
                output_grid = np.flip(output_grid, axis=1)
            
            test_batch.append({
                'inputs': input_grid,
                'outputs': output_grid
            })
            print(f"  - Added grid {size}x{size} (numpy{'[flipped]' if i % 2 == 0 else ''})")
        
        # Test the collate function
        try:
            result = custom_collate_fn_v2(test_batch, stage)
            
            # Verify results
            assert 'inputs' in result and 'outputs' in result
            assert result['inputs'].shape == (len(test_batch), target_size, target_size)
            assert result['outputs'].shape == (len(test_batch), target_size, target_size)
            assert result['inputs'].dtype == torch.long
            assert result['outputs'].dtype == torch.long
            
            print(f"✅ Success! Batch shape: {result['inputs'].shape}")
            print(f"   - All grids resized to {target_size}x{target_size}")
            print(f"   - Negative stride arrays handled correctly")
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test edge cases
    print("\n📊 Testing edge cases...")
    
    # Test with single item batch
    single_batch = [{'inputs': np.ones((5, 5)), 'outputs': np.zeros((5, 5))}]
    result = custom_collate_fn_v2(single_batch, 0)
    assert result['inputs'].shape == (1, 6, 6), f"Single batch failed: {result['inputs'].shape}"
    print("✅ Single item batch: OK")
    
    # Test with already correct size
    correct_size_batch = [{'inputs': np.ones((6, 6)), 'outputs': np.zeros((6, 6))}]
    result = custom_collate_fn_v2(correct_size_batch, 0)
    assert result['inputs'].shape == (1, 6, 6), f"Correct size failed: {result['inputs'].shape}"
    print("✅ Already correct size: OK")
    
    # Test with very large grids
    large_batch = [{'inputs': np.ones((50, 50)), 'outputs': np.zeros((50, 50))}]
    result = custom_collate_fn_v2(large_batch, 0)
    assert result['inputs'].shape == (1, 6, 6), f"Large grid failed: {result['inputs'].shape}"
    print("✅ Very large grids (50x50): OK")
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! The collate function is working correctly.")
    return True

if __name__ == "__main__":
    test_collate_function()