#!/usr/bin/env python3
"""
Quick test script for TinyGridMemorizer
"""
import torch
import sys
sys.path.append('/mnt/d/opt/AutomataNexus_Olympus_AGI2')

from src.models.tiny_grid_memorizer import TinyGridMemorizer

def test_memorizer():
    """Test the TinyGridMemorizer model to ensure no OOM"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create model
    model = TinyGridMemorizer(max_grid_size=5, memory_size=10000).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test forward pass with small batch
    batch_size = 32
    grid_size = 3
    
    # Create dummy input
    x = torch.randn(batch_size, 10, grid_size, grid_size).to(device)
    target = torch.randint(0, 10, (batch_size, grid_size, grid_size)).to(device)
    
    print("\nTesting forward pass...")
    try:
        # Test training mode
        model.train()
        output = model(x, target, mode='train')
        print(f"Training forward pass successful! Output shape: {output.shape}")
        
        # Test eval mode
        model.eval()
        with torch.no_grad():
            output = model(x, mode='eval')
        print(f"Eval forward pass successful! Output shape: {output.shape}")
        
        # Test backward pass
        model.train()
        loss = torch.nn.functional.cross_entropy(
            output.permute(0, 2, 3, 1).reshape(-1, 10),
            target.reshape(-1)
        )
        print(f"Loss computed: {loss.item():.4f}")
        
        loss.backward()
        print("Backward pass successful!")
        
        # Check memory usage
        if torch.cuda.is_available():
            print(f"\nGPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        
        print("\n✅ All tests passed! Model is ready for training.")
        
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memorizer()