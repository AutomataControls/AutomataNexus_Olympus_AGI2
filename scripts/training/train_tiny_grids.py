"""
Specialized training for tiny grids (3x3-5x5)
Uses the lightweight memorizer instead of OLYMPUS
"""
import torch
import sys
import os
sys.path.append('/content/AutomataNexus_Olympus_AGI2')

from src.models.tiny_grid_memorizer import TinyGridMemorizer, train_tiny_grid_memorizer
from scripts.training.train_olympus_ensemble_v3 import OlympusV3UltimateDataset, foundation_collate_fn
from torch.utils.data import DataLoader

# Configuration for tiny grids
TINY_GRID_CONFIG = {
    'batch_size': 256,  # Much smaller batch - we're memorizing patterns
    'learning_rate': 0.01,  # Higher LR for faster learning
    'epochs': 50,
    'memory_size': 10000,  # Reduced to 10k patterns to avoid OOM
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🔬 Training Tiny Grid Specialist (3x3-5x5)")
    print("=" * 80)
    
    # Train each grid size separately
    for grid_size in [3, 4, 5]:
        print(f"\n📐 Training {grid_size}x{grid_size} grids...")
        
        # Create dataset - use less augmentation for memorization
        dataset = OlympusV3UltimateDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=grid_size,
            stage_config={'complexity': 'tiny_grid', 'focus': 'memorization'},
            augmentation_factor=5  # Much less augmentation - we want to memorize real patterns
        )
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=TINY_GRID_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=foundation_collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=TINY_GRID_CONFIG['batch_size'],
            shuffle=False,
            collate_fn=foundation_collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Create and train model
        model = TinyGridMemorizer(max_grid_size=grid_size, memory_size=TINY_GRID_CONFIG['memory_size']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TINY_GRID_CONFIG['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TINY_GRID_CONFIG['epochs'])
        
        print(f"💾 Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"📊 Dataset has {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        best_accuracy = 0
        
        for epoch in range(TINY_GRID_CONFIG['epochs']):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, targets, _ in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs, targets, mode='train')
                
                # Loss
                target_idx = targets.argmax(dim=1) if targets.dim() == 4 else targets
                loss = torch.nn.functional.cross_entropy(
                    outputs.permute(0, 2, 3, 1).reshape(-1, 10),
                    target_idx.reshape(-1)
                )
                
                # Accuracy
                pred_grid = outputs.argmax(dim=1)
                train_correct += (pred_grid == target_idx).all(dim=(1,2)).sum().item()
                train_total += inputs.shape[0]
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs, mode='eval')
                    target_idx = targets.argmax(dim=1) if targets.dim() == 4 else targets
                    pred_grid = outputs.argmax(dim=1)
                    
                    val_correct += (pred_grid == target_idx).all(dim=(1,2)).sum().item()
                    val_total += inputs.shape[0]
            
            # Update scheduler
            scheduler.step()
            
            # Print progress
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100 if val_total > 0 else 0
            
            print(f"Epoch {epoch+1}/{TINY_GRID_CONFIG['epochs']}: "
                  f"Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Train Acc={train_acc:.2f}%, "
                  f"Val Acc={val_acc:.2f}%, "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                save_path = f'/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels/tiny_grid_{grid_size}x{grid_size}_best.pt'
                torch.save(model.state_dict(), save_path)
                print(f"💾 Saved best model with {val_acc:.2f}% validation accuracy")
            
            # Early stopping
            if train_acc > 95:
                print(f"🎯 Hit {train_acc:.2f}% training accuracy! Moving to next size.")
                break
        
        print(f"\n✅ Best {grid_size}x{grid_size} accuracy: {best_accuracy:.2f}%")
    
    print("\n🏆 Tiny grid training complete!")

if __name__ == "__main__":
    main()