# Stage 0 Exact Match Boost - Aggressive training strategy for exact matches

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple
import random


class ExactMatchBoostDataset(Dataset):
    """
    Ultra-focused dataset for Stage 0 that GUARANTEES exact match learning
    by using only simple, deterministic transformations
    """
    
    def __init__(self, num_samples: int = 50000, fixed_size: int = 5):
        self.num_samples = num_samples
        self.fixed_size = fixed_size  # Fix size for all samples
        self.samples = []
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate samples that are GUARANTEED to produce exact matches"""
        
        samples_per_type = self.num_samples // 15  # More pattern types
        
        # 1. Pure Identity (30% of data) - MUST learn to copy exactly
        for _ in range(samples_per_type * 5):  # Increased from 20% to 30%
            grid = np.random.randint(0, 3, (self.fixed_size, self.fixed_size))
            self.samples.append({
                'input': grid,
                'output': grid.copy(),
                'transform': 'identity'
            })
        
        # 2. Single color fill (10%) - Fill everything with color 1
        for _ in range(samples_per_type):
            grid = np.random.randint(0, 3, (self.fixed_size, self.fixed_size))
            output = np.ones_like(grid)
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'fill_one'
            })
        
        # 3. Clear grid (10%) - Everything becomes 0
        for _ in range(samples_per_type):
            grid = np.random.randint(1, 4, (self.fixed_size, self.fixed_size))
            output = np.zeros_like(grid)
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'clear'
            })
        
        # 4. Binary threshold (10%) - >1 becomes 1, else 0
        for _ in range(samples_per_type):
            grid = np.random.randint(0, 4, (self.fixed_size, self.fixed_size))
            output = (grid > 1).astype(int)
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'binary_threshold'
            })
        
        # 5. Flip single pixel (10%) - Change just one pixel
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            # Add one pixel
            x, y = random.randint(0, self.fixed_size-1), random.randint(0, self.fixed_size-1)
            grid[x, y] = 1
            output = grid.copy()
            # Flip it
            output[x, y] = 0
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'flip_pixel'
            })
        
        # 6. Horizontal line (10%) - Draw horizontal line at row 0
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            output = grid.copy()
            output[0, :] = 1
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'horizontal_line'
            })
        
        # 7. Vertical line (10%) - Draw vertical line at col 0
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            output = grid.copy()
            output[:, 0] = 1
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'vertical_line'
            })
        
        # 8. Corner dot (10%) - Put a dot in top-left corner
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            output = grid.copy()
            output[0, 0] = 1
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'corner_dot'
            })
        
        # 9. Simple color swap 0<->1 (10%)
        for _ in range(samples_per_type):
            grid = np.random.randint(0, 2, (self.fixed_size, self.fixed_size))
            output = 1 - grid  # Swap 0 and 1
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'swap_01'
            })
        
        # 10. Count and fill (10%) - Fill with count of non-zero
        for _ in range(samples_per_type):
            grid = np.random.randint(0, 2, (self.fixed_size, self.fixed_size))
            count = min(np.count_nonzero(grid), 9)  # Cap at 9
            output = np.full_like(grid, count)
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'count_fill'
            })
        
        # 11. Checker pattern (7%) - Create checker board
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            output = np.indices((self.fixed_size, self.fixed_size)).sum(axis=0) % 2
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'checker'
            })
        
        # 12. All ones (8%) - Everything becomes 1
        for _ in range(samples_per_type):
            grid = np.random.randint(0, 4, (self.fixed_size, self.fixed_size))
            output = np.ones_like(grid)
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'all_ones'
            })
        
        # 13. Diagonal pattern (7%) - Diagonal line
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            output = np.eye(self.fixed_size, dtype=int)
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'diagonal'
            })
        
        # 14. Border pattern (8%) - Only border is 1
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            output = np.zeros_like(grid)
            output[0, :] = 1
            output[-1, :] = 1
            output[:, 0] = 1
            output[:, -1] = 1
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'border'
            })
        
        # 15. Center dot (5%) - Put dot in center
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            output = grid.copy()
            center = self.fixed_size // 2
            output[center, center] = 1
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'center_dot'
            })
        
        # Shuffle samples
        random.shuffle(self.samples)
        print(f"Generated {len(self.samples)} exact-match training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class AggressiveLoss(nn.Module):
    """
    Loss function that HEAVILY penalizes incorrect pixels
    and gives massive rewards for exact matches
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = pred.shape
        
        # Get predictions
        pred_indices = pred.argmax(dim=1)
        target_indices = target.argmax(dim=1)
        input_indices = input_grid.argmax(dim=1)
        
        # 1. Per-pixel cross entropy with heavy weight on errors
        ce_loss = self.ce_loss(pred.permute(0, 2, 3, 1).reshape(-1, C),
                               target_indices.reshape(-1))
        
        # Triple the loss for incorrect pixels
        incorrect_mask = (pred_indices != target_indices).float().reshape(-1)
        ce_loss = ce_loss * (1 + incorrect_mask * 4)  # 5x weight on errors
        ce_loss = ce_loss.mean()
        
        # 2. Exact match bonus (negative loss) - BALANCED
        exact_matches = (pred_indices == target_indices).all(dim=[1, 2]).float()
        exact_bonus = -1.0 * exact_matches.mean()  # Much smaller bonus to maintain gradient flow
        
        # 3. Heavy penalty for copying when shouldn't
        should_not_copy = (target_indices != input_indices).any(dim=[1, 2]).float()
        did_copy = (pred_indices == input_indices).all(dim=[1, 2]).float()
        copy_penalty = 5.0 * (should_not_copy * did_copy).mean()
        
        # 4. Transformation encouragement - WITH NaN protection
        changed_pixels = (pred_indices != input_indices).float().mean(dim=[1, 2])
        target_changed = (target_indices != input_indices).float().mean(dim=[1, 2])
        transform_diff = F.mse_loss(changed_pixels, target_changed) * 1.0  # Reduced multiplier
        
        # 5. Color usage penalty - SAFER implementation
        color_penalty = 0.0
        try:
            for b in range(B):
                pred_colors = torch.unique(pred_indices[b])
                target_colors = torch.unique(target_indices[b])
                if len(target_colors) > 0 and len(pred_colors) > 0:
                    pred_set = set(pred_colors.cpu().numpy())
                    target_set = set(target_colors.cpu().numpy())
                    intersection = len(pred_set & target_set)
                    missing_colors = max(0, len(target_set) - intersection)
                    color_penalty += missing_colors * 0.1  # Reduced penalty
        except Exception:
            color_penalty = 0.0  # Fallback if any issues
        
        total_loss = ce_loss + exact_bonus + copy_penalty + transform_diff + color_penalty
        
        # NaN protection - only replace NaN, don't clamp valid values
        if torch.isnan(total_loss).any():
            total_loss = torch.tensor(2.0, device=total_loss.device, requires_grad=True)
        
        # Only clamp if loss is extremely large (>100)
        if total_loss > 100.0:
            total_loss = torch.tensor(10.0, device=total_loss.device, requires_grad=True)
        
        return {
            'total': total_loss,
            'reconstruction': ce_loss,
            'transformation': copy_penalty,  # Use copy penalty as transformation metric
            'exact_bonus': -exact_bonus,  # Show as positive in logs
            'exact_count': exact_matches.sum(),
            'copy_penalty': copy_penalty,
            'transform_diff': transform_diff
        }


def create_exact_match_curriculum(stage: int = 0, fixed_size: int = 5) -> List[Dict]:
    """
    Create curriculum that gradually increases complexity
    but maintains focus on exact matches
    """
    
    if stage == 0:
        # Ultra simple - just identity and basic fills
        samples = []
        
        # 50% identity
        for _ in range(2500):
            grid = np.random.randint(0, 2, (fixed_size, fixed_size))
            samples.append({'input': grid, 'output': grid.copy()})
        
        # 25% fill with 1
        for _ in range(1250):
            grid = np.random.randint(0, 2, (fixed_size, fixed_size))
            samples.append({'input': grid, 'output': np.ones_like(grid)})
        
        # 25% fill with 0
        for _ in range(1250):
            grid = np.random.randint(1, 3, (fixed_size, fixed_size))
            samples.append({'input': grid, 'output': np.zeros_like(grid)})
        
        return samples
    
    elif stage == 1:
        # Add simple transformations
        dataset = ExactMatchBoostDataset(10000, fixed_size=fixed_size)
        return dataset.samples[:5000]  # Use first half
    
    else:
        # Full complexity
        dataset = ExactMatchBoostDataset(10000, fixed_size=fixed_size)
        return dataset.samples


def exact_match_collate_fn(batch):
    """Custom collate function to handle dictionary samples"""
    inputs = torch.stack([torch.tensor(item['input'], dtype=torch.long) for item in batch])
    outputs = torch.stack([torch.tensor(item['output'], dtype=torch.long) for item in batch])
    return {'input': inputs, 'output': outputs}


def inject_exact_match_training(model, device='cuda', num_epochs=20):
    """
    Special pre-training phase that forces exact match learning
    """
    
    print("\n🎯 EXACT MATCH INJECTION TRAINING")
    print("="*50)
    
    # Create ultra-focused dataset with more samples
    dataset = ExactMatchBoostDataset(10000, fixed_size=5)  # Doubled samples
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=exact_match_collate_fn  # Larger batch
    )
    
    # COMPREHENSIVE: Use AggressiveLoss with proper LR for injection training
    initial_lr = 0.005  # Higher LR for better learning
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    # No scheduler for injection training - keep LR constant
    scheduler = None
    # Use AggressiveLoss for comprehensive exact match training
    loss_fn = AggressiveLoss()
    
    model.train()
    best_exact_match = 0
    
    for epoch in range(num_epochs):
        total_exact = 0
        total_samples = 0
        
        for batch in dataloader:
            # Get tensors from batch
            inputs = batch['input'].to(device)
            outputs = batch['output'].to(device)
            
            # One-hot encode
            B, H, W = inputs.shape
            inputs_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            outputs_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            # Forward pass
            if hasattr(model, '__class__') and model.__class__.__name__ == 'CHRONOS':
                pred = model([inputs_oh], target=outputs_oh)['predicted_output']
            else:
                pred = model(inputs_oh, outputs_oh, mode='train')['predicted_output']
            
            # Comprehensive AggressiveLoss calculation
            losses = loss_fn(pred, outputs_oh, inputs_oh)
            
            # DEBUG: Print some stats on first batch of first epoch
            if epoch == 0 and total_samples == 0:
                pred_classes = pred.argmax(dim=1)
                target_classes = outputs
                print(f"DEBUG - Pred range: {pred_classes.min()}-{pred_classes.max()}, Target range: {target_classes.min()}-{target_classes.max()}")
                print(f"DEBUG - Pred unique: {torch.unique(pred_classes)}, Target unique: {torch.unique(target_classes)}")
                print(f"DEBUG - Loss components: recon={losses['reconstruction']:.4f}, exact_bonus={losses['exact_bonus']:.4f}, total={losses['total']:.4f}")
            
            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Track exact matches
            total_exact += losses['exact_count'].item()
            total_samples += B
        
        exact_pct = total_exact / total_samples * 100
        print(f"Epoch {epoch+1}/{num_epochs}: Exact Match: {exact_pct:.1f}% | LR: {optimizer.param_groups[0]['lr']:.5f} | Total samples: {total_samples}")
        
        # No scheduler step - keeping LR constant for injection training
        
        # Track best performance
        if exact_pct > best_exact_match:
            best_exact_match = exact_pct
            
        # Early stopping with patience
        if exact_pct > 30:  # Lower threshold since we're being more conservative
            print("✅ Achieved >30% exact match! Stopping injection training.")
            break
        elif epoch > 5 and exact_pct < 1.0:  # If no learning after 5 epochs
            # Try reducing learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"📉 No learning detected, reducing LR to {optimizer.param_groups[0]['lr']:.5f}")
    
    return model


# Integration function for main training script
def apply_stage0_exact_boost(train_loader, model, optimizer, device='cuda'):
    """
    Apply exact match boost during regular training
    """
    
    # Every N batches, inject an exact match batch
    INJECT_EVERY = 5
    
    # Create exact match dataset
    exact_dataset = ExactMatchBoostDataset(1000)
    
    # Create aggressive loss
    aggressive_loss = AggressiveLoss()
    
    # Return enhanced training function
    def enhanced_train_step(batch_idx, batch, regular_loss_fn):
        if batch_idx % INJECT_EVERY == 0:
            # Inject exact match batch
            exact_idx = random.randint(0, len(exact_dataset) - 32)
            exact_batch = [exact_dataset[i] for i in range(exact_idx, exact_idx + 32)]
            
            # Process exact match batch with aggressive loss
            inputs = torch.stack([torch.tensor(s['input']) for s in exact_batch]).to(device)
            outputs = torch.stack([torch.tensor(s['output']) for s in exact_batch]).to(device)
            
            B, H, W = inputs.shape
            inputs_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            outputs_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            # Forward with exact match data
            if hasattr(model, '__class__') and model.__class__.__name__ == 'CHRONOS':
                pred = model([inputs_oh], target=outputs_oh)['predicted_output']
            else:
                pred = model(inputs_oh, outputs_oh, mode='train')['predicted_output']
            
            # Use aggressive loss
            return aggressive_loss(pred, outputs_oh, inputs_oh)
        else:
            # Regular training step
            return None
    
    return enhanced_train_step