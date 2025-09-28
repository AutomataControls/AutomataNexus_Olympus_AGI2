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
        # Add additional augmented samples
        self._augment_samples()
    
    def _generate_samples(self):
        """Generate samples that are GUARANTEED to produce exact matches"""
        
        samples_per_type = self.num_samples // 20  # Even more pattern types
        
        # 1. Pure Identity (30% of data) - MUST learn to copy exactly
        for _ in range(samples_per_type * 6):  # Increased from 30% to 30%
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
        
        # 16. Simple masks (5%) - Keep only certain positions
        for _ in range(samples_per_type):
            grid = np.random.randint(0, 3, (self.fixed_size, self.fixed_size))
            output = grid.copy()
            # Keep only top-left quadrant
            output[self.fixed_size//2:, :] = 0
            output[:, self.fixed_size//2:] = 0
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'quadrant_mask'
            })
        
        # 17. Row/Column operations (5%) - Fill specific row/column
        for _ in range(samples_per_type):
            grid = np.random.randint(0, 2, (self.fixed_size, self.fixed_size))
            output = grid.copy()
            if random.random() < 0.5:
                # Fill middle row
                output[self.fixed_size//2, :] = 2
            else:
                # Fill middle column
                output[:, self.fixed_size//2] = 2
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'row_col_fill'
            })
        
        # 18. Simple arithmetic (5%) - Add 1 to all non-zero
        for _ in range(samples_per_type):
            grid = np.random.randint(0, 3, (self.fixed_size, self.fixed_size))
            output = np.where(grid > 0, np.minimum(grid + 1, 9), 0)
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'increment_nonzero'
            })
        
        # 19. Connectivity patterns (5%) - Simple connected components
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            # Create two separated objects
            grid[0:2, 0:2] = 1  # Top-left object
            grid[-2:, -2:] = 2  # Bottom-right object
            output = grid.copy()
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'separated_objects'
            })
        
        # 20. Cross patterns (5%) - Draw cross
        for _ in range(samples_per_type):
            grid = np.zeros((self.fixed_size, self.fixed_size), dtype=int)
            output = grid.copy()
            center = self.fixed_size // 2
            output[center, :] = 1
            output[:, center] = 1
            self.samples.append({
                'input': grid,
                'output': output,
                'transform': 'cross_pattern'
            })
        
        # Shuffle samples
        random.shuffle(self.samples)
        print(f"Generated {len(self.samples)} exact-match training samples")
    
    def _augment_samples(self):
        """Add rotated and flipped versions of existing samples for more diversity"""
        augmented = []
        
        # Only augment a subset to avoid too much growth
        for sample in self.samples[:len(self.samples)//4]:
            input_grid = sample['input']
            output_grid = sample['output']
            
            # Skip augmentation for certain transformations that don't make sense
            if sample['transform'] in ['identity', 'count_fill', 'all_ones', 'fill_one', 'clear']:
                continue
                
            # Add 90 degree rotation
            aug_input_90 = np.rot90(input_grid, 1).copy()
            aug_output_90 = np.rot90(output_grid, 1).copy()
            augmented.append({
                'input': aug_input_90,
                'output': aug_output_90,
                'transform': sample['transform'] + '_rot90'
            })
            
            # Add horizontal flip
            aug_input_h = np.fliplr(input_grid).copy()
            aug_output_h = np.fliplr(output_grid).copy()
            augmented.append({
                'input': aug_input_h,
                'output': aug_output_h,
                'transform': sample['transform'] + '_fliph'
            })
        
        self.samples.extend(augmented)
        random.shuffle(self.samples)
        print(f"After augmentation: {len(self.samples)} total samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class AggressiveLoss(nn.Module):
    """
    Loss function that HEAVILY penalizes incorrect pixels
    and gives massive rewards for exact matches
    """
    
    def __init__(self, label_smoothing=0.0):  # Disable label smoothing initially
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        self.current_epoch = 0
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = pred.shape
        
        # Get predictions
        pred_indices = pred.argmax(dim=1)
        target_indices = target.argmax(dim=1)
        input_indices = input_grid.argmax(dim=1)
        
        # Simple approach for first few epochs
        if self.current_epoch < 3:
            # Just use plain cross entropy
            ce_loss = F.cross_entropy(pred, target_indices, reduction='mean')
            exact_matches = (pred_indices == target_indices).all(dim=[1, 2]).float()
            
            return {
                'total': ce_loss,
                'reconstruction': ce_loss,
                'transformation': torch.tensor(0.0),
                'exact_bonus': torch.tensor(0.0),
                'exact_count': exact_matches.sum(),
                'copy_penalty': torch.tensor(0.0),
                'transform_diff': torch.tensor(0.0),
                'color_penalty': torch.tensor(0.0)
            }
        
        # Full loss after warmup
        # 1. Per-pixel cross entropy with heavy weight on errors
        ce_loss = self.ce_loss(pred.permute(0, 2, 3, 1).reshape(-1, C),
                               target_indices.reshape(-1))
        
        # Focal loss style weighting for hard examples
        incorrect_mask = (pred_indices != target_indices).float().reshape(-1)
        # Focus more on pixels that are almost correct
        pred_probs = torch.softmax(pred.permute(0, 2, 3, 1).reshape(-1, C), dim=-1)
        target_probs = pred_probs.gather(1, target_indices.reshape(-1, 1)).squeeze()  
        focal_weight = (1 - target_probs) ** 2  # gamma=2 for more balanced focus
        ce_loss = ce_loss * (1 + incorrect_mask * 1 + focal_weight * 0.5)  # Further reduced
        ce_loss = ce_loss.mean()
        
        # 2. Exact match bonus (negative loss) - PROGRESSIVE
        exact_matches = (pred_indices == target_indices).all(dim=[1, 2]).float()
        # Progressive bonus that increases over time
        epoch_progress = min(1.0, self.current_epoch / 50.0) if hasattr(self, 'current_epoch') else 0.5
        bonus_weight = 1.0 + 2.0 * epoch_progress  # From 1.0 to 3.0 - more conservative
        exact_bonus = -bonus_weight * exact_matches.mean()
        
        # 3. Heavy penalty for copying when shouldn't
        should_not_copy = (target_indices != input_indices).any(dim=[1, 2]).float()
        did_copy = (pred_indices == input_indices).all(dim=[1, 2]).float()
        copy_penalty = 2.0 * (should_not_copy * did_copy).mean()  # Reduced from 5.0
        
        # 4. Transformation encouragement - WITH NaN protection
        changed_pixels = (pred_indices != input_indices).float().mean(dim=[1, 2])
        target_changed = (target_indices != input_indices).float().mean(dim=[1, 2])
        transform_diff = F.mse_loss(changed_pixels, target_changed) * 0.5  # Further reduced
        
        # 5. Color usage penalty - SAFER implementation
        color_penalty = 0.0
        try:
            batch_penalties = []
            for b in range(B):
                pred_colors = torch.unique(pred_indices[b])
                target_colors = torch.unique(target_indices[b])
                if len(target_colors) > 0 and len(pred_colors) > 0:
                    pred_set = set(pred_colors.cpu().numpy())
                    target_set = set(target_colors.cpu().numpy())
                    intersection = len(pred_set & target_set)
                    missing_colors = max(0, len(target_set) - intersection)
                    batch_penalties.append(missing_colors * 0.01)
            if batch_penalties:
                color_penalty = sum(batch_penalties) / len(batch_penalties)  # Average over batch
        except Exception:
            color_penalty = 0.0  # Fallback if any issues
        
        total_loss = ce_loss + exact_bonus + copy_penalty + transform_diff + color_penalty
        
        # NaN protection - only replace NaN, don't clamp valid values
        if torch.isnan(total_loss).any():
            print("WARNING: NaN in total loss, using fallback")
            total_loss = ce_loss  # Use just CE loss as fallback
        
        return {
            'total': total_loss,
            'reconstruction': ce_loss,
            'transformation': copy_penalty,  # Use copy penalty as transformation metric
            'exact_bonus': -exact_bonus,  # Show as positive in logs
            'exact_count': exact_matches.sum(),
            'copy_penalty': copy_penalty,
            'transform_diff': transform_diff,
            'color_penalty': color_penalty
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
    """Custom collate function to handle dictionary samples with different sizes"""
    # Find max size
    max_h = max(item['input'].shape[0] for item in batch)
    max_w = max(item['input'].shape[1] for item in batch)
    
    # Pad all to max size
    inputs = []
    outputs = []
    for item in batch:
        input_tensor = torch.tensor(item['input'], dtype=torch.long)
        output_tensor = torch.tensor(item['output'], dtype=torch.long)
        
        # Pad if needed
        h, w = input_tensor.shape
        if h < max_h or w < max_w:
            pad_h = max_h - h
            pad_w = max_w - w
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), value=0)
            output_tensor = F.pad(output_tensor, (0, pad_w, 0, pad_h), value=0)
        
        inputs.append(input_tensor)
        outputs.append(output_tensor)
    
    inputs = torch.stack(inputs)
    outputs = torch.stack(outputs)
    return {'input': inputs, 'output': outputs}


def inject_exact_match_training(model, device='cuda', num_epochs=100):
    """
    Special pre-training phase that forces exact match learning
    """
    
    print("\n🎯 EXACT MATCH INJECTION TRAINING")
    print("="*50)
    print("Enhanced with:")
    print("  • Focal loss (gamma=2)")
    print("  • Progressive exact match bonus (1x-3x)")
    print("  • Data augmentation")
    print("  • Warmup + cosine annealing")
    print("  • 300K total samples")
    
    # Initialize model weights for stable training
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    # Create ultra-focused dataset with curriculum - start with easier sizes
    datasets = []
    # Progressive difficulty: 3x3 -> 4x4 -> 5x5
    for size in [3, 4, 5]:
        datasets.append(ExactMatchBoostDataset(100000, fixed_size=size))  # Double dataset size!
    
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # With 80GB GPU, use much larger batch size
    batch_size = 512
    print(f"Using batch size: {batch_size} (optimized for 80GB GPU)")
    
    dataloader = torch.utils.data.DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=exact_match_collate_fn,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True
    )
    
    # COMPREHENSIVE: Use AggressiveLoss with proper LR for injection training
    # Start with very low LR to prevent exploding gradients
    initial_lr = 0.0001  # Even lower to prevent NaN
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5, betas=(0.9, 0.999))  # More stable betas
    
    # Linear warmup + CosineAnnealingWarmRestarts
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs * len(dataloader)  # Even lower start
    )
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Initial restart period
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    # Use AggressiveLoss for comprehensive exact match training
    loss_fn = AggressiveLoss()
    
    model.train()
    best_exact_match = 0
    patience_counter = 0
    best_model_state = None
    
    # Set current epoch for progressive bonus
    loss_fn.current_epoch = 0
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    # Initialize optimizer
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        loss_fn.current_epoch = epoch
        total_exact = 0
        total_samples = 0
        accumulated_loss = 0
        step_count = 0  # Track actual optimizer steps
        
        for batch_idx, batch in enumerate(dataloader):
            # Get tensors from batch
            inputs = batch['input'].to(device)
            outputs = batch['output'].to(device)
            
            # One-hot encode
            B, H, W = inputs.shape
            
            # Pad smaller grids to 5x5 for consistency
            if H < 5 or W < 5:
                pad_h = 5 - H
                pad_w = 5 - W
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), value=0)
                outputs = F.pad(outputs, (0, pad_w, 0, pad_h), value=0)
                B, H, W = inputs.shape
            
            inputs_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            outputs_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            # Add small input noise for regularization (only during training)
            if epoch > 10:  # Start after initial learning
                noise = torch.randn_like(inputs_oh) * 0.01
                inputs_oh = inputs_oh + noise
            
            # Forward pass without autocast for debugging
            if hasattr(model, '__class__') and model.__class__.__name__ == 'CHRONOS':
                pred = model([inputs_oh], target=outputs_oh)['predicted_output']
            else:
                pred = model(inputs_oh, outputs_oh, mode='train')['predicted_output']
            
            # Check for NaN in predictions
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"WARNING: NaN/Inf in model predictions, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Comprehensive AggressiveLoss calculation
            losses = loss_fn(pred, outputs_oh, inputs_oh)
            loss = losses['total']
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss, skipping batch")
                optimizer.zero_grad()
                continue
            
            # DEBUG: Print some stats on first batch of first epoch
            if epoch == 0 and batch_idx < 2:
                pred_classes = pred.argmax(dim=1)
                target_classes = outputs[:, :pred_classes.shape[1], :pred_classes.shape[2]]  # Handle padding
                print(f"DEBUG - Pred range: {pred_classes.min()}-{pred_classes.max()}, Target range: {target_classes.min()}-{target_classes.max()}")
                print(f"DEBUG - Pred unique: {torch.unique(pred_classes)}, Target unique: {torch.unique(target_classes)}")
                print(f"DEBUG - Loss components: recon={losses['reconstruction']:.4f}, exact_bonus={losses['exact_bonus']:.4f}, total={losses['total']:.4f}")
                print(f"DEBUG - Other: copy={losses['copy_penalty']:.4f}, transform={losses['transform_diff']:.4f}, color={losses['color_penalty']:.4f}")
            
            # Backward pass without scaler
            loss.backward()
            
            # Calculate gradient norm before clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if epoch == 0 and batch_idx < 2:
                print(f"DEBUG - Gradient norm before clipping: {total_norm:.4f}")
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Less aggressive clipping
            
            # Only skip if NaN/Inf
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"WARNING: NaN/Inf gradients detected, skipping step")
                optimizer.zero_grad()
                continue
            
            # Optimizer step without scaler
            optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Scheduler step after optimizer update
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                main_scheduler.step()
            step_count += 1
            
            # Track exact matches
            total_exact += losses['exact_count'].item()
            total_samples += B
            accumulated_loss += losses['total'].item()
        
        exact_pct = total_exact / total_samples * 100 if total_samples > 0 else 0
        avg_loss = accumulated_loss / max(1, step_count)
        print(f"Epoch {epoch+1}/{num_epochs}: Exact Match: {exact_pct:.1f}% | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.5f} | Steps: {step_count}/{len(dataloader)}")
        
        # Print detailed stats every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"📊 Stats: Best exact match so far: {best_exact_match:.1f}% | Patience: {patience_counter}/15")
        
        # Track best performance
        if exact_pct > best_exact_match:
            best_exact_match = exact_pct
            
        # Track best performance with patience
        if exact_pct > best_exact_match:
            best_exact_match = exact_pct
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping with patience
        if exact_pct > 70:  # Even higher threshold
            print("✅ Achieved >70% exact match! Stopping injection training.")
            break
        elif patience_counter > 20:  # More patience
            print(f"🛑 No improvement for 20 epochs. Best: {best_exact_match:.1f}%")
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
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