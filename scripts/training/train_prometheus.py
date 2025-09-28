"""
PROMETHEUS Individual Training Script
Creative Pattern Generation Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
import numpy as np
import json
import os
import sys
import gc
from tqdm import tqdm
from typing import Dict, List, Optional
import random

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import model and components
from src.models.prometheus_model import EnhancedPrometheusNet
from colab_training_v4_megascale_curriculum import (
    MegaScaleLoss, CurriculumMegaScaleDataset, TrainingReporter
)

# Training configuration
BATCH_SIZE = 512
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.01
NUM_EPOCHS = 300
MAX_GRID_SIZE = 30
NUM_COLORS = 10
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# Enhanced loss weights
RECONSTRUCTION_WEIGHT = 1.0
EDGE_WEIGHT = 0.3
COLOR_BALANCE_WEIGHT = 0.2
STRUCTURE_WEIGHT = 0.3
TRANSFORMATION_PENALTY = 2.0
EXACT_MATCH_BONUS = 5.0

# Curriculum settings
CURRICULUM_STAGES = 3
EPOCHS_PER_STAGE = 100

# VAE specific settings
KL_WEIGHT = 0.001  # KL divergence weight for VAE

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# Data directory
DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'


def custom_collate_fn(batch):
    """Custom collate function to handle different grid sizes"""
    # Find max dimensions
    max_h_in = max(item['inputs'].shape[0] for item in batch)
    max_w_in = max(item['inputs'].shape[1] for item in batch)
    max_h_out = max(item['outputs'].shape[0] for item in batch)
    max_w_out = max(item['outputs'].shape[1] for item in batch)
    max_h = max(max_h_in, max_h_out, MAX_GRID_SIZE)
    max_w = max(max_w_in, max_w_out, MAX_GRID_SIZE)
    
    # Pad all grids to max size
    inputs = []
    outputs = []
    for item in batch:
        # Pad inputs
        h_in, w_in = item['inputs'].shape
        pad_h = max_h - h_in
        pad_w = max_w - w_in
        padded_input = F.pad(item['inputs'], (0, pad_w, 0, pad_h), value=0)
        
        # Pad outputs
        h_out, w_out = item['outputs'].shape
        pad_h = max_h - h_out
        pad_w = max_w - w_out
        padded_output = F.pad(item['outputs'], (0, pad_w, 0, pad_h), value=0)
        
        inputs.append(padded_input)
        outputs.append(padded_output)
    
    return {
        'inputs': torch.stack(inputs),
        'outputs': torch.stack(outputs)
    }


class PrometheusLoss(MegaScaleLoss):
    """Extended loss for PROMETHEUS with VAE components"""
    def __init__(self, kl_weight=0.001):
        super().__init__()
        self.kl_weight = kl_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, input_grid: torch.Tensor,
                mu: Optional[torch.Tensor] = None, log_var: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Get base loss components
        losses = super().forward(pred, target, input_grid)
        
        # Add KL divergence loss if VAE components provided
        if mu is not None and log_var is not None:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss = kl_loss / (mu.size(0) * mu.size(1))  # Normalize by batch and latent size
            losses['kl'] = kl_loss
            losses['total'] = losses['total'] + self.kl_weight * kl_loss
        
        return losses


def train_prometheus():
    """Train PROMETHEUS model with curriculum learning"""
    print("\n" + "="*60)
    print("🔥 Training PROMETHEUS - Creative Pattern Generation")
    print("="*60)
    
    # Create model
    model = EnhancedPrometheusNet().to(device)
    print(f"Model: {model.description}")
    
    # Initialize reporter
    reporter = TrainingReporter('prometheus')
    
    # Loss function (with VAE component)
    loss_fn = PrometheusLoss(kl_weight=KL_WEIGHT)
    
    # Stage-adaptive learning rates
    stage_lrs = [LEARNING_RATE, LEARNING_RATE * 0.5, LEARNING_RATE * 0.2]
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=stage_lrs[0],
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )
    
    # Scheduler
    total_epochs = EPOCHS_PER_STAGE * CURRICULUM_STAGES
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    
    scaler = GradScaler('cuda')
    
    best_exact = 0
    best_val_loss = float('inf')
    global_epoch = 0
    patience_counter = 0
    max_patience = 20
    
    # Note: PROMETHEUS skips exact match injection due to VAE architecture
    print("\n📝 Note: PROMETHEUS skips exact match injection (VAE-based model)")
    
    # Curriculum training loop
    for stage in range(CURRICULUM_STAGES):
        print(f"\n📚 Starting Curriculum Stage {stage}")
        print("="*40)
        
        # Adjust learning rate for stage
        if stage > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = stage_lrs[stage]
            print(f"Learning rate adjusted to: {stage_lrs[stage]}")
        
        # Create dataset for this stage
        dataset = CurriculumMegaScaleDataset(
            DATA_DIR, 
            curriculum_stage=stage,
            use_arc_synthesis=True,
            synthesis_ratio=0.4 if stage == 0 else 0.3
        )
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True,
            collate_fn=custom_collate_fn
        )
        
        print(f"Stage {stage} - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        print(f"Batches per epoch: {len(train_loader):,}")
        
        # Train for this stage
        for epoch in range(EPOCHS_PER_STAGE):
            global_epoch += 1
            
            # Training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0, 'kl': 0}
            
            pbar = tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch+1}/{EPOCHS_PER_STAGE}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['inputs'].to(device, non_blocking=True)
                outputs = batch['outputs'].to(device, non_blocking=True)
                
                # Validate ranges
                inputs = torch.clamp(inputs, 0, 9)
                outputs = torch.clamp(outputs, 0, 9)
                
                # Convert to one-hot
                input_grids = F.one_hot(inputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                output_grids = F.one_hot(outputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                
                with autocast('cuda'):
                    model_outputs = model(input_grids, output_grids, mode='train')
                    pred_output = model_outputs['predicted_output']
                    mu = model_outputs['mu']
                    log_var = model_outputs['log_var']
                    
                    losses = loss_fn(pred_output, output_grids, input_grids, mu, log_var)
                    loss = losses['total'] / GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Update metrics
                train_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                train_metrics['exact'] += losses['exact_count'].item()
                train_metrics['samples'] += input_grids.size(0)
                if 'kl' in losses:
                    train_metrics['kl'] += losses['kl'].item() * input_grids.size(0)
                
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.3f}",
                    'exact': f"{losses['exact_count'].item():.0f}",
                    'kl': f"{losses.get('kl', 0):.4f}"
                })
            
            # Step scheduler
            scheduler.step()
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                val_metrics = {'loss': 0, 'exact': 0, 'pixel_acc': 0, 'samples': 0}
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        inputs = batch['inputs'].to(device, non_blocking=True)
                        outputs = batch['outputs'].to(device, non_blocking=True)
                        
                        # Validate ranges
                        inputs = torch.clamp(inputs, 0, 9)
                        outputs = torch.clamp(outputs, 0, 9)
                        
                        # Convert to one-hot
                        input_grids = F.one_hot(inputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        output_grids = F.one_hot(outputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        
                        with autocast('cuda'):
                            model_outputs = model(input_grids)
                            pred_output = model_outputs['predicted_output']
                            losses = loss_fn(pred_output, output_grids, input_grids)
                        
                        # Metrics
                        pred_indices = pred_output.argmax(dim=1)
                        target_indices = output_grids.argmax(dim=1)
                        
                        exact = (pred_indices == target_indices).all(dim=[1,2]).sum().item()
                        pixel_acc = (pred_indices == target_indices).float().mean().item()
                        
                        val_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                        val_metrics['exact'] += exact
                        val_metrics['pixel_acc'] += pixel_acc * input_grids.size(0)
                        val_metrics['samples'] += input_grids.size(0)
                
                # Calculate averages
                train_loss = train_metrics['loss'] / train_metrics['samples']
                train_exact_pct = train_metrics['exact'] / train_metrics['samples'] * 100
                train_kl = train_metrics['kl'] / train_metrics['samples'] if train_metrics['kl'] > 0 else 0
                
                val_loss = val_metrics['loss'] / val_metrics['samples']
                val_exact_pct = val_metrics['exact'] / val_metrics['samples'] * 100
                val_pixel_acc = val_metrics['pixel_acc'] / val_metrics['samples'] * 100
                
                print(f"\nGlobal Epoch {global_epoch} (Stage {stage}): "
                      f"Train Loss: {train_loss:.4f}, Train Exact: {train_exact_pct:.2f}%, KL: {train_kl:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Exact: {val_exact_pct:.2f}%, Pixel: {val_pixel_acc:.2f}%")
                
                # Add metrics to reporter
                reporter.add_metrics(
                    epoch=global_epoch,
                    stage=stage,
                    train_loss=train_loss,
                    train_exact=train_exact_pct,
                    val_loss=val_loss,
                    val_exact=val_exact_pct,
                    val_pixel_acc=val_pixel_acc,
                    lr=optimizer.param_groups[0]['lr'],
                    trans_penalty=TRANSFORMATION_PENALTY
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience and stage > 0:
                        print(f"⚠️ Early stopping triggered!")
                        break
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    os.makedirs('/content/AutomataNexus_Olympus_AGI2/arc_models_v4', exist_ok=True)
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'val_pixel_acc': val_pixel_acc,
                        'val_loss': val_loss
                    }, f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/prometheus_best.pt')
                    
                    print(f"✅ New best model! Exact: {val_exact_pct:.2f}%")
        
        # Check early stopping
        if patience_counter >= max_patience and stage > 0:
            print(f"📛 Stage {stage} terminated early")
            break
    
    # Generate report
    print(f"\n📊 Generating training report for PROMETHEUS...")
    reporter.generate_report({
        'best_exact': best_exact,
        'best_val_loss': best_val_loss,
        'total_epochs': global_epoch
    })
    print(f"✅ Report saved to: /content/AutomataNexus_Olympus_AGI2/src/models/reports/prometheus/")
    
    # Clear memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n🎉 PROMETHEUS training complete!")
    print(f"Best exact match: {best_exact:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    print("="*80)
    print("PROMETHEUS - Creative Pattern Generation Training")
    print("="*80)
    print(f"Batch size: {BATCH_SIZE} (effective {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Curriculum stages: {CURRICULUM_STAGES}")
    print(f"KL weight: {KL_WEIGHT}")
    print("="*80)
    
    train_prometheus()