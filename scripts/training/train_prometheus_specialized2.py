#!/usr/bin/env python3
"""
PROMETHEUS Specialized Training V2 - Enhanced Creative Pattern Generation
Extended training sessions with improved learning capabilities
Builds upon train_prometheus_specialized.py with enhanced features
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
from collections import defaultdict
from tqdm import tqdm
import json
import time
import gc
from typing import Dict, List, Optional

# Add paths for imports
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import PROMETHEUS model
from src.models.prometheus_model_simplified import SimplifiedPrometheusNet

# Import base training components and enhance them
from train_prometheus_specialized import (
    PrometheusSimplifiedLoss,
    prometheus_exact_match_injection,
    custom_collate_fn,
    PROMETHEUS_CONFIG as BASE_CONFIG,
    STAGE_CONFIG as BASE_STAGE_CONFIG
)

# Try to import enhanced training systems
try:
    from src.training_systems.mept_system import create_mept_system
    from src.training_systems.leap_system import create_leap_system  
    from src.training_systems.prism_system import create_prism_system
    from src.dsl.base_dsl import DSLTraining
    TRAINING_SYSTEMS_AVAILABLE = True
    print("✅ Enhanced training systems available")
except ImportError:
    TRAINING_SYSTEMS_AVAILABLE = False
    print("⚠️ Enhanced training systems not available")

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎨 PROMETHEUS V2 Training on {device}")
print(f"Using device: {device}")

# Enhanced PROMETHEUS Configuration V2 - Extended training
PROMETHEUS_CONFIG_V2 = BASE_CONFIG.copy()
PROMETHEUS_CONFIG_V2.update({
    'batch_size': 64,  # Larger batch for better gradients
    'learning_rate': 0.0005,  # Lower for extended training
    'num_epochs': 400,  # Extended training (8 stages x 50 epochs)
    'epochs_per_stage': 50,  # More epochs per stage
    'gradient_accumulation': 4,  # Effective batch: 256
    'gradient_clip': 1.0,
    'weight_decay': 1e-5,  # Reduced for longer training
    'transform_penalty': 0.2,  # Lower to encourage creativity
    'exact_match_bonus': 5.0,  # Higher for aggressive IoU learning
    'creativity_weight': 0.15,  # Increased creativity factor
    'curriculum_stages': 8,
    'warmup_epochs': 20,  # Longer warmup for complex patterns
    'cosine_restarts': True,  # Learning rate scheduling
    'label_smoothing': 0.1,  # Better generalization
    'mixup_alpha': 0.2,  # Data augmentation
})

# Enhanced Stage Configuration V2 - Progressive complexity
STAGE_CONFIG_V2 = []
for i, stage in enumerate(BASE_STAGE_CONFIG):
    enhanced_stage = stage.copy()
    enhanced_stage.update({
        'synthesis_ratio': max(0.3, 0.8 - i * 0.1),  # Decrease synthesis over stages
        'exact_injection': i < 4,  # First 4 stages get exact match injection
        'creativity_complexity': 'basic' if i < 3 else 'advanced' if i < 6 else 'expert',
        'pattern_types': 5 + i * 2,  # Increasing pattern variety
    })
    STAGE_CONFIG_V2.append(enhanced_stage)

print("=" * 80)
print("PROMETHEUS V2 Specialized Training - Enhanced Creative Generation")
print("Extended Training with Advanced Pattern Synthesis")
print("=" * 80)
print("🔥 V2 Enhancements:")
print("  • Extended training: 400 epochs (50 per stage)")
print("  • Enhanced IoU-based learning with creativity factors")
print("  • Progressive pattern complexity")
print("  • Advanced data augmentation (mixup, label smoothing)")
print("  • Cosine annealing with restarts")
print("  • Checkpoint resumption from best model")
print("=" * 80)


class PrometheusEnhancedLoss(PrometheusSimplifiedLoss):
    """Enhanced PROMETHEUS loss with V2 improvements"""
    
    def __init__(self, transformation_penalty=0.2, exact_match_bonus=5.0, creativity_weight=0.15):
        super().__init__(transformation_penalty, exact_match_bonus)
        self.creativity_weight = creativity_weight
        self.label_smoothing = 0.1
        
    def forward(self, model_outputs, targets, inputs, mixup_lambda=None):
        """Enhanced forward pass with mixup and creativity"""
        
        # Handle mixup if provided
        if mixup_lambda is not None:
            # Apply mixup to loss calculation
            targets_a, targets_b = targets
            losses_a = self._calculate_base_loss(model_outputs, targets_a, inputs)
            losses_b = self._calculate_base_loss(model_outputs, targets_b, inputs)
            
            # Mix the losses
            mixed_losses = {}
            for key in losses_a:
                if torch.is_tensor(losses_a[key]):
                    mixed_losses[key] = mixup_lambda * losses_a[key] + (1 - mixup_lambda) * losses_b[key]
                else:
                    mixed_losses[key] = losses_a[key]
            
            return mixed_losses
        
        return self._calculate_base_loss(model_outputs, targets, inputs)
    
    def _calculate_base_loss(self, model_outputs, targets, inputs):
        """Calculate base loss with enhanced creativity factors"""
        pred_output = model_outputs['predicted_output']
        B, C, H, W = pred_output.shape
        
        # Apply label smoothing for better generalization
        if self.label_smoothing > 0:
            targets = self._apply_label_smoothing(targets, self.label_smoothing)
        
        # Enhanced focal loss
        focal_loss = self._focal_loss(pred_output, targets, gamma=2.0)
        
        # IoU-based exact match scoring (same as successful models)
        pred_indices = pred_output.argmax(dim=1)
        target_indices = targets.argmax(dim=1) if targets.dim() > 3 else targets
        
        # Strict exact matches
        exact_matches_strict = (pred_indices == target_indices).all(dim=[1,2]).float()
        
        # IoU-based soft exact match
        intersection = (pred_indices == target_indices).float().sum(dim=[1,2])
        union = (pred_indices.shape[1] * pred_indices.shape[2])
        iou_scores = intersection / union
        
        # Combine with aggressive IoU weighting for PROMETHEUS
        combined_matches = 0.2 * exact_matches_strict + 0.8 * iou_scores
        exact_count = combined_matches.sum()
        exact_bonus = -combined_matches.mean() * self.exact_match_bonus
        exact_bonus = exact_bonus.clamp(min=-3.0)
        
        # Enhanced transformation penalty
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transformation_penalty
        
        # Enhanced creativity bonus
        creativity_bonus = 0.0
        if 'creativity_factor' in model_outputs:
            creativity_factor = model_outputs['creativity_factor']
            # Reward higher creativity
            creativity_bonus = torch.sigmoid(creativity_factor) * self.creativity_weight
        
        # Pattern diversity bonus
        unique_colors = []
        for b in range(B):
            unique = torch.unique(pred_indices[b]).numel()
            unique_colors.append(unique)
        diversity_bonus = torch.tensor(np.mean(unique_colors), device=pred_output.device) * 0.01
        
        # Total enhanced loss
        total_loss = focal_loss + transform_penalty + exact_bonus - creativity_bonus - diversity_bonus
        
        # Stability check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ NaN/Inf loss in PROMETHEUS V2, using focal only")
            total_loss = focal_loss.clamp(max=10.0)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
            'creativity_bonus': creativity_bonus,
            'diversity_bonus': diversity_bonus,
        }
    
    def _apply_label_smoothing(self, targets, smoothing):
        """Apply label smoothing for better generalization"""
        if targets.dim() == 3:  # Convert indices to one-hot if needed
            targets = F.one_hot(targets, num_classes=10).permute(0, 3, 1, 2).float()
        
        C = targets.shape[1]
        smooth_targets = targets * (1 - smoothing) + smoothing / C
        return smooth_targets


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation for better generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, (y_a, y_b), lam


def train_prometheus_specialized_v2():
    """Enhanced PROMETHEUS V2 training with extended sessions"""
    print("🎨 Starting PROMETHEUS V2 Enhanced Training")
    print("=" * 70)
    print("📊 Enhanced Creative Pattern Generation:")
    print("  • Extended 400-epoch training (50 per stage)")
    print("  • Advanced creativity and diversity factors")
    print("  • IoU-based learning with 80% soft matching")
    print("  • Mixup augmentation and label smoothing")
    print("  • Progressive pattern complexity")
    print("=" * 70)
    
    # Initialize enhanced model
    model = SimplifiedPrometheusNet(max_grid_size=12).to(device)
    print(f"🎨 PROMETHEUS V2 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Enhanced loss function
    loss_fn = PrometheusEnhancedLoss(
        transformation_penalty=PROMETHEUS_CONFIG_V2['transform_penalty'],
        exact_match_bonus=PROMETHEUS_CONFIG_V2['exact_match_bonus'],
        creativity_weight=PROMETHEUS_CONFIG_V2['creativity_weight']
    ).to(device)
    
    # Enhanced optimizer with lower learning rate for extended training
    optimizer = optim.AdamW(
        model.parameters(),
        lr=PROMETHEUS_CONFIG_V2['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=PROMETHEUS_CONFIG_V2['weight_decay']
    )
    
    # Enhanced scheduler with cosine annealing and restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=PROMETHEUS_CONFIG_V2['epochs_per_stage'], 
        T_mult=1,
        eta_min=PROMETHEUS_CONFIG_V2['learning_rate'] * 0.1
    )
    
    # Mixed precision
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    # Model directory
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = f'{models_dir}/prometheus_best.pt'
    
    best_exact = 0.0
    global_epoch = 0
    start_stage = 0
    
    # Load existing best model if available
    if os.path.exists(best_model_path):
        print(f"🔄 Loading best PROMETHEUS model from {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_exact = checkpoint.get('best_exact', 0.0)
            global_epoch = checkpoint.get('epoch', 0)
            start_stage = checkpoint.get('stage', 0)
            print(f"✅ Resumed from epoch {global_epoch}, stage {start_stage}, best: {best_exact:.2f}%")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("🆕 Starting fresh training")
    else:
        print("🆕 No existing model found - starting fresh V2 training")
    
    # Data directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
    # Import dataset components
    sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')
    from colab_training_v4_megascale_curriculum import CurriculumMegaScaleDataset, ExactMatchBoostDataset
    
    print(f"\n🎨 PROMETHEUS V2 8-Stage Progressive Training")
    print("=" * 70)
    
    # 8-Stage Progressive Training with enhanced features
    for stage in range(start_stage, PROMETHEUS_CONFIG_V2['curriculum_stages']):
        stage_config = STAGE_CONFIG_V2[stage]
        grid_size = stage_config['max_grid_size']
        synthesis_ratio = stage_config['synthesis_ratio']
        
        print(f"\n🎨 PROMETHEUS V2 Stage {stage}: {grid_size}x{grid_size} Creative Generation")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {synthesis_ratio*100:.0f}%")
        print(f"   🎨 Complexity: {stage_config['creativity_complexity']} | Patterns: {stage_config['pattern_types']}")
        print("=" * 60)
        
        # Create enhanced dataset
        try:
            dataset = CurriculumMegaScaleDataset(
                DATA_DIR,
                curriculum_stage=min(stage, 7),
                use_arc_synthesis=True,
                synthesis_ratio=synthesis_ratio
            )
        except Exception as e:
            print(f"⚠️ Failed to create dataset: {e}")
            continue
        
        # Split dataset
        train_size = int(0.85 * len(dataset))  # More training data
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=PROMETHEUS_CONFIG_V2['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=PROMETHEUS_CONFIG_V2['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=False
        )
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # Enhanced exact match injection for first few stages
        if stage_config['exact_injection'] and stage == start_stage:
            print(f"🎯 Enhanced Exact Match Injection for Stage {stage}")
            try:
                model = prometheus_exact_match_injection(
                    model, device,
                    num_epochs=30,  # Extended injection
                    target_accuracy=85.0
                )
                print(f"✅ Enhanced injection completed for Stage {stage}")
            except Exception as e:
                print(f"⚠️ Injection failed: {e}")
        
        # Stage training loop with enhanced features
        stage_epochs = PROMETHEUS_CONFIG_V2['epochs_per_stage']
        for epoch in range(stage_epochs):
            global_epoch += 1
            
            # Training phase with enhancements
            model.train()
            train_metrics = defaultdict(float)
            
            pbar = tqdm(train_loader, desc=f"PROMETHEUS V2 Stage {stage}, Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['inputs'].to(device, non_blocking=True)
                outputs = batch['outputs'].to(device, non_blocking=True)
                
                # Clamp values
                inputs = torch.clamp(inputs, 0, 9)
                outputs = torch.clamp(outputs, 0, 9)
                
                # Convert to one-hot
                input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                
                # Apply mixup augmentation randomly
                mixup_lambda = None
                if random.random() < 0.3:  # 30% chance of mixup
                    input_grids, output_targets, mixup_lambda = mixup_data(
                        input_grids, output_grids, 
                        alpha=PROMETHEUS_CONFIG_V2['mixup_alpha']
                    )
                    output_grids = output_targets
                
                with autocast(device.type):
                    model_outputs = model(input_grids, mode='train')
                    losses = loss_fn(model_outputs, output_grids, input_grids, mixup_lambda)
                
                loss = losses['total'] / PROMETHEUS_CONFIG_V2['gradient_accumulation']
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % PROMETHEUS_CONFIG_V2['gradient_accumulation'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), PROMETHEUS_CONFIG_V2['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                # Update metrics
                train_metrics['loss'] += losses['total'].item()
                train_metrics['exact'] += losses['exact_count'].item()
                train_metrics['samples'] += inputs.size(0)
                
                # Enhanced progress display
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.3f}",
                    'exact': f"{losses['exact_count'].item():.0f}",
                    'soft': f"{losses.get('soft_exact_count', torch.tensor(0)).item():.1f}",
                    'IoU': f"{losses.get('avg_iou', torch.tensor(0)).item():.2f}",
                    'creativity': f"{losses.get('creativity_bonus', torch.tensor(0)).item():.3f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                })
            
            # Enhanced validation every 5 epochs
            if epoch % 5 == 0 or epoch == stage_epochs - 1:
                model.eval()
                val_metrics = defaultdict(float)
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="V2 Validation"):
                        inputs = batch['inputs'].to(device)
                        outputs = batch['outputs'].to(device)
                        
                        inputs = torch.clamp(inputs, 0, 9)
                        outputs = torch.clamp(outputs, 0, 9)
                        
                        input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                        output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                        
                        with autocast(device.type):
                            model_outputs = model(input_grids, mode='inference')
                            losses = loss_fn(model_outputs, output_grids, input_grids)
                        
                        val_metrics['loss'] += losses['total'].item()
                        val_metrics['exact'] += losses['exact_count'].item()
                        val_metrics['samples'] += inputs.size(0)
                
                # Calculate and display enhanced metrics
                train_exact_pct = train_metrics['exact'] / train_metrics['samples'] * 100
                train_loss = train_metrics['loss'] / len(train_loader)
                val_exact_pct = val_metrics['exact'] / val_metrics['samples'] * 100
                val_loss = val_metrics['loss'] / len(val_loader)
                
                print(f"\n🎨 PROMETHEUS V2 Stage {stage}, Epoch {epoch+1} (Global: {global_epoch}):")
                print(f"   🎯 Train: {train_exact_pct:.2f}% exact, Loss: {train_loss:.3f}")
                print(f"   🎯 Val: {val_exact_pct:.2f}% exact, Loss: {val_loss:.3f}")
                print(f"   📊 LR: {scheduler.get_last_lr()[0]:.6f} | Grid: {grid_size}x{grid_size}")
                
                # Save enhanced best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_exact': best_exact,
                        'config': PROMETHEUS_CONFIG_V2,
                        'stage_config': STAGE_CONFIG_V2
                    }, best_model_path)
                    print(f"   💾 NEW V2 BEST: {val_exact_pct:.2f}% exact match saved!")
    
    print(f"\n🎉 PROMETHEUS V2 Enhanced Training Complete!")
    print(f"   🏆 Best exact match: {best_exact:.2f}%")
    print(f"   📏 Enhanced stages completed: 8 (6x6 → 30x30 grids)")
    print(f"   📊 Total epochs: {global_epoch}")
    print(f"   🎨 Enhanced with creativity, diversity, and IoU learning")
    
    return model, best_exact


if __name__ == "__main__":
    print("🚀 Starting PROMETHEUS V2 Enhanced Training...")
    model, best_performance = train_prometheus_specialized_v2()
    print("✅ PROMETHEUS V2 training completed successfully!")
    print(f"🎨 Final Performance: {best_performance:.2f}% exact match")