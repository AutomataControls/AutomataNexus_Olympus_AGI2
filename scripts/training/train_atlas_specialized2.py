"""
ATLAS Specialized Training Script V2 - Enhanced Spatial Transformation Expert
Improved version with better learning dynamics and stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.amp import GradScaler, autocast
import numpy as np
import json
import os
import sys
import gc
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import ATLAS model
from src.models.atlas_model import EnhancedAtlasNet

# Import ALL AutomataNexus novel training components
from src.dsl import DSLTrainingIntegration, DSLProgramGenerator
from src.dsl.atlas_dsl import ATLASDSLTraining, ATLASDSLGenerator
from src.program_synthesis.synthesis_integration import LightweightProgramSynthesizer, ProgramSynthesisDataGenerator

# ATLAS-specific program synthesis
try:
    from src.program_synthesis.atlas_synthesis import ATLASProgramSynthesizer, create_atlas_synthesis_system
    ATLAS_SYNTHESIS_AVAILABLE = True
except ImportError:
    ATLAS_SYNTHESIS_AVAILABLE = False
    print("⚠️ ATLAS-specific synthesis not available")

# PRISM System - Use ATLAS-specific version
try:
    from src.training_systems.atlas_prism import create_atlas_prism_system
    ATLAS_PRISM_AVAILABLE = True
except ImportError:
    ATLAS_PRISM_AVAILABLE = False
    print("⚠️ ATLAS-specific PRISM not available")
    # Fallback to generic version
    try:
        from src.program_synthesis.prism_system import PRISMSynthesizer, create_prism_system
        PRISM_AVAILABLE = True
    except ImportError:
        PRISM_AVAILABLE = False
        print("⚠️ Generic PRISM not available")

# MEPT and LEAP Systems - Use ATLAS-specific versions
try:
    from src.training_systems.atlas_mept import create_atlas_mept_system, AtlasMEPTLoss
    from src.training_systems.atlas_leap import create_atlas_leap_system, AtlasLEAPTrainer
    ATLAS_MEPT_LEAP_AVAILABLE = True
except ImportError:
    ATLAS_MEPT_LEAP_AVAILABLE = False
    print("⚠️ ATLAS-specific MEPT/LEAP not available")
    # Fallback to generic versions
    try:
        from src.utils.mept_system import ExperienceReplayBuffer, PatternBank, MEPTLoss, create_mept_system
        from src.utils.leap_system import AdaptivePatternGenerator, LEAPTrainer, create_leap_system
        MEPT_LEAP_AVAILABLE = True
    except ImportError:
        MEPT_LEAP_AVAILABLE = False
        print("⚠️ Generic MEPT/LEAP not available")

# LEAP-PRISM Bridge
try:
    from src.utils.leap_prism_bridge import create_leap_prism_bridge, LEAPPatternEnhancer
    LEAP_PRISM_BRIDGE_AVAILABLE = True
except ImportError:
    LEAP_PRISM_BRIDGE_AVAILABLE = False
    print("⚠️ LEAP-PRISM bridge not available")

# Exact Match Injection System
try:
    from stage0_exact_match_boost import ExactMatchBoostDataset, AggressiveLoss, inject_exact_match_training
    EXACT_BOOST_AVAILABLE = True
except ImportError:
    EXACT_BOOST_AVAILABLE = False
    print("⚠️ Exact match boost not available")

# Data systems
from src.data.arc_data_synthesis import ARCDataSynthesizer, ARCDataAugmenter
from colab_training_v4_megascale_curriculum import CurriculumMegaScaleDataset, TrainingReporter

# Enhanced ATLAS Configuration - V2
ATLAS_CONFIG_V2 = {
    # Core training params
    'batch_size': 32,  # Smaller for better gradient quality
    'learning_rate': 0.001,  # Conservative start
    'num_epochs': 400,  # 8 stages x 50 epochs
    'hidden_dim': 256,  # Increased model capacity
    
    # Memory and optimization
    'spatial_memory_size': 500,  # Larger memory
    'gradient_accumulation': 2,  # Effective batch: 64
    'gradient_clip': 1.0,  # Gradient clipping
    
    # Loss weights - carefully balanced
    'reconstruction_weight': 1.0,
    'transform_penalty': 0.3,  # Moderate penalty
    'exact_match_bonus': 3.0,  # Moderate bonus
    'focal_gamma': 1.5,  # Less aggressive focal loss
    'spatial_weight': 0.2,  # Balanced spatial loss
    'consistency_weight': 0.3,  # New: temporal consistency
    
    # Curriculum settings
    'curriculum_stages': 8,
    'epochs_per_stage': 50,  # More epochs per stage
    'warmup_epochs': 10,  # Longer warmup
    
    # Advanced features
    'use_ema': True,  # Exponential moving average
    'ema_decay': 0.999,
    'label_smoothing': 0.1,  # Prevent overconfidence
    'dropout_rate': 0.1,  # During training only
    'augmentation_prob': 0.3,  # Data augmentation
    
    # Adaptive learning
    'lr_patience': 10,  # Epochs before LR reduction
    'lr_factor': 0.5,  # LR reduction factor
    'early_stop_patience': 20,  # Early stopping
    'min_improvement': 0.001,  # Minimum improvement threshold
}

# Enhanced Stage Configuration - V2
STAGE_CONFIG_V2 = {
    0: {'max_grid_size': 6,  'synthesis_ratio': 0.9, 'lr_mult': 1.0, 'dropout': 0.0},
    1: {'max_grid_size': 8,  'synthesis_ratio': 0.8, 'lr_mult': 1.0, 'dropout': 0.05},
    2: {'max_grid_size': 10, 'synthesis_ratio': 0.7, 'lr_mult': 0.9, 'dropout': 0.1},
    3: {'max_grid_size': 13, 'synthesis_ratio': 0.6, 'lr_mult': 0.8, 'dropout': 0.1},
    4: {'max_grid_size': 16, 'synthesis_ratio': 0.5, 'lr_mult': 0.7, 'dropout': 0.15},
    5: {'max_grid_size': 20, 'synthesis_ratio': 0.4, 'lr_mult': 0.6, 'dropout': 0.15},
    6: {'max_grid_size': 25, 'synthesis_ratio': 0.3, 'lr_mult': 0.5, 'dropout': 0.2},
    7: {'max_grid_size': 30, 'synthesis_ratio': 0.2, 'lr_mult': 0.4, 'dropout': 0.2}
}

# Training flags
USE_MEPT = True and (ATLAS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_LEAP = True and (ATLAS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_PRISM = True and (ATLAS_PRISM_AVAILABLE or PRISM_AVAILABLE)
USE_EXACT_BOOST = True and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Using device: {device}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# Data directory
DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'


class EnhancedAtlasLoss(nn.Module):
    """Enhanced ATLAS loss with better stability and learning dynamics"""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
    def forward(self, pred_output, target_output, input_grid, model_outputs=None):
        """Enhanced loss calculation with multiple components"""
        B, C, H, W = pred_output.shape
        
        # 1. Base reconstruction loss with label smoothing
        if self.config['label_smoothing'] > 0:
            target_smooth = self._apply_label_smoothing(target_output)
            base_loss = F.kl_div(
                F.log_softmax(pred_output, dim=1),
                target_smooth,
                reduction='batchmean'
            )
        else:
            base_loss = F.cross_entropy(
                pred_output,
                target_output.argmax(dim=1)
            )
        
        # 2. Focal loss for hard examples
        focal_loss = self._focal_loss(pred_output, target_output, gamma=self.config['focal_gamma'])
        
        # 3. Spatial consistency loss
        spatial_loss = self._enhanced_spatial_loss(pred_output, target_output)
        
        # 4. Transformation penalty
        transform_loss = self._transformation_penalty(pred_output, input_grid)
        
        # 5. Exact match bonus (negative to reduce total loss)
        pred_indices = pred_output.argmax(dim=1)
        target_indices = target_output.argmax(dim=1)
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_bonus = -exact_matches.mean() * self.config['exact_match_bonus']
        
        # 6. Consistency regularization
        consistency_loss = self._consistency_regularization(pred_output, model_outputs)
        
        # Combine all losses
        total_loss = (
            base_loss * self.config['reconstruction_weight'] +
            focal_loss * 0.5 +  # Focal as auxiliary
            spatial_loss * self.config['spatial_weight'] +
            transform_loss * self.config['transform_penalty'] +
            exact_bonus +
            consistency_loss * self.config.get('consistency_weight', 0.3)
        )
        
        # Prevent extreme losses
        total_loss = torch.clamp(total_loss, min=-5.0, max=10.0)
        
        return {
            'total': total_loss,
            'base': base_loss,
            'focal': focal_loss,
            'spatial': spatial_loss,
            'transform': transform_loss,
            'exact_bonus': exact_bonus,
            'exact_count': exact_matches.sum(),
            'consistency': consistency_loss
        }
    
    def _apply_label_smoothing(self, targets):
        """Apply label smoothing to prevent overconfidence"""
        C = targets.shape[1]
        smooth_targets = targets * (1 - self.config['label_smoothing'])
        smooth_targets += self.config['label_smoothing'] / C
        return smooth_targets
    
    def _focal_loss(self, pred, target, gamma=1.5):
        """Focal loss for handling hard examples"""
        target_idx = target.argmax(dim=1)
        ce_loss = F.cross_entropy(pred, target_idx, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** gamma * ce_loss
        return focal.mean()
    
    def _enhanced_spatial_loss(self, pred, target):
        """Enhanced spatial consistency with edge awareness"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Basic difference
        diff = (pred_idx != target_idx).float()
        
        # Edge-aware loss
        pred_edges = self._detect_edges(pred_idx)
        target_edges = self._detect_edges(target_idx)
        edge_loss = F.mse_loss(pred_edges, target_edges)
        
        # Local consistency
        kernel = torch.ones(3, 3, device=pred.device).unsqueeze(0).unsqueeze(0) / 9
        pred_smooth = F.conv2d(pred_idx.unsqueeze(1).float(), kernel, padding=1)
        target_smooth = F.conv2d(target_idx.unsqueeze(1).float(), kernel, padding=1)
        consistency = F.mse_loss(pred_smooth, target_smooth)
        
        return diff.mean() + edge_loss * 0.5 + consistency * 0.3
    
    def _detect_edges(self, grid):
        """Simple edge detection using gradients"""
        dx = torch.abs(grid[:, 1:, :] - grid[:, :-1, :])
        dy = torch.abs(grid[:, :, 1:] - grid[:, :, :-1])
        # Pad to original size
        dx = F.pad(dx, (0, 0, 0, 1))
        dy = F.pad(dy, (0, 1, 0, 0))
        return (dx + dy).float()
    
    def _transformation_penalty(self, pred, input_grid):
        """Penalize simple copying of input"""
        pred_idx = pred.argmax(dim=1)
        input_idx = input_grid.argmax(dim=1)
        
        # Check if prediction is too similar to input
        similarity = (pred_idx == input_idx).float().mean()
        
        # Only penalize high similarity (>80%)
        penalty = torch.relu(similarity - 0.8) * 2.0
        
        return penalty
    
    def _consistency_regularization(self, pred, model_outputs):
        """Regularize model predictions for consistency"""
        if model_outputs is None:
            return torch.tensor(0.0, device=pred.device)
        
        consistency_loss = 0.0
        
        # If model provides multiple predictions, ensure consistency
        if 'auxiliary_output' in model_outputs:
            aux_pred = model_outputs['auxiliary_output']
            consistency_loss += F.mse_loss(pred, aux_pred) * 0.5
        
        return consistency_loss


class SpatialAugmentation:
    """Data augmentation specifically for spatial patterns"""
    def __init__(self, prob=0.3):
        self.prob = prob
        
    def __call__(self, input_grid, output_grid):
        """Apply spatial augmentations"""
        if random.random() > self.prob:
            return input_grid, output_grid
        
        # Random rotation (0, 90, 180, 270 degrees)
        if random.random() < 0.5:
            k = random.randint(0, 3)
            input_grid = torch.rot90(input_grid, k, dims=[-2, -1])
            output_grid = torch.rot90(output_grid, k, dims=[-2, -1])
        
        # Random flip
        if random.random() < 0.3:
            if random.random() < 0.5:
                input_grid = torch.flip(input_grid, dims=[-2])
                output_grid = torch.flip(output_grid, dims=[-2])
            else:
                input_grid = torch.flip(input_grid, dims=[-1])
                output_grid = torch.flip(output_grid, dims=[-1])
        
        return input_grid, output_grid


class ExponentialMovingAverage:
    """EMA for model weights to improve stability"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


def create_atlas_dataset_v2(stage: int, data_dir: str) -> Tuple[Dataset, Dataset]:
    """Create enhanced dataset for ATLAS training"""
    grid_size = STAGE_CONFIG_V2[stage]['max_grid_size']
    synthesis_ratio = STAGE_CONFIG_V2[stage]['synthesis_ratio']
    
    print(f"🔧 Creating enhanced dataset for stage {stage} (grid size: {grid_size})")
    
    # Load base ARC data
    dataset = CurriculumMegaScaleDataset(
        data_dir,
        curriculum_stage=stage,
        use_arc_synthesis=True,
        synthesis_ratio=synthesis_ratio
    )
    
    # Add synthetic spatial transformation examples
    synthetic_samples = []
    for _ in range(int(len(dataset) * 0.3)):  # 30% synthetic
        size = random.randint(4, min(grid_size, 8))
        
        # Create various spatial patterns
        pattern_type = random.choice(['rotation', 'reflection', 'translation', 'scaling'])
        
        if pattern_type == 'rotation':
            input_grid = torch.randint(0, 5, (size, size))
            k = random.randint(1, 3)
            output_grid = torch.rot90(input_grid, k)
        
        elif pattern_type == 'reflection':
            input_grid = torch.randint(0, 5, (size, size))
            if random.random() < 0.5:
                output_grid = torch.flip(input_grid, dims=[0])
            else:
                output_grid = torch.flip(input_grid, dims=[1])
        
        elif pattern_type == 'translation':
            input_grid = torch.zeros(size, size, dtype=torch.long)
            input_grid[1:3, 1:3] = torch.randint(1, 5, (2, 2))
            output_grid = torch.zeros(size, size, dtype=torch.long)
            shift = random.randint(1, 2)
            output_grid[1+shift:3+shift, 1+shift:3+shift] = input_grid[1:3, 1:3]
        
        else:  # scaling
            small_size = size // 2
            pattern = torch.randint(0, 5, (small_size, small_size))
            input_grid = pattern.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)[:size, :size]
            output_grid = pattern
            # Pad output to match size
            output_grid = F.pad(output_grid, (0, size-small_size, 0, size-small_size))
        
        synthetic_samples.append({
            'inputs': input_grid.numpy(),
            'outputs': output_grid.numpy()
        })
    
    # Combine datasets
    class CombinedDataset(Dataset):
        def __init__(self, base_dataset, synthetic_samples):
            self.base_dataset = base_dataset
            self.synthetic_samples = synthetic_samples
            self.total_len = len(base_dataset) + len(synthetic_samples)
            
        def __len__(self):
            return self.total_len
            
        def __getitem__(self, idx):
            if idx < len(self.base_dataset):
                return self.base_dataset[idx]
            else:
                return self.synthetic_samples[idx - len(self.base_dataset)]
    
    combined_dataset = CombinedDataset(dataset, synthetic_samples)
    
    # Split into train/val
    train_size = int(0.9 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    
    return train_dataset, val_dataset


def train_epoch_v2(model, train_loader, optimizer, loss_fn, scaler, 
                   augmentation, ema, config, stage, epoch):
    """Enhanced training epoch with better practices"""
    model.train()
    
    # Set dropout rate for this stage
    dropout_rate = STAGE_CONFIG_V2[stage]['dropout']
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
    
    metrics = defaultdict(float)
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        inputs = batch['inputs'].to(device)
        outputs = batch['outputs'].to(device)
        
        # Ensure valid range
        inputs = torch.clamp(inputs, 0, 9)
        outputs = torch.clamp(outputs, 0, 9)
        
        # Apply augmentation
        if augmentation and stage > 0:  # No augmentation for stage 0
            inputs, outputs = augmentation(inputs, outputs)
        
        # Convert to one-hot
        input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
        output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
        
        # Forward pass with mixed precision
        with autocast('cuda'):
            model_outputs = model(input_grids, output_grids, mode='train')
            pred_output = model_outputs['predicted_output']
            
            losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
            loss = losses['total'] / config['gradient_accumulation']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config['gradient_accumulation'] == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA
            if ema:
                ema.update()
        
        # Update metrics
        for key, value in losses.items():
            if key != 'total':
                metrics[key] += value.item() if torch.is_tensor(value) else value
        metrics['loss'] += losses['total'].item()
        metrics['samples'] += inputs.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.3f}",
            'exact': f"{losses['exact_count'].item():.0f}",
            'spatial': f"{losses.get('spatial', 0):.3f}"
        })
    
    # Calculate averages
    for key in metrics:
        if key != 'samples' and key != 'exact_count':
            metrics[key] /= num_batches
    
    metrics['exact_pct'] = metrics.get('exact_count', 0) / metrics['samples'] * 100
    
    return metrics


def validate_epoch_v2(model, val_loader, loss_fn, ema, config):
    """Enhanced validation with EMA"""
    model.eval()
    
    # Apply EMA weights for validation
    if ema:
        ema.apply_shadow()
    
    metrics = defaultdict(float)
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs = batch['inputs'].to(device)
            outputs = batch['outputs'].to(device)
            
            inputs = torch.clamp(inputs, 0, 9)
            outputs = torch.clamp(outputs, 0, 9)
            
            input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            with autocast('cuda'):
                model_outputs = model(input_grids, mode='inference')
                pred_output = model_outputs['predicted_output']
                
                losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
            
            # Calculate pixel accuracy
            pred_idx = pred_output.argmax(dim=1)
            target_idx = output_grids.argmax(dim=1)
            pixel_acc = (pred_idx == target_idx).float().mean()
            
            # Update metrics
            for key, value in losses.items():
                if key != 'total':
                    metrics[key] += value.item() if torch.is_tensor(value) else value
            metrics['loss'] += losses['total'].item()
            metrics['pixel_acc'] += pixel_acc.item()
            metrics['samples'] += inputs.size(0)
    
    # Restore original weights
    if ema:
        ema.restore()
    
    # Calculate averages
    for key in metrics:
        if key != 'samples' and key != 'exact_count':
            metrics[key] /= num_batches
    
    metrics['exact_pct'] = metrics.get('exact_count', 0) / metrics['samples'] * 100
    metrics['pixel_acc'] *= 100  # Convert to percentage
    
    return metrics


def train_atlas_specialized_v2():
    """Enhanced ATLAS training with improved techniques"""
    print("🌍 ATLAS Specialized Training V2 - Enhanced Edition")
    print("=" * 60)
    print("📊 Key Improvements:")
    print("  • Better loss balancing and stability")
    print("  • Exponential Moving Average (EMA)")
    print("  • Label smoothing and focal loss")
    print("  • Enhanced data augmentation")
    print("  • Adaptive learning rate with patience")
    print("  • Better curriculum progression")
    print("=" * 60)
    
    # Initialize model
    max_grid_size = STAGE_CONFIG_V2[7]['max_grid_size']
    model = EnhancedAtlasNet(
        max_grid_size=max_grid_size,
        hidden_dim=ATLAS_CONFIG_V2['hidden_dim']
    ).to(device)
    
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize loss function
    loss_fn = EnhancedAtlasLoss(ATLAS_CONFIG_V2).to(device)
    
    # Initialize optimizer with AdamW for better regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ATLAS_CONFIG_V2['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    # Initialize EMA
    ema = ExponentialMovingAverage(model, decay=ATLAS_CONFIG_V2['ema_decay']) if ATLAS_CONFIG_V2['use_ema'] else None
    
    # Initialize augmentation
    augmentation = SpatialAugmentation(prob=ATLAS_CONFIG_V2['augmentation_prob'])
    
    # Training setup
    scaler = GradScaler('cuda')
    best_exact = 0.0
    global_epoch = 0
    
    # Create results directory
    results_dir = '/content/AutomataNexus_Olympus_AGI2/results/atlas_v2'
    os.makedirs(results_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_exact': [],
        'val_exact': [],
        'learning_rates': []
    }
    
    # Stage-wise training
    for stage in range(ATLAS_CONFIG_V2['curriculum_stages']):
        print(f"\n🌍 STAGE {stage}: {STAGE_CONFIG_V2[stage]['max_grid_size']}x{STAGE_CONFIG_V2[stage]['max_grid_size']} grids")
        print("=" * 60)
        
        # Create datasets
        train_dataset, val_dataset = create_atlas_dataset_v2(stage, DATA_DIR)
        print(f"📚 Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=ATLAS_CONFIG_V2['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=ATLAS_CONFIG_V2['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Adjust learning rate for stage
        stage_lr = ATLAS_CONFIG_V2['learning_rate'] * STAGE_CONFIG_V2[stage]['lr_mult']
        for param_group in optimizer.param_groups:
            param_group['lr'] = stage_lr
        
        # Stage-specific scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximize exact match
            factor=ATLAS_CONFIG_V2['lr_factor'],
            patience=ATLAS_CONFIG_V2['lr_patience'],
            min_lr=1e-6,
            verbose=True
        )
        
        # Warmup for first epochs
        warmup_epochs = ATLAS_CONFIG_V2['warmup_epochs'] if stage == 0 else 5
        
        # Training loop for this stage
        stage_best_exact = 0.0
        patience_counter = 0
        
        for epoch in range(ATLAS_CONFIG_V2['epochs_per_stage']):
            global_epoch += 1
            
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = stage_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Train epoch
            train_metrics = train_epoch_v2(
                model, train_loader, optimizer, loss_fn, 
                scaler, augmentation, ema, ATLAS_CONFIG_V2, 
                stage, epoch
            )
            
            # Validate epoch
            val_metrics = validate_epoch_v2(
                model, val_loader, loss_fn, ema, ATLAS_CONFIG_V2
            )
            
            # Update scheduler (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step(val_metrics['exact_pct'])
            
            # Log metrics
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n📈 Epoch {epoch+1}/{ATLAS_CONFIG_V2['epochs_per_stage']} (Global: {global_epoch})")
            print(f"  Train: Loss={train_metrics['loss']:.4f}, Exact={train_metrics['exact_pct']:.2f}%")
            print(f"  Val:   Loss={val_metrics['loss']:.4f}, Exact={val_metrics['exact_pct']:.2f}%, Pixel={val_metrics['pixel_acc']:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_exact'].append(train_metrics['exact_pct'])
            history['val_exact'].append(val_metrics['exact_pct'])
            history['learning_rates'].append(current_lr)
            
            # Save best model
            if val_metrics['exact_pct'] > best_exact:
                best_exact = val_metrics['exact_pct']
                stage_best_exact = val_metrics['exact_pct']
                patience_counter = 0
                
                torch.save({
                    'epoch': global_epoch,
                    'stage': stage,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_shadow': ema.shadow if ema else None,
                    'best_exact': best_exact,
                    'config': ATLAS_CONFIG_V2
                }, f"{results_dir}/atlas_v2_best.pt")
                
                print(f"🏆 New best model! Exact: {best_exact:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping for stage
            if patience_counter >= ATLAS_CONFIG_V2['early_stop_patience'] and epoch > warmup_epochs + 10:
                print(f"⚠️ Early stopping triggered for stage {stage}")
                break
            
            # Save checkpoint
            if epoch % 10 == 0 or epoch == ATLAS_CONFIG_V2['epochs_per_stage'] - 1:
                torch.save({
                    'epoch': global_epoch,
                    'stage': stage,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_shadow': ema.shadow if ema else None,
                    'history': history
                }, f"{results_dir}/atlas_v2_checkpoint.pt")
        
        print(f"\n✅ Stage {stage} complete! Best exact: {stage_best_exact:.2f}%")
        
        # Reduce learning rate for next stage
        if stage < ATLAS_CONFIG_V2['curriculum_stages'] - 1:
            print(f"🔄 Preparing for next stage...")
            gc.collect()
            torch.cuda.empty_cache()
    
    # Training complete
    print("\n" + "=" * 60)
    print("🎉 ATLAS V2 Training Complete!")
    print(f"🏆 Best exact match: {best_exact:.2f}%")
    print(f"📊 Total epochs: {global_epoch}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    plt.plot(history['val_loss'], label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_exact'], label='Train Exact', alpha=0.7)
    plt.plot(history['val_exact'], label='Val Exact', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Exact Match %')
    plt.title('Exact Match History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rates'], alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, history


if __name__ == "__main__":
    print("=" * 80)
    print("ATLAS Specialized Training V2 - Enhanced Edition")
    print("Spatial Transformation Expert with Advanced Training Techniques")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run training
    model, history = train_atlas_specialized_v2()