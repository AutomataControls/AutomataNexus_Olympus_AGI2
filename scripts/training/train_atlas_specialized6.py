"""
ATLAS Specialized Training V6 - Ultimate 2D Spatial Reasoning Master for ARC-AGI-2
Complete grid mastery (5x5 to 30x30) with deep spatial architecture and geometric synthesis
Builds upon V5 with revolutionary spatial intelligence capabilities and massive data pipeline
Target: 90%+ performance with ultimate spatial intelligence mastery
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
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import random
from collections import defaultdict

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import enhanced ATLAS V4 model (contains V6 improvements)
from src.models.atlas_v4_enhanced import AtlasV4Enhanced

# Enhanced ATLAS V6 Configuration - Ultimate Spatial Intelligence Focus
ATLAS_V6_CONFIG = {
    # Core Training Parameters - OPTIMIZED for V6 Ultimate Performance
    'batch_size': 24,  # Balanced for deep spatial architecture
    'learning_rate': 0.00015,  # Lower for deep spatial learning
    'num_epochs': 580,  # Extended: 20 stages x 29 epochs
    'gradient_accumulation': 4,  # Effective batch 96 for stability
    'epochs_per_stage': 29,  # Deep learning per stage
    'curriculum_stages': 20,  # Complete grid mastery 5x5 -> 30x30
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.02,  # Very low - maximum spatial exploration
    'exact_match_bonus': 10.5,  # Highest bonus for ultimate precision
    'gradient_clip': 0.5,  # Stable clipping for deep architecture
    'weight_decay': 1.8e-6,  # Ultra-light regularization for ultimate spatial strategy
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'spatial_reasoning_weight': 0.75,  # Ultimate focus - spatial intelligence
    'geometric_transformation_weight': 0.68,  # Maximum geometric mastery
    'multiscale_processing_weight': 0.62,  # Ultimate multi-scale understanding
    'ensemble_coordination_weight': 0.55,  # Maximum ensemble integration
    'pattern_analysis_weight': 0.48,  # Ultimate pattern analysis
    'decision_confidence_weight': 0.45,  # Ultimate decision confidence
    'spatial_synthesis_weight': 0.42,  # NEW: Spatial synthesis integration
    'deep_spatial_weight': 0.38,  # NEW: Deep spatial transformer bonus
    
    # ATLAS V6-Specific Ultimate Enhancements
    'deep_spatial_layers': 8,  # 8-layer deep spatial reasoning
    'mega_spatial_memory': 320,  # Massive spatial pattern memory
    'advanced_ensemble_prep': True,  # Ultimate OLYMPUS preparation
    'spatial_synthesis_integration': True,  # Advanced spatial synthesis
    'complete_grid_mastery': True,  # 5x5 to 30x30 complete coverage
    'ultimate_test_time_adaptation': True,  # Advanced spatial adaptation
    
    # Advanced Training Features
    'label_smoothing': 0.015,  # Ultra-refined for spatial precision
    'pattern_diversity_bonus': True,
    'geometric_reasoning_bonus': True,
    'spatial_memory_bonus': True,
    'spatial_synthesis_bonus': True,
    'deep_spatial_bonus': True,
    'ultimate_spatial_bonus': True,  # NEW: Ultimate spatial bonus
    
    # Learning Rate Scheduling
    'warmup_epochs': 20,  # Extended warmup for deep architecture
    'cosine_restarts': True,
    'restart_multiplier': 1.45,
    'plateau_patience': 25,
}

# Enhanced 20-Stage Progressive Configuration - Complete Grid Mastery starting with ARC sizes
STAGE_CONFIG = [
    # Foundation Spatial Understanding (5x5 - 9x9) - START WITH NORMAL ARC SIZES
    {'stage': 0, 'max_grid_size': 5,  'synthesis_ratio': 0.93, 'spatial_complexity': 'micro_spatial', 'focus': 'micro_spatial_patterns'},
    {'stage': 1, 'max_grid_size': 6,  'synthesis_ratio': 0.89, 'spatial_complexity': 'basic_strategy', 'focus': 'basic_shape_recognition'},
    {'stage': 2, 'max_grid_size': 7,  'synthesis_ratio': 0.85, 'spatial_complexity': 'simple_reasoning', 'focus': 'simple_spatial_inference'},
    {'stage': 3, 'max_grid_size': 8,  'synthesis_ratio': 0.81, 'spatial_complexity': 'shape_detection', 'focus': 'shape_identification'},
    {'stage': 4, 'max_grid_size': 9,  'synthesis_ratio': 0.76, 'spatial_complexity': 'pattern_analysis', 'focus': 'spatial_pattern_analysis'},
    
    # Intermediate Spatial Reasoning (10x10 - 15x15) 
    {'stage': 5, 'max_grid_size': 10, 'synthesis_ratio': 0.72, 'spatial_complexity': 'multi_step', 'focus': 'multi_step_spatial_reasoning'},
    {'stage': 6, 'max_grid_size': 11, 'synthesis_ratio': 0.67, 'spatial_complexity': 'complex_rules', 'focus': 'complex_spatial_rule_learning'},
    {'stage': 7, 'max_grid_size': 12, 'synthesis_ratio': 0.62, 'spatial_complexity': 'geometric_planning', 'focus': 'geometric_planning'},
    {'stage': 8, 'max_grid_size': 13, 'synthesis_ratio': 0.57, 'spatial_complexity': 'ensemble_prep_basic', 'focus': 'basic_ensemble_spatial_coordination'},
    {'stage': 9, 'max_grid_size': 14, 'synthesis_ratio': 0.52, 'spatial_complexity': 'arc_spatial_basic', 'focus': 'arc_spatial_patterns'},
    {'stage': 10, 'max_grid_size': 15, 'synthesis_ratio': 0.47, 'spatial_complexity': 'meta_reasoning', 'focus': 'meta_spatial_cognitive_reasoning'},
    
    # Advanced Spatial Mastery (16x16 - 22x22)
    {'stage': 11, 'max_grid_size': 16, 'synthesis_ratio': 0.42, 'spatial_complexity': 'expert_spatial', 'focus': 'expert_spatial_analysis'},
    {'stage': 12, 'max_grid_size': 18, 'synthesis_ratio': 0.37, 'spatial_complexity': 'spatial_genius', 'focus': 'spatial_intelligence_mastery'},
    {'stage': 13, 'max_grid_size': 20, 'synthesis_ratio': 0.32, 'spatial_complexity': 'arc_spatial_advanced', 'focus': 'advanced_arc_spatial_reasoning'},
    {'stage': 14, 'max_grid_size': 22, 'synthesis_ratio': 0.27, 'spatial_complexity': 'ultimate_spatial_basic', 'focus': 'ultimate_spatial_intelligence_basic'},
    
    # Ultimate Spatial Mastery (24x24 - 30x30)
    {'stage': 15, 'max_grid_size': 24, 'synthesis_ratio': 0.24, 'spatial_complexity': 'synthesis_spatial', 'focus': 'spatial_synthesis_mastery'},
    {'stage': 16, 'max_grid_size': 26, 'synthesis_ratio': 0.21, 'spatial_complexity': 'deep_spatial_advanced', 'focus': 'deep_spatial_transformer_mastery'},
    {'stage': 17, 'max_grid_size': 28, 'synthesis_ratio': 0.17, 'spatial_complexity': 'mega_spatial', 'focus': 'mega_spatial_pattern_mastery'},
    {'stage': 18, 'max_grid_size': 30, 'synthesis_ratio': 0.14, 'spatial_complexity': 'ultimate_spatial_advanced', 'focus': 'ultimate_spatial_intelligence_advanced'},
    {'stage': 19, 'max_grid_size': 30, 'synthesis_ratio': 0.11, 'spatial_complexity': 'spatial_god_mode', 'focus': 'ultimate_spatial_god_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 140}\033[0m")
print(f"\033[96mATLAS V6 Ultimate Training - Ultimate 2D Spatial Reasoning Master for ARC-AGI-2\033[0m")
print(f"\033[96mDeep Geometric Transformers + Mega Spatial Memory + Spatial Synthesis Integration\033[0m")
print(f"\033[96mTarget: 90%+ Performance with Ultimate Spatial Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 140}\033[0m")


# Use the same loss and dataset classes as V4/V5 but with V6 config
from train_atlas_specialized4 import AtlasV4SpatialLoss, Advanced2DSpatialDataset, advanced_spatial_collate_fn


def train_atlas_specialized_v6():
    """Main training function for ATLAS V6"""
    print(f"\033[96mInitializing ATLAS V6 Ultimate Spatial Intelligence Training...\033[0m")
    
    # Initialize enhanced model (V4 model contains all V6 improvements)
    model = AtlasV4Enhanced(
        max_grid_size=30,
        d_model=256,
        num_layers=8,  # Deep architecture for ultimate performance
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load V5 weights with multiple fallback paths
    model_paths = [
        '/content/AutomataNexus_Olympus_AGI2/models/atlas_v5_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/atlas_v4_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/atlas_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/atlas_best.pt'
    ]
    
    weights_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Manual weight loading with compatibility
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Load compatible parameters manually
                model_dict = model.state_dict()
                compatible_params = {}
                
                for name, param in state_dict.items():
                    if name in model_dict and model_dict[name].shape == param.shape:
                        compatible_params[name] = param
                
                model_dict.update(compatible_params)
                model.load_state_dict(model_dict)
                
                print(f"\033[96mATLAS: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters from {model_path}\033[0m")
                weights_loaded = True
                break
            except Exception as e:
                continue
    
    if not weights_loaded:
        print(f"\033[96mWarning: Could not load existing weights, starting V6 training from scratch\033[0m")
    else:
        print(f"\033[96mSuccessfully loaded existing weights for V6 ultimate training\033[0m")
    
    # Initialize loss function
    criterion = AtlasV4SpatialLoss(ATLAS_V6_CONFIG)
    
    # Initialize optimizer with V6 learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ATLAS_V6_CONFIG['learning_rate'],
        weight_decay=ATLAS_V6_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=ATLAS_V6_CONFIG['warmup_epochs'],
        T_mult=int(ATLAS_V6_CONFIG['restart_multiplier']),
        eta_min=ATLAS_V6_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Ultimate Progressive Spatial Training - 20 Complete Spatial Intelligence Stages\033[0m")
    
    # Ultimate progressive training through spatial stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 145}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Spatial: {stage_config['spatial_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 145}\033[0m")
        
        # Create ultimate spatial dataset for this stage
        dataset = Advanced2DSpatialDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            spatial_focus=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=ATLAS_V6_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=advanced_spatial_collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Stage-specific training
        stage_performance = train_ultimate_spatial_stage(
            model, dataloader, criterion, optimizer, scheduler, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best V6 model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_performance': best_performance,
                'stage': stage_idx,
                'config': ATLAS_V6_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V6'
            }, '/content/AutomataNexus_Olympus_AGI2/models/atlas_v6_best.pt')
            print(f"\033[96mNew best V6 spatial performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 155}\033[0m")
    print(f"\033[96mATLAS V6 Ultimate Spatial Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest V6 Ultimate Spatial Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 155}\033[0m")
    
    return model, best_performance


def train_ultimate_spatial_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                 stage_idx, stage_config, training_stats):
    """Train a single ultimate spatial curriculum stage for V6"""
    model.train()
    
    epochs_for_stage = ATLAS_V6_CONFIG['epochs_per_stage']
    accumulation_steps = ATLAS_V6_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        ultimate_spatial_count = 0
        arc_spatial_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mUltimate Spatial Stage {stage_idx} Epoch {epoch}\033[0m")
        
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                outputs = model(inputs, targets, mode='train')
                loss_dict = criterion(outputs, targets, inputs)
                loss = loss_dict['total'] / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), ATLAS_V6_CONFIG['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update learning rate
                scheduler.step()
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    epoch_losses[key] += value.item()
            
            total_exact_matches += loss_dict['exact_count'].item()
            total_samples += inputs.shape[0]
            
            # Count ultimate spatial cases and ARC-specific cases
            for meta in metadata:
                if meta['spatial_analysis']['spatial_intelligence_level'] >= 4:
                    ultimate_spatial_count += 1
                if 'arc_' in stage_config.get('focus', ''):
                    arc_spatial_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'UltSpatial': f"{ultimate_spatial_count}",
                'ARC': f"{arc_spatial_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 7 == 0 or epoch == epochs_for_stage - 1:
            spatial_ratio = ultimate_spatial_count / max(total_samples, 1)
            arc_ratio = arc_spatial_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[38;2;255;204;153m⏰ ATLAS V6 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * ATLAS_V6_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Spatial: {spatial_ratio:.1%} | ARC: {arc_ratio:.1%}\033[0m")
            if epoch == epochs_for_stage - 1:
                print(f"\033[96m✅ Stage {stage_idx} complete! Final exact: {epoch_performance:.2%}\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_stage_performance


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train model
    model, best_performance = train_atlas_specialized_v6()