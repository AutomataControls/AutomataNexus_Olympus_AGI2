"""
IRIS Specialized Training V6 - Ultimate Color Intelligence Master for ARC-AGI-2
Complete grid mastery (5x5 to 30x30) with deep chromatic architecture and color synthesis
Builds upon V5 with revolutionary color intelligence capabilities and massive data pipeline
Target: 90%+ performance with ultimate color intelligence mastery
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

# Import enhanced IRIS V4 model (contains V6 improvements)
from src.models.iris_v4_enhanced import IrisV4Enhanced

# Enhanced IRIS V6 Configuration - Ultimate Color Intelligence Focus
IRIS_V6_CONFIG = {
    # Core Training Parameters - OPTIMIZED for V6 Ultimate Performance
    'batch_size': 28,  # Balanced for deep color architecture
    'learning_rate': 0.00012,  # Lower for deep color learning
    'num_epochs': 640,  # Extended: 20 stages x 32 epochs
    'gradient_accumulation': 5,  # Effective batch 140 for stability
    'epochs_per_stage': 32,  # Deep learning per stage
    'curriculum_stages': 20,  # Complete grid mastery 5x5 -> 30x30
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.025,  # Very low - maximum color exploration
    'exact_match_bonus': 10.2,  # Highest bonus for ultimate precision
    'gradient_clip': 0.5,  # Stable clipping for deep architecture
    'weight_decay': 1.5e-6,  # Ultra-light regularization for ultimate color strategy
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'chromatic_reasoning_weight': 0.72,  # Ultimate focus - color intelligence
    'color_harmony_weight': 0.65,  # Maximum color harmony understanding
    'color_space_weight': 0.58,  # Ultimate color space analysis
    'ensemble_coordination_weight': 0.52,  # Maximum ensemble integration
    'pattern_analysis_weight': 0.48,  # Ultimate pattern analysis
    'decision_confidence_weight': 0.45,  # Ultimate decision confidence
    'color_synthesis_weight': 0.42,  # NEW: Color synthesis integration
    'deep_chromatic_weight': 0.38,  # NEW: Deep chromatic transformer bonus
    
    # IRIS V6-Specific Ultimate Enhancements
    'deep_chromatic_layers': 8,  # 8-layer deep color reasoning
    'mega_color_memory': 280,  # Massive chromatic pattern memory
    'advanced_ensemble_prep': True,  # Ultimate OLYMPUS preparation
    'color_synthesis_integration': True,  # Advanced color synthesis
    'complete_grid_mastery': True,  # 5x5 to 30x30 complete coverage
    'ultimate_test_time_adaptation': True,  # Advanced color adaptation
    
    # Advanced Training Features
    'label_smoothing': 0.018,  # Ultra-refined for color precision
    'pattern_diversity_bonus': True,
    'chromatic_reasoning_bonus': True,
    'color_harmony_bonus': True,
    'color_synthesis_bonus': True,
    'deep_chromatic_bonus': True,
    'ultimate_color_bonus': True,  # NEW: Ultimate color bonus
    
    # Learning Rate Scheduling
    'warmup_epochs': 22,  # Extended warmup for deep architecture
    'cosine_restarts': True,
    'restart_multiplier': 1.4,
    'plateau_patience': 28,
}

# Enhanced 20-Stage Progressive Configuration - Complete Grid Mastery starting with ARC sizes
STAGE_CONFIG = [
    # Foundation Color Understanding (5x5 - 9x9) - START WITH NORMAL ARC SIZES
    {'stage': 0, 'max_grid_size': 5,  'synthesis_ratio': 0.94, 'color_complexity': 'micro_chromatic', 'focus': 'micro_color_patterns'},
    {'stage': 1, 'max_grid_size': 6,  'synthesis_ratio': 0.91, 'color_complexity': 'basic_strategy', 'focus': 'basic_color_recognition'},
    {'stage': 2, 'max_grid_size': 7,  'synthesis_ratio': 0.87, 'color_complexity': 'simple_reasoning', 'focus': 'simple_color_inference'},
    {'stage': 3, 'max_grid_size': 8,  'synthesis_ratio': 0.82, 'color_complexity': 'color_detection', 'focus': 'color_identification'},
    {'stage': 4, 'max_grid_size': 9,  'synthesis_ratio': 0.78, 'color_complexity': 'pattern_analysis', 'focus': 'color_pattern_analysis'},
    
    # Intermediate Color Reasoning (10x10 - 15x15) 
    {'stage': 5, 'max_grid_size': 10, 'synthesis_ratio': 0.73, 'color_complexity': 'multi_step', 'focus': 'multi_step_color_reasoning'},
    {'stage': 6, 'max_grid_size': 11, 'synthesis_ratio': 0.68, 'color_complexity': 'complex_rules', 'focus': 'complex_color_rule_learning'},
    {'stage': 7, 'max_grid_size': 12, 'synthesis_ratio': 0.63, 'color_complexity': 'chromatic_planning', 'focus': 'chromatic_planning'},
    {'stage': 8, 'max_grid_size': 13, 'synthesis_ratio': 0.58, 'color_complexity': 'ensemble_prep_basic', 'focus': 'basic_ensemble_color_coordination'},
    {'stage': 9, 'max_grid_size': 14, 'synthesis_ratio': 0.53, 'color_complexity': 'arc_chromatic_basic', 'focus': 'arc_color_patterns'},
    {'stage': 10, 'max_grid_size': 15, 'synthesis_ratio': 0.48, 'color_complexity': 'meta_reasoning', 'focus': 'meta_color_cognitive_reasoning'},
    
    # Advanced Color Mastery (16x16 - 22x22)
    {'stage': 11, 'max_grid_size': 16, 'synthesis_ratio': 0.43, 'color_complexity': 'expert_chromatic', 'focus': 'expert_color_analysis'},
    {'stage': 12, 'max_grid_size': 18, 'synthesis_ratio': 0.38, 'color_complexity': 'color_genius', 'focus': 'color_intelligence_mastery'},
    {'stage': 13, 'max_grid_size': 20, 'synthesis_ratio': 0.33, 'color_complexity': 'arc_chromatic_advanced', 'focus': 'advanced_arc_color_reasoning'},
    {'stage': 14, 'max_grid_size': 22, 'synthesis_ratio': 0.28, 'color_complexity': 'ultimate_chromatic_basic', 'focus': 'ultimate_color_intelligence_basic'},
    
    # Ultimate Color Mastery (24x24 - 30x30)
    {'stage': 15, 'max_grid_size': 24, 'synthesis_ratio': 0.25, 'color_complexity': 'synthesis_chromatic', 'focus': 'color_synthesis_mastery'},
    {'stage': 16, 'max_grid_size': 26, 'synthesis_ratio': 0.22, 'color_complexity': 'deep_chromatic_advanced', 'focus': 'deep_color_transformer_mastery'},
    {'stage': 17, 'max_grid_size': 28, 'synthesis_ratio': 0.18, 'color_complexity': 'mega_chromatic', 'focus': 'mega_color_pattern_mastery'},
    {'stage': 18, 'max_grid_size': 30, 'synthesis_ratio': 0.15, 'color_complexity': 'ultimate_chromatic_advanced', 'focus': 'ultimate_color_intelligence_advanced'},
    {'stage': 19, 'max_grid_size': 30, 'synthesis_ratio': 0.12, 'color_complexity': 'color_god_mode', 'focus': 'ultimate_color_god_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 135}\033[0m")
print(f"\033[96mIRIS V6 Ultimate Training - Ultimate Color Intelligence Master for ARC-AGI-2\033[0m")
print(f"\033[96mDeep Chromatic Transformers + Mega Color Memory + Color Synthesis Integration\033[0m")
print(f"\033[96mTarget: 90%+ Performance with Ultimate Color Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 135}\033[0m")


# Use the same loss and dataset classes as V4/V5 but with V6 config
from train_iris_specialized4 import IrisV4ChromaticLoss, AdvancedColorDataset, advanced_color_collate_fn


def train_iris_specialized_v6():
    """Main training function for IRIS V6"""
    print(f"\033[96mInitializing IRIS V6 Ultimate Color Intelligence Training...\033[0m")
    
    # Initialize enhanced model (V4 model contains all V6 improvements)
    model = IrisV4Enhanced(
        max_grid_size=30,
        d_model=256,
        num_layers=8,  # Deep architecture for ultimate performance
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load V5 weights with multiple fallback paths
    model_paths = [
        '/content/AutomataNexus_Olympus_AGI2/models/iris_v5_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/iris_v4_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/iris_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/iris_best.pt'
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
                
                print(f"\033[96mIRIS: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters from {model_path}\033[0m")
                weights_loaded = True
                break
            except Exception as e:
                continue
    
    if not weights_loaded:
        print(f"\033[96mWarning: Could not load existing weights, starting V6 training from scratch\033[0m")
    else:
        print(f"\033[96mSuccessfully loaded existing weights for V6 ultimate training\033[0m")
    
    # Initialize loss function
    criterion = IrisV4ChromaticLoss(IRIS_V6_CONFIG)
    
    # Initialize optimizer with V6 learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=IRIS_V6_CONFIG['learning_rate'],
        weight_decay=IRIS_V6_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=IRIS_V6_CONFIG['warmup_epochs'],
        T_mult=int(IRIS_V6_CONFIG['restart_multiplier']),
        eta_min=IRIS_V6_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Ultimate Progressive Color Training - 20 Complete Color Intelligence Stages\033[0m")
    
    # Ultimate progressive training through color stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 140}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Color: {stage_config['color_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 140}\033[0m")
        
        # Create ultimate color dataset for this stage
        dataset = AdvancedColorDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            color_focus=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=IRIS_V6_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=advanced_color_collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Stage-specific training
        stage_performance = train_ultimate_color_stage(
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
                'config': IRIS_V6_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V6'
            }, '/content/AutomataNexus_Olympus_AGI2/models/iris_v6_best.pt')
            print(f"\033[96mNew best V6 color performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 150}\033[0m")
    print(f"\033[96mIRIS V6 Ultimate Color Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest V6 Ultimate Color Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 150}\033[0m")
    
    return model, best_performance


def train_ultimate_color_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                               stage_idx, stage_config, training_stats):
    """Train a single ultimate color curriculum stage for V6"""
    model.train()
    
    epochs_for_stage = IRIS_V6_CONFIG['epochs_per_stage']
    accumulation_steps = IRIS_V6_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        ultimate_color_count = 0
        arc_color_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mUltimate Color Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), IRIS_V6_CONFIG['gradient_clip'])
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
            
            # Count ultimate color cases and ARC-specific cases
            for meta in metadata:
                if meta['color_analysis']['color_intelligence_level'] >= 4:
                    ultimate_color_count += 1
                if 'arc_' in stage_config.get('focus', ''):
                    arc_color_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'UltColor': f"{ultimate_color_count}",
                'ARC': f"{arc_color_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 8 == 0 or epoch == epochs_for_stage - 1:
            color_ratio = ultimate_color_count / max(total_samples, 1)
            arc_ratio = arc_color_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[38;2;255;204;153m⏰ IRIS V6 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * IRIS_V6_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Color: {color_ratio:.1%} | ARC: {arc_ratio:.1%}\033[0m")
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
    model, best_performance = train_iris_specialized_v6()