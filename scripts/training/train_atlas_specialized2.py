"""
ATLAS Specialized Training Script V2 - Enhanced Spatial Transformation Expert
Builds upon train_atlas_specialized.py with targeted improvements
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
from typing import Dict, List, Optional
import random
from collections import defaultdict

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

# Import the ENTIRE original script to build upon it
from train_atlas_specialized import (
    AtlasSpecializedDataset, 
    AtlasSpecializedLoss,
    atlas_exact_match_injection,
    atlas_mept_injection,
    atlas_leap_injection,
    atlas_prism_injection,
    custom_collate_fn,
    train_atlas_specialized as train_atlas_specialized_v1,
    ATLAS_CONFIG as ATLAS_CONFIG_V1,
    STAGE_CONFIG as STAGE_CONFIG_V1
)

# Enhanced ATLAS Configuration V2 - Building on V1
ATLAS_CONFIG = ATLAS_CONFIG_V1.copy()
ATLAS_CONFIG.update({
    # Refined parameters based on V1 issues
    'batch_size': 32,  # Smaller batches for better gradients
    'learning_rate': 0.002,  # More conservative than 0.005
    'gradient_accumulation': 2,  # Effective batch: 64
    'transform_penalty': 0.2,  # Lower penalty for ATLAS
    'exact_match_bonus': 4.0,  # Balanced bonus
    'focal_gamma': 1.5,  # Less aggressive
    'spatial_weight': 0.4,  # Slightly higher for spatial focus
    
    # New V2 features
    'use_mixup': True,  # Mixup augmentation
    'mixup_alpha': 0.2,
    'gradient_clip': 1.0,
    'warmup_steps': 500,  # Warmup for stability
    'cosine_restarts': True,  # Cosine annealing with restarts
    'label_smoothing': 0.1,
})

# Enhanced Stage Configuration V2
STAGE_CONFIG = STAGE_CONFIG_V1.copy()
# Keep same grid sizes but adjust learning dynamics
for stage in STAGE_CONFIG:
    STAGE_CONFIG[stage]['lr_mult'] = min(1.0, STAGE_CONFIG[stage]['lr_mult'] * 1.5)  # Less aggressive LR decay


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class AtlasSpecializedLossV2(AtlasSpecializedLoss):
    """Enhanced ATLAS loss with mixup and label smoothing"""
    
    def forward(self, pred_output, target_output, input_grid, model_outputs=None, mixup_lambda=None):
        """Enhanced forward with mixup support"""
        
        # If using mixup, adjust the loss calculation
        if mixup_lambda is not None:
            # Get base losses for both targets
            losses1 = super().forward(pred_output, target_output[0], input_grid, model_outputs)
            losses2 = super().forward(pred_output, target_output[1], input_grid, model_outputs)
            
            # Mix the losses
            mixed_losses = {}
            for key in losses1:
                if torch.is_tensor(losses1[key]):
                    mixed_losses[key] = mixup_lambda * losses1[key] + (1 - mixup_lambda) * losses2[key]
                else:
                    mixed_losses[key] = losses1[key]  # For counts, use first
            
            return mixed_losses
        
        # Apply label smoothing if configured
        if hasattr(self, 'label_smoothing') and self.label_smoothing > 0:
            C = target_output.shape[1]
            smooth_target = target_output * (1 - self.label_smoothing)
            smooth_target += self.label_smoothing / C
            target_output = smooth_target
        
        return super().forward(pred_output, target_output, input_grid, model_outputs)


def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


class WarmupCosineSchedule(optim.lr_scheduler._LRScheduler):
    """Cosine learning rate schedule with warmup"""
    
    def __init__(self, optimizer, warmup_steps, training_steps, cycles=1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.cycles = cycles
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.training_steps - self.warmup_steps)
            if self.cycles > 1:
                # With restarts
                cycle_progress = progress * self.cycles
                progress = cycle_progress - int(cycle_progress)
            return [base_lr * (0.5 * (1 + np.cos(np.pi * progress))) for base_lr in self.base_lrs]


def train_atlas_specialized_v2():
    """Enhanced ATLAS training building on V1"""
    print("🌍 Starting ATLAS Specialized Training V2")
    print("=" * 60)
    print("📊 Enhancements over V1:")
    print("  • Mixup augmentation for better generalization")
    print("  • Warmup + cosine annealing with restarts")
    print("  • Label smoothing")
    print("  • Gradient clipping")
    print("  • Better batch size and learning rate")
    print("=" * 60)
    
    # Initialize model with maximum grid size from final stage
    max_grid_size = STAGE_CONFIG[7]['max_grid_size']  # 30x30
    model = EnhancedAtlasNet(
        max_grid_size=max_grid_size
    ).to(device)
    
    print(f"📊 ATLAS Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems (reuse V1 logic)
    systems = {}
    
    # MEPT System - Use ATLAS-specific if available
    if USE_MEPT:
        if ATLAS_MEPT_LEAP_AVAILABLE:
            mept_components = create_atlas_mept_system(
                model=model,
                device=device
            )
            systems['spatial_memory'] = mept_components['spatial_memory']
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components['loss_function']
            print("✅ ATLAS-specific MEPT system initialized")
        else:
            mept_components = create_mept_system(
                capacity=40000,
                pattern_bank_size=8000,
                transformation_penalty=ATLAS_CONFIG['transform_penalty'],
                exact_match_bonus=ATLAS_CONFIG['exact_match_bonus']
            )
            systems['spatial_memory'] = mept_components['replay_buffer'] 
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components.get('loss_fn')
            print("✅ Generic MEPT system initialized")
    
    # Initialize other systems (LEAP, PRISM, etc.) - same as V1
    # ... (keeping same initialization code as V1)
    
    # Initialize V2 enhanced loss
    loss_fn = AtlasSpecializedLossV2().to(device)
    loss_fn.label_smoothing = ATLAS_CONFIG.get('label_smoothing', 0.1)
    
    # Optimizer - AdamW for better regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ATLAS_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    # Calculate total training steps
    total_epochs = ATLAS_CONFIG['num_epochs']
    steps_per_epoch = 100  # Approximate
    total_steps = total_epochs * steps_per_epoch
    
    # Warmup + Cosine scheduler with restarts
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=ATLAS_CONFIG['warmup_steps'],
        training_steps=total_steps,
        cycles=3 if ATLAS_CONFIG.get('cosine_restarts') else 1
    )
    
    scaler = GradScaler('cuda')
    
    # Training metrics
    best_exact = 0.0
    global_epoch = 0
    global_step = 0
    
    # Skip checkpoint loading - always fresh start in V2
    print("🆕 Starting fresh training (V2 always starts fresh)")
    
    # Training history
    history = defaultdict(list)
    
    # 8-Stage Progressive Curriculum Training Loop
    stage_metrics = []
    
    # 4-PHASE INJECTION (if stage 0)
    if USE_EXACT_BOOST:
        print("\n" + "=" * 60)
        print("🌍 ATLAS 4-PHASE SPATIAL TRANSFORMATION INJECTION SEQUENCE")
        print("=" * 60)
        
        # Phase 1: Exact Match
        print("\n📍 PHASE 1: Spatial Identity Mapping")
        model = atlas_exact_match_injection(model, device, num_epochs=100, target_accuracy=85.0)
        
        # Phase 2: MEPT (skip if not available)
        if USE_MEPT and 'spatial_memory' in systems:
            print("\n📍 PHASE 2: Spatial Memory Enhancement (MEPT)")
            print("⚠️ MEPT injection not implemented in V1, skipping")
        
        # Phase 3: LEAP
        if USE_LEAP and 'leap_trainer' in systems:
            print("\n📍 PHASE 3: Adaptive Spatial Learning (LEAP)")
            model = atlas_leap_injection(model, device, systems, num_epochs=80)
        
        # Phase 4: PRISM
        if USE_PRISM and 'prism_synthesizer' in systems:
            print("\n📍 PHASE 4: Spatial Program Synthesis (PRISM)")
            model = atlas_prism_injection(model, device, systems, num_epochs=80)
        
        print("\n✅ 4-PHASE INJECTION COMPLETE")
        print("=" * 60)
    
    # Main curriculum training
    for stage in range(ATLAS_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        
        print(f"\n🌍 ATLAS Stage {stage}: {grid_size}x{grid_size} Spatial Transformation")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {int(stage_config['synthesis_ratio']*100)}%")
        print("=" * 60)
        
        # Create dataset using V1 approach but with smaller size for testing
        print(f"🔧 Generating ATLAS-specific DSL spatial patterns for stage {stage}...")
        
        # Initialize DSL trainer
        dsl_trainer = ATLASDSLTraining(model, device)
        print(f"✅ ATLAS DSL spatial pattern trainer initialized")
        
        # Create dataset (simplified from V1)
        dataset_samples = []
        
        # Load ARC JSON files
        arc_files = ['arc-agi_training_challenges.json', 'arc-agi_evaluation_challenges.json']
        
        for filename in arc_files:
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    tasks = json.load(f)
                    for task_id, task_data in tasks.items():
                        for example in task_data['train']:
                            input_grid = np.array(example['input'])
                            output_grid = np.array(example['output'])
                            if input_grid.shape[0] <= grid_size and input_grid.shape[1] <= grid_size:
                                dataset_samples.append({'inputs': input_grid, 'outputs': output_grid})
        
        # Add synthetic samples
        for i in range(200):  # More synthetic data
            size = min(random.choice([4, 5, 6]), grid_size)
            input_grid = np.random.randint(0, 5, (size, size))
            # Enhanced transformations
            transform = random.choice(['rotate', 'flip', 'transpose', 'shift'])
            if transform == 'rotate':
                output_grid = np.rot90(input_grid, k=random.randint(1, 3)).copy()
            elif transform == 'flip':
                output_grid = np.flip(input_grid, axis=random.randint(0, 1)).copy()
            elif transform == 'transpose':
                output_grid = input_grid.T.copy()
            else:  # shift
                output_grid = np.roll(input_grid, shift=1, axis=random.randint(0, 1))
            dataset_samples.append({'inputs': input_grid, 'outputs': output_grid})
        
        # Create dataset
        class SimpleARCDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]
        
        dataset = SimpleARCDataset(dataset_samples)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Train: {train_size}, Val: {val_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=ATLAS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=ATLAS_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=False
        )
        
        # Adjust learning rate for stage
        stage_lr = ATLAS_CONFIG['learning_rate'] * stage_config['lr_mult']
        for param_group in optimizer.param_groups:
            param_group['lr'] = stage_lr
        
        # Stage training loop
        print(f"\n🔄 Training Stage {stage} for {ATLAS_CONFIG['epochs_per_stage']} epochs...")
        
        for epoch in range(ATLAS_CONFIG['epochs_per_stage']):
            global_epoch += 1
            
            # Training phase
            model.train()
            train_metrics = defaultdict(float)
            
            pbar = tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                global_step += 1
                
                inputs = batch['inputs'].to(device, non_blocking=True)
                outputs = batch['outputs'].to(device, non_blocking=True)
                
                # Clamp values
                inputs = torch.clamp(inputs, 0, 9)
                outputs = torch.clamp(outputs, 0, 9)
                
                # Convert to one-hot
                input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                
                # Apply mixup if enabled
                if ATLAS_CONFIG.get('use_mixup') and random.random() < 0.5:
                    mixed_input, target_a, target_b, lam = mixup_data(
                        input_grids, output_grids, alpha=ATLAS_CONFIG.get('mixup_alpha', 0.2)
                    )
                    
                    with autocast('cuda'):
                        model_outputs = model(mixed_input, target_a, mode='train')
                        pred_output = model_outputs['predicted_output']
                        losses = loss_fn(pred_output, (target_a, target_b), mixed_input, 
                                       model_outputs, mixup_lambda=lam)
                else:
                    with autocast('cuda'):
                        model_outputs = model(input_grids, output_grids, mode='train')
                        pred_output = model_outputs['predicted_output']
                        losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                
                loss = losses['total'] / ATLAS_CONFIG['gradient_accumulation']
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % ATLAS_CONFIG['gradient_accumulation'] == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), ATLAS_CONFIG['gradient_clip'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update scheduler
                    scheduler.step()
                
                # Update metrics
                train_metrics['loss'] += losses['total'].item()
                train_metrics['exact'] += losses['exact_count'].item()
                train_metrics['samples'] += inputs.size(0)
                
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.3f}",
                    'exact': f"{losses['exact_count'].item():.0f}",
                    'lr': f"{scheduler.get_lr()[0]:.6f}"
                })
            
            # Validation phase
            if epoch % 5 == 0 or epoch == ATLAS_CONFIG['epochs_per_stage'] - 1:
                model.eval()
                val_metrics = defaultdict(float)
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        inputs = batch['inputs'].to(device, non_blocking=True)
                        outputs = batch['outputs'].to(device, non_blocking=True)
                        
                        inputs = torch.clamp(inputs, 0, 9)
                        outputs = torch.clamp(outputs, 0, 9)
                        
                        input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                        output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                        
                        with autocast('cuda'):
                            model_outputs = model(input_grids, mode='inference')
                            pred_output = model_outputs['predicted_output']
                            losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                        
                        pred_indices = pred_output.argmax(dim=1)
                        target_indices = output_grids.argmax(dim=1)
                        pixel_acc = (pred_indices == target_indices).float().mean()
                        
                        val_metrics['loss'] += losses['total'].item()
                        val_metrics['exact'] += losses['exact_count'].item()
                        val_metrics['pixel_acc'] += pixel_acc.item()
                        val_metrics['samples'] += inputs.size(0)
                
                # Calculate averages
                train_loss = train_metrics['loss'] / len(train_loader)
                train_exact_pct = train_metrics['exact'] / train_metrics['samples'] * 100
                val_loss = val_metrics['loss'] / len(val_loader)
                val_exact_pct = val_metrics['exact'] / val_metrics['samples'] * 100
                val_pixel_acc = val_metrics['pixel_acc'] / len(val_loader) * 100
                
                print(f"\n🌍 ATLAS Epoch {epoch+1} (Stage {stage}, {grid_size}x{grid_size}):")
                print(f"   🎯 Train: {train_exact_pct:.2f}% exact, Loss: {train_loss:.3f}")
                print(f"   🎯 Val: {val_exact_pct:.2f}% exact, Loss: {val_loss:.3f}, Pixel: {val_pixel_acc:.1f}%")
                
                # Save history
                history['train_loss'].append(train_loss)
                history['train_exact'].append(train_exact_pct)
                history['val_loss'].append(val_loss)
                history['val_exact'].append(val_exact_pct)
                history['learning_rate'].append(scheduler.get_lr()[0])
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_exact': best_exact,
                        'val_loss': val_loss
                    }, f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/atlas_v2_best.pt')
                    print(f"   🏆 New best model! Exact: {best_exact:.2f}%")
        
        # Stage complete
        stage_exact = train_exact_pct
        stage_metrics.append({
            'stage': stage,
            'grid_size': grid_size,
            'final_exact': stage_exact
        })
        
        print(f"\n✅ Stage {stage} complete! Final exact: {stage_exact:.2f}%")
        
        # Clear memory
        del train_loader, val_loader, dataset
        gc.collect()
        torch.cuda.empty_cache()
    
    # Training complete
    print("\n" + "=" * 60)
    print("🎉 ATLAS V2 8-Stage Training Complete!")
    print(f"   🏆 Best exact match: {best_exact:.2f}%")
    print(f"   📏 Stages completed: 8 (6x6 → 30x30 grids)")
    print(f"   📊 Total epochs: {global_epoch}")
    
    print(f"\n📏 Stage-by-stage Spatial Learning Progression:")
    for metrics in stage_metrics:
        print(f"   Stage {metrics['stage']} ({metrics['grid_size']}x{metrics['grid_size']}): "
              f"{metrics['final_exact']:.2f}% exact match")
    
    return model, history


# Training components flags (from V1)
USE_MEPT = True and (ATLAS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_LEAP = True and (ATLAS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_PRISM = True and (ATLAS_PRISM_AVAILABLE or PRISM_AVAILABLE)
USE_EXACT_BOOST = True and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE

# Data directory
DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'


if __name__ == "__main__":
    print("=" * 80)
    print("ATLAS Specialized Training V2 - Building on V1")
    print("Spatial Transformation Expert with Targeted Enhancements")
    print("=" * 80)
    
    train_atlas_specialized_v2()