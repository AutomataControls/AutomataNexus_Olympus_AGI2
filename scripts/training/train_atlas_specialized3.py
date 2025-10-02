"""
ATLAS Specialized Training Script V3 - Conservative Spatial Learning
Builds upon V2 with slower, more careful training to prevent accuracy degradation
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

# PRISM System - Use ATLAS-specific version
try:
    from src.training_systems.atlas_prism import create_atlas_prism_system
    ATLAS_PRISM_AVAILABLE = True
except ImportError:
    ATLAS_PRISM_AVAILABLE = False

# Generic PRISM as fallback
try:
    from src.training_systems.prism_system import create_prism_system
    PRISM_AVAILABLE = True
except ImportError:
    PRISM_AVAILABLE = False

# MEPT System - Use ATLAS-specific version
try:
    from src.training_systems.atlas_mept import create_atlas_mept_system
    ATLAS_MEPT_LEAP_AVAILABLE = True
except ImportError:
    ATLAS_MEPT_LEAP_AVAILABLE = False

# Generic MEPT as fallback
try:
    from src.training_systems.mept_system import create_mept_system
    MEPT_AVAILABLE = True
except ImportError:
    MEPT_AVAILABLE = False

# LEAP System - Use ATLAS-specific version
try:
    from src.training_systems.atlas_leap import create_atlas_leap_system
    from src.training_systems.leap_system import create_leap_system
    LEAP_AVAILABLE = True
except ImportError:
    LEAP_AVAILABLE = False

# LEAP-PRISM Bridge
try:
    from src.training_systems.leap_prism_bridge import create_leap_prism_bridge
    LEAP_PRISM_BRIDGE_AVAILABLE = True
except ImportError:
    LEAP_PRISM_BRIDGE_AVAILABLE = False

# Exact match boost system
try:
    from src.training_systems.exact_match_boost import atlas_exact_match_injection
    EXACT_BOOST_AVAILABLE = True
except ImportError:
    EXACT_BOOST_AVAILABLE = False

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

# Ultra-Conservative ATLAS Configuration V3 - Slow and Steady
ATLAS_CONFIG = ATLAS_CONFIG_V1.copy()
ATLAS_CONFIG.update({
    # Much slower learning for stability
    'batch_size': 16,  # Very small batches for precise gradients
    'learning_rate': 0.0005,  # Much lower starting LR
    'gradient_accumulation': 4,  # Effective batch: 64
    'transform_penalty': 0.1,  # Very low penalty
    'exact_match_bonus': 2.0,  # Conservative bonus
    'focal_gamma': 1.0,  # Minimal focal loss
    'spatial_weight': 0.2,  # Lower spatial weight
    
    # Conservative augmentation
    'use_mixup': False,  # Disable mixup for now
    'gradient_clip': 0.5,  # Gentle gradient clipping
    'warmup_steps': 1000,  # Longer warmup
    'cosine_restarts': False,  # No restarts, just steady decay
    'label_smoothing': 0.05,  # Minimal smoothing
    
    # V3 specific - much longer training
    'epochs_per_stage': 80,  # Double the epochs
    'patience': 20,  # More patience for early stopping
    'min_lr': 1e-6,  # Lower minimum LR
    'lr_decay_factor': 0.8,  # Gentler LR decay
})

# Conservative Stage Configuration V3 - Preserve Learning
STAGE_CONFIG = STAGE_CONFIG_V1.copy()
for stage in STAGE_CONFIG:
    # Much gentler LR multipliers to prevent degradation
    STAGE_CONFIG[stage]['lr_mult'] = max(0.7, STAGE_CONFIG[stage]['lr_mult'])
    # Longer training per stage
    STAGE_CONFIG[stage]['epochs'] = 80

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Configuration flags
USE_DSL = True and DSLTrainingIntegration is not None
USE_MEPT = True and (ATLAS_MEPT_LEAP_AVAILABLE or MEPT_AVAILABLE)
USE_LEAP = True and LEAP_AVAILABLE
USE_PRISM = True and (ATLAS_PRISM_AVAILABLE or PRISM_AVAILABLE)
USE_EXACT_BOOST = True and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE


class AtlasSpecializedLossV3(nn.Module):
    """Ultra-conservative loss for ATLAS V3 - gentle learning"""
    
    def __init__(self):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
        self.label_smoothing = 0.05  # Very minimal smoothing
        
        # Ultra-conservative weights
        self.weights = {
            'base': 1.0,
            'spatial': 0.1,  # Much lower spatial weight
            'transform': 0.05,  # Minimal transform penalty
            'exact': 1.5,  # Gentle exact match bonus
            'focal': 0.5,  # Minimal focal loss
        }
    
    def forward(self, predictions, targets, model_outputs=None):
        batch_size = predictions.shape[0]
        
        # Convert predictions to class indices
        pred_indices = predictions.argmax(dim=1)
        target_indices = targets.argmax(dim=1) if targets.dim() > 2 else targets
        
        # Base cross entropy with label smoothing
        base_loss = F.cross_entropy(
            predictions.view(-1, predictions.shape[-1]), 
            target_indices.view(-1),
            label_smoothing=self.label_smoothing
        )
        
        # Very gentle exact match bonus
        exact_matches = (pred_indices == target_indices).float()
        exact_match_rate = exact_matches.mean()
        exact_bonus = -exact_match_rate * self.weights['exact']
        
        # Minimal spatial consistency loss
        spatial_loss = self._gentle_spatial_loss(pred_indices, target_indices)
        
        # Conservative focal loss
        focal_loss = self._gentle_focal_loss(predictions, target_indices)
        
        total_loss = (
            self.weights['base'] * base_loss +
            self.weights['spatial'] * spatial_loss +
            self.weights['focal'] * focal_loss +
            exact_bonus
        )
        
        return total_loss
    
    def _gentle_spatial_loss(self, pred_indices, target_indices):
        """Very gentle spatial consistency loss"""
        if pred_indices.shape[-1] < 3 or pred_indices.shape[-2] < 3:
            return torch.tensor(0.0, device=pred_indices.device)
        
        diff = (pred_indices != target_indices).float()
        return diff.mean() * 0.1  # Very small spatial penalty
    
    def _gentle_focal_loss(self, pred, target, gamma=1.0):
        """Minimal focal loss for hard examples"""
        target_idx = target.argmax(dim=-1) if target.dim() > 2 else target
        ce_loss = F.cross_entropy(pred.view(-1, pred.shape[-1]), target_idx.view(-1), reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** gamma * ce_loss
        return focal.mean()


class WarmupCosineSchedule:
    """Gentle warmup + cosine decay scheduler"""
    
    def __init__(self, optimizer, warmup_steps, training_steps, cycles=1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.cycles = cycles
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr_mult = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.training_steps - self.warmup_steps)
            lr_mult = 0.5 * (1 + np.cos(np.pi * progress))
            lr_mult = max(lr_mult, 0.01)  # Prevent LR from going too low
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult


def train_atlas_specialized_v3():
    """Ultra-conservative ATLAS training V3 - slow and steady wins the race"""
    print("🌍 Starting ATLAS Specialized Training V3")
    print("============================================================")
    print("📊 V3 Philosophy: Slow, Conservative, Steady Learning")
    print("  • Ultra-small learning rates for stability")
    print("  • Gentle gradient updates")
    print("  • Conservative loss functions")
    print("  • Checkpoint recovery enabled")
    print("  • Extended training time per stage")
    print("============================================================")
    
    # Initialize model with maximum grid size from final stage
    max_grid_size = STAGE_CONFIG[7]['max_grid_size']  # Final stage size (30x30)
    model = EnhancedAtlasNet(
        max_grid_size=max_grid_size
    ).to(device)
    
    print(f"📊 ATLAS Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
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
    
    # LEAP System - Use ATLAS-specific if available
    if USE_LEAP:
        if ATLAS_MEPT_LEAP_AVAILABLE:
            leap_components = create_atlas_leap_system(model=model, device=device)
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components.get('pattern_generator')
            systems['weak_detector'] = leap_components.get('detector')
            print("✅ ATLAS-specific LEAP system initialized")
        else:
            leap_components = create_leap_system(device)
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components.get('pattern_generator')
            systems['weak_detector'] = leap_components.get('detector')
            print("✅ Generic LEAP system initialized")
    
    # PRISM System - Use ATLAS-specific if available
    if USE_PRISM:
        if ATLAS_PRISM_AVAILABLE:
            prism_components = create_atlas_prism_system(model=model, device=device)
            systems['prism_synthesizer'] = prism_components['synthesizer']
            systems['prism_library'] = prism_components['program_bank']
            print("✅ ATLAS-specific PRISM system initialized")
        else:
            prism_components = create_prism_system()
            systems['prism_synthesizer'] = prism_components['synthesizer']
            print("✅ Generic PRISM system initialized")
    
    # Initialize V3 ultra-conservative loss
    loss_fn = AtlasSpecializedLossV3().to(device)
    
    # Ultra-conservative optimizer - much lower LR
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ATLAS_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=1e-5,  # Very gentle weight decay
        eps=1e-8
    )
    
    # Calculate total training steps
    total_epochs = ATLAS_CONFIG['epochs_per_stage'] * 8  # 8 stages
    steps_per_epoch = 100  # Approximate
    total_steps = total_epochs * steps_per_epoch
    
    # Gentle warmup + cosine scheduler
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=ATLAS_CONFIG['warmup_steps'],
        training_steps=total_steps
    )
    
    scaler = GradScaler('cuda')
    
    # Data directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
    # Training metrics
    best_exact = 0.0
    global_epoch = 0
    global_step = 0
    
    # V3 FEATURE: Load from best checkpoint if available
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = f'{models_dir}/atlas_checkpoint.pt'
    best_model_path = f'{models_dir}/atlas_best.pt'
    
    # Try to load from atlas_best.pt first
    start_stage = 0
    if os.path.exists(best_model_path):
        print(f"🔄 Loading best model from {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                global_epoch = checkpoint['epoch']
            if 'stage' in checkpoint:
                start_stage = min(checkpoint.get('stage', 0), 7)  # Don't exceed max stage
            best_exact = checkpoint.get('best_exact', 0.0)
            print(f"✅ Loaded from best model: epoch {global_epoch}, stage {start_stage}, accuracy {best_exact:.2f}%")
        except Exception as e:
            print(f"⚠️ Failed to load best model: {e}")
            print("🔄 Starting fresh training")
    elif os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_epoch = checkpoint['epoch']
            start_stage = checkpoint.get('stage', 0)
            best_exact = checkpoint.get('best_exact', 0.0)
            print(f"✅ Resumed from checkpoint: epoch {global_epoch}, stage {start_stage}, accuracy {best_exact:.2f}%")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("🔄 Starting fresh training")
    else:
        print("🆕 No existing models found - starting fresh training")
    
    # Training history
    history = defaultdict(list)
    
    # 8-Stage Progressive Curriculum Training Loop
    stage_metrics = []
    
    # Conservative 4-PHASE INJECTION (only if starting fresh)
    if start_stage == 0 and USE_EXACT_BOOST:
        print("\n" + "=" * 60)
        print("🌍 ATLAS CONSERVATIVE 4-PHASE INJECTION SEQUENCE")
        print("=" * 60)
        
        # Phase 1: Gentle Exact Match (lower target)
        print("\n📍 PHASE 1: Conservative Spatial Identity Mapping")
        model = atlas_exact_match_injection(model, device, num_epochs=60, target_accuracy=70.0)
        
        # Phase 2: MEPT (skip if not available)
        if USE_MEPT and 'spatial_memory' in systems:
            print("\n📍 PHASE 2: Gentle Spatial Memory Enhancement (MEPT)")
            print("⚠️ Using conservative MEPT settings")
        
        # Phase 3: LEAP
        if USE_LEAP and 'leap_trainer' in systems:
            print("\n📍 PHASE 3: Conservative Adaptive Spatial Learning (LEAP)")
            model = atlas_leap_injection(model, device, systems, num_epochs=60)
        
        # Phase 4: PRISM
        if USE_PRISM and 'prism_synthesizer' in systems:
            print("\n📍 PHASE 4: Conservative Spatial Program Synthesis (PRISM)")
            model = atlas_prism_injection(model, device, systems, num_epochs=60)
        
        print("\n✅ CONSERVATIVE 4-PHASE INJECTION COMPLETE")
        print("=" * 60)
    
    # Main curriculum training
    for stage in range(start_stage, ATLAS_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        
        print(f"\n🌍 ATLAS Stage {stage}: {grid_size}x{grid_size} Conservative Spatial Learning")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {int(stage_config['synthesis_ratio']*100)}%")
        print("=" * 60)
        
        # Very gentle learning rate adjustment
        if stage > start_stage:
            lr_mult = stage_config['lr_mult']
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_mult
            print(f"🔧 Conservative LR adjustment: {optimizer.param_groups[0]['lr']:.6f} (mult: {lr_mult:.2f})")
        
        # Create dataset
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
                            # Only use grids that fit in current stage size
                            if input_grid.shape[0] <= grid_size and input_grid.shape[1] <= grid_size:
                                dataset_samples.append({'inputs': input_grid, 'outputs': output_grid})
        
        # Add validation samples
        for filename in arc_files:
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    tasks = json.load(f)
                    for task_id, task_data in tasks.items():
                        if 'test' in task_data:
                            for example in task_data['test']:
                                input_grid = np.array(example['input'])
                                output_grid = np.array(example['output'])
                                if input_grid.shape[0] <= grid_size and input_grid.shape[1] <= grid_size:
                                    dataset_samples.append({'inputs': input_grid, 'outputs': output_grid})
        
        if not dataset_samples:
            print(f"⚠️ No suitable samples for stage {stage}, skipping")
            continue
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Total samples: {len(dataset_samples)}")
        
        # Create dataset and dataloaders
        dataset = AtlasSpecializedDataset(
            dataset_samples, 
            max_grid_size=grid_size,
            augment=False  # Disable augmentation for conservative training
        )
        
        # Conservative train/val split
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=ATLAS_CONFIG['batch_size'],
            shuffle=True, 
            collate_fn=custom_collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=ATLAS_CONFIG['batch_size'],
            shuffle=False, 
            collate_fn=custom_collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Train this stage with extra patience and gentle updates
        stage_best_exact = 0.0
        patience_counter = 0
        
        print(f"\n🔄 Conservative training Stage {stage} for {ATLAS_CONFIG['epochs_per_stage']} epochs...")
        
        for epoch in range(ATLAS_CONFIG['epochs_per_stage']):
            model.train()
            train_loss = 0.0
            train_exact = 0
            train_samples = 0
            
            # Training loop with gradient accumulation
            pbar = tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                
                with autocast(device_type='cuda'):
                    outputs = model(inputs, targets, mode='training')
                    predictions = outputs['predicted_output']
                    loss = loss_fn(predictions, targets, outputs)
                    
                    # Gradient accumulation
                    loss = loss / ATLAS_CONFIG['gradient_accumulation']
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % ATLAS_CONFIG['gradient_accumulation'] == 0:
                    # Gentle gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ATLAS_CONFIG['gradient_clip'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Calculate exact matches
                with torch.no_grad():
                    pred_indices = predictions.argmax(dim=1)
                    target_indices = targets.argmax(dim=1) if targets.dim() > 2 else targets
                    exact_matches = (pred_indices == target_indices).all(dim=(-1, -2)).sum().item()
                    
                    train_exact += exact_matches
                    train_samples += inputs.size(0)
                    train_loss += loss.item() * ATLAS_CONFIG['gradient_accumulation']
                
                # Update progress bar
                current_exact = (train_exact / train_samples) * 100
                pbar.set_postfix({
                    'loss': f"{train_loss/(batch_idx+1):.3f}",
                    'exact': f"{exact_matches}/{inputs.size(0)}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
            
            global_epoch += 1
            
            # Validation every 10 epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == ATLAS_CONFIG['epochs_per_stage'] - 1:
                model.eval()
                val_loss = 0.0
                val_exact = 0
                val_samples = 0
                val_pixel_acc = 0
                
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc="Validation")
                    for batch in val_pbar:
                        inputs = batch['inputs'].to(device)
                        targets = batch['targets'].to(device)
                        
                        with autocast(device_type='cuda'):
                            outputs = model(inputs, targets, mode='training')
                            predictions = outputs['predicted_output']
                            loss = loss_fn(predictions, targets, outputs)
                        
                        pred_indices = predictions.argmax(dim=1)
                        target_indices = targets.argmax(dim=1) if targets.dim() > 2 else targets
                        exact_matches = (pred_indices == target_indices).all(dim=(-1, -2)).sum().item()
                        pixel_matches = (pred_indices == target_indices).float().mean().item() * 100
                        
                        val_exact += exact_matches
                        val_samples += inputs.size(0)
                        val_loss += loss.item()
                        val_pixel_acc += pixel_matches
                
                val_exact_pct = (val_exact / val_samples) * 100
                val_pixel_pct = val_pixel_acc / len(val_loader)
                
                print(f"\n🌍 ATLAS Epoch {epoch+1} (Stage {stage}, {grid_size}x{grid_size}):")
                print(f"   🎯 Train: {current_exact:.2f}% exact, Loss: {train_loss/len(train_loader):.3f}")
                print(f"   🎯 Val: {val_exact_pct:.2f}% exact, Loss: {val_loss/len(val_loader):.3f}, Pixel: {val_pixel_pct:.1f}%")
                
                # Track best model - save more frequently in V3
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    stage_best_exact = val_exact_pct
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': global_epoch,
                        'stage': stage,
                        'best_exact': best_exact,
                        'config': ATLAS_CONFIG
                    }, best_model_path)
                    
                    print(f"   🏆 New best model! Exact: {best_exact:.2f}%")
                else:
                    patience_counter += 1
                
                # Save checkpoint every 20 epochs
                if (epoch + 1) % 20 == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': global_epoch,
                        'stage': stage,
                        'best_exact': best_exact,
                        'config': ATLAS_CONFIG
                    }, checkpoint_path)
                
                # Record metrics
                history['train_exact'].append(current_exact)
                history['val_exact'].append(val_exact_pct)
                history['train_loss'].append(train_loss/len(train_loader))
                history['val_loss'].append(val_loss/len(val_loader))
                
                # Early stopping with much more patience in V3
                if patience_counter >= ATLAS_CONFIG['patience']:
                    print(f"⏰ Early stopping after {patience_counter} epochs without improvement")
                    break
        
        # Stage complete
        stage_metrics.append({
            'stage': stage,
            'grid_size': grid_size,
            'final_exact': stage_best_exact,
            'epochs': epoch + 1
        })
        
        print(f"\n✅ Stage {stage} complete! Final exact: {stage_best_exact:.2f}%")
        
        # Clear memory
        del train_loader, val_loader, dataset
        gc.collect()
        torch.cuda.empty_cache()
    
    # Training complete
    print("\n" + "=" * 60)
    print("🎉 ATLAS V3 Conservative Training Complete!")
    print(f"   🏆 Best exact match: {best_exact:.2f}%")
    print(f"   📏 Stages completed: {len(stage_metrics)} (6x6 → 30x30 grids)")
    print(f"   📊 Total epochs: {global_epoch}")
    
    print(f"\n📏 Stage-by-stage Conservative Learning Progression:")
    for metrics in stage_metrics:
        print(f"   Stage {metrics['stage']} ({metrics['grid_size']}x{metrics['grid_size']}): "
              f"{metrics['final_exact']:.2f}% exact match ({metrics['epochs']} epochs)")
    
    return model, history


if __name__ == "__main__":
    print("=" * 80)
    print("ATLAS Specialized Training V3 - Conservative & Steady")
    print("Ultra-conservative approach to prevent accuracy degradation")
    print("=" * 80)
    
    try:
        model, history = train_atlas_specialized_v3()
        print("\n✅ ATLAS V3 training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise