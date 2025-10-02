"""
IRIS Specialized Training Script V2 - Enhanced Color Pattern Recognition Expert
Builds upon train_iris_specialized.py with targeted improvements
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

# Import IRIS model
from src.models.iris_model import EnhancedIrisNet

# Import ALL AutomataNexus novel training components
from src.dsl import DSLTrainingIntegration, DSLProgramGenerator
from src.dsl.iris_dsl import IRISDSLTraining, IRISDSLGenerator
from src.program_synthesis.synthesis_integration import LightweightProgramSynthesizer, ProgramSynthesisDataGenerator

# IRIS-specific program synthesis
try:
    from src.program_synthesis.iris_synthesis import IRISProgramSynthesizer, create_iris_synthesis_system
    IRIS_SYNTHESIS_AVAILABLE = True
except ImportError:
    IRIS_SYNTHESIS_AVAILABLE = False
    print("⚠️ IRIS-specific synthesis not available")

# PRISM System - Use IRIS-specific version
try:
    from src.training_systems.iris_prism import create_iris_prism_system
    IRIS_PRISM_AVAILABLE = True
except ImportError:
    IRIS_PRISM_AVAILABLE = False
    print("⚠️ IRIS-specific PRISM not available")
    # Fallback to generic version
    try:
        from src.program_synthesis.prism_system import PRISMSynthesizer, create_prism_system
        PRISM_AVAILABLE = True
    except ImportError:
        PRISM_AVAILABLE = False
        print("⚠️ Generic PRISM not available")

# MEPT and LEAP Systems - Use IRIS-specific versions
try:
    from src.training_systems.iris_mept import create_iris_mept_system, IrisMEPTLoss
    from src.training_systems.iris_leap import create_iris_leap_system, IrisLEAPTrainer
    IRIS_MEPT_LEAP_AVAILABLE = True
except ImportError:
    IRIS_MEPT_LEAP_AVAILABLE = False
    print("⚠️ IRIS-specific MEPT/LEAP not available")
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
from train_iris_specialized import (
    IrisSpecializedDataset, 
    IrisSpecializedLoss,
    iris_exact_match_injection,
    iris_mept_injection,
    iris_leap_injection,
    iris_prism_injection,
    custom_collate_fn,
    train_iris_specialized as train_iris_specialized_v1,
    IRIS_CONFIG as IRIS_CONFIG_V1,
    STAGE_CONFIG as STAGE_CONFIG_V1
)

# Enhanced IRIS Configuration V2 - Building on V1
IRIS_CONFIG = IRIS_CONFIG_V1.copy()
IRIS_CONFIG.update({
    # Refined parameters based on V1 issues
    'batch_size': 128,  # Smaller than V1 for better gradients
    'learning_rate': 0.001,  # More conservative than V1
    'gradient_accumulation': 2,  # Effective batch: 256
    'transform_penalty': 0.2,  # Lower for color transformations
    'exact_match_bonus': 2.5,  # Balanced bonus
    'color_mapping_weight': 0.15,  # More conservative
    'color_consistency_weight': 0.15,
    'color_diversity_weight': 0.25,  # Slightly higher for diversity
    'lstm_rule_weight': 0.05,  # Very conservative
    
    # New V2 features
    'use_mixup': True,  # Color mixup augmentation
    'mixup_alpha': 0.3,  # Higher for color mixing
    'gradient_clip': 0.5,  # More aggressive clipping
    'warmup_steps': 1000,  # Longer warmup
    'cosine_restarts': True,  # Cosine annealing with restarts
    'label_smoothing': 0.05,  # Less smoothing for color precision
    'color_augmentation': True,  # New color-specific augmentation
    'perceptual_loss_weight': 0.1,  # New perceptual color loss
})

# Enhanced Stage Configuration V2
STAGE_CONFIG = STAGE_CONFIG_V1.copy()
# Adjust learning rates more conservatively
for stage in STAGE_CONFIG:
    STAGE_CONFIG[stage]['lr_mult'] = 1.0 - (stage * 0.1)  # More gradual decay

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class IrisSpecializedLossV2(IrisSpecializedLoss):
    """Enhanced IRIS loss with mixup and perceptual color loss"""
    
    def __init__(self):
        super().__init__()
        # Add perceptual loss weight
        self.weights['perceptual'] = IRIS_CONFIG.get('perceptual_loss_weight', 0.1)
    
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
        
        # Get base loss
        base_losses = super().forward(pred_output, target_output, input_grid, model_outputs)
        
        # Add perceptual color loss
        perceptual_loss = self._perceptual_color_loss(pred_output, target_output)
        base_losses['perceptual'] = perceptual_loss
        base_losses['total'] = base_losses['total'] + perceptual_loss * self.weights['perceptual']
        
        return base_losses
    
    def _perceptual_color_loss(self, pred, target):
        """Perceptual color similarity loss"""
        # Color distance in perceptual space
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Define perceptual color distances (simplified)
        # Colors that are perceptually similar have lower distance
        color_distances = torch.tensor([
            [0, 3, 4, 5, 4, 5, 6, 7, 8, 9],  # 0 (black)
            [3, 0, 2, 3, 4, 5, 6, 7, 8, 9],  # 1 (blue)
            [4, 2, 0, 2, 3, 4, 5, 6, 7, 8],  # 2 (red)
            [5, 3, 2, 0, 2, 3, 4, 5, 6, 7],  # 3 (green)
            [4, 4, 3, 2, 0, 2, 3, 4, 5, 6],  # 4 (yellow)
            [5, 5, 4, 3, 2, 0, 2, 3, 4, 5],  # 5 (gray)
            [6, 6, 5, 4, 3, 2, 0, 2, 3, 4],  # 6 (magenta)
            [7, 7, 6, 5, 4, 3, 2, 0, 2, 3],  # 7 (orange)
            [8, 8, 7, 6, 5, 4, 3, 2, 0, 2],  # 8 (light blue)
            [9, 9, 8, 7, 6, 5, 4, 3, 2, 0],  # 9 (brown)
        ], dtype=torch.float32, device=pred.device) / 9.0  # Normalize
        
        # Calculate perceptual distance
        pred_flat = pred_idx.flatten()
        target_flat = target_idx.flatten()
        
        perceptual_dist = 0
        for i in range(len(pred_flat)):
            perceptual_dist += color_distances[pred_flat[i], target_flat[i]]
        
        return perceptual_dist / len(pred_flat)


def color_augmentation(input_grid, output_grid):
    """Apply color-specific augmentation"""
    if random.random() < 0.3:
        # Color shift
        shift = random.randint(1, 9)
        input_shifted = (input_grid + shift) % 10
        output_shifted = (output_grid + shift) % 10
        return input_shifted, output_shifted
    elif random.random() < 0.3:
        # Color swap
        c1, c2 = random.sample(range(10), 2)
        input_swapped = input_grid.clone()
        output_swapped = output_grid.clone()
        mask1 = input_grid == c1
        mask2 = input_grid == c2
        input_swapped[mask1] = c2
        input_swapped[mask2] = c1
        mask1 = output_grid == c1
        mask2 = output_grid == c2
        output_swapped[mask1] = c2
        output_swapped[mask2] = c1
        return input_swapped, output_swapped
    return input_grid, output_grid


def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation for color patterns"""
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


def train_iris_specialized_v2():
    """Enhanced IRIS training building on V1"""
    print("🎨 Starting IRIS Specialized Training V2")
    print("=" * 60)
    print("📊 Enhancements over V1:")
    print("  • Color mixup augmentation")
    print("  • Perceptual color loss")
    print("  • Color-specific augmentation")
    print("  • Warmup + cosine annealing with restarts")
    print("  • Label smoothing for color precision")
    print("  • Better batch size and learning rate")
    print("=" * 60)
    
    # Initialize model with maximum grid size from final stage
    max_grid_size = STAGE_CONFIG[7]['max_grid_size']  # 30x30
    model = EnhancedIrisNet(
        max_grid_size=max_grid_size
    ).to(device)
    
    print(f"📊 IRIS Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
    systems = {}
    
    # MEPT System - Use IRIS-specific if available
    if USE_MEPT:
        if IRIS_MEPT_LEAP_AVAILABLE:
            mept_components = create_iris_mept_system()
            systems['replay_buffer'] = mept_components['replay_buffer']
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components['loss_fn']
            print("✅ IRIS-specific MEPT system initialized")
        else:
            mept_components = create_mept_system(
                capacity=50000,
                pattern_bank_size=10000,
                transformation_penalty=IRIS_CONFIG['transform_penalty'],
                exact_match_bonus=IRIS_CONFIG['exact_match_bonus']
            )
            systems['replay_buffer'] = mept_components['replay_buffer']
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components.get('loss_fn')
            print("✅ Generic MEPT system initialized")
    
    # LEAP System - Use IRIS-specific if available
    if USE_LEAP:
        if IRIS_MEPT_LEAP_AVAILABLE:
            leap_components = create_iris_leap_system()
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['pattern_generator']
            print("✅ IRIS-specific LEAP system initialized")
        else:
            leap_components = create_leap_system(device)
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['pattern_generator']
            print("✅ Generic LEAP system initialized")
    
    # PRISM System - Use IRIS-specific if available
    if USE_PRISM:
        if IRIS_PRISM_AVAILABLE:
            prism_components = create_iris_prism_system()
            systems['prism_synthesizer'] = prism_components['synthesizer']
            systems['program_library'] = prism_components['program_library']
            print("✅ IRIS-specific PRISM system initialized")
        else:
            prism_components = create_prism_system()
            systems['prism_synthesizer'] = prism_components['synthesizer']
            systems['program_library'] = prism_components['program_library']
            print("✅ Generic PRISM system initialized")
    
    # Initialize V2 enhanced loss
    loss_fn = IrisSpecializedLossV2().to(device)
    loss_fn.label_smoothing = IRIS_CONFIG.get('label_smoothing', 0.05)
    
    # Optimizer - AdamW for better regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=IRIS_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=5e-5
    )
    
    # Calculate total training steps
    total_epochs = IRIS_CONFIG['num_epochs']
    steps_per_epoch = 100  # Approximate
    total_steps = total_epochs * steps_per_epoch
    
    # Warmup + Cosine scheduler with restarts
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=IRIS_CONFIG['warmup_steps'],
        training_steps=total_steps,
        cycles=3 if IRIS_CONFIG.get('cosine_restarts') else 1
    )
    
    scaler = GradScaler('cuda')
    
    # Data directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
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
        print("🌈 IRIS 4-PHASE COLOR PERCEPTION INJECTION SEQUENCE")
        print("=" * 60)
        
        # Phase 1: Exact Match
        print("\n🎨 PHASE 1: Color Identity Mapping")
        model = iris_exact_match_injection(model, device, num_epochs=120, target_accuracy=88.0)
        
        # Phase 2: MEPT
        if USE_MEPT:
            print("\n🎨 PHASE 2: Color Memory Enhancement (MEPT)")
            model = iris_mept_injection(model, device, num_epochs=100, target_accuracy=90.0)
        
        # Phase 3: LEAP
        if USE_LEAP:
            print("\n🎨 PHASE 3: Adaptive Color Learning (LEAP)")
            model = iris_leap_injection(model, device, num_epochs=100, target_accuracy=90.0)
        
        # Phase 4: PRISM
        if USE_PRISM:
            print("\n🎨 PHASE 4: Color Program Synthesis (PRISM)")
            model = iris_prism_injection(model, device, num_epochs=100)
        
        print("\n✅ 4-PHASE INJECTION COMPLETE")
        print("=" * 60)
    
    # Main curriculum training
    for stage in range(IRIS_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        
        print(f"\n🎨 IRIS Stage {stage}: {grid_size}x{grid_size} Color Pattern Recognition")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {int(stage_config['synthesis_ratio']*100)}%")
        print("=" * 60)
        
        # Create dataset using V1 approach but with enhancements
        print(f"🔧 Generating IRIS-specific DSL color patterns for stage {stage}...")
        
        # Initialize DSL trainer
        dsl_samples = IRISDSLTraining.create_iris_dsl_samples(curriculum_stage=stage)
        print(f"✅ Created {len(dsl_samples)} IRIS DSL color pattern samples")
        
        # Create dataset
        dataset_samples = []
        dataset_samples.extend(dsl_samples)
        
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
        
        # Add synthetic color patterns
        for i in range(300):  # More synthetic data for color learning
            size = min(random.choice([4, 5, 6]), grid_size)
            input_grid = np.random.randint(0, 8, (size, size))  # Use fewer colors initially
            
            # Color transformations
            transform = random.choice(['gradient', 'blocks', 'stripes', 'shift'])
            if transform == 'gradient':
                output_grid = np.zeros_like(input_grid)
                for row in range(size):
                    output_grid[row, :] = (row * 7) // size + 1
            elif transform == 'blocks':
                output_grid = np.zeros_like(input_grid)
                block_size = size // 2
                for bi in range(2):
                    for bj in range(2):
                        color = bi * 2 + bj + 1
                        output_grid[bi*block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size] = color
            elif transform == 'stripes':
                output_grid = np.zeros_like(input_grid)
                for col in range(size):
                    output_grid[:, col] = (col % 4) + 1
            else:  # shift
                output_grid = (input_grid + 2) % 8
            
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
        
        # Limit dataset size
        if len(dataset) > 20000:
            dataset = torch.utils.data.Subset(dataset, list(range(20000)))
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Train: {train_size}, Val: {val_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=IRIS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=IRIS_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=False
        )
        
        # Adjust learning rate for stage
        stage_lr = IRIS_CONFIG['learning_rate'] * stage_config['lr_mult']
        for param_group in optimizer.param_groups:
            param_group['lr'] = stage_lr
        
        # Stage training loop
        print(f"\n🔄 Training Stage {stage} for {IRIS_CONFIG['epochs_per_stage']} epochs...")
        
        for epoch in range(IRIS_CONFIG['epochs_per_stage']):
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
                
                # Apply color augmentation
                if IRIS_CONFIG.get('color_augmentation') and random.random() < 0.5:
                    inputs, outputs = color_augmentation(inputs, outputs)
                
                # Convert to one-hot
                input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                
                # Apply mixup if enabled
                if IRIS_CONFIG.get('use_mixup') and random.random() < 0.5:
                    mixed_input, target_a, target_b, lam = mixup_data(
                        input_grids, output_grids, alpha=IRIS_CONFIG.get('mixup_alpha', 0.3)
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
                
                loss = losses['total'] / IRIS_CONFIG['gradient_accumulation']
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % IRIS_CONFIG['gradient_accumulation'] == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), IRIS_CONFIG['gradient_clip'])
                    
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
            if epoch % 5 == 0 or epoch == IRIS_CONFIG['epochs_per_stage'] - 1:
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
                
                print(f"\n🎨 IRIS Epoch {epoch+1} (Stage {stage}, {grid_size}x{grid_size}):")
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
                    }, f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/iris_best.pt')
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
    print("🎉 IRIS V2 8-Stage Training Complete!")
    print(f"   🏆 Best exact match: {best_exact:.2f}%")
    print(f"   📏 Stages completed: 8 (6x6 → 30x30 grids)")
    print(f"   📊 Total epochs: {global_epoch}")
    
    print(f"\n📏 Stage-by-stage Color Learning Progression:")
    for metrics in stage_metrics:
        print(f"   Stage {metrics['stage']} ({metrics['grid_size']}x{metrics['grid_size']}): "
              f"{metrics['final_exact']:.2f}% exact match")
    
    return model, history


# Training components flags (from V1)
USE_MEPT = True and (IRIS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_LEAP = True and (IRIS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_PRISM = True and (IRIS_PRISM_AVAILABLE or PRISM_AVAILABLE)
USE_EXACT_BOOST = True and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE


if __name__ == "__main__":
    print("=" * 80)
    print("IRIS Specialized Training V2 - Building on V1")
    print("Color Pattern Recognition Expert with Targeted Enhancements")
    print("=" * 80)
    
    train_iris_specialized_v2()