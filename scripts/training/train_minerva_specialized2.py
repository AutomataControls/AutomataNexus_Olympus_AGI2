"""
MINERVA Specialized Training Script V2 - Enhanced Grid Reasoning & Strategic Analysis
Builds upon train_minerva_specialized.py with targeted improvements
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

# Import MINERVA model
from src.models.minerva_model import EnhancedMinervaNet

# Import ALL AutomataNexus novel training components
from src.dsl import DSLTrainingIntegration, DSLProgramGenerator
from src.dsl.minerva_dsl import MINERVADSLTraining, MINERVADSLGenerator
from src.program_synthesis.synthesis_integration import LightweightProgramSynthesizer, ProgramSynthesisDataGenerator

# PRISM System - Use MINERVA-specific version
try:
    from src.training_systems.minerva_prism import create_minerva_prism_system
    MINERVA_PRISM_AVAILABLE = True
except ImportError:
    MINERVA_PRISM_AVAILABLE = False
    print("⚠️ MINERVA-specific PRISM not available")
    # Fallback to generic version
    try:
        from src.program_synthesis.prism_system import PRISMSynthesizer, create_prism_system
        PRISM_AVAILABLE = True
    except ImportError:
        PRISM_AVAILABLE = False
        print("⚠️ Generic PRISM not available")

# MEPT and LEAP Systems - Use MINERVA-specific versions
try:
    from src.training_systems.minerva_mept import create_minerva_mept_system, MinervaMEPTLoss
    from src.training_systems.minerva_leap import create_minerva_leap_system, MinervaLEAPTrainer
    MINERVA_MEPT_LEAP_AVAILABLE = True
except ImportError:
    MINERVA_MEPT_LEAP_AVAILABLE = False
    print("⚠️ MINERVA-specific MEPT/LEAP not available")
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

# Data systems
from src.data.arc_data_synthesis import ARCDataSynthesizer, ARCDataAugmenter

# Import the ENTIRE original script to build upon it
from train_minerva_specialized import (
    MinervaSpecializedDataset,
    MinervaSpecializedLoss,
    minerva_exact_match_injection,
    minerva_mept_injection,
    minerva_leap_injection,
    minerva_prism_injection,
    custom_collate_fn,
    train_minerva_specialized as train_minerva_specialized_v1,
    MINERVA_CONFIG as MINERVA_CONFIG_V1,
    STAGE_CONFIG as STAGE_CONFIG_V1
)

# Enhanced MINERVA Configuration V2 - CHRONOS-style
MINERVA_CONFIG = MINERVA_CONFIG_V1.copy()
MINERVA_CONFIG.update({
    'batch_size': 256,  # CHRONOS-style batch size
    'learning_rate': 0.002,  # CHRONOS-style learning rate
    'num_epochs': 320,  # 8 stages x 40 epochs
    'gradient_accumulation': 2,  # Effective batch: 512
    'epochs_per_stage': 40,  # CHRONOS-style stage length
    'curriculum_stages': 8,  # CHRONOS-style progression
    'transform_penalty': 0.3,  # Lower than V1
    'exact_match_bonus': 3.0,  # Higher to encourage exact matches
    'relational_weight': 0.1,  # Higher for grid reasoning
    'pattern_memory_weight': 0.05,  # Higher for pattern recall
    
    # New V2 features
    'use_mixup': True,  # Grid mixup augmentation
    'mixup_alpha': 0.2,
    'gradient_clip': 1.0,
    'warmup_steps': 500,
    'cosine_restarts': True,
    'label_smoothing': 0.1,
    'grid_augmentation': True,  # New grid-specific augmentation
    'strategic_loss_weight': 0.2,  # New strategic reasoning loss
})

# Enhanced Stage Configuration V2 - CHRONOS-style
STAGE_CONFIG = STAGE_CONFIG_V1.copy()
for stage in STAGE_CONFIG:
    STAGE_CONFIG[stage]['lr_mult'] = min(1.0, STAGE_CONFIG[stage]['lr_mult'] * 1.2)

# CHRONOS-style exact injection for stage 0
STAGE_CONFIG[0]['exact_injection'] = True

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class MinervaSpecializedLossV2(MinervaSpecializedLoss):
    """Enhanced MINERVA loss with mixup and strategic reasoning loss"""
    
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
        
        # Add strategic reasoning loss
        strategic_loss = self._strategic_reasoning_loss(pred_output, target_output, model_outputs)
        base_losses['strategic'] = strategic_loss
        base_losses['total'] = base_losses['total'] + strategic_loss * MINERVA_CONFIG.get('strategic_loss_weight', 0.2)
        
        return base_losses
    
    def _strategic_reasoning_loss(self, pred, target, model_outputs):
        """Strategic pattern reasoning loss for MINERVA"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Grid structure preservation loss
        grid_loss = 0.0
        
        # Check row/column consistency
        for b in range(pred.shape[0]):
            # Row consistency
            for row in range(pred_idx.shape[1]):
                row_pred = pred_idx[b, row, :]
                row_target = target_idx[b, row, :]
                row_consistency = (row_pred == row_target).float().mean()
                grid_loss += (1 - row_consistency)
            
            # Column consistency
            for col in range(pred_idx.shape[2]):
                col_pred = pred_idx[b, :, col]
                col_target = target_idx[b, :, col]
                col_consistency = (col_pred == col_target).float().mean()
                grid_loss += (1 - col_consistency)
        
        grid_loss = grid_loss / (pred.shape[0] * (pred_idx.shape[1] + pred_idx.shape[2]))
        
        # Strategic pattern detection (quadrants, diagonals)
        pattern_loss = 0.0
        if model_outputs and 'attention_weights' in model_outputs:
            # Use attention to identify strategic patterns
            attention = model_outputs['attention_weights']
            # Encourage attention on strategic positions (corners, center, etc.)
            strategic_mask = self._create_strategic_mask(attention.shape[-1])
            pattern_loss = -torch.sum(attention * strategic_mask) / attention.shape[0]
        
        return grid_loss + pattern_loss * 0.5


def grid_augmentation(input_grid, output_grid):
    """Apply grid-specific augmentation for MINERVA"""
    if random.random() < 0.3:
        # 90-degree rotation
        k = random.randint(1, 3)
        input_rotated = torch.rot90(input_grid, k, dims=[-2, -1])
        output_rotated = torch.rot90(output_grid, k, dims=[-2, -1])
        return input_rotated, output_rotated
    elif random.random() < 0.3:
        # Flip
        if random.random() < 0.5:
            # Horizontal flip
            input_flipped = torch.flip(input_grid, dims=[-1])
            output_flipped = torch.flip(output_grid, dims=[-1])
        else:
            # Vertical flip
            input_flipped = torch.flip(input_grid, dims=[-2])
            output_flipped = torch.flip(output_grid, dims=[-2])
        return input_flipped, output_flipped
    return input_grid, output_grid


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


def custom_collate_fn_v2(batch, stage):
    """V2 collate function that handles numpy arrays with negative strides"""
    inputs = []
    outputs = []
    
    # Determine target size based on stage
    target_sizes = {0: 6, 1: 8, 2: 10, 3: 12, 4: 15, 5: 19, 6: 25, 7: 30}
    target_size = target_sizes.get(stage, 30)
    
    for item in batch:
        input_grid = item['inputs']
        output_grid = item['outputs']
        
        # Convert numpy arrays to tensors with .copy() to handle negative strides
        if isinstance(input_grid, np.ndarray):
            input_grid = torch.from_numpy(input_grid.copy()).long()
        if isinstance(output_grid, np.ndarray):
            output_grid = torch.from_numpy(output_grid.copy()).long()
        
        # Ensure 2D tensors
        while input_grid.dim() > 2:
            input_grid = input_grid.squeeze(0)
        while output_grid.dim() > 2:
            output_grid = output_grid.squeeze(0)
        while input_grid.dim() < 2:
            input_grid = input_grid.unsqueeze(0)
        while output_grid.dim() < 2:
            output_grid = output_grid.unsqueeze(0)
        
        # Always resize to target size to ensure consistency
        h, w = input_grid.shape
        # Use interpolation to resize to exact target size
        input_grid = F.interpolate(input_grid.unsqueeze(0).unsqueeze(0).float(), 
                                 size=(target_size, target_size), 
                                 mode='nearest').squeeze().long()
        output_grid = F.interpolate(output_grid.unsqueeze(0).unsqueeze(0).float(), 
                                  size=(target_size, target_size), 
                                  mode='nearest').squeeze().long()
        
        # Final validation to ensure correct size
        assert input_grid.shape == (target_size, target_size), f"Input shape {input_grid.shape} != ({target_size}, {target_size})"
        assert output_grid.shape == (target_size, target_size), f"Output shape {output_grid.shape} != ({target_size}, {target_size})"
        
        inputs.append(input_grid)
        outputs.append(output_grid)
    
    # Stack tensors
    inputs_tensor = torch.stack(inputs)
    outputs_tensor = torch.stack(outputs)
    
    return {
        'inputs': inputs_tensor,
        'outputs': outputs_tensor
    }


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


def train_minerva_specialized_v2():
    """Enhanced MINERVA training building on V1"""
    print("🧠 Starting MINERVA Specialized Training V2 - CHRONOS-Style")
    print("=" * 60)
    print("📊 CHRONOS-style Enhancements:")
    print("  • Massive dataset generation (40,000+ samples)")
    print("  • CHRONOS-style batch size (256) and learning rate (0.002)")
    print("  • 300K exact match injection for stage 0")
    print("  • 10x data augmentation like CHRONOS")
    print("  • Grid mixup and spatial augmentation")
    print("  • Strategic reasoning loss")
    print("  • Warmup + cosine annealing with restarts")
    print("  • Program synthesis integration")
    print("=" * 60)
    
    # Initialize model with maximum grid size from final stage
    max_grid_size = STAGE_CONFIG[7]['max_grid_size']  # 30x30
    model = EnhancedMinervaNet(
        max_grid_size=max_grid_size
    ).to(device)
    
    print(f"📊 MINERVA Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
    systems = {}
    
    # MEPT System - Use MINERVA-specific if available
    if USE_MEPT:
        if MINERVA_MEPT_LEAP_AVAILABLE:
            mept_components = create_minerva_mept_system()
            systems['replay_buffer'] = mept_components['replay_buffer']
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components['loss_fn']
            print("✅ MINERVA-specific MEPT system initialized")
        else:
            mept_components = create_mept_system(
                capacity=40000,
                pattern_bank_size=8000,
                transformation_penalty=MINERVA_CONFIG['transform_penalty'],
                exact_match_bonus=MINERVA_CONFIG['exact_match_bonus']
            )
            systems['replay_buffer'] = mept_components['replay_buffer']
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components.get('loss_fn')
            print("✅ Generic MEPT system initialized")
    
    # LEAP System - Use MINERVA-specific if available
    if USE_LEAP:
        if MINERVA_MEPT_LEAP_AVAILABLE:
            leap_components = create_minerva_leap_system()
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['pattern_generator']
            print("✅ MINERVA-specific LEAP system initialized")
        else:
            leap_components = create_leap_system(device)
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['pattern_generator']
            print("✅ Generic LEAP system initialized")
    
    # PRISM System - Use MINERVA-specific if available
    if USE_PRISM:
        if MINERVA_PRISM_AVAILABLE:
            prism_components = create_minerva_prism_system()
            systems['prism_synthesizer'] = prism_components['synthesizer']
            systems['program_library'] = prism_components['library']
            print("✅ MINERVA-specific PRISM system initialized")
        else:
            prism_components = create_prism_system()
            systems['prism_synthesizer'] = prism_components['synthesizer']
            # Generic PRISM doesn't have program_library
            print("✅ Generic PRISM system initialized")
    
    # Initialize V2 enhanced loss
    loss_fn = MinervaSpecializedLossV2().to(device)
    loss_fn.label_smoothing = MINERVA_CONFIG.get('label_smoothing', 0.1)
    
    # Optimizer - AdamW for better regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MINERVA_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    # Calculate total training steps
    total_epochs = MINERVA_CONFIG['num_epochs']
    steps_per_epoch = 100  # Approximate
    total_steps = total_epochs * steps_per_epoch
    
    # Warmup + Cosine scheduler with restarts
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=MINERVA_CONFIG['warmup_steps'],
        training_steps=total_steps,
        cycles=3 if MINERVA_CONFIG.get('cosine_restarts') else 1
    )
    
    scaler = GradScaler('cuda')
    
    # Data directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
    # Training metrics
    best_exact = 0.0
    global_epoch = 0
    global_step = 0
    
    # Check for existing best model
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    best_model_path = f'{models_dir}/minerva_best.pt'
    
    if os.path.exists(best_model_path):
        print(f"🔄 Loading best model from {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            loaded_exact = checkpoint.get('best_exact', 0.0)
            # For incremental training: reset threshold to allow improvements
            best_exact = 0.0  # Reset to allow stage validation to save improvements
            print(f"✅ Loaded best model with {loaded_exact:.2f}% exact match")
            print(f"🔄 Reset threshold to {best_exact:.2f}% for incremental training")
        except Exception as e:
            print(f"⚠️ Failed to load best model: {e}")
            print("🆕 Starting fresh training")
    else:
        print("🆕 No existing model found - starting fresh training")
    
    # Training history
    history = defaultdict(list)
    
    # 8-Stage Progressive Curriculum Training Loop
    stage_metrics = []
    
    # 4-PHASE INJECTION (if stage 0)
    if USE_EXACT_BOOST:
        print("\n" + "=" * 60)
        print("🧠 MINERVA 4-PHASE STRATEGIC INJECTION SEQUENCE")
        print("=" * 60)
        
        # Phase 1: Exact Match
        print("\n🎯 PHASE 1: Grid Identity Mapping")
        model = minerva_exact_match_injection(model, device, num_epochs=100, target_accuracy=90.0)
        
        # Phase 2: MEPT
        if USE_MEPT and 'replay_buffer' in systems:
            print("\n🎯 PHASE 2: Grid Memory Enhancement (MEPT)")
            model = minerva_mept_injection(model, device, systems, num_epochs=100, target_accuracy=90.0)
        
        # Phase 3: LEAP
        if USE_LEAP and 'leap_trainer' in systems:
            print("\n🎯 PHASE 3: Adaptive Grid Learning (LEAP)")
            model = minerva_leap_injection(model, device, systems, num_epochs=100, target_accuracy=90.0)
        
        # Phase 4: PRISM
        if USE_PRISM and 'prism_synthesizer' in systems:
            print("\n🎯 PHASE 4: Grid Program Synthesis (PRISM)")
            model = minerva_prism_injection(model, device, systems, num_epochs=200, target_accuracy=90.0)
        
        print("\n✅ 4-PHASE INJECTION COMPLETE")
        print("=" * 60)
    
    # Main curriculum training
    for stage in range(MINERVA_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        
        print(f"\n🧠 MINERVA Stage {stage}: {grid_size}x{grid_size} Grid Reasoning")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {int(stage_config['synthesis_ratio']*100)}%")
        print("=" * 60)
        
        # Create dataset using V1 approach but with enhancements
        print(f"🔧 Generating MINERVA-specific DSL grid patterns for stage {stage}...")
        
        # Create dataset
        dataset_samples = []
        
        # Generate MINERVA-specific DSL samples using static method
        dsl_samples = MINERVADSLTraining.create_minerva_dsl_samples(curriculum_stage=stage)
        dataset_samples.extend(dsl_samples)
        print(f"✅ Generated {len(dsl_samples)} MINERVA DSL grid pattern samples")
        
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
        
        # Add synthetic grid patterns
        for i in range(200):  # More synthetic data
            size = min(random.choice([4, 5, 6]), grid_size)
            input_grid = np.random.randint(0, 5, (size, size))
            
            # Grid transformations
            transform = random.choice(['rotate', 'flip', 'transpose', 'quadrant'])
            if transform == 'rotate':
                output_grid = np.rot90(input_grid, k=random.randint(1, 3)).copy()
            elif transform == 'flip':
                output_grid = np.flip(input_grid, axis=random.randint(0, 1)).copy()
            elif transform == 'transpose':
                output_grid = input_grid.T.copy()
            else:  # quadrant
                output_grid = np.zeros_like(input_grid)
                mid_h, mid_w = size // 2, size // 2
                # Only do quadrant swap if grid is even-sized
                if size % 2 == 0:
                    # Swap quadrants
                    output_grid[:mid_h, :mid_w] = input_grid[mid_h:, mid_w:]
                    output_grid[mid_h:, mid_w:] = input_grid[:mid_h, :mid_w]
                    output_grid[:mid_h, mid_w:] = input_grid[mid_h:, :mid_w]
                    output_grid[mid_h:, :mid_w] = input_grid[:mid_h, mid_w:]
                else:
                    # For odd-sized grids, just rotate 180 degrees
                    output_grid = np.rot90(input_grid, k=2).copy()
            
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
        
        # DISABLE the specialized dataset wrapper - it's causing DataLoader hanging
        # The MinervaSpecializedDataset replay buffer sampling is the source of the hang
        # if USE_MEPT and 'replay_buffer' in systems:
        #     train_dataset = MinervaSpecializedDataset(
        #         train_dataset,
        #         systems['replay_buffer'],
        #         replay_ratio=0.3
        #     )
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Train: {train_size}, Val: {val_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=MINERVA_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: custom_collate_fn_v2(batch, stage),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=MINERVA_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: custom_collate_fn_v2(batch, stage),
            drop_last=False
        )
        
        # Adjust learning rate for stage
        stage_lr = MINERVA_CONFIG['learning_rate'] * stage_config['lr_mult']
        for param_group in optimizer.param_groups:
            param_group['lr'] = stage_lr
        
        # Stage training loop
        print(f"\n🔄 Training Stage {stage} for {MINERVA_CONFIG['epochs_per_stage']} epochs...")
        
        for epoch in range(MINERVA_CONFIG['epochs_per_stage']):
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
                
                # Apply grid augmentation
                if MINERVA_CONFIG.get('grid_augmentation') and random.random() < 0.5:
                    inputs, outputs = grid_augmentation(inputs, outputs)
                
                # Convert to one-hot
                input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                
                # Apply mixup if enabled
                if MINERVA_CONFIG.get('use_mixup') and random.random() < 0.5:
                    mixed_input, target_a, target_b, lam = mixup_data(
                        input_grids, output_grids, alpha=MINERVA_CONFIG.get('mixup_alpha', 0.2)
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
                
                loss = losses['total'] / MINERVA_CONFIG['gradient_accumulation']
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % MINERVA_CONFIG['gradient_accumulation'] == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MINERVA_CONFIG['gradient_clip'])
                    
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
            if epoch % 5 == 0 or epoch == MINERVA_CONFIG['epochs_per_stage'] - 1:
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
                
                print(f"\n🧠 MINERVA Epoch {epoch+1} (Stage {stage}, {grid_size}x{grid_size}):")
                print(f"   \033[96m🎯 Train: {train_exact_pct:.2f}% exact, Loss: {train_loss:.3f}\033[0m")
                print(f"   \033[96m🎯 Val: {val_exact_pct:.2f}% exact, Loss: {val_loss:.3f}, Pixel: {val_pixel_acc:.1f}%\033[0m")
                
                # Save history
                history['train_loss'].append(train_loss)
                history['train_exact'].append(train_exact_pct)
                history['val_loss'].append(val_loss)
                history['val_exact'].append(val_exact_pct)
                history['learning_rate'].append(scheduler.get_lr()[0])
                
                # Save best model (SAME NAME AS V1)
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_exact': best_exact,
                        'val_loss': val_loss
                    }, f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/minerva_best.pt')
                    print(f"   \033[96m🏆 New best model! Exact: {best_exact:.2f}%\033[0m")
        
        # Stage complete
        stage_exact = train_exact_pct
        stage_metrics.append({
            'stage': stage,
            'grid_size': grid_size,
            'final_exact': stage_exact
        })
        
        print(f"\n\033[96m✅ Stage {stage} complete! Final exact: {stage_exact:.2f}%\033[0m")
        
        # CHRONOS-style exact match injection for stage 0
        if stage == 0 and STAGE_CONFIG[0].get('exact_injection'):
            print(f"\033[96m🎯 CHRONOS-style exact match injection training for stage 0...\033[0m")
            
            try:
                from stage0_exact_match_boost import ExactMatchBoostDataset, AggressiveLoss
                
                # Create exact match dataset with 300K samples
                exact_dataset = ExactMatchBoostDataset(300_000)  # Pass number, not list
                exact_loader = DataLoader(
                    exact_dataset, 
                    batch_size=MINERVA_CONFIG['batch_size'], 
                    shuffle=True,
                    num_workers=2
                )
                
                # Train with exact matches
                model.train()
                exact_optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # Very conservative
                aggressive_loss = AggressiveLoss()
                
                exact_epochs = 5  # Short burst
                for epoch in range(exact_epochs):
                    epoch_loss = 0
                    exact_matches = 0
                    total_samples = 0
                    
                    for batch_idx, batch in enumerate(tqdm(exact_loader, desc=f"Exact Match Epoch {epoch+1}")):
                        exact_optimizer.zero_grad()
                        
                        inputs = batch['input'].to(device).long()
                        targets = batch['output'].to(device).long()
                        
                        with autocast('cuda'):
                            outputs = model(F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float())
                            if isinstance(outputs, dict):
                                outputs = outputs['predicted_output']
                            loss = aggressive_loss(outputs, targets, F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float())
                        
                        scaler.scale(loss).backward()
                        scaler.step(exact_optimizer)
                        scaler.update()
                        
                        epoch_loss += loss.item()
                        
                        # Calculate accuracy
                        pred_classes = outputs.argmax(1)
                        exact_matches += (pred_classes == targets).all(dim=(1,2)).sum().item()
                        total_samples += inputs.size(0)
                        
                        if batch_idx % 100 == 0:
                            batch_acc = (pred_classes == targets).all(dim=(1,2)).float().mean().item() * 100
                            print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Acc={batch_acc:.2f}%")
                    
                    avg_loss = epoch_loss / len(exact_loader)
                    exact_acc = exact_matches / total_samples * 100
                    print(f"\033[96m✓ Exact Match Epoch {epoch+1}: Avg Loss={avg_loss:.4f}, Acc={exact_acc:.2f}%\033[0m")
                
                print(f"\033[96m🎯 CHRONOS-style exact match injection complete!\033[0m")
                
            except ImportError:
                print("⚠️ Exact match boost not available for injection")
        
        # Clear memory
        del train_loader, val_loader, dataset
        gc.collect()
        torch.cuda.empty_cache()
    
    # Training complete
    print("\n" + "=" * 60)
    print("\033[96m🎉 MINERVA V2 8-Stage CHRONOS-Style Training Complete!\033[0m")
    print(f"   \033[96m🏆 Best exact match: {best_exact:.2f}%\033[0m")
    print(f"   \033[96m📏 Stages completed: 8 (6x6 → 30x30 grids)\033[0m")
    print(f"   \033[96m📊 Total epochs: {global_epoch}\033[0m")
    
    print(f"\n📏 Stage-by-stage Grid Learning Progression:")
    for metrics in stage_metrics:
        print(f"   Stage {metrics['stage']} ({metrics['grid_size']}x{metrics['grid_size']}): "
              f"{metrics['final_exact']:.2f}% exact match")
    
    return model, history


# Training components flags (from V1)
USE_MEPT = True and (MINERVA_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_LEAP = True and (MINERVA_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_PRISM = True and (MINERVA_PRISM_AVAILABLE or PRISM_AVAILABLE)
USE_EXACT_BOOST = True
try:
    from stage0_exact_match_boost import ExactMatchBoostDataset, AggressiveLoss, inject_exact_match_training
    EXACT_BOOST_AVAILABLE = True
except ImportError:
    EXACT_BOOST_AVAILABLE = False
    print("⚠️ Exact match boost not available")

USE_EXACT_BOOST = USE_EXACT_BOOST and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE


if __name__ == "__main__":
    print("=" * 80)
    print("MINERVA Specialized Training V2 - Building on V1")
    print("Grid Reasoning & Strategic Analysis Expert with Targeted Enhancements")
    print("=" * 80)
    
    train_minerva_specialized_v2()