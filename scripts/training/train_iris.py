"""
IRIS Individual Training Script - V4 Enhanced
Feature Extraction Specialist Training with MEPT, LEAP, and PRISM
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
from collections import defaultdict

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import model and components
from src.models.iris_model import EnhancedIrisNet
from src.dsl import DSLTrainingIntegration, DSLProgramGenerator
from src.program_synthesis.synthesis_integration import LightweightProgramSynthesizer, ProgramSynthesisDataGenerator
try:
    from src.program_synthesis.prism_system import PRISMSynthesizer, create_prism_system
    PRISM_AVAILABLE = True
except ImportError:
    PRISM_AVAILABLE = False
from src.data.arc_data_synthesis import ARCDataSynthesizer, ARCDataAugmenter

# Import MEPT and LEAP components
try:
    from src.utils.mept_system import ExperienceReplayBuffer, PatternBank, MEPTLoss, create_mept_system
    from src.utils.leap_system import AdaptivePatternGenerator, LEAPTrainer, create_leap_system
    MEPT_LEAP_AVAILABLE = True
except ImportError:
    MEPT_LEAP_AVAILABLE = False
    print("⚠️ MEPT/LEAP components not available")

# Define MEPTAugmentedDataset here since it's not in mept_system.py
class MEPTAugmentedDataset(Dataset):
    """Dataset that combines regular data with replay buffer samples"""
    def __init__(self, base_dataset, replay_buffer, replay_ratio=0.3):
        self.base_dataset = base_dataset
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # With probability replay_ratio, return a replay sample
        if random.random() < self.replay_ratio and len(self.replay_buffer.buffer) > 0:
            experiences = self.replay_buffer.sample(1, exact_ratio=0.7)
            if experiences:
                exp = experiences[0]
                # Convert from one-hot back to indices if needed
                input_tensor = exp['input']
                output_tensor = exp['output']
                
                # Check if one-hot encoded (4D tensor with channels)
                if input_tensor.dim() == 4:
                    # Convert from one-hot to indices: [C, H, W] -> [H, W]
                    input_tensor = input_tensor.argmax(dim=0)
                elif input_tensor.dim() == 3:
                    # Already indices, just remove batch dim if present
                    input_tensor = input_tensor.squeeze(0)
                    
                if output_tensor.dim() == 4:
                    output_tensor = output_tensor.argmax(dim=0)
                elif output_tensor.dim() == 3:
                    output_tensor = output_tensor.squeeze(0)
                
                return {
                    'inputs': input_tensor,
                    'outputs': output_tensor
                }
        
        # Otherwise return regular sample
        return self.base_dataset[idx]

# Import LEAP-PRISM bridge
try:
    from src.utils.leap_prism_bridge import create_leap_prism_bridge, LEAPPatternEnhancer
    LEAP_PRISM_BRIDGE_AVAILABLE = True
except ImportError:
    LEAP_PRISM_BRIDGE_AVAILABLE = False

# Import exact match boost
try:
    from stage0_exact_match_boost import ExactMatchBoostDataset, AggressiveLoss, inject_exact_match_training
    EXACT_BOOST_AVAILABLE = True
except ImportError:
    EXACT_BOOST_AVAILABLE = False
    print("⚠️ Exact match boost not available")

# Import training components from V4
from colab_training_v4_megascale_curriculum import (
    CurriculumMegaScaleDataset, TrainingReporter
)

BATCH_SIZE = 512
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.005
NUM_EPOCHS = 300
MAX_GRID_SIZE = 25
NUM_COLORS = 10

import os
import multiprocessing

cpu_count = multiprocessing.cpu_count()
if torch.cuda.is_available():
    NUM_WORKERS = min(8, cpu_count)
    PIN_MEMORY = True
else:
    NUM_WORKERS = min(2, cpu_count)
    PIN_MEMORY = False

PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

RECONSTRUCTION_WEIGHT = 1.0
EDGE_WEIGHT = 0.3
COLOR_BALANCE_WEIGHT = 0.2
STRUCTURE_WEIGHT = 0.3
TRANSFORMATION_PENALTY = 0.5
EXACT_MATCH_BONUS = 5.0

CURRICULUM_STAGES = 3
EPOCHS_PER_STAGE = 100

USE_MEPT = True
USE_LEAP = True
USE_PRISM = True and PRISM_AVAILABLE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'

print(f"\n⚙️ IRIS V4 Training Configuration:")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Transformation penalty: {TRANSFORMATION_PENALTY}")
print(f"  Exact match bonus: {EXACT_MATCH_BONUS}")
print(f"  MEPT: {'Enabled' if USE_MEPT else 'Disabled'}")
print(f"  LEAP: {'Enabled' if USE_LEAP else 'Disabled'}")
print(f"  PRISM: {'Enabled' if USE_PRISM else 'Disabled'}")


def custom_collate_fn(batch):
    """Custom collate function to handle different grid sizes and data formats"""
    inputs = []
    outputs = []
    
    for item in batch:
        inp = item['inputs']
        out = item['outputs']
        
        if inp.dim() == 3:
            inp = inp.squeeze(0)
        if out.dim() == 3:
            out = out.squeeze(0)
        
        if inp.dim() != 2 or out.dim() != 2:
            print(f"Warning: Unexpected tensor dimensions - input: {inp.shape}, output: {out.shape}")
            continue
        
        inputs.append(inp)
        outputs.append(out)
    
    if not inputs:
        return {
            'inputs': torch.zeros(1, MAX_GRID_SIZE, MAX_GRID_SIZE, dtype=torch.long),
            'outputs': torch.zeros(1, MAX_GRID_SIZE, MAX_GRID_SIZE, dtype=torch.long)
        }
    
    max_h = max(inp.shape[0] for inp in inputs)
    max_w = max(inp.shape[1] for inp in inputs)
    max_h = max(max_h, max(out.shape[0] for out in outputs))
    max_w = max(max_w, max(out.shape[1] for out in outputs))
    
    max_h = min(max_h, MAX_GRID_SIZE)
    max_w = min(max_w, MAX_GRID_SIZE)
    
    padded_inputs = []
    padded_outputs = []
    
    for inp, out in zip(inputs, outputs):
        h_in, w_in = inp.shape
        pad_h = max_h - h_in
        pad_w = max_w - w_in
        padded_input = F.pad(inp, (0, pad_w, 0, pad_h), value=0)
        
        h_out, w_out = out.shape
        pad_h = max_h - h_out
        pad_w = max_w - w_out
        padded_output = F.pad(out, (0, pad_w, 0, pad_h), value=0)
        
        padded_inputs.append(padded_input)
        padded_outputs.append(padded_output)
    
    return {
        'inputs': torch.stack(padded_inputs),
        'outputs': torch.stack(padded_outputs)
    }


def train_iris():
    """Train IRIS with V4 enhancements including MEPT, LEAP, and PRISM"""
    print("\n👁️ Training IRIS - V4 Enhanced with MEPT, LEAP, and PRISM")
    print("="*60)
    
    # Initialize model
    model = EnhancedIrisNet(max_grid_size=MAX_GRID_SIZE).to(device)
    
    # Initialize MEPT components if enabled
    if USE_MEPT:
        print("\n🧠 Initializing MEPT (Memory-Enhanced Progressive Training)")
        replay_buffer = ExperienceReplayBuffer(capacity=100000)
        pattern_bank = PatternBank(max_patterns=20000)
        loss_fn = MEPTLoss(replay_buffer, pattern_bank, use_mept=True,
                          transformation_penalty=TRANSFORMATION_PENALTY,
                          exact_match_bonus=EXACT_MATCH_BONUS)
        print(f"✅ MEPT initialized with:")
        print(f"   Replay buffer capacity: {replay_buffer.capacity}")
        print(f"   Pattern bank capacity: {pattern_bank.max_patterns}")
        print(f"   Transformation penalty: {TRANSFORMATION_PENALTY}")
        print(f"   Exact match bonus: {EXACT_MATCH_BONUS}")
    else:
        # Fallback to regular loss
        replay_buffer = ExperienceReplayBuffer(capacity=1)
        pattern_bank = PatternBank(max_patterns=1)
        loss_fn = MEPTLoss(replay_buffer, pattern_bank, use_mept=False,
                          transformation_penalty=TRANSFORMATION_PENALTY,
                          exact_match_bonus=EXACT_MATCH_BONUS)
    
    # Initialize LEAP components if enabled
    if USE_LEAP:
        print("\n🎯 Initializing LEAP (Learning Enhancement through Adaptive Patterns)")
        leap_trainer = LEAPTrainer(device=device)
        print("✅ LEAP initialized for Stage 0 adaptive pattern training")
    
    # Initialize LEAP-PRISM bridge
    leap_prism_bridge = None
    if USE_LEAP and USE_PRISM and LEAP_PRISM_BRIDGE_AVAILABLE:
        print("\n🌉 Initializing LEAP-PRISM Bridge")
        leap_prism_bridge = None  # Will be initialized after synthesizer
    
    # Initialize program synthesizer
    if USE_PRISM:
        print("\n🔮 Initializing PRISM (Program Reasoning through Inductive Synthesis)")
        prism_system = create_prism_system()
        synthesizer = prism_system['synthesizer']
        print("✅ PRISM initialized with meta-programming and constraint solving")
    else:
        synthesizer = LightweightProgramSynthesizer()
    
    # Create LEAP-PRISM bridge now
    if USE_LEAP and USE_PRISM and LEAP_PRISM_BRIDGE_AVAILABLE:
        leap_prism_bridge = create_leap_prism_bridge(leap_trainer, synthesizer)
        print("✅ LEAP-PRISM Bridge created to enhance pattern learning")
    
    synthesis_stats = {
        'total_attempts': 0,
        'successful_syntheses': 0,
        'exact_improvements': 0,
        'prism_meta_programs': defaultdict(int)
    }
    
    # Initialize reporter
    reporter = TrainingReporter('iris')
    
    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )
    
    # Stage-specific learning rate scheduling instead of global decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_STAGE)
    
    scaler = GradScaler('cuda')
    
    best_exact = 0
    best_val_loss = float('inf')
    global_epoch = 0
    patience_counter = 0
    max_patience = 20
    
    # Check for existing checkpoint
    os.makedirs('/content/AutomataNexus_Olympus_AGI2/arc_models_v4', exist_ok=True)
    checkpoint_path = f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/iris_checkpoint.pt'
    
    resume_stage = 0
    resume_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"📂 Found checkpoint, loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_epoch = checkpoint.get('global_epoch', 0)
        resume_stage = checkpoint.get('stage', 0)
        best_exact = checkpoint.get('best_exact', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ Resumed from epoch {global_epoch} (Stage {resume_stage})")
    
    # EXACT MATCH PRE-TRAINING for Stage 0
    if EXACT_BOOST_AVAILABLE and resume_stage == 0 and global_epoch == 0:
        print(f"\n🎯 Running EXACT MATCH INJECTION for IRIS")
        model = inject_exact_match_training(model, device=device, num_epochs=50, target_accuracy=99.0)
        print("✅ Exact match injection complete!")
    
    # CURRICULUM LOOP
    for stage in range(resume_stage, CURRICULUM_STAGES):
        print(f"\n📚 Starting Curriculum Stage {stage}")
        print("="*40)
        
        # Reset learning rate for each new stage to prevent decay issues
        if stage > resume_stage:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
            # Reset scheduler for the new stage
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_STAGE)
        
        # Create dataset for this stage
        dataset = CurriculumMegaScaleDataset(
            DATA_DIR, 
            curriculum_stage=stage,
            use_arc_synthesis=True,
            synthesis_ratio=0.4 if stage == 0 else 0.3
        )
        
        # Limit Stage 1+ dataset size to prevent hanging
        if stage > 0 and len(dataset) > 10000:
            print(f"⚠️ Stage {stage}: Limiting dataset from {len(dataset):,} to 10,000 samples")
            dataset = torch.utils.data.Subset(dataset, list(range(10000)))
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Initialize augmenter
        arc_augmenter = ARCDataAugmenter(device=device)
        
        # Apply MEPT augmentation if enabled
        if USE_MEPT and replay_buffer.get_stats()['total_experiences'] > 100 and stage == 0:
            # Only apply MEPT augmentation for Stage 0 to prevent Stage 1+ hanging
            print(f"🔄 Applying MEPT augmentation with replay buffer...")
            train_dataset = MEPTAugmentedDataset(
                train_dataset,
                replay_buffer,
                replay_ratio=0.3 if stage == 0 else 0.2
            )
        elif stage > 0 and USE_MEPT:
        
        # Create data loaders with adaptive configuration
        # Use stage-adaptive configuration to prevent hanging
        stage_workers = NUM_WORKERS if stage == 0 else 0
        stage_batch_size = BATCH_SIZE if stage == 0 else BATCH_SIZE // 2
        
        train_loader_kwargs = {
            'dataset': train_dataset,
            'batch_size': stage_batch_size,
            'shuffle': True,
            'num_workers': stage_workers,
            'pin_memory': PIN_MEMORY if stage_workers > 0 else False,
            'persistent_workers': stage_workers > 0,
            'collate_fn': custom_collate_fn,
            'drop_last': True if stage > 0 else False  # Add drop_last for Stage 1+
        }
        
        val_loader_kwargs = {
            'dataset': val_dataset,
            'batch_size': stage_batch_size,
            'shuffle': False,
            'num_workers': stage_workers,
            'pin_memory': PIN_MEMORY if stage_workers > 0 else False,
            'persistent_workers': stage_workers > 0,
            'collate_fn': custom_collate_fn
        }
        
        # Add prefetch_factor only if workers > 0
        if stage_workers > 0 and PREFETCH_FACTOR is not None:
            train_loader_kwargs['prefetch_factor'] = PREFETCH_FACTOR
            val_loader_kwargs['prefetch_factor'] = PREFETCH_FACTOR
        
        # Force simple DataLoader for Stage 1+ to prevent hanging
        if stage > 0:
            train_loader_kwargs.pop('prefetch_factor', None)
            val_loader_kwargs.pop('prefetch_factor', None)
        
        train_loader = DataLoader(**train_loader_kwargs)
        val_loader = DataLoader(**val_loader_kwargs)
        
        print(f"Stage {stage} - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        print(f"  Workers: {stage_workers}, Batch size: {stage_batch_size}, Drop last: {stage > 0}")
        if stage > 0:
            print(f"  ⚡ Stage 1+ optimizations active to prevent hanging")
        
        # Train for this stage
        start_epoch = resume_epoch if stage == resume_stage else 0
        for epoch in range(start_epoch, EPOCHS_PER_STAGE):
            if not (stage == resume_stage and epoch == start_epoch and global_epoch > 0):
                global_epoch += 1
            
            # Training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
            
            # Create exact match dataset for Stage 0
            if stage == 0 and EXACT_BOOST_AVAILABLE:
                exact_dataset = ExactMatchBoostDataset(1000, fixed_size=6)
                aggressive_loss = AggressiveLoss()
            
            pbar = tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch+1}/{EPOCHS_PER_STAGE}")
            optimizer.zero_grad()
            
            # LEAP integration
            leap_batch_counter = 0
            
            for batch_idx, batch in enumerate(pbar):
                # Process batch similar to V4 training...
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
                    losses = loss_fn(pred_output, output_grids, input_grids)
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
                
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.3f}",
                    'exact': f"{losses['exact_count'].item():.0f}",
                    'trans': f"{losses.get('transformation', torch.tensor(0)).item():.2f}"
                })
                
                # LEAP training for Stage 0
                if USE_LEAP and stage == 0 and batch_idx % 3 == 0:
                    leap_batch = leap_trainer.generate_leap_batch(batch_size=64)
                    leap_inputs = leap_batch['inputs'].to(device)
                    leap_outputs = leap_batch['outputs'].to(device)
                    pattern_types = leap_batch['pattern_types']
                    
                    # Pad to MAX_GRID_SIZE
                    if leap_inputs.shape[1] < MAX_GRID_SIZE or leap_inputs.shape[2] < MAX_GRID_SIZE:
                        pad_h = MAX_GRID_SIZE - leap_inputs.shape[1]
                        pad_w = MAX_GRID_SIZE - leap_inputs.shape[2]
                        leap_inputs = F.pad(leap_inputs, (0, pad_w, 0, pad_h), value=0)
                        leap_outputs = F.pad(leap_outputs, (0, pad_w, 0, pad_h), value=0)
                    
                    # Convert to one-hot
                    leap_input_oh = F.one_hot(leap_inputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                    leap_output_oh = F.one_hot(leap_outputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                    
                    with autocast('cuda'):
                        leap_pred = model(leap_input_oh, leap_output_oh, mode='train')['predicted_output']
                        leap_losses = loss_fn(leap_pred, leap_output_oh, leap_input_oh)
                        leap_loss = leap_losses['total'] / GRADIENT_ACCUMULATION_STEPS
                    
                    scaler.scale(leap_loss).backward()
                    
                    # Update pattern statistics
                    leap_trainer.update_pattern_stats(pattern_types, leap_pred, leap_output_oh)
                    leap_batch_counter += 1
                    
                    # Analyze failed patterns with LEAP-PRISM bridge
                    if leap_prism_bridge and leap_batch_counter % 10 == 0:
                        pred_indices = leap_pred.argmax(dim=1)
                        target_indices = leap_output_oh.argmax(dim=1)
                        
                        for i, pattern_type in enumerate(pattern_types):
                            if not (pred_indices[i] == target_indices[i]).all():
                                analysis = leap_prism_bridge.analyze_failed_leap_pattern(
                                    pattern_type,
                                    leap_inputs[i].cpu().numpy(),
                                    leap_outputs[i].cpu().numpy(),
                                    pred_indices[i].cpu().numpy()
                                )
                                
                                if leap_batch_counter % 50 == 0 and analysis['synthesis_hint']:
                                    print(f"\n💡 LEAP-PRISM insight for {pattern_type}: Use {analysis['synthesis_hint']}")
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                val_metrics = {'loss': 0, 'exact': 0, 'pixel_acc': 0, 'samples': 0}
                synthesis_metrics = {'attempts': 0, 'successes': 0, 'exact_via_synthesis': 0}
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        inputs = batch['inputs'].to(device, non_blocking=True)
                        outputs = batch['outputs'].to(device, non_blocking=True)
                        
                        inputs = torch.clamp(inputs, 0, 9)
                        outputs = torch.clamp(outputs, 0, 9)
                        
                        input_grids = F.one_hot(inputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        output_grids = F.one_hot(outputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        
                        with autocast('cuda'):
                            model_outputs = model(input_grids)
                            pred_output = model_outputs['predicted_output']
                            losses = loss_fn(pred_output, output_grids, input_grids)
                        
                        pred_indices = pred_output.argmax(dim=1)
                        target_indices = output_grids.argmax(dim=1)
                        
                        exact = (pred_indices == target_indices).all(dim=[1,2]).sum().item()
                        pixel_acc = (pred_indices == target_indices).float().mean().item()
                        
                        val_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                        val_metrics['exact'] += exact
                        val_metrics['pixel_acc'] += pixel_acc * input_grids.size(0)
                        val_metrics['samples'] += input_grids.size(0)
                        
                        # Try program synthesis on a subset
                        if synthesis_metrics['attempts'] < 50:
                            for i in range(min(5, input_grids.size(0))):
                                synthesis_metrics['attempts'] += 1
                                input_np = inputs[i].cpu().numpy()
                                output_np = outputs[i].cpu().numpy()
                                
                                if USE_PRISM and hasattr(synthesizer, 'synthesize'):
                                    program = synthesizer.synthesize(input_np, output_np, time_limit=2.0)
                                    if program:
                                        synthesis_metrics['successes'] += 1
                                        if hasattr(program, 'meta_program'):
                                            synthesis_stats['prism_meta_programs'][str(program.meta_program)] += 1
                                        if not (pred_indices[i] == target_indices[i]).all():
                                            synthesis_metrics['exact_via_synthesis'] += 1
                
                # Calculate averages
                train_loss = train_metrics['loss'] / train_metrics['samples']
                train_exact_pct = train_metrics['exact'] / train_metrics['samples'] * 100
                val_loss = val_metrics['loss'] / val_metrics['samples']
                val_exact_pct = val_metrics['exact'] / val_metrics['samples'] * 100
                val_pixel_acc = val_metrics['pixel_acc'] / val_metrics['samples'] * 100
                
                print(f"\nGlobal Epoch {global_epoch} (Stage {stage}): "
                      f"Train Loss: {train_loss:.4f}, Train Exact: {train_exact_pct:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Exact: {val_exact_pct:.2f}%, Pixel: {val_pixel_acc:.2f}%")
                
                # Report MEPT status
                if USE_MEPT:
                    buffer_stats = replay_buffer.get_stats()
                    print(f"📊 MEPT Status:")
                    print(f"   Experience Buffer: {buffer_stats['total_experiences']:,} samples "
                          f"({buffer_stats['exact_matches']:,} exact matches)")
                    print(f"   Pattern Bank: {len(pattern_bank.patterns):,} unique patterns")
                    print(f"   Loss weights: {loss_fn.weights}")
                
                # Report LEAP status
                if USE_LEAP and stage == 0:
                    leap_report = leap_trainer.get_performance_report()
                    if leap_report:
                        print(leap_report)
                
                # Report synthesis results
                if synthesis_metrics['attempts'] > 0:
                    synthesis_success_rate = synthesis_metrics['successes'] / synthesis_metrics['attempts'] * 100
                    print(f"🔧 Program Synthesis: {synthesis_success_rate:.1f}% success rate, "
                          f"{synthesis_metrics['exact_via_synthesis']} additional exact matches")
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'val_loss': val_loss
                    }, f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/iris_best.pt')
                    print(f"✅ New best model! Exact: {val_exact_pct:.2f}%")
                
                # Save checkpoint
                torch.save({
                    'epoch': global_epoch,
                    'global_epoch': global_epoch,
                    'stage': stage,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_exact': val_exact_pct,
                    'best_exact': best_exact,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience and stage > 0:
                        print(f"⚠️ Early stopping triggered!")
                        break
            
            # Step scheduler ONCE per epoch (not per batch!)
            scheduler.step()
        
        if patience_counter >= max_patience and stage > 0:
            break
        
        # Clear resume_epoch after first stage
        if stage == resume_stage:
            resume_epoch = 0
    
    # Generate report
    print(f"\n📊 Generating training report for IRIS...")
    reporter.generate_report({
        'best_exact': best_exact,
        'best_val_loss': best_val_loss,
        'total_epochs': global_epoch
    })
    
    # Report synthesis statistics
    if synthesis_stats['total_attempts'] > 0:
        overall_success_rate = synthesis_stats['successful_syntheses'] / synthesis_stats['total_attempts'] * 100
        print(f"\n📊 Program Synthesis Summary:")
        print(f"  Total synthesis attempts: {synthesis_stats['total_attempts']}")
        print(f"  Successful syntheses: {synthesis_stats['successful_syntheses']} ({overall_success_rate:.1f}%)")
        print(f"  Additional exact matches via synthesis: {synthesis_stats['exact_improvements']}")
        
        if USE_PRISM and synthesis_stats['prism_meta_programs']:
            print(f"\n🔮 PRISM Meta-Program Usage:")
            for meta_program, count in sorted(synthesis_stats['prism_meta_programs'].items(), 
                                             key=lambda x: x[1], reverse=True):
                print(f"    {meta_program}: {count} times")
    
    print("\n✅ IRIS training complete!")
    
    # Clear memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    print("="*80)
    print("IRIS V4 Enhanced Training Script")
    print("With MEPT, LEAP, and PRISM Integration")
    print("="*80)
    
    train_iris()