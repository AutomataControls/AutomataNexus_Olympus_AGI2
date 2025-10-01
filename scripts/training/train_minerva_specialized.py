"""
MINERVA Specialized Training Script - Grid Reasoning & Strategic Analysis
Integrates ALL AutomataNexus novel training methods for MINERVA's unique architecture
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
import time
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

# MINERVA-specific injection modules for ARC competition
def minerva_exact_match_injection(model, device, num_epochs=100, target_accuracy=90.0):
    """MINERVA-specific exact match injection using MINERVA_CONFIG - EXTENDED FOR ARC COMPETITION"""
    print("🎯 MINERVA EXACT MATCH INJECTION - EXTENDED")
    print("=" * 50)
    print(f"  Batch size: {MINERVA_CONFIG['batch_size']}")
    print(f"  Learning rate: {MINERVA_CONFIG['learning_rate']*5} -> {MINERVA_CONFIG['learning_rate']*15} (with warmup)")
    print(f"  Transform penalty: {MINERVA_CONFIG['transform_penalty']}")
    print(f"  Exact match bonus: {MINERVA_CONFIG['exact_match_bonus']}")
    print(f"  Epochs: {num_epochs} (EXTENDED)")
    
    # AGGRESSIVE exact match training for MINERVA
    model.train()
    # DISABLE DROPOUT for exact match training
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.p = 0.0  # Disable all dropout
    
    # Start with lower LR for warmup
    base_lr = MINERVA_CONFIG['learning_rate'] * 2  # Reduced multiplier
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.0)  # No weight decay for exact match
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=base_lr*0.001)
    
    # Create MORE DIVERSE exact match patterns
    patterns = []
    
    # 1. Very simple 2x2 patterns first (EASIEST)
    for i in range(100):
        size = 2
        color = (i % 9) + 1
        pattern = torch.full((size, size), color, dtype=torch.long)
        patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # 2. Then 3x3 patterns
    for i in range(100):
        size = 3
        color = (i % 9) + 1
        pattern = torch.full((size, size), color, dtype=torch.long)
        patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # 3. Mixed size patterns
    for i in range(100):
        size = random.choice([4, 5, 6])
        color = (i % 9) + 1
        pattern = torch.full((size, size), color, dtype=torch.long)
        patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # 2. Simple geometric patterns (more diverse)
    for i in range(100):  # Increased
        size = random.choice([5, 6, 7])
        pattern = torch.zeros((size, size), dtype=torch.long)
        
        if i % 4 == 0:
            # Cross pattern
            pattern[size//2, :] = 1
            pattern[:, size//2] = 1
        elif i % 4 == 1:
            # Diagonal
            torch.diagonal(pattern).fill_(2)
        elif i % 4 == 2:
            # Checkerboard
            pattern[::2, ::2] = 3
            pattern[1::2, 1::2] = 3
        else:
            # Frame
            pattern[0, :] = 4
            pattern[-1, :] = 4
            pattern[:, 0] = 4
            pattern[:, -1] = 4
        patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # 3. Two-color patterns (more variations)
    for i in range(100):  # Increased
        size = random.choice([5, 6])
        pattern = torch.zeros((size, size), dtype=torch.long)
        
        if i % 3 == 0:
            pattern[:size//2, :] = 1
            pattern[size//2:, :] = 2
        elif i % 3 == 1:
            pattern[:, :size//2] = 3
            pattern[:, size//2:] = 4
        else:
            # Quadrants
            pattern[:size//2, :size//2] = 5
            pattern[size//2:, size//2:] = 6
        patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # 4. Simple stripes (new)
    for i in range(50):
        size = 6
        pattern = torch.zeros((size, size), dtype=torch.long)
        if i % 2 == 0:
            # Horizontal stripes
            for row in range(size):
                pattern[row, :] = (row % 3) + 1
        else:
            # Vertical stripes
            for col in range(size):
                pattern[:, col] = (col % 3) + 1
        patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # Shuffle patterns for better training
    random.shuffle(patterns)
    
    # Training loop with warmup and aggressive optimization
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0
        
        # Warmup phase with increasing LR
        if epoch < 10:
            new_lr = base_lr + (MINERVA_CONFIG['learning_rate'] * 15 - base_lr) * (epoch / 10)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # Shuffle patterns each epoch
        random.shuffle(patterns)
        
        for batch_start in range(0, len(patterns), MINERVA_CONFIG['batch_size']):
            batch_patterns = patterns[batch_start:batch_start + MINERVA_CONFIG['batch_size']]
            
            # Find max size in batch before stacking
            max_h = max(p['inputs'].shape[0] for p in batch_patterns)
            max_w = max(p['inputs'].shape[1] for p in batch_patterns)
            
            # Pad all patterns to max size
            padded_inputs = []
            padded_outputs = []
            for p in batch_patterns:
                inp = p['inputs']
                out = p['outputs']
                h, w = inp.shape
                pad_h = max_h - h
                pad_w = max_w - w
                if pad_h > 0 or pad_w > 0:
                    padded_inp = F.pad(inp, (0, pad_w, 0, pad_h), value=0)
                    padded_out = F.pad(out, (0, pad_w, 0, pad_h), value=0)
                else:
                    padded_inp = inp
                    padded_out = out
                padded_inputs.append(padded_inp)
                padded_outputs.append(padded_out)
            
            inputs = torch.stack(padded_inputs).to(device)
            outputs = torch.stack(padded_outputs).to(device)
            
            input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            output_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            optimizer.zero_grad()
            # Pass mode='exact_match' to potentially help model adapt
            pred = model(input_oh, output_oh, mode='train')['predicted_output']
            
            # AGGRESSIVE loss for exact matching
            loss = F.cross_entropy(pred, outputs, reduction='none')
            
            # Add exact match bonus (progressive)
            pred_idx = pred.argmax(dim=1)
            exact_matches = (pred_idx == outputs).all(dim=[1,2]).float()
            
            # Per-sample loss weighting - focus on mistakes
            # Reduce loss for correct predictions to encourage learning
            loss_weights = torch.ones_like(loss)
            loss_weights[exact_matches.bool()] = 0.1  # Much lower weight for correct predictions
            
            # Apply weights
            weighted_loss = (loss * loss_weights).mean()
            
            # Simple exact match reward
            exact_reward = -exact_matches.mean() * min(2.0, 0.5 + epoch / 50)
            
            # Total loss
            total_loss = weighted_loss + exact_reward
            
            # Ensure minimum loss
            total_loss = torch.clamp(total_loss, min=0.01)
            
            # Add L2 regularization for stability
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            total_loss = total_loss + 0.0001 * l2_reg
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            correct += exact_matches.sum().item()
            total += outputs.size(0)
            epoch_loss += total_loss.item()
        
        acc = correct / total * 100
        avg_loss = epoch_loss / (len(patterns) // MINERVA_CONFIG['batch_size'] + 1)
        
        if epoch >= 10:  # After warmup
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}: {acc:.1f}% exact match | Loss: {avg_loss:.3f} | LR: {current_lr:.5f}")
        
        # Track improvement
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if acc >= target_accuracy:
            print(f"🏆 TARGET REACHED: {acc:.1f}% >= {target_accuracy}%")
            break
        elif epoch == num_epochs - 1:
            print(f"⚠️ INJECTION COMPLETE: {acc:.1f}% (best: {best_acc:.1f}%, target: {target_accuracy}%)")
    
    return model


def minerva_mept_injection(model, device, systems, num_epochs=100, target_accuracy=90.0):
    """MINERVA MEPT (Memory-Enhanced Pattern Training) injection"""
    print("🧠 MINERVA MEPT INJECTION")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% pattern recall")
    print(f"  Epochs: {num_epochs}")
    
    if 'replay_buffer' not in systems or 'pattern_bank' not in systems:
        print("⚠️ MEPT systems not available, skipping")
        return model
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=MINERVA_CONFIG['learning_rate']*5, momentum=0.9)
    
    # Generate grid reasoning patterns for MEPT
    patterns = []
    for i in range(100):
        size = 6
        # Grid patterns that MINERVA excels at
        pattern = torch.zeros((size, size), dtype=torch.long)
        
        if i % 4 == 0:  # Checkerboard
            pattern[::2, ::2] = 1
            pattern[1::2, 1::2] = 1
        elif i % 4 == 1:  # Stripes
            pattern[::2, :] = 2
        elif i % 4 == 2:  # Quadrants
            pattern[:size//2, :size//2] = 3
            pattern[size//2:, size//2:] = 3
        else:  # Frame
            pattern[0, :] = 4
            pattern[-1, :] = 4
            pattern[:, 0] = 4
            pattern[:, -1] = 4
            
        patterns.append({'inputs': pattern, 'outputs': pattern})
    
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        
        for batch_start in range(0, len(patterns), 32):
            batch = patterns[batch_start:batch_start + 32]
            inputs = torch.stack([p['inputs'] for p in batch]).to(device)
            outputs = torch.stack([p['outputs'] for p in batch]).to(device)
            
            input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            output_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            optimizer.zero_grad()
            pred = model(input_oh, output_oh, mode='train')['predicted_output']
            loss = F.cross_entropy(pred, outputs)
            loss.backward()
            optimizer.step()
            
            pred_idx = pred.argmax(dim=1)
            exact = (pred_idx == outputs).all(dim=[1,2])
            correct += exact.sum().item()
            total += len(batch)
            
            # Store successful patterns
            for i, is_exact in enumerate(exact):
                if is_exact:
                    systems['replay_buffer'].add(
                        input_oh[i], output_oh[i], pred_idx[i],
                        loss.item(), is_exact=True
                    )
        
        acc = correct / total * 100
        print(f"MEPT Epoch {epoch+1}/{num_epochs}: {acc:.1f}% recall")
        if acc >= target_accuracy:
            print(f"🏆 MEPT TARGET REACHED: {acc:.1f}%")
            break
    
    return model


def minerva_leap_injection(model, device, systems, num_epochs=100, target_accuracy=90.0):
    """MINERVA LEAP (Learning Enhancement through Adaptive Patterns) injection"""
    print("🎯 MINERVA LEAP INJECTION")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% pattern mastery")
    print(f"  Epochs: {num_epochs}")
    
    if 'leap_trainer' not in systems:
        print("⚠️ LEAP system not available, skipping")
        return model
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=MINERVA_CONFIG['learning_rate']*3, momentum=0.9)
    best_acc = 0
    
    for epoch in range(num_epochs):
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0
        batches_per_epoch = 10  # Generate multiple batches per epoch
        
        for batch_idx in range(batches_per_epoch):
            # Generate LEAP patterns with basic complexity for easier learning
            leap_batch = systems['leap_trainer'].generate_leap_batch(
                batch_size=32,  # Smaller batch size
                stage=0,
                grid_size=6,
                complexity='basic'  # Start with basic patterns
            )
            
            inputs = leap_batch['inputs'].to(device)
            outputs = leap_batch['outputs'].to(device)
            
            # Ensure proper tensor dimensions
            if inputs.dim() == 4:
                inputs = inputs.squeeze(1)
            if outputs.dim() == 4:
                outputs = outputs.squeeze(1)
            
            input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            output_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            optimizer.zero_grad()
            pred = model(input_oh, output_oh, mode='train')['predicted_output']
            loss = F.cross_entropy(pred, outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred_idx = pred.argmax(dim=1)
            exact = (pred_idx == outputs).all(dim=[1,2])
            batch_correct = exact.sum().item()
            
            epoch_correct += batch_correct
            epoch_total += outputs.size(0)
            epoch_loss += loss.item()
            
            # Update LEAP statistics
            systems['leap_trainer'].update_pattern_stats(
                leap_batch['pattern_types'], pred, output_oh
            )
        
        acc = epoch_correct / epoch_total * 100
        avg_loss = epoch_loss / batches_per_epoch
        
        if acc > best_acc:
            best_acc = acc
        
        print(f"LEAP Epoch {epoch+1}/{num_epochs}: {acc:.1f}% mastery | Loss: {avg_loss:.3f} | Best: {best_acc:.1f}%")
        
        if acc >= target_accuracy:
            print(f"🏆 LEAP TARGET REACHED: {acc:.1f}%")
            break
        elif epoch == num_epochs - 1:
            print(f"⚠️ LEAP COMPLETE: {acc:.1f}% (best: {best_acc:.1f}%, target: {target_accuracy}%)")
    
    return model


def minerva_prism_injection(model, device, systems, num_epochs=100, target_accuracy=90.0):
    """MINERVA PRISM (Program Reasoning through Inductive Synthesis) injection"""
    print("🔍 MINERVA PRISM INJECTION")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% program synthesis")
    print(f"  Epochs: {num_epochs}")
    
    if 'prism_synthesizer' not in systems:
        print("⚠️ PRISM system not available, skipping")
        return model
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=MINERVA_CONFIG['learning_rate']*2, momentum=0.9)
    best_acc = 0
    
    # Generate MORE program-based patterns for better learning
    patterns = []
    for i in range(300):  # Increased from 50 to 300
        size = random.choice([4, 5, 6])  # Varied sizes
        # Create more diverse base patterns
        if i % 4 == 0:
            base = torch.randint(0, 4, (size, size))
        elif i % 4 == 1:
            base = torch.zeros((size, size), dtype=torch.long)
            base[::2, ::2] = 1
            base[1::2, 1::2] = 2
        elif i % 4 == 2:
            base = torch.arange(size*size).reshape(size, size) % 3
        else:
            base = torch.randint(0, 3, (size, size))
        
        # Apply simple transformations
        if i % 5 == 0:
            output = torch.rot90(base, 1)  # Rotate 90
        elif i % 5 == 1:
            output = torch.flip(base, [0])  # Flip vertical
        elif i % 5 == 2:
            output = torch.flip(base, [1])  # Flip horizontal
        elif i % 5 == 3:
            output = base.T  # Transpose
        else:
            output = base + 1  # Simple increment
            
        patterns.append({'inputs': base, 'outputs': output})
    
    for epoch in range(num_epochs):
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0
        
        # Shuffle patterns each epoch
        random.shuffle(patterns)
        
        for batch_start in range(0, len(patterns), 32):  # Batch size 32
            batch = patterns[batch_start:batch_start + 32]
            
            # Handle variable sizes by padding
            max_size = max(p['inputs'].shape[0] for p in batch)
            padded_inputs = []
            padded_outputs = []
            
            for p in batch:
                inp = p['inputs']
                out = p['outputs']
                if inp.shape[0] < max_size:
                    pad = max_size - inp.shape[0]
                    inp = F.pad(inp, (0, pad, 0, pad), value=0)
                    out = F.pad(out, (0, pad, 0, pad), value=0)
                padded_inputs.append(inp)
                padded_outputs.append(out)
            
            inputs = torch.stack(padded_inputs).to(device)
            outputs = torch.stack(padded_outputs).to(device)
            
            input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            output_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            optimizer.zero_grad()
            pred = model(input_oh, output_oh, mode='train')['predicted_output']
            loss = F.cross_entropy(pred, outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred_idx = pred.argmax(dim=1)
            exact = (pred_idx == outputs).all(dim=[1,2])
            batch_correct = exact.sum().item()
            
            epoch_correct += batch_correct
            epoch_total += len(batch)
            epoch_loss += loss.item()
        
        acc = epoch_correct / epoch_total * 100
        avg_loss = epoch_loss / (len(patterns) // 32 + 1)
        
        if acc > best_acc:
            best_acc = acc
        
        print(f"PRISM Epoch {epoch+1}/{num_epochs}: {acc:.1f}% synthesis | Loss: {avg_loss:.3f} | Best: {best_acc:.1f}%")
        
        if acc >= target_accuracy:
            print(f"🏆 PRISM TARGET REACHED: {acc:.1f}%")
            break
        elif epoch == num_epochs - 1:
            print(f"⚠️ PRISM COMPLETE: {acc:.1f}% (best: {best_acc:.1f}%, target: {target_accuracy}%)")
    
    return model

EXACT_BOOST_AVAILABLE = True

# Data systems
from src.data.arc_data_synthesis import ARCDataSynthesizer, ARCDataAugmenter

# MINERVA-Specific Configuration with 8-Stage Progressive Curriculum - OPTIMIZED V2
MINERVA_CONFIG = {
    'batch_size': 64,  # Further reduced to prevent hanging
    'learning_rate': 0.003,  # Base learning rate
    'num_epochs': 400,  # 8 stages x 50 epochs - longer training
    'hidden_dim': 256,
    'pattern_memory_size': 200,
    'gradient_accumulation': 2,  # Effective batch: 384
    'transform_penalty': 0.5,  # Positive value to discourage identity copying
    'exact_match_bonus': 2.0,  # Moderate bonus to avoid negative losses
    'curriculum_stages': 8,  # Progressive 8-stage curriculum
    'epochs_per_stage': 50,  # Extended for better convergence
    'attention_heads': 8,
    'relational_weight': 0.05,  # Slightly increased for grid reasoning
    'pattern_memory_weight': 0.01,  # Re-enabled with minimal weight
    'adaptive_lr': True,  # Enable adaptive learning rate per stage
    'warmup_epochs': 10,  # Warmup for larger grids
    'grid_size_penalty': 0.1  # Penalty scaling with grid complexity
}

# 8-Stage Progressive Grid Size Curriculum - OPTIMIZED FOR PERFORMANCE SCALING
STAGE_CONFIG = {
    0: {'max_grid_size': 6,  'synthesis_ratio': 0.7, 'exact_injection': True,  'leap_complexity': 'basic', 'lr_mult': 1.0},
    1: {'max_grid_size': 7,  'synthesis_ratio': 0.6, 'exact_injection': True,  'leap_complexity': 'basic', 'lr_mult': 0.8},
    2: {'max_grid_size': 9,  'synthesis_ratio': 0.6, 'exact_injection': False, 'leap_complexity': 'simple', 'lr_mult': 0.7},
    3: {'max_grid_size': 12, 'synthesis_ratio': 0.5, 'exact_injection': False, 'leap_complexity': 'simple', 'lr_mult': 0.6},
    4: {'max_grid_size': 15, 'synthesis_ratio': 0.5, 'exact_injection': False, 'leap_complexity': 'medium', 'lr_mult': 0.5},
    5: {'max_grid_size': 19, 'synthesis_ratio': 0.4, 'exact_injection': False, 'leap_complexity': 'medium', 'lr_mult': 0.4},
    6: {'max_grid_size': 24, 'synthesis_ratio': 0.4, 'exact_injection': False, 'leap_complexity': 'complex', 'lr_mult': 0.3},
    7: {'max_grid_size': 30, 'synthesis_ratio': 0.3, 'exact_injection': False, 'leap_complexity': 'complex', 'lr_mult': 0.2}
}

# Training components flags
USE_MEPT = True and (MINERVA_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_LEAP = True and (MINERVA_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_PRISM = True and (MINERVA_PRISM_AVAILABLE or PRISM_AVAILABLE)
USE_EXACT_BOOST = True and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 MINERVA Training on {device}")


class MinervaSpecializedDataset(Dataset):
    """MINERVA-optimized dataset with grid attention focus"""
    def __init__(self, base_dataset, replay_buffer=None, replay_ratio=0.3):
        self.base_dataset = base_dataset
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # MEPT replay integration
        if (self.replay_buffer and random.random() < self.replay_ratio and 
            len(self.replay_buffer.buffer) > 0):
            experiences = self.replay_buffer.sample(1, strategy='mixed')  # Use MINERVA's mixed strategy
            if experiences:
                exp = experiences[0]
                input_tensor = exp['input']
                output_tensor = exp['output']
                
                # Handle tensor dimensions
                if input_tensor.dim() == 4:
                    input_tensor = input_tensor.argmax(dim=0)
                elif input_tensor.dim() == 3:
                    input_tensor = input_tensor.squeeze(0)
                    
                if output_tensor.dim() == 4:
                    output_tensor = output_tensor.argmax(dim=0)
                elif output_tensor.dim() == 3:
                    output_tensor = output_tensor.squeeze(0)
                
                return {
                    'inputs': input_tensor,
                    'outputs': output_tensor
                }
        
        return self.base_dataset[idx]


class MinervaSpecializedLoss(nn.Module):
    """MINERVA-specific loss - SIMPLIFIED FOR STABILITY"""
    def __init__(self):
        super().__init__()
        self.weights = {
            'reconstruction': 1.0,
            'transformation': MINERVA_CONFIG['transform_penalty'],
            'exact_match': MINERVA_CONFIG['exact_match_bonus']
        }
        
    def forward(self, pred_output, target_output, input_grid, model_outputs=None):
        """SIMPLIFIED MINERVA loss - based on working IRIS approach"""
        B, C, H, W = pred_output.shape
        
        # Simple focal loss like IRIS
        focal_loss = self._focal_loss(pred_output, target_output, gamma=2.0)
        
        # Exact match detection and bonus
        pred_indices = pred_output.argmax(dim=1)
        target_indices = target_output.argmax(dim=1)
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_count = exact_matches.sum()
        exact_bonus = -exact_matches.mean() * self.weights['exact_match']
        exact_bonus = exact_bonus.clamp(min=-2.0)  # Prevent excessive negative contribution
        
        # Simple transformation penalty
        input_indices = input_grid.argmax(dim=1)
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.weights['transformation']
        
        # SIMPLE total loss - only 3 components like IRIS
        total_loss = focal_loss + transform_penalty + exact_bonus
        
        # Simple stability check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ NaN/Inf loss, using focal only")
            total_loss = focal_loss.clamp(max=10.0)
        
        # Prevent extremely negative losses that indicate instability
        if total_loss < -5.0:
            print(f"⚠️ Loss too negative ({total_loss.item():.3f}), clamping")
            total_loss = total_loss.clamp(min=-5.0)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'exact_count': exact_count,
            'relational': torch.tensor(0.001, device=total_loss.device),  # Dummy for display
            'pattern_memory': torch.tensor(0.0, device=total_loss.device)  # Dummy for display
        }
    
    def _focal_loss(self, pred, target, gamma=1.5):
        """Focal loss for hard pixel classification"""
        target_idx = target.argmax(dim=1)
        ce_loss = F.cross_entropy(pred, target_idx, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** gamma * ce_loss
        return focal.mean()
    
    def _edge_aware_loss(self, pred, target):
        """Emphasize boundaries for object detection"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Compute edges using Sobel-like filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(pred.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        target_edges_x = F.conv2d(target_idx.float().unsqueeze(1), sobel_x, padding=1)
        target_edges_y = F.conv2d(target_idx.float().unsqueeze(1), sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        # Weight loss by edge strength
        edge_weight = 1.0 + 2.0 * torch.sigmoid(target_edges)
        pixel_loss = F.cross_entropy(pred, target_idx, reduction='none')
        
        return (pixel_loss * edge_weight.squeeze(1)).mean()
    
    def _color_balance_loss(self, pred, target):
        """Preserve color distribution"""
        pred_colors = F.softmax(pred, dim=1).sum(dim=[2, 3])
        target_colors = target.sum(dim=[2, 3])
        return F.mse_loss(pred_colors, target_colors)
    
    def _structure_loss(self, pred, target):
        """Preserve structural connectivity"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Use erosion/dilation to check structure preservation
        kernel = torch.ones(3, 3).to(pred.device)
        kernel = kernel.view(1, 1, 3, 3)
        
        # Simplified structure check using local consistency
        pred_local = F.conv2d(pred_idx.float().unsqueeze(1), kernel, padding=1)
        target_local = F.conv2d(target_idx.float().unsqueeze(1), kernel, padding=1)
        
        return F.mse_loss(pred_local, target_local)


def custom_collate_fn(batch, stage=0):
    """MINERVA-optimized collate function with stage-specific grid sizes"""
    inputs = []
    outputs = []
    target_size = STAGE_CONFIG[stage]['max_grid_size']
    
    for i, item in enumerate(batch):
        try:
            input_grid = item['inputs']
            output_grid = item['outputs']
            
            # Convert to tensor if needed
            if not isinstance(input_grid, torch.Tensor):
                input_grid = torch.tensor(input_grid, dtype=torch.long)
            if not isinstance(output_grid, torch.Tensor):
                output_grid = torch.tensor(output_grid, dtype=torch.long)
            
            # Force to 2D by squeezing all extra dimensions
            while input_grid.dim() > 2:
                input_grid = input_grid.squeeze(0)
            while output_grid.dim() > 2:
                output_grid = output_grid.squeeze(0)
            
            # Handle 1D tensors by reshaping
            if input_grid.dim() == 1:
                size = int(input_grid.numel() ** 0.5)
                if size * size == input_grid.numel():
                    input_grid = input_grid.view(size, size)
                else:
                    input_grid = input_grid.view(1, -1)
            
            if output_grid.dim() == 1:
                size = int(output_grid.numel() ** 0.5)
                if size * size == output_grid.numel():
                    output_grid = output_grid.view(size, size)
                else:
                    output_grid = output_grid.view(1, -1)
            
            # ALWAYS create new tensors of exact target size
            new_input = torch.zeros(target_size, target_size, dtype=torch.long)
            new_output = torch.zeros(target_size, target_size, dtype=torch.long)
            
            # Get actual dimensions
            input_h, input_w = input_grid.shape
            output_h, output_w = output_grid.shape
            
            # Copy what fits
            copy_h_in = min(input_h, target_size)
            copy_w_in = min(input_w, target_size)
            copy_h_out = min(output_h, target_size)  
            copy_w_out = min(output_w, target_size)
            
            new_input[:copy_h_in, :copy_w_in] = input_grid[:copy_h_in, :copy_w_in]
            new_output[:copy_h_out, :copy_w_out] = output_grid[:copy_h_out, :copy_w_out]
            
            inputs.append(new_input)
            outputs.append(new_output)
            
        except Exception as e:
            print(f"Error processing batch item {i}: {e}")
            print(f"Input shape: {input_grid.shape if 'input_grid' in locals() else 'unknown'}")
            print(f"Output shape: {output_grid.shape if 'output_grid' in locals() else 'unknown'}")
            # Create dummy tensors as fallback
            inputs.append(torch.zeros(target_size, target_size, dtype=torch.long))
            outputs.append(torch.zeros(target_size, target_size, dtype=torch.long))
    
    return {
        'inputs': torch.stack(inputs),
        'outputs': torch.stack(outputs)
    }


def train_minerva_specialized():
    """Main MINERVA specialized training function"""
    print("🧠 Starting MINERVA Specialized Training")
    print("=" * 60)
    
    # Initialize model with maximum grid size from final stage
    max_grid_size = STAGE_CONFIG[7]['max_grid_size']  # Final stage size (30x30)
    model = EnhancedMinervaNet(
        max_grid_size=max_grid_size,
        hidden_dim=MINERVA_CONFIG['hidden_dim']
    ).to(device)
    
    print(f"📊 MINERVA Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
    systems = {}
    
    # MEPT System - Use MINERVA-specific if available
    if USE_MEPT:
        if MINERVA_MEPT_LEAP_AVAILABLE:
            mept_components = create_minerva_mept_system(
                capacity=50000,  # Integer capacity for replay buffer
                pattern_bank_size=10000,
                transformation_penalty=MINERVA_CONFIG['transform_penalty'],
                exact_match_bonus=MINERVA_CONFIG['exact_match_bonus']
            )
            systems['replay_buffer'] = mept_components['replay_buffer']
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components['loss_fn']  # MINERVA-specific loss
            print("✅ MINERVA-specific MEPT system initialized")
        elif MEPT_LEAP_AVAILABLE:
            mept_components = create_mept_system(
                capacity=50000,  # Integer capacity for replay buffer
                pattern_bank_size=10000,
                transformation_penalty=MINERVA_CONFIG['transform_penalty'],
                exact_match_bonus=MINERVA_CONFIG['exact_match_bonus']
            )
            systems['replay_buffer'] = mept_components['replay_buffer']
            systems['pattern_bank'] = mept_components['pattern_bank']
            print("✅ Generic MEPT system initialized")
    
    # LEAP System - Use MINERVA-specific if available
    if USE_LEAP:
        if MINERVA_MEPT_LEAP_AVAILABLE:
            leap_components = create_minerva_leap_system(device)
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['pattern_generator']
            print("✅ MINERVA-specific LEAP system initialized")
        elif MEPT_LEAP_AVAILABLE:
            leap_components = create_leap_system(device)
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['pattern_generator']
            systems['weak_detector'] = leap_components['detector']
            print("✅ Generic LEAP system initialized")
    
    # PRISM System - Use MINERVA-specific if available
    if USE_PRISM:
        if MINERVA_PRISM_AVAILABLE:
            prism_components = create_minerva_prism_system()
            systems['prism_synthesizer'] = prism_components['synthesizer']
            systems['prism_library'] = prism_components['library']
            print("✅ MINERVA-specific PRISM system initialized")
        elif PRISM_AVAILABLE:
            systems['prism_synthesizer'] = create_prism_system()
            print("✅ Generic PRISM system initialized")
    
    # LEAP-PRISM Bridge
    if USE_LEAP_PRISM_BRIDGE and USE_LEAP and USE_PRISM:
        systems['leap_prism_bridge'] = create_leap_prism_bridge(
            systems['leap_trainer'], systems['prism_synthesizer']
        )
        print("✅ LEAP-PRISM bridge initialized")
    
    # Initialize specialized loss
    if USE_MEPT and 'loss_fn' in systems:
        # Use MINERVA-specific MEPT loss if available
        loss_fn = systems['loss_fn'].to(device)
        print("✅ Using MINERVA-specific MEPT loss function")
    else:
        loss_fn = MinervaSpecializedLoss().to(device)
        print("✅ Using MinervaSpecializedLoss")
    
    # Optimizer - SGD with Nesterov for grid attention stability
    optimizer = optim.SGD(
        model.parameters(),
        lr=MINERVA_CONFIG['learning_rate'],
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4
    )
    
    # Scheduler with warmup for stability
    def get_lr_with_warmup(epoch, warmup_epochs=10):
        if epoch < warmup_epochs:
            # Linear warmup from 0.1x to 1.0x learning rate
            return MINERVA_CONFIG['learning_rate'] * (0.1 + 0.9 * epoch / warmup_epochs)
        else:
            # Cosine annealing after warmup
            import math
            progress = (epoch - warmup_epochs) / (MINERVA_CONFIG['num_epochs'] - warmup_epochs)
            return MINERVA_CONFIG['learning_rate'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: get_lr_with_warmup(epoch) / MINERVA_CONFIG['learning_rate']
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Ultra-stable parameter initialization
    def initialize_weights_stable(m):
        if isinstance(m, nn.Linear):
            # Xavier uniform with reduced variance
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # He initialization with reduced variance
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Apply stable initialization
    model.apply(initialize_weights_stable)
    print("✅ Applied ultra-stable weight initialization")
    
    # Data directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
    # Training metrics
    best_exact = 0.0
    global_epoch = 0
    start_stage = 0
    
    # Check for existing checkpoint
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    checkpoint_path = f'{models_dir}/minerva_checkpoint.pt'
    best_model_path = f'{models_dir}/minerva_best.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_epoch = checkpoint['epoch']
            start_stage = checkpoint.get('stage', 0)
            best_exact = checkpoint.get('best_exact', 0.0)
            print(f"✅ Resumed from epoch {global_epoch}, stage {start_stage}, best: {best_exact:.2f}%")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("🔄 Starting fresh training")
    elif os.path.exists(best_model_path):
        print(f"🔄 Loading best model from {best_model_path}")
        try:
            best_checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            best_exact = best_checkpoint.get('best_exact', 0.0)
            print(f"✅ Loaded best model with {best_exact:.2f}% exact match")
        except Exception as e:
            print(f"⚠️ Failed to load best model: {e}")
            print("🔄 Starting fresh training")
    else:
        print("🆕 No existing models found - starting fresh training")
    
    # 8-Stage Progressive Curriculum Training Loop
    stage_metrics = []  # Track learning progression
    
    for stage in range(start_stage, MINERVA_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        synthesis_ratio = stage_config['synthesis_ratio']
        lr_multiplier = stage_config['lr_mult']
        
        print(f"\n🎯 MINERVA Stage {stage}: {grid_size}x{grid_size} Grid Reasoning")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {synthesis_ratio*100:.0f}% | LEAP: {stage_config['leap_complexity']}")
        print("=" * 60)
        
        # Adaptive learning rate adjustment for larger grids
        if MINERVA_CONFIG['adaptive_lr'] and stage > 0:
            adjusted_lr = MINERVA_CONFIG['learning_rate'] * lr_multiplier
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lr
            print(f"🔧 Adaptive LR: {adjusted_lr:.4f} (multiplier: {lr_multiplier:.1f})")
        
        # Warmup for complex grids (Stage 3+)
        warmup_needed = stage >= 3 and grid_size >= 12
        if warmup_needed:
            print(f"🔥 Complex grid warmup enabled for {grid_size}x{grid_size}")
        
        # Add DSL-generated samples for MINERVA's grid reasoning
        print(f"🔧 Adding MINERVA-specific DSL samples for stage {stage}...")
        dsl_samples = MINERVADSLTraining.create_minerva_dsl_samples(curriculum_stage=stage)
        print(f"✅ Added {len(dsl_samples)} MINERVA DSL samples for {grid_size}x{grid_size} grids")
        
        # Load basic ARC training data
        import json
        train_file = os.path.join(DATA_DIR, 'arc-agi_training_challenges.json')
        with open(train_file, 'r') as f:
            arc_data = json.load(f)
        
        # Create simple dataset from raw ARC data
        dataset_samples = []
        for task_id, task_data in list(arc_data.items())[:1000]:  # Limit samples
            for example in task_data['train']:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                if input_grid.shape[0] <= grid_size and input_grid.shape[1] <= grid_size:
                    dataset_samples.append({'inputs': input_grid, 'outputs': output_grid})
        
        # Convert to torch dataset and ensure minimum size
        class SimpleARCDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
                # Ensure minimum dataset size by repeating samples
                while len(self.samples) < 5000:
                    self.samples.extend(samples[:min(1000, len(samples))])
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]
        
        dataset = SimpleARCDataset(dataset_samples)
        
        # Limit dataset size for efficient training
        if len(dataset) > 15000:  # Reasonable limit
            print(f"⚠️ Reducing dataset from {len(dataset):,} to 15,000 samples for efficiency")
            dataset = torch.utils.data.Subset(dataset, list(range(15000)))
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Apply MINERVA-specialized dataset wrapper
        if USE_MEPT and 'replay_buffer' in systems:
            train_dataset = MinervaSpecializedDataset(
                train_dataset, 
                systems['replay_buffer'],
                replay_ratio=0.3 if stage == 0 else 0.2
            )
        
        # Data loaders with stage-specific grid sizes
        # Use 0 workers to avoid hanging issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=MINERVA_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,  # Disable when using 0 workers
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            persistent_workers=False  # Can't use with 0 workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=MINERVA_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,  # Disable when using 0 workers
            collate_fn=lambda batch: custom_collate_fn(batch, stage)
        )
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # MINERVA exact match injection for Stage 0 and 1
        exact_dataset = stage_config['exact_injection'] and EXACT_BOOST_AVAILABLE
        if exact_dataset:
            print(f"✅ Stage {stage} MINERVA exact match injection enabled ({grid_size}x{grid_size})")
        
        # Stage-specific performance tracking
        stage_start_time = time.time()
        stage_exact_progression = []
        
        # Stage training loop
        for epoch in range(MINERVA_CONFIG['epochs_per_stage']):
            global_epoch += 1
            
            # Extended exact match injection training (Stage 0 and 1, FIRST EPOCH ONLY)
            target_acc = 50.0 if stage == 0 else 60.0  # Realistic targets for injection
            
            if exact_dataset and stage_config['exact_injection'] and epoch == 0:  # ONLY FIRST EPOCH
                print(f"🔥 Running MINERVA 4-PHASE INJECTION SEQUENCE for Stage {stage}")
                print("=" * 60)
                
                # Phase 1: Exact Match Injection
                print(f"\n📍 PHASE 1/4: EXACT MATCH")
                model = minerva_exact_match_injection(
                    model, device=device,
                    num_epochs=100,  # Keep high for more training
                    target_accuracy=50.0  # More realistic target
                )
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
                
                # Phase 2: MEPT Injection
                print(f"\n📍 PHASE 2/4: MEPT (Memory-Enhanced Pattern Training)")
                model = minerva_mept_injection(
                    model, device=device, systems=systems,
                    num_epochs=100,  # Keep high for more training
                    target_accuracy=75.0  # Keep at 75% since it was achieving this
                )
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
                
                # Phase 3: LEAP Injection
                print(f"\n📍 PHASE 3/4: LEAP (Learning Enhancement)")
                model = minerva_leap_injection(
                    model, device=device, systems=systems,
                    num_epochs=150,  # MORE epochs since it's at 0%
                    target_accuracy=25.0  # Very low target to start
                )
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
                
                # Phase 4: PRISM Injection
                print(f"\n📍 PHASE 4/4: PRISM (Program Synthesis)")
                model = minerva_prism_injection(
                    model, device=device, systems=systems,
                    num_epochs=150,  # MORE epochs since it's at 0%
                    target_accuracy=20.0  # Very low target to start
                )
                
                print(f"\n💉 ALL 4 INJECTIONS COMPLETED - Stage {stage}, Epoch {global_epoch}")
                print("=" * 60)
                # Final cleanup
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
            else:
                if stage < 2:  # Only log for relevant stages
                    print(f"⏭️ Skipping exact injection: Stage {stage}, Epoch {epoch}")
            
            # Main training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
            
            print(f"🔄 Starting training loop for Stage {stage}, Epoch {epoch+1}, DataLoader batches: {len(train_loader)}")
            
            pbar = tqdm(train_loader, desc=f"MINERVA Stage {stage}, Epoch {epoch+1}", 
                       colour='cyan', bar_format='{l_bar}{bar:30}{r_bar}')
            optimizer.zero_grad()
            
            # Timeout detection to prevent hanging
            last_batch_time = time.time()
            stuck_counter = 0
            batch_count = 0
            
            try:
                for batch_idx, batch in enumerate(pbar):
                    batch_count += 1
                    if batch_count % 5 == 1:
                        print(f"  Processing batch {batch_count}/{len(train_loader)}...")
                    current_time = time.time()
                    if current_time - last_batch_time > 60:  # 60 seconds timeout
                        stuck_counter += 1
                        print(f"⚠️ Warning: Batch {batch_idx} taking too long (stuck counter: {stuck_counter})")
                        if stuck_counter > 3:
                            print("❌ Training appears stuck, breaking epoch")
                            break
                    
                    last_batch_time = current_time
                    
                    # MINERVA DSL augmentation - SAFE VERSION
                    if batch_idx % 10 == 0 and dsl_samples:  # Every 10th batch (reduced frequency)
                        try:
                            batch = MINERVADSLTraining.augment_batch_with_minerva_dsl(
                                batch, curriculum_stage=stage, dsl_ratio=0.1  # Reduced ratio
                            )
                        except Exception as e:
                            print(f"⚠️ DSL augmentation error: {e}")
                            # Continue with original batch
                    
                    inputs = batch['inputs'].to(device, non_blocking=True)
                    outputs = batch['outputs'].to(device, non_blocking=True)
                    
                    # Clamp values
                    inputs = torch.clamp(inputs, 0, 9)
                    outputs = torch.clamp(outputs, 0, 9)
                
                    # Convert to one-hot
                    input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                    output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                    
                    with autocast('cuda'):
                        # MINERVA forward pass
                        model_outputs = model(input_grids, output_grids, mode='train')
                        pred_output = model_outputs['predicted_output']
                        
                        # Specialized loss
                        losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                        loss = losses['total'] / MINERVA_CONFIG['gradient_accumulation']
                        
                        # EMERGENCY loss validation - skip problematic batches
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"⚠️ Skipping NaN/Inf loss at batch {batch_idx}")
                            continue
                        
                        # Skip extremely high losses that cause gradient explosions
                        if loss.abs() > 50.0:  # Raised threshold - 10.0 was too aggressive
                            print(f"⚠️ EMERGENCY: Skipping explosive loss {loss.item():.2f} at batch {batch_idx}")
                            continue
                        
                        # Skip if pattern memory component is problematic
                        if 'pattern_memory' in losses and losses['pattern_memory'] < -2.0:
                            print(f"⚠️ Skipping batch {batch_idx} due to pattern memory instability")
                            continue
                    
                    scaler.scale(loss).backward()
                
                    if (batch_idx + 1) % MINERVA_CONFIG['gradient_accumulation'] == 0:
                        scaler.unscale_(optimizer)
                        
                        # Normal gradient clipping like IRIS
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if grad_norm > 5.0:
                            print(f"⚠️ Large gradient norm: {grad_norm:.2f}, clipped to 1.0")
                        
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    # Update metrics
                    train_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                    train_metrics['exact'] += losses['exact_count'].item()
                    train_metrics['samples'] += input_grids.size(0)
                
                    # Handle different loss function outputs
                    postfix_dict = {
                        'loss': f"{losses['total'].item():.3f}",
                        'exact': f"{losses['exact_count'].item():.0f}"
                    }
                    
                    # Add optional keys if they exist
                    if 'relational' in losses:
                        postfix_dict['relational'] = f"{losses['relational'].item():.3f}"
                    elif 'spatial' in losses:
                        postfix_dict['spatial'] = f"{losses['spatial'].item():.3f}"
                        
                    if 'pattern_memory' in losses:
                        postfix_dict['pattern'] = f"{losses['pattern_memory'].item():.3f}"
                    elif 'object' in losses:
                        postfix_dict['object'] = f"{losses['object'].item():.3f}"
                    
                    pbar.set_postfix(postfix_dict)
                    
                    # LEAP training integration with stage-specific complexity
                    if USE_LEAP and 'leap_trainer' in systems and batch_idx % 3 == 0:
                        # Adjust LEAP complexity based on current stage
                        leap_complexity = stage_config['leap_complexity']
                        leap_grid_size = min(grid_size, 12)  # Cap LEAP at 12x12 for stability
                        
                        # Generate LEAP batch with stage-specific parameters for MINERVA
                        if MINERVA_MEPT_LEAP_AVAILABLE and hasattr(systems['leap_trainer'], 'generate_leap_batch'):
                            leap_batch = systems['leap_trainer'].generate_leap_batch(
                                batch_size=max(32, 64 - stage*8),  # Reduce batch size for larger grids
                                stage=stage,
                                grid_size=leap_grid_size
                            )
                        else:
                            # Fallback for generic LEAP
                            leap_batch = systems['leap_trainer'].generate_leap_batch(
                                batch_size=max(32, 64 - stage*8)  # Reduce batch size for larger grids
                            )
                        leap_inputs = leap_batch['inputs'].to(device)
                        leap_outputs = leap_batch['outputs'].to(device)
                        
                        # Ensure proper grid size for current stage
                        H, W = leap_inputs.shape[-2:]
                        if H < grid_size or W < grid_size:
                            pad_h = grid_size - H
                            pad_w = grid_size - W
                            leap_inputs = F.pad(leap_inputs, (0, pad_w, 0, pad_h), value=0)
                            leap_outputs = F.pad(leap_outputs, (0, pad_w, 0, pad_h), value=0)
                        
                        leap_input_oh = F.one_hot(leap_inputs, num_classes=10).permute(0, 3, 1, 2).float()
                        leap_output_oh = F.one_hot(leap_outputs, num_classes=10).permute(0, 3, 1, 2).float()
                        
                        with autocast('cuda'):
                            leap_model_outputs = model(leap_input_oh, leap_output_oh, mode='train')
                            leap_pred = leap_model_outputs['predicted_output']
                            leap_losses = loss_fn(leap_pred, leap_output_oh, leap_input_oh, leap_model_outputs)
                            leap_loss = leap_losses['total'] / MINERVA_CONFIG['gradient_accumulation']
                        
                        scaler.scale(leap_loss).backward()
                        
                        # Update LEAP pattern statistics
                        systems['leap_trainer'].update_pattern_stats(
                            leap_batch['pattern_types'], leap_pred, leap_output_oh
                        )
                
                    # MEPT experience collection
                    if USE_MEPT and 'replay_buffer' in systems:
                        pred_indices = pred_output.argmax(dim=1)
                        target_indices = output_grids.argmax(dim=1)
                        exact_matches = (pred_indices == target_indices).all(dim=[1,2])
                        
                        for i in range(input_grids.size(0)):
                            if exact_matches[i]:
                                systems['replay_buffer'].add(
                                    input_grids[i],
                                    output_grids[i],
                                    pred_indices[i],
                                    losses['total'].item(),
                                    is_exact=True
                                )
                
                # End of training loop
            except Exception as e:
                print(f"❌ Error in training loop: {e}")
                print("Attempting to continue to next epoch...")
                continue
                
            scheduler.step()
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                val_metrics = {'loss': 0, 'exact': 0, 'pixel_acc': 0, 'samples': 0}
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation", colour='cyan'):
                        inputs = batch['inputs'].to(device, non_blocking=True)
                        outputs = batch['outputs'].to(device, non_blocking=True)
                        
                        inputs = torch.clamp(inputs, 0, 9)
                        outputs = torch.clamp(outputs, 0, 9)
                        
                        input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                        output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                        
                        with autocast('cuda'):
                            model_outputs = model(input_grids, output_grids, mode='train')
                            pred_output = model_outputs['predicted_output']
                            losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                        
                        pred_indices = pred_output.argmax(dim=1)
                        target_indices = output_grids.argmax(dim=1)
                        
                        exact = (pred_indices == target_indices).all(dim=[1,2]).sum().item()
                        pixel_acc = (pred_indices == target_indices).float().mean().item()
                        
                        val_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                        val_metrics['exact'] += exact
                        val_metrics['pixel_acc'] += pixel_acc * input_grids.size(0)
                        val_metrics['samples'] += input_grids.size(0)
                
                # Calculate metrics with zero division protection
                train_loss = train_metrics['loss'] / max(train_metrics['samples'], 1)
                train_exact_pct = train_metrics['exact'] / max(train_metrics['samples'], 1) * 100
                val_loss = val_metrics['loss'] / max(val_metrics['samples'], 1)
                val_exact_pct = val_metrics['exact'] / max(val_metrics['samples'], 1) * 100
                val_pixel_acc = val_metrics['pixel_acc'] / max(val_metrics['samples'], 1) * 100
                
                # Track stage progression for early stopping
                stage_exact_progression.append(val_exact_pct)
                
                # Track learning progress
                current_metrics = {
                    'train_exact': train_exact_pct,
                    'val_exact': val_exact_pct,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'pixel_acc': val_pixel_acc,
                    'stage': stage,
                    'grid_size': grid_size
                }
                stage_metrics.append(current_metrics)
                
                # Calculate learning trends
                if len(stage_metrics) > 1:
                    prev = stage_metrics[-2]
                    exact_trend = val_exact_pct - prev['val_exact']
                    loss_trend = val_loss - prev['val_loss']
                    trend_icon = "📈" if exact_trend > 0 else "📉" if exact_trend < 0 else "➡️"
                    trend_text = f"({exact_trend:+.2f}%)"
                else:
                    trend_icon = "🎆"
                    trend_text = "(baseline)"
                
                # Enhanced learning indicators
                print(f"\n🧠 MINERVA Epoch {global_epoch} (Stage {stage}, {grid_size}x{grid_size}):")
                print(f"   📏 GRID SIZE: {grid_size}x{grid_size} | LEARNING: {trend_icon} {trend_text}")
                print(f"   🎯 Train: {train_exact_pct:.2f}% exact, Loss: {train_loss:.3f}")
                print(f"   🎯 Val: {val_exact_pct:.2f}% exact, Loss: {val_loss:.3f}, Pixel: {val_pixel_acc:.1f}%")
                
                # Stage progress indicator
                stage_progress = (epoch + 1) / MINERVA_CONFIG['epochs_per_stage'] * 100
                total_progress = (stage * MINERVA_CONFIG['epochs_per_stage'] + epoch + 1) / (MINERVA_CONFIG['curriculum_stages'] * MINERVA_CONFIG['epochs_per_stage']) * 100
                print(f"   📏 Stage Progress: {stage_progress:.0f}% | Total Progress: {total_progress:.0f}%")
                
                # Enhanced system status reports
                if USE_MEPT and 'replay_buffer' in systems:
                    buffer_stats = systems['replay_buffer'].get_stats()
                    exact_rate = (buffer_stats['exact_matches'] / max(1, buffer_stats['total_experiences'])) * 100
                    print(f"   📊 MEPT: {buffer_stats['total_experiences']:,} experiences | {buffer_stats['exact_matches']:,} exact ({exact_rate:.1f}% rate)")
                
                if USE_LEAP and 'leap_trainer' in systems:
                    leap_report = systems['leap_trainer'].get_performance_report()
                    if leap_report and "0.0%" not in leap_report:
                        print(f"   🎯 LEAP: {leap_report}")
                    else:
                        print(f"   ⚠️ LEAP: Pattern learning stuck at 0.0% - needs complexity adjustment for {grid_size}x{grid_size} grids")
                
                # Learning status analysis
                if val_exact_pct >= 5.0:
                    status = f"🏆 EXCELLENT learning for {grid_size}x{grid_size} grids!"
                elif val_exact_pct >= 1.0:
                    status = f"📈 GOOD progress on {grid_size}x{grid_size} patterns"
                elif val_exact_pct >= 0.1:
                    status = f"🔄 LEARNING basics for {grid_size}x{grid_size} grids"
                else:
                    status = f"⚠️ Still learning {grid_size}x{grid_size} fundamentals"
                print(f"   📊 STATUS: {status}")
                
                # Create models directory if needed
                models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
                os.makedirs(models_dir, exist_ok=True)
                
                # Save checkpoint every validation
                checkpoint_path = f'{models_dir}/minerva_checkpoint.pt'
                torch.save({
                    'epoch': global_epoch,
                    'stage': stage,
                    'grid_size': grid_size,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_exact': val_exact_pct,
                    'best_exact': best_exact,
                    'val_loss': val_loss,
                    'config': MINERVA_CONFIG,
                    'stage_config': STAGE_CONFIG
                }, checkpoint_path)
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    best_model_path = f'{models_dir}/minerva_best.pt'
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'grid_size': grid_size,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'val_loss': val_loss,
                        'config': MINERVA_CONFIG,
                        'stage_config': STAGE_CONFIG
                    }, best_model_path)
                    print(f"   💾 NEW BEST: {val_exact_pct:.2f}% exact match saved!")
        
        # Stage completion summary
        stage_duration = time.time() - stage_start_time
        final_exact = stage_exact_progression[-1] if stage_exact_progression else 0.0
        
        print(f"\n📏 Stage {stage} ({grid_size}x{grid_size}) COMPLETED:")
        print(f"   ⏱️ Duration: {stage_duration/60:.1f} minutes")
        print(f"   🎯 Final exact match: {final_exact:.2f}%")
        
        # Learning progression analysis
        if len(stage_exact_progression) >= 2:
            improvement = final_exact - stage_exact_progression[0]
            trend = "📈 IMPROVING" if improvement > 0.5 else "➡️ STABLE" if improvement > -0.5 else "📉 DECLINING"
            print(f"   📊 Learning trend: {trend} ({improvement:+.2f}%)")
        
        # Early progression criteria
        if stage >= 2 and final_exact < 0.5:  # If stuck on larger grids
            print(f"   ⚠️ WARNING: Low performance on {grid_size}x{grid_size} grids - consider adjusting learning rate")
        elif final_exact >= 3.0:
            print(f"   🏆 EXCELLENT: Strong performance on {grid_size}x{grid_size} grids!")
    
    # Final training summary
    print(f"\n🎉 MINERVA 8-Stage Training Complete!")
    print(f"   🏆 Best exact match: {best_exact:.2f}%")
    print(f"   📏 Stages completed: {MINERVA_CONFIG['curriculum_stages']} (6x6 → 30x30 grids)")
    print(f"   📊 Total epochs: {global_epoch}")
    
    # Stage-by-stage progress summary
    if stage_metrics:
        print(f"\n📏 Stage-by-stage Learning Progression:")
        for i, stage_config in enumerate(STAGE_CONFIG.values()):
            stage_final = [m for m in stage_metrics if m['stage'] == i]
            if stage_final:
                final_exact = stage_final[-1]['val_exact']
                grid_size = stage_config['max_grid_size']
                print(f"   Stage {i} ({grid_size}x{grid_size}): {final_exact:.2f}% exact match")
    
    return model, best_exact


if __name__ == "__main__":
    train_minerva_specialized()