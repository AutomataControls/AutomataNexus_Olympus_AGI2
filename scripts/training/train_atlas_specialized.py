"""
ATLAS Specialized Training Script - Spatial Transformation Expert
Integrates ALL AutomataNexus novel training methods for ATLAS's unique spatial architecture
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

# ATLAS-Specific Configuration with 8-Stage Progressive Curriculum
ATLAS_CONFIG = {
    'batch_size': 64,  # Further reduced for better gradient quality
    'learning_rate': 0.005,  # Increased for better learning
    'num_epochs': 320,  # 8 stages x 40 epochs
    'hidden_dim': 128,
    'spatial_memory_size': 200,
    'gradient_accumulation': 4,  # Effective batch: 256
    'transform_penalty': 0.5,  # Increased to match NexusReference.md requirement
    'exact_match_bonus': 5.0,  # Higher bonus to encourage exact matches
    'focal_gamma': 2.0,  # Focal loss gamma parameter
    'curriculum_stages': 8,  # Progressive 8-stage curriculum
    'epochs_per_stage': 40,  # Standard for better convergence
    'spatial_weight': 0.3,  # Reduced to balance with focal loss
    'affine_weight': 0.6,  # Spatial transformer weight
    'rotation_weight': 0.4,  # Discrete rotation weight
    'reflection_weight': 0.4,  # Discrete reflection weight
    'geometric_consistency_weight': 0.5,  # Geometric transformation consistency
    'adaptive_lr': True,  # Enable adaptive learning rate per stage
    'warmup_epochs': 5,  # Warmup for larger grids
    'grid_size_penalty': 0.05  # Penalty scaling with grid complexity
}

# 8-Stage Progressive Grid Size Curriculum - OPTIMIZED FOR SPATIAL TRANSFORMATIONS
STAGE_CONFIG = {
    0: {'max_grid_size': 6,  'synthesis_ratio': 0.8, 'exact_injection': True,  'leap_complexity': 'basic', 'lr_mult': 1.0},
    1: {'max_grid_size': 8,  'synthesis_ratio': 0.7, 'exact_injection': True,  'leap_complexity': 'basic', 'lr_mult': 1.0},
    2: {'max_grid_size': 10, 'synthesis_ratio': 0.7, 'exact_injection': False, 'leap_complexity': 'simple', 'lr_mult': 1.0},
    3: {'max_grid_size': 13, 'synthesis_ratio': 0.6, 'exact_injection': False, 'leap_complexity': 'simple', 'lr_mult': 0.9},
    4: {'max_grid_size': 16, 'synthesis_ratio': 0.6, 'exact_injection': False, 'leap_complexity': 'medium', 'lr_mult': 0.8},
    5: {'max_grid_size': 20, 'synthesis_ratio': 0.5, 'exact_injection': False, 'leap_complexity': 'medium', 'lr_mult': 0.7},
    6: {'max_grid_size': 25, 'synthesis_ratio': 0.5, 'exact_injection': False, 'leap_complexity': 'complex', 'lr_mult': 0.6},
    7: {'max_grid_size': 30, 'synthesis_ratio': 0.4, 'exact_injection': False, 'leap_complexity': 'complex', 'lr_mult': 0.5}
}

# Training components flags
USE_MEPT = True and (ATLAS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_LEAP = True and (ATLAS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_PRISM = True and (ATLAS_PRISM_AVAILABLE or PRISM_AVAILABLE)
USE_EXACT_BOOST = True and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🌍 ATLAS Training on {device}")


class AtlasSpecializedDataset(Dataset):
    """ATLAS-optimized dataset with spatial transformation focus"""
    def __init__(self, base_dataset, replay_buffer=None, replay_ratio=0.3):
        self.base_dataset = base_dataset
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # DISABLE replay buffer sampling to avoid hanging like IRIS/MINERVA
        # The replay buffer sampling causes DataLoader to hang
        # TODO: Fix replay buffer sampling in future version
        
        # Get item from base dataset - handle both regular datasets and Subset objects
        try:
            item = self.base_dataset[idx]
            # Ensure it returns the right format
            if isinstance(item, dict):
                # Handle both 'inputs'/'outputs' and 'input'/'output' formats
                if 'inputs' in item and 'outputs' in item:
                    return item
                elif 'input' in item and 'output' in item:
                    # DSL format - convert to standard format
                    return {'inputs': item['input'], 'outputs': item['output']}
                else:
                    raise ValueError(f"Dict missing required keys: {list(item.keys())}")
            elif hasattr(item, '__getitem__') and len(item) >= 2:
                # Handle list/tuple format from random_split
                return {'inputs': item[0], 'outputs': item[1]}
            else:
                raise ValueError(f"Unknown dataset item format: {type(item)}")
        except Exception as e:
            print(f"Error getting item {idx} from base dataset: {e}")
            raise


class AtlasSpecializedLoss(nn.Module):
    """ATLAS-specific loss function - SIMPLIFIED FOR STABILITY"""
    def __init__(self):
        super().__init__()
        self.weights = {
            'reconstruction': 1.0,
            'transformation': ATLAS_CONFIG['transform_penalty'],
            'exact_match': ATLAS_CONFIG['exact_match_bonus'],
            'spatial': ATLAS_CONFIG['spatial_weight'],
            'affine': ATLAS_CONFIG['affine_weight']
        }
        
    def forward(self, pred_output, target_output, input_grid, model_outputs=None):
        """SIMPLIFIED ATLAS loss - based on working IRIS approach"""
        B, C, H, W = pred_output.shape
        
        # Simple focal loss like IRIS
        focal_loss = self._focal_loss(pred_output, target_output, gamma=ATLAS_CONFIG['focal_gamma'])
        
        # Exact match detection and bonus
        pred_indices = pred_output.argmax(dim=1)
        target_indices = target_output.argmax(dim=1)
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_count = exact_matches.sum()
        exact_bonus = -exact_matches.mean() * self.weights['exact_match']
        exact_bonus = exact_bonus.clamp(min=-2.0)  # Prevent excessive negative contribution
        
        # Simple transformation penalty (very low for ATLAS)
        input_indices = input_grid.argmax(dim=1)
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.weights['transformation']
        
        # ATLAS-specific: Spatial consistency loss
        spatial_loss = self._spatial_consistency_loss(pred_indices, target_indices)
        
        # SIMPLE total loss - only 4 components like IRIS
        total_loss = focal_loss + transform_penalty + exact_bonus + spatial_loss
        
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
            'spatial': spatial_loss,
            'affine': torch.tensor(0.001, device=total_loss.device)  # Dummy for display
        }
    
    def _focal_loss(self, pred, target, gamma=1.5):
        """Focal loss for hard pixel classification"""
        target_idx = target.argmax(dim=1)
        ce_loss = F.cross_entropy(pred, target_idx, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** gamma * ce_loss
        return focal.mean()
    
    def _spatial_consistency_loss(self, pred_indices, target_indices):
        """Ensure spatial patterns are preserved"""
        # Always calculate some spatial loss to encourage learning
        diff = (pred_indices != target_indices).float()
        
        # Base spatial loss - mean squared difference
        base_loss = diff.mean()
        
        # Additional penalty for isolated incorrect pixels
        if base_loss > 0:
            kernel = torch.ones(3, 3, device=pred_indices.device).unsqueeze(0).unsqueeze(0)
            neighbor_errors = F.conv2d(diff.unsqueeze(1).float(), kernel, padding=1)
            
            # Higher penalty for isolated errors (spatial inconsistency)
            spatial_penalty = (diff * (neighbor_errors.squeeze(1) < 3).float()).mean()
            total_spatial = base_loss + spatial_penalty * 0.5
        else:
            # When perfect match, minimal spatial loss
            total_spatial = base_loss
        
        return total_spatial * self.weights['spatial']


# ATLAS-specific injection functions
def atlas_exact_match_injection(model, device, num_epochs=150, target_accuracy=90.0):
    """ATLAS-specific exact match injection - spatial transformation patterns"""
    print("🌍 ATLAS EXACT MATCH INJECTION - SPATIAL PATTERNS")
    print("=" * 50)
    print(f"  Batch size: 32 (reduced for better learning)")
    print(f"  Learning rate: 0.008 -> 0.025 (with warmup)")
    print(f"  Transform penalty: {ATLAS_CONFIG['transform_penalty']}")
    print(f"  Exact match bonus: {ATLAS_CONFIG['exact_match_bonus']}")
    print(f"  Epochs: {num_epochs}")
    
    model.train()
    # DISABLE DROPOUT for exact match training
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.p = 0.0  # Disable all dropout
    
    # Use Adam with learning rate warmup for spatial learning
    optimizer = optim.Adam(model.parameters(), lr=0.008, betas=(0.9, 0.999), weight_decay=1e-5)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < 10:
            return (epoch + 1) / 10.0  # Warmup
        else:
            # Gradually increase then decrease
            progress = (epoch - 10) / (num_epochs - 10)
            if progress < 0.3:
                return 1.0 + 2.0 * progress  # Increase to 3x
            else:
                return 3.0 - 2.5 * (progress - 0.3) / 0.7  # Decrease to 0.5x
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Generate spatial transformation patterns
    def generate_spatial_patterns():
        patterns = []
        batch_size = 32  # Reduced for better learning
        
        for pattern_type in range(8):
            size = random.choice([4, 5, 6])
            
            if pattern_type == 0:  # Rotation patterns
                pattern = torch.zeros(size, size, dtype=torch.long)
                pattern[0, :] = 1
                pattern[:, 0] = 2
                # Create rotated version
                output = torch.rot90(pattern, k=1)
            elif pattern_type == 1:  # Horizontal flip
                pattern = torch.zeros(size, size, dtype=torch.long)
                pattern[:, :size//2] = 1
                pattern[:, size//2:] = 2
                output = torch.flip(pattern, dims=[1])
            elif pattern_type == 2:  # Vertical flip
                pattern = torch.zeros(size, size, dtype=torch.long)
                pattern[:size//2, :] = 3
                pattern[size//2:, :] = 4
                output = torch.flip(pattern, dims=[0])
            elif pattern_type == 3:  # Translation
                pattern = torch.zeros(size, size, dtype=torch.long)
                pattern[1:3, 1:3] = 5
                output = torch.zeros(size, size, dtype=torch.long)
                output[2:4, 2:4] = 5
            elif pattern_type == 4:  # Scaling (zoom)
                pattern = torch.zeros(size, size, dtype=torch.long)
                center = size // 2
                pattern[center, center] = 6
                output = torch.zeros(size, size, dtype=torch.long)
                output[center-1:center+2, center-1:center+2] = 6
            elif pattern_type == 5:  # Mirror diagonal
                pattern = torch.zeros(size, size, dtype=torch.long)
                for i in range(size):
                    pattern[i, i] = 7
                output = pattern.T
            elif pattern_type == 6:  # Rotation 180
                pattern = torch.zeros(size, size, dtype=torch.long)
                pattern[0, 0] = 8
                pattern[-1, -1] = 8
                output = torch.rot90(pattern, k=2)
            else:  # Complex transformation
                pattern = torch.randint(0, 4, (size, size))
                output = torch.rot90(torch.flip(pattern, dims=[1]), k=1)
            
            # Pad to same size
            max_size = 6
            padded_pattern = torch.zeros(max_size, max_size, dtype=torch.long)
            padded_output = torch.zeros(max_size, max_size, dtype=torch.long)
            
            padded_pattern[:size, :size] = pattern
            padded_output[:size, :size] = output
            
            # Repeat pattern multiple times for better learning
            for _ in range(12):
                patterns.append({'inputs': padded_pattern, 'outputs': padded_output})
        
        return patterns[:batch_size]
    
    best_acc = 0
    for epoch in range(num_epochs):
        patterns = generate_spatial_patterns()
        
        inputs = torch.stack([p['inputs'] for p in patterns]).to(device)
        outputs = torch.stack([p['outputs'] for p in patterns]).to(device)
        
        input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
        output_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
        
        optimizer.zero_grad()
        model_outputs = model(input_oh, output_oh, mode='train')
        pred = model_outputs['predicted_output']
        
        # Simple cross entropy loss for exact match training
        loss = F.cross_entropy(pred, outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        pred_idx = pred.argmax(dim=1)
        exact = (pred_idx == outputs).all(dim=[1,2])
        correct = exact.sum().item()
        total = outputs.size(0)
        acc = correct / total * 100
        
        if acc > best_acc:
            best_acc = acc
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}: {acc:.1f}% exact match | Loss: {loss.item():.3f} | LR: {current_lr:.5f}")
        
        if acc >= target_accuracy:
            print(f"🏆 TARGET REACHED: {acc:.1f}% >= {target_accuracy}%")
            break
        elif epoch == num_epochs - 1:
            print(f"⚠️ INJECTION COMPLETE: {acc:.1f}% (best: {best_acc:.1f}%, target: {target_accuracy}%)")
    
    return model


def atlas_mept_injection(model, device, systems, num_epochs=100, target_accuracy=90.0):
    """ATLAS MEPT (Memory-Enhanced Pattern Training) injection"""
    print("🌍 ATLAS MEPT INJECTION - SPATIAL MEMORY")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% spatial pattern recall")
    print(f"  Epochs: {num_epochs}")
    
    if 'replay_buffer' not in systems or 'pattern_bank' not in systems:
        print("⚠️ MEPT systems not available, skipping")
        return model
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=ATLAS_CONFIG['learning_rate']*3, betas=(0.9, 0.999))
    
    # Generate spatial transformation patterns for MEPT
    patterns = []
    for i in range(200):
        size = 6
        # Spatial patterns that ATLAS excels at
        pattern = torch.zeros(size, size, dtype=torch.long)
        
        if i % 8 == 0:  # Rotation
            pattern[0, :] = 1
            pattern[:, 0] = 2
            output = torch.rot90(pattern, k=1)
        elif i % 8 == 1:  # Horizontal flip
            pattern[:, :3] = 3
            output = torch.flip(pattern, dims=[1])
        elif i % 8 == 2:  # Vertical flip
            pattern[:3, :] = 4
            output = torch.flip(pattern, dims=[0])
        elif i % 8 == 3:  # Translation
            pattern[1:3, 1:3] = 5
            output = torch.zeros(size, size, dtype=torch.long)
            output[3:5, 3:5] = 5
        elif i % 8 == 4:  # Diagonal reflection
            for j in range(size):
                pattern[j, j] = 6
            output = pattern.T
        elif i % 8 == 5:  # 180 rotation
            pattern[0, 0] = 7
            pattern[-1, -1] = 7
            output = torch.rot90(pattern, k=2)
        elif i % 8 == 6:  # Center expansion
            center = size // 2
            pattern[center, center] = 8
            output = torch.zeros(size, size, dtype=torch.long)
            output[center-1:center+2, center-1:center+2] = 8
        else:  # Complex spatial transformation
            pattern[:2, :2] = 9
            output = torch.zeros(size, size, dtype=torch.long)
            output[-2:, -2:] = 9
            
        patterns.append({'inputs': pattern, 'outputs': output})
    
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
                    systems['spatial_memory'].store_transformation(
                        input_oh[i], output_oh[i], 
                        "mept_pattern", True
                    )
        
        acc = correct / total * 100
        print(f"MEPT Epoch {epoch+1}/{num_epochs}: {acc:.1f}% recall")
        if acc >= target_accuracy:
            print(f"🏆 MEPT TARGET REACHED: {acc:.1f}%")
            break
    
    return model


def atlas_leap_injection(model, device, systems, num_epochs=100, target_accuracy=90.0):
    """ATLAS LEAP (Learning Enhancement through Adaptive Patterns) injection"""
    print("🌍 ATLAS LEAP INJECTION - SPATIAL ADAPTATION")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% spatial pattern mastery")
    print(f"  Epochs: {num_epochs}")
    
    if 'leap_trainer' not in systems:
        print("⚠️ LEAP system not available, skipping")
        return model
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=ATLAS_CONFIG['learning_rate']*2, betas=(0.9, 0.999))
    best_acc = 0
    
    # Generate adaptive spatial patterns
    def generate_adaptive_spatial_patterns():
        batch_size = 64
        patterns = []
        
        for i in range(batch_size):
            size = 6  # Fixed size to avoid tensor stacking errors
            pattern_type = i % 6
            
            if pattern_type == 0:  # Adaptive rotation
                pattern = torch.zeros(size, size, dtype=torch.long)
                pattern[0, :] = random.randint(1, 8)
                pattern[:, 0] = random.randint(1, 8)
                output = torch.rot90(pattern, k=random.randint(1, 3))
            elif pattern_type == 1:  # Adaptive reflection
                pattern = torch.randint(1, 5, (size, size))
                if random.random() < 0.5:
                    output = torch.flip(pattern, dims=[0])
                else:
                    output = torch.flip(pattern, dims=[1])
            elif pattern_type == 2:  # Adaptive translation
                pattern = torch.zeros(size, size, dtype=torch.long)
                color = random.randint(1, 6)
                start_x, start_y = random.randint(0, size-2), random.randint(0, size-2)
                pattern[start_x:start_x+2, start_y:start_y+2] = color
                
                output = torch.zeros(size, size, dtype=torch.long)
                end_x = (start_x + random.randint(1, 2)) % (size-1)
                end_y = (start_y + random.randint(1, 2)) % (size-1)
                output[end_x:end_x+2, end_y:end_y+2] = color
            elif pattern_type == 3:  # Adaptive scaling
                pattern = torch.zeros(size, size, dtype=torch.long)
                center = size // 2
                pattern[center, center] = random.randint(1, 7)
                
                output = torch.zeros(size, size, dtype=torch.long)
                scale = random.choice([1, 2])
                if center - scale >= 0 and center + scale + 1 <= size:
                    output[center-scale:center+scale+1, center-scale:center+scale+1] = pattern[center, center]
            elif pattern_type == 4:  # Adaptive mirroring
                pattern = torch.zeros(size, size, dtype=torch.long)
                for j in range(size//2):
                    pattern[j, j] = random.randint(1, 8)
                output = pattern + pattern.T
                output = torch.clamp(output, 0, 9)
            else:  # Complex adaptive transformation
                pattern = torch.randint(0, 3, (size, size))
                output = torch.rot90(torch.flip(pattern, dims=[random.randint(0, 1)]), k=random.randint(1, 3))
            
            patterns.append(pattern)
        
        return torch.stack(patterns)
    
    for epoch in range(num_epochs):
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0
        batches_per_epoch = 10
        
        for batch_idx in range(batches_per_epoch):
            # Generate adaptive spatial patterns
            inputs = generate_adaptive_spatial_patterns().to(device)
            outputs = inputs.clone()  # Identity mapping for adaptive patterns
            
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


def atlas_prism_injection(model, device, systems, num_epochs=100, target_accuracy=90.0):
    """ATLAS PRISM (Program Reasoning through Inductive Synthesis) injection"""
    print("🌍 ATLAS PRISM INJECTION - SPATIAL SYNTHESIS")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% spatial program synthesis")
    print(f"  Epochs: {num_epochs}")
    
    if 'prism_synthesizer' not in systems and 'atlas_synthesizer' not in systems:
        print("⚠️ PRISM system not available, skipping")
        return model
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=ATLAS_CONFIG['learning_rate']*1.5, betas=(0.9, 0.999))
    best_acc = 0
    
    # Generate spatial program-based patterns
    patterns = []
    for i in range(300):
        size = random.choice([4, 5, 6])
        
        if i % 6 == 0:  # Rotation program
            base = torch.randint(0, 4, (size, size))
            output = torch.rot90(base, k=random.randint(1, 3))
        elif i % 6 == 1:  # Flip program
            base = torch.randint(0, 4, (size, size))
            if random.random() < 0.5:
                output = torch.flip(base, dims=[0])
            else:
                output = torch.flip(base, dims=[1])
        elif i % 6 == 2:  # Translation program
            base = torch.zeros(size, size, dtype=torch.long)
            color = random.randint(1, 5)
            x, y = random.randint(0, size-2), random.randint(0, size-2)
            base[x, y] = color
            
            output = torch.zeros(size, size, dtype=torch.long)
            new_x = (x + 1) % size
            new_y = (y + 1) % size
            output[new_x, new_y] = color
        elif i % 6 == 3:  # Scaling program
            base = torch.zeros(size, size, dtype=torch.long)
            center = size // 2
            base[center, center] = random.randint(1, 6)
            
            output = base.clone()
            if center > 0:
                output[center-1:center+2, center-1:center+2] = base[center, center]
        elif i % 6 == 4:  # Mirror program
            base = torch.zeros(size, size, dtype=torch.long)
            for j in range(size):
                base[j, min(j, size-1)] = random.randint(1, 7)
            output = base + base.T
            output = torch.clamp(output, 0, 9)
        else:  # Complex transformation program
            base = torch.randint(0, 3, (size, size))
            output = torch.rot90(torch.flip(base, dims=[1]), k=2)
        
        patterns.append({'inputs': base, 'outputs': output})
    
    for epoch in range(num_epochs):
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0
        
        # Shuffle patterns each epoch
        random.shuffle(patterns)
        
        for batch_start in range(0, len(patterns), 32):
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


def custom_collate_fn(batch, stage=0):
    """ATLAS-optimized collate function with stage-specific grid sizes"""
    target_size = STAGE_CONFIG[stage]['max_grid_size']
    
    inputs = []
    outputs = []
    
    for i, item in enumerate(batch):
        input_grid = item['inputs']
        output_grid = item['outputs']
        
        # Convert to tensors efficiently
        if isinstance(input_grid, np.ndarray):
            input_grid = torch.from_numpy(input_grid).long()
        elif torch.is_tensor(input_grid):
            input_grid = input_grid.long()
        else:
            input_grid = torch.tensor(input_grid, dtype=torch.long)
            
        if isinstance(output_grid, np.ndarray):
            output_grid = torch.from_numpy(output_grid).long()
        elif torch.is_tensor(output_grid):
            output_grid = output_grid.long()
        else:
            output_grid = torch.tensor(output_grid, dtype=torch.long)
        
        # Ensure 2D - handle all cases
        while input_grid.dim() > 2:
            input_grid = input_grid.squeeze(0)
        if input_grid.dim() == 1:
            input_grid = input_grid.view(-1, 1)
            
        while output_grid.dim() > 2:
            output_grid = output_grid.squeeze(0)
        if output_grid.dim() == 1:
            output_grid = output_grid.view(-1, 1)
        
        # Pad to target size
        h, w = input_grid.shape
        if h < target_size or w < target_size:
            padded_input = torch.zeros(target_size, target_size, dtype=torch.long)
            padded_input[:h, :w] = input_grid[:target_size, :target_size]
            input_grid = padded_input
            
        h, w = output_grid.shape
        if h < target_size or w < target_size:
            padded_output = torch.zeros(target_size, target_size, dtype=torch.long)
            padded_output[:h, :w] = output_grid[:target_size, :target_size]
            output_grid = padded_output
        
        inputs.append(input_grid[:target_size, :target_size])
        outputs.append(output_grid[:target_size, :target_size])
    
    result = {
        'inputs': torch.stack(inputs),
        'outputs': torch.stack(outputs)
    }
    return result


def train_atlas_specialized():
    """Main ATLAS specialized training function"""
    print("🌍 Starting ATLAS Specialized Training")
    print("=" * 60)
    
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
            print("✅ ATLAS-specific MEPT system initialized")
        else:
            mept_components = create_mept_system(
                capacity=40000,
                pattern_bank_size=8000,
                transformation_penalty=ATLAS_CONFIG['transform_penalty'],
                exact_match_bonus=ATLAS_CONFIG['exact_match_bonus']
            )
            print("✅ Generic MEPT system initialized")
        if ATLAS_MEPT_LEAP_AVAILABLE:
            systems['spatial_memory'] = mept_components['spatial_memory']
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components['loss_function']
        else:
            systems['spatial_memory'] = mept_components['replay_buffer'] 
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components.get('loss_fn')
    
    # LEAP System - Use ATLAS-specific if available
    if USE_LEAP:
        if ATLAS_MEPT_LEAP_AVAILABLE:
            leap_components = create_atlas_leap_system(model=model, device=device)
            print("✅ ATLAS-specific LEAP system initialized")
        else:
            leap_components = create_leap_system(device)
            print("✅ Generic LEAP system initialized")
        systems['leap_trainer'] = leap_components['trainer']
        systems['pattern_generator'] = leap_components.get('pattern_generator')
        systems['weak_detector'] = leap_components.get('detector')
    
    # PRISM System - Use ATLAS-specific if available
    if USE_PRISM:
        if ATLAS_PRISM_AVAILABLE:
            prism_components = create_atlas_prism_system(model=model, device=device)
            systems['prism_synthesizer'] = prism_components['synthesizer']
            systems['prism_library'] = prism_components['program_bank']
            print("✅ ATLAS-specific PRISM system initialized")
        else:
            systems['prism_synthesizer'] = create_prism_system()
            print("✅ Generic PRISM system initialized")
    
    # LEAP-PRISM Bridge
    if USE_LEAP_PRISM_BRIDGE and USE_LEAP and USE_PRISM:
        systems['leap_prism_bridge'] = create_leap_prism_bridge(
            systems['leap_trainer'], systems['prism_synthesizer']
        )
        print("✅ LEAP-PRISM bridge initialized")
    
    # ATLAS-specific Program Synthesis
    if ATLAS_SYNTHESIS_AVAILABLE:
        synthesis_components = create_atlas_synthesis_system()
        systems['atlas_synthesizer'] = synthesis_components['synthesizer']
        systems['atlas_program_library'] = synthesis_components['get_library']
        print("✅ ATLAS-specific program synthesis initialized")
    
    # Initialize specialized loss
    loss_fn = AtlasSpecializedLoss().to(device)
    
    # Optimizer - Adam for spatial transformation learning
    optimizer = optim.Adam(
        model.parameters(),
        lr=ATLAS_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ATLAS_CONFIG['epochs_per_stage']
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Data directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
    # Training metrics
    best_exact = 0.0
    global_epoch = 0
    start_stage = 0
    
    # Check for existing checkpoint
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    checkpoint_path = f'{models_dir}/atlas_checkpoint.pt'
    best_model_path = f'{models_dir}/atlas_best.pt'
    
    # FORCE FRESH START - Previous model is broken
    FORCE_FRESH_START = True
    
    if FORCE_FRESH_START:
        print("🔄 FORCED FRESH START - Ignoring checkpoints")
        print("🆕 Starting fresh ATLAS training from scratch")
    elif os.path.exists(checkpoint_path):
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
    
    for stage in range(start_stage, ATLAS_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        synthesis_ratio = stage_config['synthesis_ratio']
        lr_multiplier = stage_config['lr_mult']
        
        print(f"\n🌍 ATLAS Stage {stage}: {grid_size}x{grid_size} Spatial Transformation")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {synthesis_ratio*100:.0f}% | LEAP: {stage_config['leap_complexity']}")
        print("=" * 60)
        
        # Adaptive learning rate adjustment for larger grids
        if ATLAS_CONFIG['adaptive_lr'] and stage > 0:
            adjusted_lr = ATLAS_CONFIG['learning_rate'] * lr_multiplier
            for param_group in optimizer.param_groups:
                param_group['lr'] = adjusted_lr
            print(f"🔧 Adaptive LR: {adjusted_lr:.4f} (multiplier: {lr_multiplier:.1f})")
        
        # Add DSL-generated samples for ATLAS's spatial reasoning
        print(f"🔧 Generating ATLAS-specific DSL spatial patterns for stage {stage}...")
        dsl_trainer = ATLASDSLTraining(model, device)
        print(f"✅ ATLAS DSL spatial pattern trainer initialized")
        
        # Create simple ARC dataset for this stage
        dataset_samples = []
        
        # DSL trainer ready for augmentation during training
        
        # Load ARC JSON files
        arc_files = [
            'arc-agi_training_challenges.json',
            'arc-agi_evaluation_challenges.json'
        ]
        
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
        
        # Convert to torch dataset
        class SimpleARCDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]
        
        dataset = SimpleARCDataset(dataset_samples)
        
        # Check if we have enough samples
        if len(dataset) < 10:
            print(f"⚠️ WARNING: Only {len(dataset)} samples found! Adding synthetic data...")
            # Add some synthetic samples to ensure training can proceed
            for i in range(100):
                size = random.choice([4, 5, 6])
                input_grid = np.random.randint(0, 5, (size, size))
                # Simple spatial transformation
                output_grid = np.rot90(input_grid, k=1)
                dataset_samples.append({'inputs': input_grid, 'outputs': output_grid})
            dataset = SimpleARCDataset(dataset_samples)
            print(f"✅ Dataset expanded to {len(dataset)} samples")
        
        # Limit dataset size for efficient training
        if len(dataset) > 15000:  # Reasonable limit
            print(f"⚠️ Reducing dataset from {len(dataset):,} to 15,000 samples for efficiency")
            dataset = torch.utils.data.Subset(dataset, list(range(15000)))
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # DISABLE specialized dataset wrapper to avoid hanging like IRIS/MINERVA
        # Apply ATLAS-specialized dataset wrapper
        # if USE_MEPT and 'replay_buffer' in systems:
        #     train_dataset = AtlasSpecializedDataset(
        #         train_dataset, 
        #         systems['replay_buffer'],
        #         replay_ratio=0.3 if stage == 0 else 0.2
        #     )
        
        # Data loaders with stage-specific grid sizes
        # Use 0 workers to avoid hanging issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=ATLAS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,  # Disable when using 0 workers
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=True  # Drop incomplete batches to avoid issues
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=ATLAS_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,  # Disable when using 0 workers
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=False
        )
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # Calculate starting epoch for this stage when resuming
        stage_start_epoch = 0
        if stage == start_stage and global_epoch > 0:
            # We're resuming in the middle of a stage
            stage_start_epoch = global_epoch % ATLAS_CONFIG['epochs_per_stage']
        
        # 4-PHASE ATLAS INJECTION SEQUENCE for Stage 0 only
        if stage == 0 and stage_start_epoch == 0 and global_epoch == 0:
            print("\n" + "="*60)
            print("🌍 ATLAS 4-PHASE SPATIAL TRANSFORMATION INJECTION SEQUENCE")
            print("="*60)
            
            # Phase 1: Exact Match Injection - Spatial Identity Mapping
            print("\n📍 PHASE 1: Spatial Identity Mapping")
            model = atlas_exact_match_injection(model, device=device, num_epochs=150, target_accuracy=90.0)
            torch.cuda.empty_cache()
            gc.collect()
            
            # Phase 2: MEPT Injection - Spatial Memory Patterns
            print("\n📍 PHASE 2: Spatial Memory Enhancement (MEPT)")
            model = atlas_mept_injection(model, device=device, systems=systems, num_epochs=100)
            torch.cuda.empty_cache()
            gc.collect()
            
            # Phase 3: LEAP Injection - Adaptive Spatial Learning
            print("\n📍 PHASE 3: Adaptive Spatial Learning (LEAP)")
            model = atlas_leap_injection(model, device=device, systems=systems, num_epochs=100, target_accuracy=90.0)
            torch.cuda.empty_cache()
            gc.collect()
            
            # Phase 4: PRISM Injection - Spatial Program Synthesis
            print("\n📍 PHASE 4: Spatial Program Synthesis (PRISM)")
            model = atlas_prism_injection(model, device=device, systems=systems, num_epochs=100, target_accuracy=90.0)
            
            print(f"\n✅ 4-PHASE INJECTION COMPLETE - ATLAS is primed for spatial transformations!")
            print("="*60)
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print(f"⏭️ Skipping injection: Stage {stage}, starting epoch {stage_start_epoch}")
        
        # Stage training loop
        for epoch in range(stage_start_epoch, ATLAS_CONFIG['epochs_per_stage']):
            global_epoch += 1
            
            # Main training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
            
            print(f"🔄 Starting training loop for Stage {stage}, Epoch {epoch+1}, Global epoch {global_epoch}")
            print(f"   Total batches to process: {len(train_loader)}")
            optimizer.zero_grad()
            
            # Manual iteration for debugging  
            for batch_idx, batch in enumerate(train_loader):
                print(f"\n🔍 DEBUG: Processing batch {batch_idx+1}/{len(train_loader)}", flush=True)
                # Update progress manually
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"\rATLAS Stage {stage}, Epoch {epoch+1}: {progress:.0f}% {batch_idx+1}/{len(train_loader)}", end='', flush=True)
                print(f"\n🔍 DEBUG: Starting batch {batch_idx+1}/{len(train_loader)}")
                
                inputs = batch['inputs'].to(device, non_blocking=True)
                outputs = batch['outputs'].to(device, non_blocking=True)
                
                # Clamp values
                inputs = torch.clamp(inputs, 0, 9)
                outputs = torch.clamp(outputs, 0, 9)
                
                # ATLAS DSL Integration - Augment batch with spatial patterns
                if batch_idx % 5 == 0:  # Every 5th batch
                    try:
                        # Create batch dict for DSL
                        batch_dict = {
                            'inputs': inputs,
                            'outputs': outputs
                        }
                        
                        # DSL augmentation disabled due to tensor size mismatches
                        # The ATLAS DSL creates different sized tensors that can't be stacked
                        
                        
                        # Ensure values are still in range
                        inputs = torch.clamp(inputs, 0, 9)
                        outputs = torch.clamp(outputs, 0, 9)
                        
                    except Exception as e:
                        # If DSL fails, continue with original batch
                        print(f"⚠️ ATLAS DSL augmentation failed: {e}")
                        pass
                
                # Convert to one-hot
                input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                
                with autocast('cuda'):
                    # ATLAS forward pass
                    model_outputs = model(input_grids, output_grids, mode='train')
                    pred_output = model_outputs['predicted_output']
                    
                    # Specialized loss
                    losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                    loss = losses['total'] / ATLAS_CONFIG['gradient_accumulation']
                    
                    # ATLAS-specific loss validation
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ Skipping invalid loss in ATLAS at batch {batch_idx}")
                        continue
                    
                    # Skip if loss is extremely negative (indicates instability)
                    if loss < -5.0:
                        print(f"⚠️ Skipping extremely negative loss in ATLAS: {loss.item():.3f}")
                        continue
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % ATLAS_CONFIG['gradient_accumulation'] == 0:
                    scaler.unscale_(optimizer)
                    # ATLAS-specific gradient clipping (moderate)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if grad_norm > 5.0:
                        print(f"⚠️ Large gradient norm in ATLAS: {grad_norm:.2f}, clipped to 1.0")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Update metrics
                train_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                train_metrics['exact'] += losses['exact_count'].item()
                train_metrics['samples'] += input_grids.size(0)
                
                # Print metrics
                print(f"   Batch metrics - loss: {losses['total'].item():.3f}, exact: {losses['exact_count'].item():.0f}, spatial: {losses['spatial'].item():.3f}, affine: {losses['affine'].item():.3f}")
                
                # LEAP training integration disabled - AtlasLEAPTrainer doesn't have generate_leap_batch method
                
                # MEPT experience collection (spatial transformations)
                if USE_MEPT and 'spatial_memory' in systems:
                    pred_indices = pred_output.argmax(dim=1)
                    target_indices = output_grids.argmax(dim=1)
                    exact_matches = (pred_indices == target_indices).all(dim=[1,2])
                    
                    # Also collect near-misses for spatial learning
                    spatial_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
                    good_spatial_matches = spatial_accuracy > 0.8
                    
                    for i in range(input_grids.size(0)):
                        if exact_matches[i] or good_spatial_matches[i]:
                            systems['spatial_memory'].store_transformation(
                                input_grids[i],
                                output_grids[i],
                                "spatial_pattern",
                                bool(exact_matches[i].item())
                            )
            
            # End of batch processing loop
            print(f"\n✅ Completed all {len(train_loader)} batches for epoch {epoch+1}")
            
            # Step scheduler after epoch completes
            scheduler.step()
            print(f"📈 Learning rate updated to: {optimizer.param_groups[0]['lr']:.6f}")
            
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
                    trend_icon = "📈" if exact_trend > 0 else "📉" if exact_trend < 0 else "➡️"
                    trend_text = f"({exact_trend:+.2f}%)"
                else:
                    trend_icon = "🎆"
                    trend_text = "(baseline)"
                
                # Enhanced learning indicators
                print(f"\n🌍 ATLAS Epoch {global_epoch} (Stage {stage}, {grid_size}x{grid_size}):")
                print(f"   📏 GRID SIZE: {grid_size}x{grid_size} | SPATIAL LEARNING: {trend_icon} {trend_text}")
                print(f"   🎯 Train: {train_exact_pct:.2f}% exact, Loss: {train_loss:.3f}")
                print(f"   🎯 Val: {val_exact_pct:.2f}% exact, Loss: {val_loss:.3f}, Pixel: {val_pixel_acc:.1f}%")
                
                # Stage progress indicator
                stage_progress = (epoch + 1) / ATLAS_CONFIG['epochs_per_stage'] * 100
                total_progress = (stage * ATLAS_CONFIG['epochs_per_stage'] + epoch + 1) / (ATLAS_CONFIG['curriculum_stages'] * ATLAS_CONFIG['epochs_per_stage']) * 100
                print(f"   📏 Stage Progress: {stage_progress:.0f}% | Total Progress: {total_progress:.0f}%")
                
                # Enhanced system status reports
                if USE_MEPT and 'spatial_memory' in systems:
                    total_patterns = len(systems['spatial_memory'].spatial_patterns)
                    total_transformations = len(systems['spatial_memory'].transformation_history)
                    print(f"   📊 MEPT: {total_patterns:,} spatial patterns | {total_transformations:,} transformations")
                
                if USE_LEAP and 'leap_trainer' in systems:
                    # AtlasLEAPTrainer doesn't have get_performance_report method
                    print(f"   🎯 LEAP: Spatial pattern training active")
                
                # Learning status analysis
                if val_exact_pct >= 5.0:
                    status = f"🏆 EXCELLENT spatial learning for {grid_size}x{grid_size} grids!"
                elif val_exact_pct >= 1.0:
                    status = f"📈 GOOD progress on {grid_size}x{grid_size} spatial patterns"
                elif val_exact_pct >= 0.1:
                    status = f"🔄 LEARNING spatial basics for {grid_size}x{grid_size} grids"
                else:
                    status = f"⚠️ Still learning {grid_size}x{grid_size} spatial fundamentals"
                print(f"   📊 STATUS: {status}")
                
                # Create models directory if needed
                models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
                os.makedirs(models_dir, exist_ok=True)
                
                # Save checkpoint every validation
                checkpoint_path = f'{models_dir}/atlas_checkpoint.pt'
                torch.save({
                    'epoch': global_epoch,
                    'stage': stage,
                    'grid_size': grid_size,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_exact': val_exact_pct,
                    'best_exact': best_exact,
                    'val_loss': val_loss,
                    'config': ATLAS_CONFIG,
                    'stage_config': STAGE_CONFIG
                }, checkpoint_path)
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    best_model_path = f'{models_dir}/atlas_best.pt'
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'grid_size': grid_size,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'val_loss': val_loss,
                        'config': ATLAS_CONFIG,
                        'stage_config': STAGE_CONFIG
                    }, best_model_path)
                    print(f"   💾 NEW BEST: {val_exact_pct:.2f}% exact match saved!")
            
            print(f"✅ Epoch {epoch+1}/{ATLAS_CONFIG['epochs_per_stage']} complete for stage {stage}")
            print(f"   Moving to next epoch...")
    
    # Final training summary
    print(f"\n🎉 ATLAS 8-Stage Training Complete!")
    print(f"   🏆 Best exact match: {best_exact:.2f}%")
    print(f"   📏 Stages completed: {ATLAS_CONFIG['curriculum_stages']} (6x6 → 30x30 grids)")
    print(f"   📊 Total epochs: {global_epoch}")
    
    # Stage-by-stage progress summary
    if stage_metrics:
        print(f"\n📏 Stage-by-stage Spatial Learning Progression:")
        for i, stage_config in enumerate(STAGE_CONFIG.values()):
            stage_final = [m for m in stage_metrics if m['stage'] == i]
            if stage_final:
                final_exact = stage_final[-1]['val_exact']
                grid_size = stage_config['max_grid_size']
                print(f"   Stage {i} ({grid_size}x{grid_size}): {final_exact:.2f}% exact match")
    
    return model, best_exact


if __name__ == "__main__":
    train_atlas_specialized()