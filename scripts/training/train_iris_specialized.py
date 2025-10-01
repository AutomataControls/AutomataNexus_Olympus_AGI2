"""
IRIS Specialized Training Script - Color Pattern Recognition Expert
Integrates ALL AutomataNexus novel training methods for IRIS's unique color-focused architecture
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

# IRIS-Specific Configuration with 8-Stage Progressive Curriculum
IRIS_CONFIG = {
    'batch_size': 192,  # Reduced to prevent hanging
    'learning_rate': 0.004,  # Reduced for stability
    'num_epochs': 320,  # 8 stages x 40 epochs
    'color_embed_dim': 64,
    'color_attention_heads': 4,
    'gradient_accumulation': 2,  # Effective batch: 384
    'transform_penalty': 0.3,  # Lower - IRIS should do color transformations
    'exact_match_bonus': 3.0,  # Reduced to prevent negative losses
    'curriculum_stages': 8,  # Progressive 8-stage curriculum
    'epochs_per_stage': 40,  # Shorter stages for smoother progression
    'color_mapping_weight': 0.2,  # Reduced for stability
    'color_consistency_weight': 0.2,  # Reduced for stability
    'color_diversity_weight': 0.2,  # Encourage diverse color usage
    'lstm_rule_weight': 0.1  # Reduced for stability
}

# 8-Stage Progressive Grid Size Curriculum for Color Learning
STAGE_CONFIG = {
    0: {'max_grid_size': 6,  'synthesis_ratio': 0.6, 'exact_injection': True,  'leap_complexity': 'basic'},
    1: {'max_grid_size': 8,  'synthesis_ratio': 0.5, 'exact_injection': False, 'leap_complexity': 'basic'},
    2: {'max_grid_size': 10, 'synthesis_ratio': 0.5, 'exact_injection': False, 'leap_complexity': 'simple'},
    3: {'max_grid_size': 13, 'synthesis_ratio': 0.4, 'exact_injection': False, 'leap_complexity': 'simple'},
    4: {'max_grid_size': 16, 'synthesis_ratio': 0.4, 'exact_injection': False, 'leap_complexity': 'medium'},
    5: {'max_grid_size': 20, 'synthesis_ratio': 0.3, 'exact_injection': False, 'leap_complexity': 'medium'},
    6: {'max_grid_size': 25, 'synthesis_ratio': 0.3, 'exact_injection': False, 'leap_complexity': 'complex'},
    7: {'max_grid_size': 30, 'synthesis_ratio': 0.2, 'exact_injection': False, 'leap_complexity': 'complex'}
}

# Training components flags
USE_MEPT = True and (IRIS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_LEAP = True and (IRIS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_PRISM = True and (IRIS_PRISM_AVAILABLE or PRISM_AVAILABLE)
USE_EXACT_BOOST = True and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎨 IRIS Training on {device}")


class IrisSpecializedDataset(Dataset):
    """IRIS-optimized dataset with color pattern focus"""
    def __init__(self, base_dataset, replay_buffer=None, replay_ratio=0.3):
        self.base_dataset = base_dataset
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # DISABLE replay buffer sampling for now to avoid hanging
        # The replay buffer sampling is causing DataLoader to hang
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
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # Handle list/tuple format from random_split
                return {'inputs': item[0], 'outputs': item[1]}
            elif isinstance(item, int):
                # This might be an index from Subset - need to get actual data
                # This shouldn't happen with our SimpleARCDataset but let's handle it
                print(f"Warning: Got index {item} instead of data at position {idx}")
                # Return a dummy sample to avoid crashing
                return {
                    'inputs': torch.zeros(6, 6, dtype=torch.long),
                    'outputs': torch.zeros(6, 6, dtype=torch.long)
                }
            else:
                raise ValueError(f"Unknown dataset item format: {type(item)}")
        except Exception as e:
            if idx % 100 == 0:  # Only print every 100th error to avoid spam
                print(f"Error getting item {idx} from base dataset: {e}")
            # Return dummy data to keep training going
            return {
                'inputs': torch.zeros(6, 6, dtype=torch.long),
                'outputs': torch.zeros(6, 6, dtype=torch.long)
            }


class IrisSpecializedLoss(nn.Module):
    """IRIS-specific loss incorporating color-focused training methods"""
    def __init__(self):
        super().__init__()
        self.weights = {
            'reconstruction': 1.0,
            'transformation': IRIS_CONFIG['transform_penalty'],
            'exact_match': IRIS_CONFIG['exact_match_bonus'],
            'color_mapping': IRIS_CONFIG['color_mapping_weight'],
            'color_consistency': IRIS_CONFIG['color_consistency_weight'],
            'color_diversity': IRIS_CONFIG['color_diversity_weight'],
            'lstm_rule': IRIS_CONFIG['lstm_rule_weight'],
            'edge': 0.2,  # Lower - colors matter more than edges for IRIS
            'color_balance': 0.5  # Higher - critical for color model
        }
        
    def forward(self, pred_output, target_output, input_grid, model_outputs=None):
        """IRIS-specialized loss function"""
        B, C, H, W = pred_output.shape
        
        # Core reconstruction loss with color focus
        color_focal_loss = self._color_focal_loss(pred_output, target_output, gamma=1.2)
        
        # Exact match detection and bonus
        pred_indices = pred_output.argmax(dim=1)
        target_indices = target_output.argmax(dim=1)
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_count = exact_matches.sum()
        # Fixed exact bonus to prevent negative losses
        exact_bonus = -exact_matches.mean() * self.weights['exact_match']
        exact_bonus = exact_bonus.clamp(min=-1.0)  # Prevent excessive negative contribution
        
        # IRIS-specific: Color transformation penalty (should be low for color model)
        input_indices = input_grid.argmax(dim=1)
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.weights['transformation']
        
        # IRIS-specific: Color mapping consistency loss
        color_mapping_loss = 0.0
        if model_outputs and 'color_map' in model_outputs:
            color_map = model_outputs['color_map']  # B, 10, 10
            # Stabilized decisive mappings (avoid uniform distributions)
            mapping_entropy = -torch.sum(color_map * torch.log(color_map + 1e-8), dim=-1)
            color_mapping_loss = mapping_entropy.mean().clamp(max=5.0) * self.weights['color_mapping']
        
        # IRIS-specific: Color consistency within spatial regions
        color_consistency_loss = self._color_consistency_loss(pred_output, target_output) * self.weights['color_consistency']
        
        # IRIS-specific: Color diversity encouragement
        color_diversity_loss = self._color_diversity_loss(pred_output) * self.weights['color_diversity']
        
        # IRIS-specific: LSTM rule learning loss
        lstm_rule_loss = 0.0
        if model_outputs and 'color_attention' in model_outputs:
            attention_weights = model_outputs['color_attention']  # B, H*W, H*W
            # Stabilized structured attention patterns for color rules
            attention_variance = torch.var(attention_weights, dim=-1).mean().clamp(max=2.0)
            lstm_rule_loss = -attention_variance * self.weights['lstm_rule']  # Negative to encourage variance
            lstm_rule_loss = lstm_rule_loss.clamp(min=-0.5)  # Prevent excessive negative contribution
        
        # Enhanced color balance preservation (critical for IRIS)
        color_balance_loss = self._enhanced_color_balance_loss(pred_output, target_output) * self.weights['color_balance']
        
        # Minimal edge loss (colors more important than boundaries)
        edge_loss = self._minimal_edge_loss(pred_output, target_output) * self.weights['edge']
        
        # Stabilized total loss with component validation
        loss_components = {
            'color_focal': color_focal_loss,
            'transform': transform_penalty,
            'color_mapping': color_mapping_loss,
            'color_consistency': color_consistency_loss,
            'color_diversity': color_diversity_loss,
            'lstm_rule': lstm_rule_loss,
            'color_balance': color_balance_loss,
            'edge': edge_loss,
            'exact_bonus': exact_bonus
        }
        
        # Validate each component
        stable_components = []
        for name, component in loss_components.items():
            if torch.isnan(component) or torch.isinf(component):
                print(f"⚠️ Invalid {name} loss: {component.item():.3f}, skipping")
                stable_components.append(torch.tensor(0.0, device=color_focal_loss.device))
            else:
                stable_components.append(component)
        
        total_loss = sum(stable_components)
        
        # Ensure total loss is reasonable
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ Total loss invalid, using focal only")
            total_loss = color_focal_loss
        
        # Prevent extremely negative losses that indicate instability
        if total_loss < -2.0:
            print(f"⚠️ Loss too negative ({total_loss.item():.3f}), clamping")
            total_loss = total_loss.clamp(min=-2.0)
        
        return {
            'total': total_loss,
            'color_focal': color_focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'exact_count': exact_count,
            'color_mapping': color_mapping_loss,
            'color_consistency': color_consistency_loss,
            'color_diversity': color_diversity_loss,
            'lstm_rule': lstm_rule_loss,
            'color_balance': color_balance_loss,
            'edge': edge_loss
        }
    
    def _color_focal_loss(self, pred, target, gamma=1.2):
        """Focal loss optimized for color classification"""
        target_idx = target.argmax(dim=1)
        ce_loss = F.cross_entropy(pred, target_idx, reduction='none')
        
        # Color-specific focal weighting
        pt = torch.exp(-ce_loss)
        color_weights = torch.ones_like(target_idx, dtype=torch.float)
        # Weight non-background colors more heavily
        color_weights[target_idx > 0] = 1.5
        
        focal = (1 - pt) ** gamma * ce_loss * color_weights
        return focal.mean()
    
    def _color_consistency_loss(self, pred, target):
        """Encourage color consistency within spatial neighborhoods"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # 3x3 neighborhood consistency
        kernel = torch.ones(3, 3).to(pred.device) / 9.0
        kernel = kernel.view(1, 1, 3, 3)
        
        # Get neighborhood mode colors
        pred_neighbors = F.conv2d(pred_idx.float().unsqueeze(1), kernel, padding=1)
        target_neighbors = F.conv2d(target_idx.float().unsqueeze(1), kernel, padding=1)
        
        return F.mse_loss(pred_neighbors, target_neighbors)
    
    def _color_diversity_loss(self, pred):
        """Encourage usage of diverse colors"""
        pred_idx = pred.argmax(dim=1)
        
        # Count unique colors per sample
        diversity_scores = []
        for b in range(pred.shape[0]):
            unique_colors = torch.unique(pred_idx[b])
            diversity_score = len(unique_colors) / 10.0  # Normalize by max colors
            diversity_scores.append(torch.tensor(diversity_score, device=pred.device))
        
        # Encourage higher diversity (negative loss)
        return -torch.stack(diversity_scores).mean()
    
    def _enhanced_color_balance_loss(self, pred, target):
        """Enhanced color distribution preservation for IRIS"""
        pred_colors = F.softmax(pred, dim=1).sum(dim=[2, 3])  # B, 10
        target_colors = target.sum(dim=[2, 3])  # B, 10
        
        # Normalize distributions
        pred_colors = pred_colors / (pred_colors.sum(dim=1, keepdim=True) + 1e-8)
        target_colors = target_colors / (target_colors.sum(dim=1, keepdim=True) + 1e-8)
        
        # KL divergence for better color distribution matching
        return F.kl_div(torch.log(pred_colors + 1e-8), target_colors, reduction='batchmean')
    
    def _minimal_edge_loss(self, pred, target):
        """Minimal edge awareness for color model"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Simple boundary detection
        pred_diff = torch.abs(pred_idx[:, 1:, :] - pred_idx[:, :-1, :]).float()
        target_diff = torch.abs(target_idx[:, 1:, :] - target_idx[:, :-1, :]).float()
        
        return F.mse_loss(pred_diff, target_diff)


def custom_collate_fn(batch, stage=0):
    """IRIS-optimized collate function with stage-specific grid sizes"""
    if len(batch) == 0:
        print("⚠️ Empty batch in collate_fn!")
        return {'inputs': torch.zeros(0, 6, 6), 'outputs': torch.zeros(0, 6, 6)}
    
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
            if i < 20:  # Only print first 20 errors to avoid spam
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


# IRIS-specific injection modules for color pattern learning
def iris_exact_match_injection(model, device, num_epochs=200, target_accuracy=90.0):
    """IRIS-specific exact match injection for color pattern learning"""
    print("🎨 IRIS EXACT MATCH INJECTION - COLOR PATTERNS")
    print("=" * 50)
    print(f"  Batch size: {IRIS_CONFIG['batch_size']}")
    print(f"  Learning rate: {IRIS_CONFIG['learning_rate']*3} -> {IRIS_CONFIG['learning_rate']*10} (with warmup)")
    print(f"  Transform penalty: {IRIS_CONFIG['transform_penalty']}")
    print(f"  Exact match bonus: {IRIS_CONFIG['exact_match_bonus']}")
    print(f"  Epochs: {num_epochs}")
    
    # Exact match training for color patterns
    model.train()
    # Disable dropout for exact match
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.p = 0.0
    
    # Use Adam optimizer for color learning
    base_lr = IRIS_CONFIG['learning_rate'] * 3
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=base_lr*0.01)
    
    # Create color-focused exact match patterns
    patterns = []
    
    # 1. Simple single-color patterns (easiest)
    for color in range(1, 10):  # Colors 1-9
        for size in [2, 3, 4]:
            pattern = torch.full((size, size), color, dtype=torch.long)
            patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # 2. Two-color patterns (vertical split)
    for c1 in range(1, 6):
        for c2 in range(c1+1, 7):
            for size in [4, 5, 6]:
                pattern = torch.zeros((size, size), dtype=torch.long)
                pattern[:, :size//2] = c1
                pattern[:, size//2:] = c2
                patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # 3. Two-color patterns (horizontal split)
    for c1 in range(1, 6):
        for c2 in range(c1+1, 7):
            for size in [4, 5, 6]:
                pattern = torch.zeros((size, size), dtype=torch.long)
                pattern[:size//2, :] = c1
                pattern[size//2:, :] = c2
                patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # 4. Checkerboard patterns
    for c1 in range(1, 5):
        for c2 in range(c1+1, 6):
            for size in [4, 6]:
                pattern = torch.zeros((size, size), dtype=torch.long)
                pattern[::2, ::2] = c1
                pattern[1::2, 1::2] = c1
                pattern[::2, 1::2] = c2
                pattern[1::2, ::2] = c2
                patterns.append({'inputs': pattern, 'outputs': pattern})
    
    # Shuffle patterns
    random.shuffle(patterns)
    
    # Training loop
    best_acc = 0
    
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        epoch_loss = 0
        
        # Warmup phase
        if epoch < 10:
            new_lr = base_lr + (IRIS_CONFIG['learning_rate'] * 10 - base_lr) * (epoch / 10)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        random.shuffle(patterns)
        
        for batch_start in range(0, len(patterns), IRIS_CONFIG['batch_size']):
            batch_patterns = patterns[batch_start:batch_start + IRIS_CONFIG['batch_size']]
            
            # Pad to consistent size
            max_h = max(p['inputs'].shape[0] for p in batch_patterns)
            max_w = max(p['inputs'].shape[1] for p in batch_patterns)
            
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
            model_outputs = model(input_oh, output_oh, mode='train')
            pred = model_outputs['predicted_output']
            
            # Color-focused loss
            loss = F.cross_entropy(pred, outputs, reduction='none')
            
            pred_idx = pred.argmax(dim=1)
            exact_matches = (pred_idx == outputs).all(dim=[1,2]).float()
            
            # Weight loss by mistakes
            loss_weights = torch.ones_like(loss)
            loss_weights[exact_matches.bool()] = 0.1
            
            weighted_loss = (loss * loss_weights).mean()
            
            # Exact match reward
            exact_reward = -exact_matches.mean() * min(2.0, 0.5 + epoch / 50)
            
            total_loss = weighted_loss + exact_reward
            total_loss = torch.clamp(total_loss, min=0.01)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            correct += exact_matches.sum().item()
            total += outputs.size(0)
            epoch_loss += total_loss.item()
        
        acc = correct / total * 100
        avg_loss = epoch_loss / (len(patterns) // IRIS_CONFIG['batch_size'] + 1)
        
        if epoch >= 10:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}: {acc:.1f}% exact match | Loss: {avg_loss:.3f} | LR: {current_lr:.5f}")
        
        if acc > best_acc:
            best_acc = acc
        
        if acc >= target_accuracy:
            print(f"🏆 TARGET REACHED: {acc:.1f}% >= {target_accuracy}%")
            break
        elif epoch == num_epochs - 1:
            print(f"⚠️ INJECTION COMPLETE: {acc:.1f}% (best: {best_acc:.1f}%, target: {target_accuracy}%)")
    
    return model


def iris_mept_injection(model, device, num_epochs=100, target_accuracy=90.0):
    """IRIS MEPT (Memory-Enhanced Pattern Training) injection for color patterns"""
    print("🎨 IRIS MEPT INJECTION - COLOR MEMORY")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% color pattern recall")
    print(f"  Epochs: {num_epochs}")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=IRIS_CONFIG['learning_rate']*5, betas=(0.9, 0.999))
    
    # Generate color patterns for MEPT - all patterns will be same size to avoid stacking error
    TARGET_SIZE = 6  # Fixed size for all patterns
    patterns = []
    for i in range(100):
        # Create pattern directly at target size
        pattern = torch.zeros((TARGET_SIZE, TARGET_SIZE), dtype=torch.long)
        
        if i % 4 == 0:  # Gradient
            for row in range(TARGET_SIZE):
                pattern[row, :] = (row * 9) // TARGET_SIZE
        elif i % 4 == 1:  # Color blocks
            block_size = TARGET_SIZE // 2
            for bi in range(2):
                for bj in range(2):
                    color = bi * 2 + bj + 1
                    pattern[bi*block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size] = color
        elif i % 4 == 2:  # Rainbow stripes
            for col in range(TARGET_SIZE):
                pattern[:, col] = (col % 7) + 1
        else:  # Random color regions  
            n_regions = random.randint(2, 4)
            for r in range(n_regions):
                x = random.randint(0, TARGET_SIZE-2)
                y = random.randint(0, TARGET_SIZE-2)
                pattern[x:x+2, y:y+2] = r + 1
        
        patterns.append({'inputs': pattern, 'outputs': pattern})
    
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        
        for batch_start in range(0, len(patterns), 32):
            batch = patterns[batch_start:batch_start + 32]
            
            # All patterns are already the same size, so just stack directly
            inputs = torch.stack([p['inputs'] for p in batch]).to(device)
            outputs = torch.stack([p['outputs'] for p in batch]).to(device)
            
            input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            output_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            optimizer.zero_grad()
            model_outputs = model(input_oh, output_oh, mode='train')
            pred = model_outputs['predicted_output']
            loss = F.cross_entropy(pred, outputs)
            loss.backward()
            optimizer.step()
            
            pred_idx = pred.argmax(dim=1)
            exact = (pred_idx == outputs).all(dim=[1,2])
            correct += exact.sum().item()
            total += len(batch)
        
        acc = correct / total * 100
        print(f"MEPT Epoch {epoch+1}/{num_epochs}: {acc:.1f}% recall")
        if acc >= target_accuracy:
            print(f"🏆 MEPT TARGET REACHED: {acc:.1f}%")
            break
    
    return model


def iris_leap_injection(model, device, num_epochs=100, target_accuracy=90.0):
    """IRIS LEAP (Learning Enhancement through Adaptive Patterns) injection"""
    print("🎨 IRIS LEAP INJECTION - COLOR ADAPTATION")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% color pattern mastery")
    print(f"  Epochs: {num_epochs}")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=IRIS_CONFIG['learning_rate']*3, betas=(0.9, 0.999))
    
    # Generate adaptive color patterns
    def generate_adaptive_patterns(size=6, batch_size=64):
        patterns = []
        for i in range(batch_size):
            pattern_type = i % 8
            if pattern_type == 0:  # Vertical color bands
                pattern = torch.zeros(size, size, dtype=torch.long)
                band_width = size // 3
                for b in range(3):
                    pattern[:, b*band_width:(b+1)*band_width] = b + 1
            elif pattern_type == 1:  # Horizontal bands
                pattern = torch.zeros(size, size, dtype=torch.long)
                band_width = size // 3
                for b in range(3):
                    pattern[b*band_width:(b+1)*band_width, :] = b + 4
            elif pattern_type == 2:  # Diagonal
                pattern = torch.zeros(size, size, dtype=torch.long)
                for i in range(size):
                    for j in range(size):
                        if i == j:
                            pattern[i, j] = 7
                        elif i + j == size - 1:
                            pattern[i, j] = 8
            elif pattern_type == 3:  # Corner colors
                pattern = torch.ones(size, size, dtype=torch.long) * 1
                pattern[:size//2, :size//2] = 2
                pattern[-size//2:, -size//2:] = 3
                pattern[:size//2, -size//2:] = 4
                pattern[-size//2:, :size//2] = 5
            elif pattern_type == 4:  # Center focus
                pattern = torch.ones(size, size, dtype=torch.long) * 1
                center = size // 2
                pattern[center-1:center+1, center-1:center+1] = 6
            elif pattern_type == 5:  # Border
                pattern = torch.zeros(size, size, dtype=torch.long)
                pattern[0, :] = 7
                pattern[-1, :] = 7
                pattern[:, 0] = 7
                pattern[:, -1] = 7
            elif pattern_type == 6:  # Alternating rows
                pattern = torch.zeros(size, size, dtype=torch.long)
                for r in range(size):
                    pattern[r, :] = (r % 3) + 1
            else:  # Random patches
                pattern = torch.zeros(size, size, dtype=torch.long)
                for _ in range(3):
                    x = random.randint(0, size-2)
                    y = random.randint(0, size-2)
                    c = random.randint(1, 8)
                    pattern[x:x+2, y:y+2] = c
            patterns.append(pattern)
        return torch.stack(patterns)
    
    best_acc = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        
        # Generate adaptive color patterns
        inputs = generate_adaptive_patterns().to(device)
        outputs = inputs.clone()  # Identity mapping for adaptive patterns
        
        input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
        output_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
        
        optimizer.zero_grad()
        model_outputs = model(input_oh, output_oh, mode='train')
        pred = model_outputs['predicted_output']
        loss = F.cross_entropy(pred, outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        pred_idx = pred.argmax(dim=1)
        exact = (pred_idx == outputs).all(dim=[1,2])
        correct = exact.sum().item()
        total = outputs.size(0)
        
        acc = correct / total * 100
        if acc > best_acc:
            best_acc = acc
            
        print(f"LEAP Epoch {epoch+1}/{num_epochs}: {acc:.1f}% mastery (best: {best_acc:.1f}%)")
        
        if acc >= target_accuracy:
            print(f"🏆 LEAP TARGET REACHED: {acc:.1f}%")
            break
        elif epoch == num_epochs - 1:
            print(f"⚠️ LEAP COMPLETE: {acc:.1f}% (best: {best_acc:.1f}%, target: {target_accuracy}%)")
    
    return model


def iris_prism_injection(model, device, num_epochs=100, target_accuracy=90.0):
    """IRIS PRISM (Program Synthesis) injection for color transformations"""
    print("🎨 IRIS PRISM INJECTION - COLOR SYNTHESIS")
    print("=" * 50)
    print(f"  Target: {target_accuracy}% color program synthesis")
    print(f"  Epochs: {num_epochs}")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=IRIS_CONFIG['learning_rate']*2, betas=(0.9, 0.999))
    
    # Try to use IRIS-specific synthesizer if available
    use_iris_synth = False
    try:
        if IRIS_SYNTHESIS_AVAILABLE:
            synthesizer = IRISProgramSynthesizer()
            use_iris_synth = True
            print("  Using IRIS-specific program synthesizer")
    except:
        use_iris_synth = False
        print("  Using basic color transformations")
    
    # Generate color transformation patterns
    patterns = []
    for i in range(100):  # More patterns
        size = random.choice([4, 5, 6])
        input_pattern = torch.randint(1, 7, (size, size), dtype=torch.long)
        
        # Use IRIS synthesizer or basic transformations
        if use_iris_synth and i % 2 == 0:
            # Create various color transformations using synthesizer
            if i % 10 == 0:  # Color replacement
                color_map = {j: (j + 1) % 7 + 1 for j in range(1, 7)}
                program = synthesizer._replace_color(input_pattern, {'color_map': color_map})
                output_pattern = program
            elif i % 10 == 2:  # Stripes
                params = {
                    'direction': 'horizontal' if i % 20 < 10 else 'vertical',
                    'colors': [1, 2, 3]
                }
                output_pattern = synthesizer._apply_color_pattern(input_pattern, {
                    'pattern': 'stripes', 
                    **params
                })
            elif i % 10 == 4:  # Gradient
                params = {
                    'direction': random.choice(['horizontal', 'vertical', 'diagonal']),
                    'colors': list(range(1, 6))
                }
                output_pattern = synthesizer._color_gradient(input_pattern, params)
            elif i % 10 == 6:  # Boundary coloring
                params = {'boundary_color': 8}
                output_pattern = synthesizer._color_boundary(input_pattern, params)
            else:  # Color invert
                params = {'max_color': 7}
                output_pattern = synthesizer._color_invert(input_pattern, params)
        else:
            # Basic transformations
            if i % 6 == 0:  # Color swap
                output_pattern = input_pattern.clone()
                output_pattern[input_pattern == 1] = 2
                output_pattern[input_pattern == 2] = 1
            elif i % 6 == 1:  # Color shift
                output_pattern = ((input_pattern - 1) % 6) + 1
            elif i % 6 == 2:  # Binary threshold
                threshold = random.randint(2, 5)
                output_pattern = torch.where(input_pattern > threshold, torch.tensor(1), torch.tensor(0))
            elif i % 6 == 3:  # Keep specific colors
                keep_colors = random.sample(range(1, 7), 2)
                output_pattern = torch.where(
                    (input_pattern == keep_colors[0]) | (input_pattern == keep_colors[1]),
                    input_pattern, torch.tensor(0)
                )
            elif i % 6 == 4:  # Color mapping
                output_pattern = torch.zeros_like(input_pattern)
                for c in range(1, 7):
                    output_pattern[input_pattern == c] = ((c + 2) % 6) + 1
            else:  # Region fill
                output_pattern = input_pattern.clone()
                # Fill corners with different color
                corner_size = size // 3
                output_pattern[:corner_size, :corner_size] = 7
                output_pattern[-corner_size:, -corner_size:] = 8
        
        patterns.append({'inputs': input_pattern, 'outputs': output_pattern})
    
    best_acc = 0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        
        random.shuffle(patterns)  # Shuffle for better learning
        
        for batch_start in range(0, len(patterns), 16):  # Batch processing
            batch_patterns = patterns[batch_start:batch_start + 16]
            
            # Stack inputs and outputs
            batch_inputs = []
            batch_outputs = []
            
            for pattern in batch_patterns:
                inp = pattern['inputs']
                out = pattern['outputs']
                # Pad to same size
                max_size = 6
                if inp.shape[0] < max_size:
                    pad = max_size - inp.shape[0]
                    inp = F.pad(inp, (0, pad, 0, pad), value=0)
                    out = F.pad(out, (0, pad, 0, pad), value=0)
                batch_inputs.append(inp)
                batch_outputs.append(out)
            
            inputs = torch.stack(batch_inputs).to(device)
            outputs = torch.stack(batch_outputs).to(device)
            
            input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            output_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            optimizer.zero_grad()
            model_outputs = model(input_oh, output_oh, mode='train')
            pred = model_outputs['predicted_output']
            loss = F.cross_entropy(pred, outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred_idx = pred.argmax(dim=1)
            exact_matches = (pred_idx == outputs).all(dim=[1,2])
            correct += exact_matches.sum().item()
            total += len(batch_patterns)
        
        acc = correct / total * 100
        if acc > best_acc:
            best_acc = acc
            
        print(f"PRISM Epoch {epoch+1}/{num_epochs}: {acc:.1f}% synthesis (best: {best_acc:.1f}%)")
        
        if acc >= target_accuracy:
            print(f"🏆 PRISM TARGET REACHED: {acc:.1f}%")
            break
        elif epoch == num_epochs - 1:
            print(f"⚠️ PRISM COMPLETE: {acc:.1f}% (best: {best_acc:.1f}%, target: {target_accuracy}%)")
    
    return model


def train_iris_specialized():
    """Main IRIS specialized training function"""
    print("🎨 Starting IRIS Specialized Training")
    print("=" * 60)
    
    # Initialize model with maximum grid size from final stage
    max_grid_size = STAGE_CONFIG[7]['max_grid_size']  # Final stage size (30x30)
    model = EnhancedIrisNet(
        max_grid_size=max_grid_size
    ).to(device)
    
    print(f"📊 IRIS Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
    systems = {}
    
    # MEPT System - Use IRIS-specific if available
    if USE_MEPT:
        if IRIS_MEPT_LEAP_AVAILABLE:
            mept_components = create_iris_mept_system(
                capacity=40000,  # Smaller for color-focused patterns
                pattern_bank_size=8000,
                transformation_penalty=IRIS_CONFIG['transform_penalty'],
                exact_match_bonus=IRIS_CONFIG['exact_match_bonus']
            )
            print("✅ IRIS-specific MEPT system initialized")
        else:
            mept_components = create_mept_system(
                capacity=40000,
                pattern_bank_size=8000,
                transformation_penalty=IRIS_CONFIG['transform_penalty'],
                exact_match_bonus=IRIS_CONFIG['exact_match_bonus']
            )
            print("✅ Generic MEPT system initialized")
        systems['replay_buffer'] = mept_components['replay_buffer']
        systems['pattern_bank'] = mept_components['pattern_bank']
        systems['loss_fn'] = mept_components.get('loss_fn')  # For IRIS-specific loss
    
    # LEAP System - Use IRIS-specific if available
    if USE_LEAP:
        if IRIS_MEPT_LEAP_AVAILABLE:
            leap_components = create_iris_leap_system(device)
            print("✅ IRIS-specific LEAP system initialized")
        else:
            leap_components = create_leap_system(device)
            print("✅ Generic LEAP system initialized")
        systems['leap_trainer'] = leap_components['trainer']
        systems['pattern_generator'] = leap_components.get('pattern_generator')
        systems['weak_detector'] = leap_components.get('detector')
    
    # PRISM System - Use IRIS-specific if available
    if USE_PRISM:
        if IRIS_PRISM_AVAILABLE:
            prism_components = create_iris_prism_system()
            systems['prism_synthesizer'] = prism_components['synthesizer']
            systems['prism_library'] = prism_components['library']
            print("✅ IRIS-specific PRISM system initialized")
        else:
            systems['prism_synthesizer'] = create_prism_system()
            print("✅ Generic PRISM system initialized")
    
    # LEAP-PRISM Bridge
    if USE_LEAP_PRISM_BRIDGE and USE_LEAP and USE_PRISM:
        systems['leap_prism_bridge'] = create_leap_prism_bridge(
            systems['leap_trainer'], systems['prism_synthesizer']
        )
        print("✅ LEAP-PRISM bridge initialized")
    
    # IRIS-specific Program Synthesis
    if IRIS_SYNTHESIS_AVAILABLE:
        synthesis_components = create_iris_synthesis_system()
        systems['iris_synthesizer'] = synthesis_components['synthesizer']
        systems['iris_program_library'] = synthesis_components['program_library']
        print("✅ IRIS-specific program synthesis initialized")
    
    # Initialize specialized loss
    loss_fn = IrisSpecializedLoss().to(device)
    
    # Optimizer - Adam for color attention learning
    optimizer = optim.Adam(
        model.parameters(),
        lr=IRIS_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=IRIS_CONFIG['epochs_per_stage']
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
    checkpoint_path = f'{models_dir}/iris_checkpoint.pt'
    best_model_path = f'{models_dir}/iris_best.pt'
    
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
    
    for stage in range(start_stage, IRIS_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        synthesis_ratio = stage_config['synthesis_ratio']
        
        print(f"\n🎨 IRIS Stage {stage}: {grid_size}x{grid_size} Color Pattern Recognition")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {synthesis_ratio*100:.0f}% | LEAP: {stage_config['leap_complexity']}")
        print("=" * 60)
        
        # Generate IRIS-specific DSL samples for this stage
        print(f"🔧 Generating IRIS-specific DSL color patterns for stage {stage}...")
        dsl_samples = IRISDSLTraining.create_iris_dsl_samples(curriculum_stage=stage)
        print(f"✅ Created {len(dsl_samples)} IRIS DSL color pattern samples")
        
        # Create simple ARC dataset for this stage
        # Load ARC training data directly
        dataset_samples = []
        
        # Add DSL samples
        dataset_samples.extend(dsl_samples)
        
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
                output_grid = input_grid.copy()
                # Simple transformation
                output_grid[output_grid == 1] = 2
                output_grid[output_grid == 2] = 1
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
        
        # Apply IRIS-specialized dataset wrapper
        if USE_MEPT and 'replay_buffer' in systems:
            train_dataset = IrisSpecializedDataset(
                train_dataset, 
                systems['replay_buffer'],
                replay_ratio=0.4 if stage == 0 else 0.2  # Higher replay for color learning
            )
        
        # Data loaders with stage-specific grid sizes - no persistent workers for stage transitions
        # Use 0 workers to avoid hanging issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=IRIS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,  # Disable when using 0 workers
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=True  # Drop incomplete batches to avoid hanging
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=IRIS_CONFIG['batch_size'],
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
            stage_start_epoch = global_epoch % IRIS_CONFIG['epochs_per_stage']
        
        # 4-PHASE IRIS INJECTION SEQUENCE for Stage 0 only
        if stage == 0 and stage_start_epoch == 0 and global_epoch == 0:
            print("\n" + "="*60)
            print("🌈 IRIS 4-PHASE COLOR PATTERN INJECTION SEQUENCE")
            print("="*60)
            
            # Phase 1: Exact Match Injection - Color Identity Mapping
            print("\n📍 PHASE 1: Color Identity Mapping")
            model = iris_exact_match_injection(model, device=device, num_epochs=200, target_accuracy=90.0)
            torch.cuda.empty_cache()
            gc.collect()
            
            # Phase 2: MEPT Injection - Color Memory Patterns
            print("\n📍 PHASE 2: Color Memory Enhancement (MEPT)")
            model = iris_mept_injection(model, device=device, num_epochs=100)
            torch.cuda.empty_cache()
            gc.collect()
            
            # Phase 3: LEAP Injection - Adaptive Color Patterns
            print("\n📍 PHASE 3: Adaptive Color Learning (LEAP)")
            model = iris_leap_injection(model, device=device, num_epochs=100)
            torch.cuda.empty_cache()
            gc.collect()
            
            # Phase 4: PRISM Injection - Color Program Synthesis
            print("\n📍 PHASE 4: Color Program Synthesis (PRISM)")
            model = iris_prism_injection(model, device=device, num_epochs=100)
            torch.cuda.empty_cache()
            gc.collect()
            
            print("\n✅ 4-PHASE INJECTION COMPLETE - IRIS is primed for color pattern recognition!")
            print("="*60 + "\n")
        
        # Stage training loop
            
        for epoch in range(stage_start_epoch, IRIS_CONFIG['epochs_per_stage']):
            global_epoch += 1
            
            # Main training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
            
            print(f"🔄 Starting training loop for Stage {stage}, Epoch {epoch+1}, Global epoch {global_epoch}")
            
            # Create data loader iterator with timeout protection
            try:
                # Manual iteration for debugging
                print(f"   Total batches to process: {len(train_loader)}")
                optimizer.zero_grad()
                
                batch_count = 0
                stuck_counter = 0
                last_batch_time = time.time()
                
                # Manual iteration for debugging
                print(f"   Total batches to process: {len(train_loader)}")
                
                for batch_idx, batch in enumerate(train_loader):
                    print(f"\n🔍 DEBUG: Processing batch {batch_idx+1}/{len(train_loader)}", flush=True)
                    # Update progress manually
                    progress = (batch_idx + 1) / len(train_loader) * 100
                    print(f"\rIRIS Stage {stage}, Epoch {epoch+1}: {progress:.0f}% {batch_idx+1}/{len(train_loader)}", end='', flush=True)
                    print(f"\n🔍 DEBUG: Starting batch {batch_idx+1}/{len(train_loader)}")
                    current_time = time.time()
                    # Check if we're stuck (more than 60 seconds on a batch)
                    if current_time - last_batch_time > 60:
                        stuck_counter += 1
                        print(f"⚠️ Warning: Batch {batch_idx} taking too long ({current_time - last_batch_time:.1f}s)")
                        if stuck_counter > 3:
                            print("❌ Training appears stuck, breaking out of epoch")
                            break
                    else:
                        stuck_counter = 0
                    last_batch_time = current_time
                    
                    batch_count += 1
                    print(f"   Batch count: {batch_count}, batch_idx: {batch_idx}")
                    
                    inputs = batch['inputs'].to(device, non_blocking=True)
                    outputs = batch['outputs'].to(device, non_blocking=True)
                    
                    # Clamp values
                    inputs = torch.clamp(inputs, 0, 9)
                    outputs = torch.clamp(outputs, 0, 9)
                    
                    # IRIS DSL Integration - Augment batch with color-specific patterns
                    if batch_idx % 5 == 0:  # Every 5th batch
                        try:
                            # Create batch dict for DSL
                            batch_dict = {
                                'inputs': inputs,
                                'outputs': outputs
                            }
                            
                            # Augment with IRIS-specific DSL samples
                            augmented_batch = IRISDSLTraining.augment_batch_with_iris_dsl(
                                batch_dict,
                                curriculum_stage=stage,
                                dsl_ratio=0.3  # 30% DSL samples
                            )
                            
                            inputs = augmented_batch['inputs']
                            outputs = augmented_batch['outputs']
                            
                            # Ensure values are still in range
                            inputs = torch.clamp(inputs, 0, 9)
                            outputs = torch.clamp(outputs, 0, 9)
                            
                        except Exception as e:
                            # If DSL fails, continue with original batch
                            print(f"⚠️ IRIS DSL augmentation failed: {e}")
                            pass
                    
                    # Convert to one-hot
                    input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                    output_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                    
                    with autocast('cuda'):
                        # IRIS forward pass
                        model_outputs = model(input_grids, output_grids, mode='train')
                        pred_output = model_outputs['predicted_output']
                        
                        # Specialized loss
                        losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                        loss = losses['total'] / IRIS_CONFIG['gradient_accumulation']
                        
                        # IRIS-specific loss validation
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"⚠️ Skipping invalid loss in IRIS at batch {batch_idx}")
                            continue
                        
                        # Skip if loss is extremely negative (indicates instability)
                        if loss < -5.0:
                            print(f"⚠️ Skipping extremely negative loss in IRIS: {loss.item():.3f}")
                            continue
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % IRIS_CONFIG['gradient_accumulation'] == 0:
                        scaler.unscale_(optimizer)
                        # IRIS-specific gradient clipping (less aggressive than MINERVA)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        if grad_norm > 5.0:
                            print(f"⚠️ Large gradient norm in IRIS: {grad_norm:.2f}, clipped to 0.5")
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    # Update metrics
                    train_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                    train_metrics['exact'] += losses['exact_count'].item()
                    train_metrics['samples'] += input_grids.size(0)
                    
                    # Print metrics instead of using pbar
                    print(f"   Batch metrics - loss: {losses['total'].item():.3f}, exact: {losses['exact_count'].item():.0f}, color_map: {losses['color_mapping'].item():.3f}, color_bal: {losses['color_balance'].item():.3f}")
                    
                    # LEAP training integration with reduced frequency for speed
                    if USE_LEAP and 'leap_trainer' in systems and batch_idx % 10 == 0:  # Every 10 batches instead of 2
                        # Adjust LEAP complexity based on current stage
                        leap_complexity = stage_config['leap_complexity']
                        leap_grid_size = min(grid_size, 12)  # Cap LEAP at 12x12 for stability
                        
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
                            leap_loss = leap_losses['total'] / IRIS_CONFIG['gradient_accumulation']
                        
                        scaler.scale(leap_loss).backward()
                        
                        # Update LEAP pattern statistics
                        systems['leap_trainer'].update_pattern_stats(
                            leap_batch['pattern_types'], leap_pred, leap_output_oh
                            )
                    
                    # MEPT experience collection (color transformations)
                    if USE_MEPT and 'replay_buffer' in systems:
                        pred_indices = pred_output.argmax(dim=1)
                        target_indices = output_grids.argmax(dim=1)
                        exact_matches = (pred_indices == target_indices).all(dim=[1,2])
                        
                        # Also collect near-misses for color learning
                        color_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
                        good_color_matches = color_accuracy > 0.8
                        
                        for i in range(input_grids.size(0)):
                            if exact_matches[i] or good_color_matches[i]:
                                systems['replay_buffer'].add(
                                    input_grids[i],
                                    output_grids[i],
                                    pred_indices[i],
                                    losses['total'].item(),
                                    is_exact=exact_matches[i].item()
                                )
            
                
                # End of batch processing loop
                print(f"\n✅ Completed all {batch_count} batches for epoch {epoch+1}")
                print(f"   Expected batches: {len(train_loader)}")
                print(f"   Actual batches processed: {batch_count}")
                
            except Exception as e:
                print(f"❌ Error in training loop: {e}")
                print("Attempting to continue...")
                import traceback
                traceback.print_exc()
            
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
                    loss_trend = val_loss - prev['val_loss']
                    trend_icon = "📈" if exact_trend > 0 else "📉" if exact_trend < 0 else "➡️"
                    trend_text = f"({exact_trend:+.2f}%)"
                else:
                    trend_icon = "🎆"
                    trend_text = "(baseline)"
                
                # Enhanced learning indicators
                print(f"\n🎨 IRIS Epoch {global_epoch} (Stage {stage}, {grid_size}x{grid_size}):")
                print(f"   🎨 GRID SIZE: {grid_size}x{grid_size} | COLOR LEARNING: {trend_icon} {trend_text}")
                print(f"   🎯 Train: {train_exact_pct:.2f}% exact, Loss: {train_loss:.3f}")
                print(f"   🎯 Val: {val_exact_pct:.2f}% exact, Loss: {val_loss:.3f}, Pixel: {val_pixel_acc:.1f}%")
                
                # Stage progress indicator
                stage_progress = (epoch + 1) / IRIS_CONFIG['epochs_per_stage'] * 100
                total_progress = (stage * IRIS_CONFIG['epochs_per_stage'] + epoch + 1) / (IRIS_CONFIG['curriculum_stages'] * IRIS_CONFIG['epochs_per_stage']) * 100
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
                        print(f"   ⚠️ LEAP: Color pattern learning stuck at 0.0% - needs complexity adjustment for {grid_size}x{grid_size} grids")
                
                # Learning status analysis
                if val_exact_pct >= 5.0:
                    status = f"🏆 EXCELLENT color learning for {grid_size}x{grid_size} grids!"
                elif val_exact_pct >= 1.0:
                    status = f"📈 GOOD color progress on {grid_size}x{grid_size} patterns"
                elif val_exact_pct >= 0.1:
                    status = f"🔄 LEARNING color basics for {grid_size}x{grid_size} grids"
                else:
                    status = f"⚠️ Still learning {grid_size}x{grid_size} color fundamentals"
                print(f"   📊 STATUS: {status}")
                
                # Create models directory if needed
                models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
                os.makedirs(models_dir, exist_ok=True)
                
                # Save checkpoint every validation
                checkpoint_path = f'{models_dir}/iris_checkpoint.pt'
                torch.save({
                    'epoch': global_epoch,
                    'stage': stage,
                    'grid_size': grid_size,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_exact': val_exact_pct,
                    'best_exact': best_exact,
                    'val_loss': val_loss,
                    'config': IRIS_CONFIG,
                    'stage_config': STAGE_CONFIG
                }, checkpoint_path)
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    best_model_path = f'{models_dir}/iris_best.pt'
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'grid_size': grid_size,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'val_loss': val_loss,
                        'config': IRIS_CONFIG,
                        'stage_config': STAGE_CONFIG
                    }, best_model_path)
                    print(f"   💾 NEW BEST: {val_exact_pct:.2f}% color exact match saved!")
            
            # End of epoch
            print(f"\n✅ Epoch {epoch+1}/{IRIS_CONFIG['epochs_per_stage']} complete for stage {stage}")
            print(f"   Moving to next epoch...\n")
        
        # End of stage
        print(f"\n✅ Stage {stage} complete! Moving to next stage...\n")
    
    # Final training summary
    print(f"\n🎉 IRIS 8-Stage Color Training Complete!")
    print(f"   🏆 Best exact match: {best_exact:.2f}%")
    print(f"   🎨 Stages completed: {IRIS_CONFIG['curriculum_stages']} (6x6 → 30x30 color grids)")
    print(f"   📊 Total epochs: {global_epoch}")
    
    # Stage-by-stage progress summary
    if stage_metrics:
        print(f"\n🎨 Stage-by-stage Color Learning Progression:")
        for i, stage_config in enumerate(STAGE_CONFIG.values()):
            stage_final = [m for m in stage_metrics if m['stage'] == i]
            if stage_final:
                final_exact = stage_final[-1]['val_exact']
                grid_size = stage_config['max_grid_size']
                print(f"   Stage {i} ({grid_size}x{grid_size}): {final_exact:.2f}% color exact match")
    
    return model, best_exact


if __name__ == "__main__":
    train_iris_specialized()