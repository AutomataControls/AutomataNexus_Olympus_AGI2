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
from src.program_synthesis.synthesis_integration import LightweightProgramSynthesizer, ProgramSynthesisDataGenerator

# PRISM System
try:
    from src.program_synthesis.prism_system import PRISMSynthesizer, create_prism_system
    PRISM_AVAILABLE = True
except ImportError:
    PRISM_AVAILABLE = False
    print("⚠️ PRISM not available")

# MEPT and LEAP Systems
try:
    from src.utils.mept_system import ExperienceReplayBuffer, PatternBank, MEPTLoss, create_mept_system
    from src.utils.leap_system import AdaptivePatternGenerator, LEAPTrainer, create_leap_system
    MEPT_LEAP_AVAILABLE = True
except ImportError:
    MEPT_LEAP_AVAILABLE = False
    print("⚠️ MEPT/LEAP not available")

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
from colab_training_v4_megascale_curriculum import CurriculumMegaScaleDataset, TrainingReporter

# IRIS-Specific Configuration
IRIS_CONFIG = {
    'batch_size': 384,  # Larger for IRIS's efficient color processing
    'learning_rate': 0.006,  # Moderate for color attention learning
    'num_epochs': 300,
    'max_grid_size': 30,
    'color_embed_dim': 64,
    'color_attention_heads': 4,
    'gradient_accumulation': 2,  # Effective batch: 768
    'transform_penalty': 0.3,  # Lower - IRIS should do color transformations
    'exact_match_bonus': 8.0,  # Higher for color precision
    'curriculum_stages': 3,
    'epochs_per_stage': 100,
    'color_mapping_weight': 0.5,  # IRIS-specific loss component
    'color_consistency_weight': 0.4,  # IRIS-specific loss component
    'color_diversity_weight': 0.2,  # Encourage diverse color usage
    'lstm_rule_weight': 0.3  # Pattern-based rule learning
}

# Training components flags
USE_MEPT = True and MEPT_LEAP_AVAILABLE
USE_LEAP = True and MEPT_LEAP_AVAILABLE
USE_PRISM = True and PRISM_AVAILABLE
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
        # MEPT replay integration - prioritize color transformation patterns
        if (self.replay_buffer and random.random() < self.replay_ratio and 
            len(self.replay_buffer.buffer) > 0):
            experiences = self.replay_buffer.sample(1, exact_ratio=0.7)  # Favor color matches
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
        exact_bonus = -exact_matches.mean() * self.weights['exact_match']
        
        # IRIS-specific: Color transformation penalty (should be low for color model)
        input_indices = input_grid.argmax(dim=1)
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.weights['transformation']
        
        # IRIS-specific: Color mapping consistency loss
        color_mapping_loss = 0.0
        if model_outputs and 'color_map' in model_outputs:
            color_map = model_outputs['color_map']  # B, 10, 10
            # Encourage decisive mappings (avoid uniform distributions)
            mapping_entropy = -torch.sum(color_map * torch.log(color_map + 1e-8), dim=-1)
            color_mapping_loss = mapping_entropy.mean() * self.weights['color_mapping']
        
        # IRIS-specific: Color consistency within spatial regions
        color_consistency_loss = self._color_consistency_loss(pred_output, target_output) * self.weights['color_consistency']
        
        # IRIS-specific: Color diversity encouragement
        color_diversity_loss = self._color_diversity_loss(pred_output) * self.weights['color_diversity']
        
        # IRIS-specific: LSTM rule learning loss
        lstm_rule_loss = 0.0
        if model_outputs and 'color_attention' in model_outputs:
            attention_weights = model_outputs['color_attention']  # B, H*W, H*W
            # Encourage structured attention patterns for color rules
            attention_variance = torch.var(attention_weights, dim=-1).mean()
            lstm_rule_loss = -attention_variance * self.weights['lstm_rule']  # Negative to encourage variance
        
        # Enhanced color balance preservation (critical for IRIS)
        color_balance_loss = self._enhanced_color_balance_loss(pred_output, target_output) * self.weights['color_balance']
        
        # Minimal edge loss (colors more important than boundaries)
        edge_loss = self._minimal_edge_loss(pred_output, target_output) * self.weights['edge']
        
        # Total loss
        total_loss = (color_focal_loss + transform_penalty + color_mapping_loss + 
                     color_consistency_loss + color_diversity_loss + lstm_rule_loss +
                     color_balance_loss + edge_loss + exact_bonus)
        
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


def custom_collate_fn(batch):
    """IRIS-optimized collate function with guaranteed size consistency"""
    inputs = []
    outputs = []
    target_size = IRIS_CONFIG['max_grid_size']
    
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


def train_iris_specialized():
    """Main IRIS specialized training function"""
    print("🎨 Starting IRIS Specialized Training")
    print("=" * 60)
    
    # Initialize model
    model = EnhancedIrisNet(
        max_grid_size=IRIS_CONFIG['max_grid_size']
    ).to(device)
    
    print(f"📊 IRIS Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
    systems = {}
    
    # MEPT System
    if USE_MEPT:
        mept_components = create_mept_system(
            capacity=40000,  # Smaller for color-focused patterns
            pattern_bank_size=8000,
            transformation_penalty=IRIS_CONFIG['transform_penalty'],
            exact_match_bonus=IRIS_CONFIG['exact_match_bonus']
        )
        systems['replay_buffer'] = mept_components['replay_buffer']
        systems['pattern_bank'] = mept_components['pattern_bank']
        print("✅ MEPT system initialized")
    
    # LEAP System  
    if USE_LEAP:
        leap_components = create_leap_system(device)
        systems['leap_trainer'] = leap_components['trainer']
        systems['pattern_generator'] = leap_components['pattern_generator']
        systems['weak_detector'] = leap_components['detector']
        print("✅ LEAP system initialized")
    
    # PRISM System
    if USE_PRISM:
        systems['prism_synthesizer'] = create_prism_system()
        print("✅ PRISM system initialized")
    
    # LEAP-PRISM Bridge
    if USE_LEAP_PRISM_BRIDGE and USE_LEAP and USE_PRISM:
        systems['leap_prism_bridge'] = create_leap_prism_bridge(
            systems['leap_trainer'], systems['prism_synthesizer']
        )
        print("✅ LEAP-PRISM bridge initialized")
    
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
    
    # Data directory and models directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
    
    # Training metrics
    best_exact = 0.0
    global_epoch = 0
    
    # Curriculum training loop
    for stage in range(IRIS_CONFIG['curriculum_stages']):
        print(f"\n🎨 IRIS Stage {stage}: Color Pattern Focus")
        print("=" * 50)
        
        # Create curriculum dataset
        dataset = CurriculumMegaScaleDataset(
            DATA_DIR,
            curriculum_stage=stage,
            use_arc_synthesis=True,
            synthesis_ratio=0.5 if stage == 0 else 0.3  # More synthesis for color patterns
        )
        
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
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=IRIS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=IRIS_CONFIG['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        print(f"📚 Stage {stage} - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # Exact match injection for Stage 0 (color-focused)
        exact_dataset = None
        if stage == 0 and USE_EXACT_BOOST:
            try:
                exact_dataset = ExactMatchBoostDataset(1200, fixed_size=6)  # Smaller grids for color focus
                print("✅ Color-focused exact match injection dataset created")
            except Exception as e:
                print(f"⚠️ Could not create exact match dataset: {e}")
        
        # Stage training loop
        for epoch in range(IRIS_CONFIG['epochs_per_stage']):
            global_epoch += 1
            
            # Exact match injection training (Stage 0 only, first 5 epochs)
            if exact_dataset and stage == 0 and epoch < 5:  # Only first 5 epochs of Stage 0
                model = inject_exact_match_training(
                    model, device=device,
                    num_epochs=1,
                    target_accuracy=92.0  # Slightly lower for complex color patterns
                )
                print(f"🎨 Color injection completed - Epoch {global_epoch}")
            
            # Main training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
            
            pbar = tqdm(train_loader, desc=f"IRIS Stage {stage}, Epoch {epoch+1}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['inputs'].to(device, non_blocking=True)
                outputs = batch['outputs'].to(device, non_blocking=True)
                
                # Clamp values
                inputs = torch.clamp(inputs, 0, 9)
                outputs = torch.clamp(outputs, 0, 9)
                
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
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % IRIS_CONFIG['gradient_accumulation'] == 0:
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
                    'color_map': f"{losses['color_mapping'].item():.3f}",
                    'color_bal': f"{losses['color_balance'].item():.3f}"
                })
                
                # LEAP training integration (color-focused patterns)
                if USE_LEAP and 'leap_trainer' in systems and batch_idx % 2 == 0:
                    leap_batch = systems['leap_trainer'].generate_leap_batch(batch_size=64)
                    leap_inputs = leap_batch['inputs'].to(device)
                    leap_outputs = leap_batch['outputs'].to(device)
                    
                    # Ensure proper grid size
                    H, W = leap_inputs.shape[-2:]
                    if H < IRIS_CONFIG['max_grid_size'] or W < IRIS_CONFIG['max_grid_size']:
                        pad_h = IRIS_CONFIG['max_grid_size'] - H
                        pad_w = IRIS_CONFIG['max_grid_size'] - W
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
            
            scheduler.step()
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                val_metrics = {'loss': 0, 'exact': 0, 'pixel_acc': 0, 'samples': 0}
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
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
                
                # Calculate metrics
                train_loss = train_metrics['loss'] / train_metrics['samples']
                train_exact_pct = train_metrics['exact'] / train_metrics['samples'] * 100
                val_loss = val_metrics['loss'] / val_metrics['samples']
                val_exact_pct = val_metrics['exact'] / val_metrics['samples'] * 100
                val_pixel_acc = val_metrics['pixel_acc'] / val_metrics['samples'] * 100
                
                print(f"\n🎨 IRIS Epoch {global_epoch} (Stage {stage}):")
                print(f"   Train Loss: {train_loss:.4f}, Train Exact: {train_exact_pct:.2f}%")
                print(f"   Val Loss: {val_loss:.4f}, Val Exact: {val_exact_pct:.2f}%, Pixel: {val_pixel_acc:.2f}%")
                
                # System status reports
                if USE_MEPT and 'replay_buffer' in systems:
                    buffer_stats = systems['replay_buffer'].get_stats()
                    print(f"   📊 MEPT: {buffer_stats['total_experiences']:,} experiences, "
                          f"{buffer_stats['exact_matches']:,} color matches")
                
                if USE_LEAP and 'leap_trainer' in systems:
                    leap_report = systems['leap_trainer'].get_performance_report()
                    if leap_report:
                        print(f"   🎯 LEAP: {leap_report}")
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    model_path = f'/content/AutomataNexus_Olympus_AGI2/models/iris_specialized_best.pth'
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'config': IRIS_CONFIG
                    }, model_path)
                    print(f"   💾 New best model saved: {val_exact_pct:.2f}%")
    
    print(f"\n🎉 IRIS Training Complete! Best exact match: {best_exact:.2f}%")
    return model, best_exact


if __name__ == "__main__":
    train_iris_specialized()