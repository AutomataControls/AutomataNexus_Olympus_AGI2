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
from collections import defaultdict

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import MINERVA model
from src.models.minerva_model import EnhancedMinervaNet

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

# MINERVA-Specific Configuration
MINERVA_CONFIG = {
    'batch_size': 256,  # Smaller for MINERVA's complex attention
    'learning_rate': 0.008,  # Higher for grid attention learning
    'num_epochs': 300,
    'max_grid_size': 30,
    'hidden_dim': 256,
    'pattern_memory_size': 200,
    'gradient_accumulation': 2,  # Effective batch: 512
    'transform_penalty': 0.5,
    'exact_match_bonus': 7.0,  # Higher for MINERVA's precision
    'curriculum_stages': 3,
    'epochs_per_stage': 100,
    'attention_heads': 8,
    'relational_weight': 0.4,  # MINERVA-specific loss component
    'pattern_memory_weight': 0.3
}

# Training components flags
USE_MEPT = True and MEPT_LEAP_AVAILABLE
USE_LEAP = True and MEPT_LEAP_AVAILABLE
USE_PRISM = True and PRISM_AVAILABLE
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
            experiences = self.replay_buffer.sample(1, exact_ratio=0.8)  # Favor exact matches
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
    """MINERVA-specific loss incorporating all novel training methods"""
    def __init__(self):
        super().__init__()
        self.weights = {
            'reconstruction': 1.0,
            'transformation': MINERVA_CONFIG['transform_penalty'],
            'exact_match': MINERVA_CONFIG['exact_match_bonus'],
            'relational': MINERVA_CONFIG['relational_weight'],
            'pattern_memory': MINERVA_CONFIG['pattern_memory_weight'],
            'edge': 0.3,
            'color_balance': 0.2,
            'structure': 0.3
        }
        
    def forward(self, pred_output, target_output, input_grid, model_outputs=None):
        """MINERVA-specialized loss function"""
        B, C, H, W = pred_output.shape
        
        # Core reconstruction loss with focal weighting
        focal_loss = self._focal_loss(pred_output, target_output, gamma=1.5)
        
        # Exact match detection and bonus
        pred_indices = pred_output.argmax(dim=1)
        target_indices = target_output.argmax(dim=1)
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_count = exact_matches.sum()
        exact_bonus = -exact_matches.mean() * self.weights['exact_match']
        
        # MINERVA-specific: Transformation penalty
        input_indices = input_grid.argmax(dim=1)
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.weights['transformation']
        
        # MINERVA-specific: Relational reasoning loss
        relational_loss = 0.0
        if model_outputs and 'features' in model_outputs:
            features = model_outputs['features']
            # Encourage spatial consistency in relational features
            spatial_grad = torch.gradient(features, dim=[2, 3])
            relational_loss = sum(g.abs().mean() for g in spatial_grad) * self.weights['relational']
        
        # MINERVA-specific: Pattern memory utilization
        pattern_memory_loss = 0.0
        if model_outputs and 'transform_params' in model_outputs:
            transform_params = model_outputs['transform_params']
            # Encourage diverse pattern usage
            param_diversity = -torch.var(transform_params, dim=0).mean()
            pattern_memory_loss = param_diversity * self.weights['pattern_memory']
        
        # Edge-aware loss for precise object boundaries
        edge_loss = self._edge_aware_loss(pred_output, target_output) * self.weights['edge']
        
        # Color balance preservation
        color_loss = self._color_balance_loss(pred_output, target_output) * self.weights['color_balance']
        
        # Structure preservation
        structure_loss = self._structure_loss(pred_output, target_output) * self.weights['structure']
        
        # Total loss
        total_loss = (focal_loss + transform_penalty + edge_loss + color_loss + 
                     structure_loss + relational_loss + pattern_memory_loss + exact_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'exact_count': exact_count,
            'relational': relational_loss,
            'pattern_memory': pattern_memory_loss,
            'edge': edge_loss,
            'color': color_loss,
            'structure': structure_loss
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


def custom_collate_fn(batch):
    """MINERVA-optimized collate function with proper tensor handling"""
    inputs = []
    outputs = []
    
    for item in batch:
        input_grid = item['inputs']
        output_grid = item['outputs']
        
        # Convert to tensor if needed
        if not isinstance(input_grid, torch.Tensor):
            input_grid = torch.tensor(input_grid, dtype=torch.long)
        if not isinstance(output_grid, torch.Tensor):
            output_grid = torch.tensor(output_grid, dtype=torch.long)
        
        # Ensure proper size and type
        if input_grid.shape[-1] > MINERVA_CONFIG['max_grid_size']:
            input_grid = input_grid[:MINERVA_CONFIG['max_grid_size'], :MINERVA_CONFIG['max_grid_size']]
        if output_grid.shape[-1] > MINERVA_CONFIG['max_grid_size']:
            output_grid = output_grid[:MINERVA_CONFIG['max_grid_size'], :MINERVA_CONFIG['max_grid_size']]
        
        # Pad to consistent size for grid attention
        H, W = input_grid.shape[-2:]
        if H < MINERVA_CONFIG['max_grid_size'] or W < MINERVA_CONFIG['max_grid_size']:
            pad_h = MINERVA_CONFIG['max_grid_size'] - H
            pad_w = MINERVA_CONFIG['max_grid_size'] - W
            input_grid = F.pad(input_grid, (0, pad_w, 0, pad_h), value=0)
            output_grid = F.pad(output_grid, (0, pad_w, 0, pad_h), value=0)
        
        inputs.append(input_grid)
        outputs.append(output_grid)
    
    return {
        'inputs': torch.stack(inputs),
        'outputs': torch.stack(outputs)
    }


def train_minerva_specialized():
    """Main MINERVA specialized training function"""
    print("🧠 Starting MINERVA Specialized Training")
    print("=" * 60)
    
    # Initialize model
    model = EnhancedMinervaNet(
        max_grid_size=MINERVA_CONFIG['max_grid_size'],
        hidden_dim=MINERVA_CONFIG['hidden_dim']
    ).to(device)
    
    print(f"📊 MINERVA Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
    systems = {}
    
    # MEPT System
    if USE_MEPT:
        mept_components = create_mept_system(
            capacity=50000,  # Integer capacity for replay buffer
            pattern_bank_size=10000,
            transformation_penalty=MINERVA_CONFIG['transform_penalty'],
            exact_match_bonus=MINERVA_CONFIG['exact_match_bonus']
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
    loss_fn = MinervaSpecializedLoss().to(device)
    
    # Optimizer - SGD with Nesterov for grid attention stability
    optimizer = optim.SGD(
        model.parameters(),
        lr=MINERVA_CONFIG['learning_rate'],
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MINERVA_CONFIG['epochs_per_stage']
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Data directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
    # Training metrics
    best_exact = 0.0
    global_epoch = 0
    
    # Curriculum training loop
    for stage in range(MINERVA_CONFIG['curriculum_stages']):
        print(f"\n🎯 MINERVA Stage {stage}: Grid Reasoning Focus")
        print("=" * 50)
        
        # Create curriculum dataset
        dataset = CurriculumMegaScaleDataset(
            DATA_DIR,
            curriculum_stage=stage,
            use_arc_synthesis=True,
            synthesis_ratio=0.4 if stage == 0 else 0.3
        )
        
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
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=MINERVA_CONFIG['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=MINERVA_CONFIG['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        print(f"📚 Stage {stage} - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # Exact match injection for Stage 0
        exact_dataset = None
        if stage == 0 and USE_EXACT_BOOST:
            try:
                exact_dataset = ExactMatchBoostDataset(1500, fixed_size=8)  # Larger for MINERVA
                print("✅ Exact match injection dataset created")
            except Exception as e:
                print(f"⚠️ Could not create exact match dataset: {e}")
        
        # Stage training loop
        for epoch in range(MINERVA_CONFIG['epochs_per_stage']):
            global_epoch += 1
            
            # Exact match injection training (Stage 0 only)
            if exact_dataset and epoch < 50:  # First 50 epochs
                model = inject_exact_match_training(
                    model, device=device,
                    num_epochs=1,
                    target_accuracy=95.0
                )
                print(f"💉 Exact injection completed - Epoch {global_epoch}")
            
            # Main training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
            
            pbar = tqdm(train_loader, desc=f"MINERVA Stage {stage}, Epoch {epoch+1}")
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
                    # MINERVA forward pass
                    model_outputs = model(input_grids, output_grids, mode='train')
                    pred_output = model_outputs['predicted_output']
                    
                    # Specialized loss
                    losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                    loss = losses['total'] / MINERVA_CONFIG['gradient_accumulation']
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % MINERVA_CONFIG['gradient_accumulation'] == 0:
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
                    'relational': f"{losses['relational'].item():.3f}",
                    'pattern': f"{losses['pattern_memory'].item():.3f}"
                })
                
                # LEAP training integration
                if USE_LEAP and 'leap_trainer' in systems and batch_idx % 3 == 0:
                    leap_batch = systems['leap_trainer'].generate_leap_batch(batch_size=64)
                    leap_inputs = leap_batch['inputs'].to(device)
                    leap_outputs = leap_batch['outputs'].to(device)
                    
                    # Ensure proper grid size
                    H, W = leap_inputs.shape[-2:]
                    if H < MINERVA_CONFIG['max_grid_size'] or W < MINERVA_CONFIG['max_grid_size']:
                        pad_h = MINERVA_CONFIG['max_grid_size'] - H
                        pad_w = MINERVA_CONFIG['max_grid_size'] - W
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
                            systems['replay_buffer'].add_experience(
                                input_grids[i].cpu(),
                                output_grids[i].cpu(),
                                exact_match=True
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
                
                print(f"\n🧠 MINERVA Epoch {global_epoch} (Stage {stage}):")
                print(f"   Train Loss: {train_loss:.4f}, Train Exact: {train_exact_pct:.2f}%")
                print(f"   Val Loss: {val_loss:.4f}, Val Exact: {val_exact_pct:.2f}%, Pixel: {val_pixel_acc:.2f}%")
                
                # System status reports
                if USE_MEPT and 'replay_buffer' in systems:
                    buffer_stats = systems['replay_buffer'].get_stats()
                    print(f"   📊 MEPT: {buffer_stats['total_experiences']:,} experiences, "
                          f"{buffer_stats['exact_matches']:,} exact matches")
                
                if USE_LEAP and 'leap_trainer' in systems:
                    leap_report = systems['leap_trainer'].get_performance_report()
                    if leap_report:
                        print(f"   🎯 LEAP: {leap_report}")
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    model_path = f'/content/AutomataNexus_Olympus_AGI2/models/minerva_specialized_best.pth'
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'config': MINERVA_CONFIG
                    }, model_path)
                    print(f"   💾 New best model saved: {val_exact_pct:.2f}%")
    
    print(f"\n🎉 MINERVA Training Complete! Best exact match: {best_exact:.2f}%")
    return model, best_exact


if __name__ == "__main__":
    train_minerva_specialized()