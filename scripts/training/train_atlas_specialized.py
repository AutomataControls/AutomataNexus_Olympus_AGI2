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

# ATLAS-Specific Configuration
ATLAS_CONFIG = {
    'batch_size': 320,  # Medium for spatial complexity
    'learning_rate': 0.005,  # Standard for spatial learning
    'num_epochs': 300,
    'max_grid_size': 30,  # Larger for spatial transformations
    'gradient_accumulation': 2,  # Effective batch: 640
    'transform_penalty': 0.2,  # Very low - ATLAS should do spatial transformations
    'exact_match_bonus': 7.0,  # High for spatial precision
    'curriculum_stages': 3,
    'epochs_per_stage': 100,
    'spatial_weight': 0.6,  # ATLAS-specific loss component
    'affine_weight': 0.4,  # Spatial transformer weight
    'rotation_weight': 0.3,  # Discrete rotation weight
    'reflection_weight': 0.3,  # Discrete reflection weight
    'geometric_consistency_weight': 0.5  # Geometric transformation consistency
}

# Training components flags
USE_MEPT = True and MEPT_LEAP_AVAILABLE
USE_LEAP = True and MEPT_LEAP_AVAILABLE
USE_PRISM = True and PRISM_AVAILABLE
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
        # MEPT replay integration - prioritize spatial transformation patterns
        if (self.replay_buffer and random.random() < self.replay_ratio and 
            len(self.replay_buffer.buffer) > 0):
            experiences = self.replay_buffer.sample(1, exact_ratio=0.7)  # Favor spatial matches
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


class AtlasSpecializedLoss(nn.Module):
    """ATLAS-specific loss incorporating spatial-focused training methods"""
    def __init__(self):
        super().__init__()
        self.weights = {
            'reconstruction': 1.0,
            'transformation': ATLAS_CONFIG['transform_penalty'],
            'exact_match': ATLAS_CONFIG['exact_match_bonus'],
            'spatial': ATLAS_CONFIG['spatial_weight'],
            'affine': ATLAS_CONFIG['affine_weight'],
            'rotation': ATLAS_CONFIG['rotation_weight'],
            'reflection': ATLAS_CONFIG['reflection_weight'],
            'geometric_consistency': ATLAS_CONFIG['geometric_consistency_weight'],
            'edge': 0.3,  # Moderate - spatial boundaries matter
            'color_balance': 0.2  # Lower - focus on spatial not color
        }
        
    def forward(self, pred_output, target_output, input_grid, model_outputs=None):
        """ATLAS-specialized loss function"""
        B, C, H, W = pred_output.shape
        
        # Core reconstruction loss with spatial focus
        spatial_focal_loss = self._spatial_focal_loss(pred_output, target_output, gamma=1.4)
        
        # Exact match detection and bonus
        pred_indices = pred_output.argmax(dim=1)
        target_indices = target_output.argmax(dim=1)
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_count = exact_matches.sum()
        exact_bonus = -exact_matches.mean() * self.weights['exact_match']
        
        # ATLAS-specific: Spatial transformation penalty (should be very low)
        input_indices = input_grid.argmax(dim=1)
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.weights['transformation']
        
        # ATLAS-specific: Affine transformation consistency
        affine_loss = 0.0
        if model_outputs and 'theta' in model_outputs:
            theta = model_outputs['theta']  # B, 2, 3
            # Encourage reasonable affine transformations (not too extreme)
            scale_regularization = torch.norm(theta[:, :2, :2], dim=(1, 2))  # Scale/rotation part
            translation_regularization = torch.norm(theta[:, :, 2], dim=1)  # Translation part
            affine_loss = (scale_regularization.mean() + translation_regularization.mean() * 0.1) * self.weights['affine']
        
        # ATLAS-specific: Rotation prediction consistency
        rotation_loss = 0.0
        if model_outputs and 'rotation_logits' in model_outputs:
            rotation_logits = model_outputs['rotation_logits']  # B, 4
            # Encourage confident rotation predictions
            rotation_entropy = -torch.sum(F.softmax(rotation_logits, dim=1) * F.log_softmax(rotation_logits, dim=1), dim=1)
            rotation_loss = rotation_entropy.mean() * self.weights['rotation']
        
        # ATLAS-specific: Reflection prediction consistency
        reflection_loss = 0.0
        if model_outputs and 'reflection_logits' in model_outputs:
            reflection_logits = model_outputs['reflection_logits']  # B, 3
            # Encourage confident reflection predictions
            reflection_entropy = -torch.sum(F.softmax(reflection_logits, dim=1) * F.log_softmax(reflection_logits, dim=1), dim=1)
            reflection_loss = reflection_entropy.mean() * self.weights['reflection']
        
        # ATLAS-specific: Geometric consistency (spatial relationships preserved)
        geometric_consistency_loss = self._geometric_consistency_loss(pred_output, target_output, input_grid) * self.weights['geometric_consistency']
        
        # Enhanced edge loss for spatial boundaries
        edge_loss = self._enhanced_edge_loss(pred_output, target_output) * self.weights['edge']
        
        # Minimal color balance (spatial transformations more important)
        color_balance_loss = self._minimal_color_balance_loss(pred_output, target_output) * self.weights['color_balance']
        
        # Total loss
        total_loss = (spatial_focal_loss + transform_penalty + affine_loss + 
                     rotation_loss + reflection_loss + geometric_consistency_loss +
                     edge_loss + color_balance_loss + exact_bonus)
        
        return {
            'total': total_loss,
            'spatial_focal': spatial_focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'exact_count': exact_count,
            'affine': affine_loss,
            'rotation': rotation_loss,
            'reflection': reflection_loss,
            'geometric_consistency': geometric_consistency_loss,
            'edge': edge_loss,
            'color_balance': color_balance_loss
        }
    
    def _spatial_focal_loss(self, pred, target, gamma=1.4):
        """Focal loss optimized for spatial transformations"""
        target_idx = target.argmax(dim=1)
        ce_loss = F.cross_entropy(pred, target_idx, reduction='none')
        
        # Spatial-specific focal weighting
        pt = torch.exp(-ce_loss)
        spatial_weights = torch.ones_like(target_idx, dtype=torch.float)
        # Weight edge pixels more heavily for spatial learning
        edge_mask = self._detect_edges(target_idx)
        spatial_weights[edge_mask] = 1.5
        
        focal = (1 - pt) ** gamma * ce_loss * spatial_weights
        return focal.mean()
    
    def _detect_edges(self, grid_batch):
        """Detect edge pixels for spatial weighting"""
        edges = torch.zeros_like(grid_batch, dtype=torch.bool)
        for b in range(grid_batch.shape[0]):
            grid = grid_batch[b]
            # Simple edge detection using differences
            diff_h = torch.abs(grid[1:, :] - grid[:-1, :]) > 0
            diff_w = torch.abs(grid[:, 1:] - grid[:, :-1]) > 0
            edges[b, 1:, :] |= diff_h
            edges[b, :-1, :] |= diff_h
            edges[b, :, 1:] |= diff_w
            edges[b, :, :-1] |= diff_w
        return edges
    
    def _geometric_consistency_loss(self, pred, target, input_grid):
        """Encourage preservation of geometric relationships"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        input_idx = input_grid.argmax(dim=1)
        
        # Check if spatial relationships are preserved
        # Simple approach: compare center of mass for each color
        consistency_loss = 0.0
        for b in range(pred.shape[0]):
            pred_centers = self._get_color_centers(pred_idx[b])
            target_centers = self._get_color_centers(target_idx[b])
            
            if len(pred_centers) > 0 and len(target_centers) > 0:
                # Compare relative distances between centers
                pred_dists = self._get_center_distances(pred_centers)
                target_dists = self._get_center_distances(target_centers)
                
                if len(pred_dists) > 0 and len(target_dists) > 0:
                    min_len = min(len(pred_dists), len(target_dists))
                    pred_dists = pred_dists[:min_len]
                    target_dists = target_dists[:min_len]
                    consistency_loss += F.mse_loss(pred_dists, target_dists)
        
        return consistency_loss / max(1, pred.shape[0])
    
    def _get_color_centers(self, grid):
        """Get center of mass for each color"""
        centers = []
        for color in torch.unique(grid):
            if color > 0:  # Skip background
                mask = (grid == color)
                if mask.any():
                    y_coords, x_coords = torch.where(mask)
                    center_y = y_coords.float().mean()
                    center_x = x_coords.float().mean()
                    centers.append(torch.stack([center_y, center_x]))
        return centers
    
    def _get_center_distances(self, centers):
        """Get distances between centers"""
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = torch.norm(centers[i] - centers[j])
                distances.append(dist)
        return torch.stack(distances) if distances else torch.tensor([])
    
    def _enhanced_edge_loss(self, pred, target):
        """Enhanced edge awareness for spatial transformations"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(pred.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        pred_edges_x = F.conv2d(pred_idx.float().unsqueeze(1), sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred_idx.float().unsqueeze(1), sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
        
        target_edges_x = F.conv2d(target_idx.float().unsqueeze(1), sobel_x, padding=1)
        target_edges_y = F.conv2d(target_idx.float().unsqueeze(1), sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
        
        return F.mse_loss(pred_edges, target_edges)
    
    def _minimal_color_balance_loss(self, pred, target):
        """Minimal color distribution preservation for spatial model"""
        pred_colors = F.softmax(pred, dim=1).sum(dim=[2, 3])  # B, 10
        target_colors = target.sum(dim=[2, 3])  # B, 10
        
        # Normalize distributions
        pred_colors = pred_colors / (pred_colors.sum(dim=1, keepdim=True) + 1e-8)
        target_colors = target_colors / (target_colors.sum(dim=1, keepdim=True) + 1e-8)
        
        return F.mse_loss(pred_colors, target_colors)


def custom_collate_fn(batch):
    """ATLAS-optimized collate function with guaranteed size consistency"""
    inputs = []
    outputs = []
    target_size = ATLAS_CONFIG['max_grid_size']
    
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


def train_atlas_specialized():
    """Main ATLAS specialized training function"""
    print("🌍 Starting ATLAS Specialized Training")
    print("=" * 60)
    
    # Initialize model
    model = EnhancedAtlasNet(
        max_grid_size=ATLAS_CONFIG['max_grid_size']
    ).to(device)
    
    print(f"📊 ATLAS Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
    systems = {}
    
    # MEPT System
    if USE_MEPT:
        mept_components = create_mept_system(
            capacity=70000,  # Large for spatial patterns
            pattern_bank_size=15000,
            transformation_penalty=ATLAS_CONFIG['transform_penalty'],
            exact_match_bonus=ATLAS_CONFIG['exact_match_bonus']
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
    loss_fn = AtlasSpecializedLoss().to(device)
    
    # Optimizer - SGD with Nesterov for spatial transformation stability
    optimizer = optim.SGD(
        model.parameters(),
        lr=ATLAS_CONFIG['learning_rate'],
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4
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
    
    # Curriculum training loop
    for stage in range(start_stage, ATLAS_CONFIG['curriculum_stages']):
        print(f"\n🌍 ATLAS Stage {stage}: Spatial Transformation Focus")
        print("=" * 50)
        
        # Create curriculum dataset - EFFICIENT SIZE
        dataset = CurriculumMegaScaleDataset(
            DATA_DIR,
            curriculum_stage=stage,
            use_arc_synthesis=True,
            synthesis_ratio=max(0.2, (0.4 if stage == 0 else 0.3) * 0.5)  # Reduce by 50%
        )
        
        # Limit dataset size for efficient training
        if len(dataset) > 15000:  # Reasonable limit
            print(f"⚠️ Reducing dataset from {len(dataset):,} to 15,000 samples for efficiency")
            dataset = torch.utils.data.Subset(dataset, list(range(15000)))
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Apply ATLAS-specialized dataset wrapper
        if USE_MEPT and 'replay_buffer' in systems:
            train_dataset = AtlasSpecializedDataset(
                train_dataset, 
                systems['replay_buffer'],
                replay_ratio=0.3 if stage == 0 else 0.2
            )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=ATLAS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=ATLAS_CONFIG['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        print(f"📚 Stage {stage} - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # Exact match injection for Stage 0 (spatial-focused)
        exact_dataset = None
        if stage == 0 and USE_EXACT_BOOST:
            try:
                exact_dataset = ExactMatchBoostDataset(1400, fixed_size=8)  # Larger grids for spatial focus
                print("✅ Spatial-focused exact match injection dataset created")
            except Exception as e:
                print(f"⚠️ Could not create exact match dataset: {e}")
        
        # Stage training loop
        for epoch in range(ATLAS_CONFIG['epochs_per_stage']):
            global_epoch += 1
            
            # Exact match injection training (Stage 0 only, FIRST EPOCH ONLY)
            if exact_dataset and stage == 0 and epoch == 0:  # ONLY FIRST EPOCH
                print(f"🔥 Running exact injection: Stage {stage}, Epoch {epoch}")
                model = inject_exact_match_training(
                    model, device=device,
                    num_epochs=1,
                    target_accuracy=94.0  # Slightly lower for complex spatial patterns
                )
                print(f"🌍 Spatial injection completed - Epoch {global_epoch}")
            else:
                print(f"⏭️ Skipping exact injection: Stage {stage}, Epoch {epoch}")
            
            # Main training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
            
            pbar = tqdm(train_loader, desc=f"ATLAS Stage {stage}, Epoch {epoch+1}", 
                       colour='cyan', bar_format='{l_bar}{bar:30}{r_bar}')
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
                    # ATLAS forward pass
                    model_outputs = model(input_grids, output_grids, mode='train')
                    pred_output = model_outputs['predicted_output']
                    
                    # Specialized loss
                    losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                    loss = losses['total'] / ATLAS_CONFIG['gradient_accumulation']
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % ATLAS_CONFIG['gradient_accumulation'] == 0:
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
                    'affine': f"{losses['affine']:.3f}",
                    'spatial': f"{losses['geometric_consistency']:.3f}"
                })
                
                # LEAP training integration (spatial-focused patterns)
                if USE_LEAP and 'leap_trainer' in systems and batch_idx % 3 == 0:
                    leap_batch = systems['leap_trainer'].generate_leap_batch(batch_size=64)
                    leap_inputs = leap_batch['inputs'].to(device)
                    leap_outputs = leap_batch['outputs'].to(device)
                    
                    # Ensure proper grid size
                    H, W = leap_inputs.shape[-2:]
                    if H < ATLAS_CONFIG['max_grid_size'] or W < ATLAS_CONFIG['max_grid_size']:
                        pad_h = ATLAS_CONFIG['max_grid_size'] - H
                        pad_w = ATLAS_CONFIG['max_grid_size'] - W
                        leap_inputs = F.pad(leap_inputs, (0, pad_w, 0, pad_h), value=0)
                        leap_outputs = F.pad(leap_outputs, (0, pad_w, 0, pad_h), value=0)
                    
                    leap_input_oh = F.one_hot(leap_inputs, num_classes=10).permute(0, 3, 1, 2).float()
                    leap_output_oh = F.one_hot(leap_outputs, num_classes=10).permute(0, 3, 1, 2).float()
                    
                    with autocast('cuda'):
                        leap_model_outputs = model(leap_input_oh, leap_output_oh, mode='train')
                        leap_pred = leap_model_outputs['predicted_output']
                        leap_losses = loss_fn(leap_pred, leap_output_oh, leap_input_oh, leap_model_outputs)
                        leap_loss = leap_losses['total'] / ATLAS_CONFIG['gradient_accumulation']
                    
                    scaler.scale(leap_loss).backward()
                    
                    # Update LEAP pattern statistics
                    systems['leap_trainer'].update_pattern_stats(
                        leap_batch['pattern_types'], leap_pred, leap_output_oh
                    )
                
                # MEPT experience collection (spatial transformations)
                if USE_MEPT and 'replay_buffer' in systems:
                    pred_indices = pred_output.argmax(dim=1)
                    target_indices = output_grids.argmax(dim=1)
                    exact_matches = (pred_indices == target_indices).all(dim=[1,2])
                    
                    # Also collect spatial patterns with good transformation quality
                    spatial_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
                    good_spatial_matches = spatial_accuracy > 0.85
                    
                    for i in range(input_grids.size(0)):
                        if exact_matches[i] or good_spatial_matches[i]:
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
                
                # Calculate metrics
                train_loss = train_metrics['loss'] / train_metrics['samples']
                train_exact_pct = train_metrics['exact'] / train_metrics['samples'] * 100
                val_loss = val_metrics['loss'] / val_metrics['samples']
                val_exact_pct = val_metrics['exact'] / val_metrics['samples'] * 100
                val_pixel_acc = val_metrics['pixel_acc'] / val_metrics['samples'] * 100
                
                print(f"\n🌍 ATLAS Epoch {global_epoch} (Stage {stage}):")
                print(f"   Train Loss: {train_loss:.4f}, Train Exact: {train_exact_pct:.2f}%")
                print(f"   Val Loss: {val_loss:.4f}, Val Exact: {val_exact_pct:.2f}%, Pixel: {val_pixel_acc:.2f}%")
                
                # System status reports
                if USE_MEPT and 'replay_buffer' in systems:
                    buffer_stats = systems['replay_buffer'].get_stats()
                    print(f"   📊 MEPT: {buffer_stats['total_experiences']:,} experiences, "
                          f"{buffer_stats['exact_matches']:,} spatial matches")
                
                if USE_LEAP and 'leap_trainer' in systems:
                    leap_report = systems['leap_trainer'].get_performance_report()
                    if leap_report:
                        print(f"   🎯 LEAP: {leap_report}")
                
                # Create models directory if needed
                models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
                os.makedirs(models_dir, exist_ok=True)
                
                # Save checkpoint every validation
                checkpoint_path = f'{models_dir}/atlas_checkpoint.pt'
                torch.save({
                    'epoch': global_epoch,
                    'stage': stage,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_exact': val_exact_pct,
                    'best_exact': best_exact,
                    'val_loss': val_loss,
                    'config': ATLAS_CONFIG
                }, checkpoint_path)
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    best_model_path = f'{models_dir}/atlas_best.pt'
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'val_loss': val_loss,
                        'config': ATLAS_CONFIG
                    }, best_model_path)
                    print(f"   💾 NEW BEST: {val_exact_pct:.2f}% exact match saved!")
    
    print(f"\n🎉 ATLAS Training Complete! Best exact match: {best_exact:.2f}%")
    return model, best_exact


if __name__ == "__main__":
    train_atlas_specialized()