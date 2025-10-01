"""
CHRONOS Specialized Training Script - Temporal Sequence Analysis Expert
Integrates ALL AutomataNexus novel training methods for CHRONOS's unique temporal architecture
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

# Import CHRONOS model
from src.models.chronos_model import EnhancedChronosNet

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

# CHRONOS-Specific Configuration with 8-Stage Progressive Curriculum
CHRONOS_CONFIG = {
    'batch_size': 256,  # Smaller for temporal complexity
    'learning_rate': 0.002,  # Further reduced for temporal stability
    'num_epochs': 320,  # 8 stages x 40 epochs
    'sequence_length': 3,  # Max sequence for temporal analysis
    'hidden_dim': 256,
    'gradient_accumulation': 2,  # Effective batch: 512
    'transform_penalty': 0.3,  # Lower - CHRONOS should do temporal transformations
    'exact_match_bonus': 2.5,  # Reduced to prevent negative losses
    'curriculum_stages': 8,  # Progressive 8-stage curriculum
    'epochs_per_stage': 40,  # Shorter stages for smoother progression
    'temporal_weight': 0.2,  # Reduced for stability
    'movement_weight': 0.15,  # Reduced for stability
    'object_tracking_weight': 0.1,  # Reduced for stability
    'sequence_consistency_weight': 0.15  # Reduced for stability
}

# 8-Stage Progressive Grid Size Curriculum for Temporal Learning
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
USE_MEPT = True and MEPT_LEAP_AVAILABLE
USE_LEAP = True and MEPT_LEAP_AVAILABLE
USE_PRISM = True and PRISM_AVAILABLE
USE_EXACT_BOOST = True and EXACT_BOOST_AVAILABLE
USE_LEAP_PRISM_BRIDGE = True and LEAP_PRISM_BRIDGE_AVAILABLE

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⏰ CHRONOS Training on {device}")


class ChronosSpecializedDataset(Dataset):
    """CHRONOS-optimized dataset with temporal sequence focus"""
    def __init__(self, base_dataset, replay_buffer=None, replay_ratio=0.3):
        self.base_dataset = base_dataset
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # MEPT replay integration - prioritize temporal patterns
        if (self.replay_buffer and random.random() < self.replay_ratio and 
            len(self.replay_buffer.buffer) > 0):
            experiences = self.replay_buffer.sample(1, exact_ratio=0.7)  # Favor temporal matches
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


class ChronosSpecializedLoss(nn.Module):
    """CHRONOS-specific loss incorporating temporal-focused training methods"""
    def __init__(self):
        super().__init__()
        self.weights = {
            'reconstruction': 1.0,
            'transformation': CHRONOS_CONFIG['transform_penalty'],
            'exact_match': CHRONOS_CONFIG['exact_match_bonus'],
            'temporal': CHRONOS_CONFIG['temporal_weight'],
            'movement': CHRONOS_CONFIG['movement_weight'],
            'object_tracking': CHRONOS_CONFIG['object_tracking_weight'],
            'sequence_consistency': CHRONOS_CONFIG['sequence_consistency_weight'],
            'edge': 0.2,  # Lower - temporal patterns matter more
            'color_balance': 0.3  # Moderate for object tracking
        }
        
    def forward(self, pred_output, target_output, input_grid, model_outputs=None):
        """CHRONOS-specialized loss function"""
        B, C, H, W = pred_output.shape
        
        # Core reconstruction loss with temporal focus
        temporal_focal_loss = self._temporal_focal_loss(pred_output, target_output, gamma=1.3)
        
        # Exact match detection and bonus
        pred_indices = pred_output.argmax(dim=1)
        target_indices = target_output.argmax(dim=1)
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_count = exact_matches.sum()
        # Fixed exact bonus to prevent negative losses in temporal model
        exact_bonus = -exact_matches.mean() * self.weights['exact_match']
        exact_bonus = exact_bonus.clamp(min=-1.5)  # Prevent excessive negative contribution
        
        # CHRONOS-specific: Temporal transformation penalty (should be low for temporal model)
        input_indices = input_grid.argmax(dim=1)
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.weights['transformation']
        
        # CHRONOS-specific: Movement prediction consistency
        movement_loss = 0.0
        if model_outputs and 'movement_params' in model_outputs:
            movement_params = model_outputs['movement_params']  # B, 128
            # Stabilized movement parameters for temporal sequences
            movement_variance = torch.var(movement_params, dim=1).mean().clamp(max=3.0)
            movement_loss = -movement_variance * self.weights['movement']  # Negative to encourage variance
            movement_loss = movement_loss.clamp(min=-0.5)  # Prevent excessive negative contribution
        
        # CHRONOS-specific: Temporal attention consistency
        temporal_loss = 0.0
        if model_outputs and 'attention_weights' in model_outputs:
            attention_weights = model_outputs['attention_weights']  # B, seq_len, seq_len
            # Stabilized temporal attention patterns
            attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
            temporal_loss = attention_entropy.mean().clamp(max=4.0) * self.weights['temporal']
        
        # CHRONOS-specific: Object tracking consistency
        object_tracking_loss = self._object_tracking_loss(pred_output, target_output) * self.weights['object_tracking']
        
        # CHRONOS-specific: Sequence consistency (temporal coherence)
        sequence_consistency_loss = self._sequence_consistency_loss(pred_output, input_grid) * self.weights['sequence_consistency']
        
        # Enhanced color balance for object tracking
        color_balance_loss = self._enhanced_color_balance_loss(pred_output, target_output) * self.weights['color_balance']
        
        # Minimal edge loss (temporal patterns more important than boundaries)
        edge_loss = self._minimal_edge_loss(pred_output, target_output) * self.weights['edge']
        
        # Stabilized total loss with temporal-specific validation
        loss_components = {
            'temporal_focal': temporal_focal_loss,
            'transform': transform_penalty,
            'movement': movement_loss,
            'temporal': temporal_loss,
            'object_tracking': object_tracking_loss,
            'sequence_consistency': sequence_consistency_loss,
            'color_balance': color_balance_loss,
            'edge': edge_loss,
            'exact_bonus': exact_bonus
        }
        
        # Validate each temporal component
        stable_components = []
        for name, component in loss_components.items():
            # Convert to tensor if it's a float/scalar
            if not isinstance(component, torch.Tensor):
                component = torch.tensor(component, device=temporal_focal_loss.device)
            
            # Update the components dict with the tensor version
            loss_components[name] = component
            
            if torch.isnan(component) or torch.isinf(component):
                print(f"⚠️ Invalid {name} loss in temporal model: {component.item():.3f}, skipping")
                stable_components.append(torch.tensor(0.0, device=temporal_focal_loss.device))
            else:
                stable_components.append(component)
        
        total_loss = sum(stable_components)
        
        # Temporal-specific stability checks
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ Total temporal loss invalid, using focal only")
            total_loss = temporal_focal_loss
        
        # Prevent extremely negative losses in temporal sequences
        if total_loss < -3.0:
            print(f"⚠️ Temporal loss too negative ({total_loss.item():.3f}), clamping")
            total_loss = total_loss.clamp(min=-3.0)
        
        return {
            'total': total_loss,
            'temporal_focal': temporal_focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'exact_count': exact_count,
            'movement': movement_loss,
            'temporal': temporal_loss,
            'object_tracking': object_tracking_loss,
            'sequence_consistency': sequence_consistency_loss,
            'color_balance': color_balance_loss,
            'edge': edge_loss
        }
    
    def _temporal_focal_loss(self, pred, target, gamma=1.3):
        """Focal loss optimized for temporal patterns"""
        target_idx = target.argmax(dim=1)
        ce_loss = F.cross_entropy(pred, target_idx, reduction='none')
        
        # Temporal-specific focal weighting
        pt = torch.exp(-ce_loss)
        temporal_weights = torch.ones_like(target_idx, dtype=torch.float)
        # Weight changed pixels more heavily for temporal learning
        changed_mask = target_idx != target.argmax(dim=1)
        temporal_weights[changed_mask] = 1.8
        
        focal = (1 - pt) ** gamma * ce_loss * temporal_weights
        return focal.mean()
    
    def _object_tracking_loss(self, pred, target):
        """Encourage object consistency for tracking"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Simple object consistency check using connected components
        pred_objects = self._get_object_centers(pred_idx)
        target_objects = self._get_object_centers(target_idx)
        
        if len(pred_objects) == 0 or len(target_objects) == 0:
            return torch.tensor(0.0).to(pred.device)
        
        # Compute distance between object centers
        center_distances = []
        for pred_center in pred_objects:
            if target_objects:
                min_dist = min(torch.norm(pred_center - target_center).item() 
                              for target_center in target_objects)
                center_distances.append(min_dist)
        
        if center_distances:
            return torch.tensor(np.mean(center_distances)).to(pred.device)
        return torch.tensor(0.0).to(pred.device)
    
    def _get_object_centers(self, grid_batch):
        """Get centers of non-zero objects in grid batch"""
        centers = []
        for b in range(grid_batch.shape[0]):
            grid = grid_batch[b]
            nonzero_positions = torch.nonzero(grid, as_tuple=False)
            if len(nonzero_positions) > 0:
                center = nonzero_positions.float().mean(dim=0)
                centers.append(center)
        return centers
    
    def _sequence_consistency_loss(self, pred, input_grid):
        """Encourage temporal consistency in predictions"""
        pred_idx = pred.argmax(dim=1)
        input_idx = input_grid.argmax(dim=1)
        
        # Check for smooth transitions (avoid abrupt changes)
        pred_gradients = torch.gradient(pred_idx.float(), dim=[1, 2])
        input_gradients = torch.gradient(input_idx.float(), dim=[1, 2])
        
        # Compare gradient magnitudes for smoothness
        pred_grad_mag = sum(g.abs().mean() for g in pred_gradients)
        input_grad_mag = sum(g.abs().mean() for g in input_gradients)
        
        return F.mse_loss(pred_grad_mag, input_grad_mag)
    
    def _enhanced_color_balance_loss(self, pred, target):
        """Enhanced color distribution preservation for object tracking"""
        pred_colors = F.softmax(pred, dim=1).sum(dim=[2, 3])  # B, 10
        target_colors = target.sum(dim=[2, 3])  # B, 10
        
        # Normalize distributions
        pred_colors = pred_colors / (pred_colors.sum(dim=1, keepdim=True) + 1e-8)
        target_colors = target_colors / (target_colors.sum(dim=1, keepdim=True) + 1e-8)
        
        # KL divergence for better color distribution matching
        return F.kl_div(torch.log(pred_colors + 1e-8), target_colors, reduction='batchmean')
    
    def _minimal_edge_loss(self, pred, target):
        """Minimal edge awareness for temporal model"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Simple boundary detection
        pred_diff = torch.abs(pred_idx[:, 1:, :] - pred_idx[:, :-1, :]).float()
        target_diff = torch.abs(target_idx[:, 1:, :] - target_idx[:, :-1, :]).float()
        
        return F.mse_loss(pred_diff, target_diff)


def custom_collate_fn(batch, stage=0):
    """CHRONOS-optimized collate function with stage-specific grid sizes"""
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


def train_chronos_specialized():
    """Main CHRONOS specialized training function"""
    print("⏰ Starting CHRONOS Specialized Training")
    print("=" * 60)
    
    # Initialize model with maximum grid size from final stage
    max_grid_size = STAGE_CONFIG[7]['max_grid_size']  # Final stage size (30x30)
    model = EnhancedChronosNet(
        max_grid_size=max_grid_size,
        hidden_dim=CHRONOS_CONFIG['hidden_dim']
    ).to(device)
    
    print(f"📊 CHRONOS Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize all AutomataNexus training systems
    systems = {}
    
    # MEPT System
    if USE_MEPT:
        mept_components = create_mept_system(
            capacity=60000,  # Medium for temporal patterns
            pattern_bank_size=12000,
            transformation_penalty=CHRONOS_CONFIG['transform_penalty'],
            exact_match_bonus=CHRONOS_CONFIG['exact_match_bonus']
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
    loss_fn = ChronosSpecializedLoss().to(device)
    
    # Optimizer - Adam for temporal sequence learning
    optimizer = optim.Adam(
        model.parameters(),
        lr=CHRONOS_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CHRONOS_CONFIG['epochs_per_stage']
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Data directory
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
    # Training metrics
    best_exact = 0.0
    global_epoch = 0
    
    # 8-Stage Progressive Curriculum Training Loop
    stage_metrics = []  # Track learning progression
    
    for stage in range(CHRONOS_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        synthesis_ratio = stage_config['synthesis_ratio']
        
        print(f"\n⏰ CHRONOS Stage {stage}: {grid_size}x{grid_size} Temporal Sequence Analysis")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {synthesis_ratio*100:.0f}% | LEAP: {stage_config['leap_complexity']}")
        print("=" * 60)
        
        # Create curriculum dataset with stage-specific grid size
        dataset = CurriculumMegaScaleDataset(
            DATA_DIR,
            curriculum_stage=min(stage, 2),  # Cap at stage 2 for compatibility
            use_arc_synthesis=True,
            synthesis_ratio=synthesis_ratio
        )
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Apply CHRONOS-specialized dataset wrapper
        if USE_MEPT and 'replay_buffer' in systems:
            train_dataset = ChronosSpecializedDataset(
                train_dataset, 
                systems['replay_buffer'],
                replay_ratio=0.3 if stage == 0 else 0.2
            )
        
        # Data loaders with stage-specific grid sizes
        train_loader = DataLoader(
            train_dataset,
            batch_size=CHRONOS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CHRONOS_CONFIG['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, stage)
        )
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # Exact match injection for Stage 0 only
        exact_dataset = None
        if stage_config['exact_injection'] and USE_EXACT_BOOST:
            try:
                exact_dataset = ExactMatchBoostDataset(1300, fixed_size=grid_size)
                print(f"✅ Stage {stage} temporal-focused exact match injection dataset created ({grid_size}x{grid_size})")
            except Exception as e:
                print(f"⚠️ Could not create exact match dataset: {e}")
        
        # Stage training loop
        for epoch in range(CHRONOS_CONFIG['epochs_per_stage']):
            global_epoch += 1
            
            # Exact match injection training (Stage 0 only, first 5 epochs)
            if exact_dataset and stage == 0 and epoch < 5:  # Only first 5 epochs of Stage 0
                model = inject_exact_match_training(
                    model, device=device,
                    num_epochs=1,
                    target_accuracy=93.0  # Slightly lower for complex temporal patterns
                )
                print(f"⏰ Temporal injection completed - Epoch {global_epoch}")
            
            # Main training
            model.train()
            train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
            
            pbar = tqdm(train_loader, desc=f"CHRONOS Stage {stage}, Epoch {epoch+1}", 
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
                    # CHRONOS forward pass - expects list of tensors for sequence
                    model_outputs = model([input_grids])
                    pred_output = model_outputs['predicted_output']
                    
                    # Specialized loss
                    losses = loss_fn(pred_output, output_grids, input_grids, model_outputs)
                    loss = losses['total'] / CHRONOS_CONFIG['gradient_accumulation']
                    
                    # CHRONOS-specific temporal loss validation
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ Skipping invalid temporal loss at batch {batch_idx}")
                        continue
                    
                    # Skip if loss is extremely negative (temporal instability)
                    if loss < -4.0:
                        print(f"⚠️ Skipping extremely negative temporal loss: {loss.item():.3f}")
                        continue
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % CHRONOS_CONFIG['gradient_accumulation'] == 0:
                    # Pre-check for explosive gradients before unscaling
                    pre_norm = sum(p.grad.data.norm(2)**2 for p in model.parameters() if p.grad is not None)**0.5
                    if pre_norm > 20.0:  # Temporal-specific threshold
                        print(f"⚠️ CHRONOS: Pre-norm {pre_norm:.2f} too high, skipping update")
                        optimizer.zero_grad()
                        scaler.update()
                        continue
                    
                    scaler.unscale_(optimizer)
                    # CHRONOS-specific temporal gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                    if grad_norm > 3.0:
                        print(f"⚠️ Large gradient norm in CHRONOS: {grad_norm:.2f}, clipped to 0.3")
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
                    'temporal': f"{losses['temporal']:.3f}",
                    'movement': f"{losses['movement']:.3f}"
                })
                
                # LEAP training integration with stage-specific complexity
                if USE_LEAP and 'leap_trainer' in systems and batch_idx % 3 == 0:
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
                        leap_model_outputs = model([leap_input_oh])
                        leap_pred = leap_model_outputs['predicted_output']
                        leap_losses = loss_fn(leap_pred, leap_output_oh, leap_input_oh, leap_model_outputs)
                        leap_loss = leap_losses['total'] / CHRONOS_CONFIG['gradient_accumulation']
                    
                    scaler.scale(leap_loss).backward()
                    
                    # Update LEAP pattern statistics
                    systems['leap_trainer'].update_pattern_stats(
                        leap_batch['pattern_types'], leap_pred, leap_output_oh
                    )
                
                # MEPT experience collection (temporal transformations)
                if USE_MEPT and 'replay_buffer' in systems:
                    pred_indices = pred_output.argmax(dim=1)
                    target_indices = output_grids.argmax(dim=1)
                    exact_matches = (pred_indices == target_indices).all(dim=[1,2])
                    
                    # Also collect temporal patterns with good accuracy
                    temporal_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
                    good_temporal_matches = temporal_accuracy > 0.75
                    
                    for i in range(input_grids.size(0)):
                        if exact_matches[i] or good_temporal_matches[i]:
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
                            model_outputs = model([input_grids])
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
                print(f"\n⏰ CHRONOS Epoch {global_epoch} (Stage {stage}, {grid_size}x{grid_size}):")
                print(f"   ⏰ GRID SIZE: {grid_size}x{grid_size} | TEMPORAL LEARNING: {trend_icon} {trend_text}")
                print(f"   🎯 Train: {train_exact_pct:.2f}% exact, Loss: {train_loss:.3f}")
                print(f"   🎯 Val: {val_exact_pct:.2f}% exact, Loss: {val_loss:.3f}, Pixel: {val_pixel_acc:.1f}%")
                
                # Stage progress indicator
                stage_progress = (epoch + 1) / CHRONOS_CONFIG['epochs_per_stage'] * 100
                total_progress = (stage * CHRONOS_CONFIG['epochs_per_stage'] + epoch + 1) / (CHRONOS_CONFIG['curriculum_stages'] * CHRONOS_CONFIG['epochs_per_stage']) * 100
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
                        print(f"   ⚠️ LEAP: Temporal pattern learning stuck at 0.0% - needs complexity adjustment for {grid_size}x{grid_size} grids")
                
                # Learning status analysis
                if val_exact_pct >= 5.0:
                    status = f"🏆 EXCELLENT temporal learning for {grid_size}x{grid_size} grids!"
                elif val_exact_pct >= 1.0:
                    status = f"📈 GOOD temporal progress on {grid_size}x{grid_size} sequences"
                elif val_exact_pct >= 0.1:
                    status = f"🔄 LEARNING temporal basics for {grid_size}x{grid_size} grids"
                else:
                    status = f"⚠️ Still learning {grid_size}x{grid_size} temporal fundamentals"
                print(f"   📊 STATUS: {status}")
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    model_path = f'/content/AutomataNexus_Olympus_AGI2/src/models/chronos_specialized_best.pth'
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'grid_size': grid_size,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'config': CHRONOS_CONFIG,
                        'stage_config': STAGE_CONFIG
                    }, model_path)
                    print(f"   💾 NEW BEST: {val_exact_pct:.2f}% temporal exact match saved!")
    
    # Final training summary
    print(f"\n🎉 CHRONOS 8-Stage Training Complete!")
    print(f"   🏆 Best exact match: {best_exact:.2f}%")
    print(f"   📏 Stages completed: {CHRONOS_CONFIG['curriculum_stages']} (6x6 → 30x30 grids)")
    print(f"   📊 Total epochs: {global_epoch}")
    
    # Stage-by-stage progress summary
    if stage_metrics:
        print(f"\n📏 Stage-by-stage Temporal Learning Progression:")
        for i, stage_config in enumerate(STAGE_CONFIG.values()):
            stage_final = [m for m in stage_metrics if m['stage'] == i]
            if stage_final:
                final_exact = stage_final[-1]['val_exact']
                grid_size = stage_config['max_grid_size']
                print(f"   Stage {i} ({grid_size}x{grid_size}): {final_exact:.2f}% temporal exact match")
    
    return model, best_exact


if __name__ == "__main__":
    train_chronos_specialized()