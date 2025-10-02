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
import time
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
from src.dsl.chronos_dsl import CHRONOSDSLTraining, CHRONOSDSLGenerator
from src.program_synthesis.synthesis_integration import LightweightProgramSynthesizer, ProgramSynthesisDataGenerator

# PRISM System - Use CHRONOS-specific version
try:
    from src.training_systems.chronos_prism import create_chronos_prism_system
    CHRONOS_PRISM_AVAILABLE = True
except ImportError:
    CHRONOS_PRISM_AVAILABLE = False
    print("⚠️ CHRONOS-specific PRISM not available")
    # Fallback to generic version
    try:
        from src.program_synthesis.prism_system import PRISMSynthesizer, create_prism_system
        PRISM_AVAILABLE = True
    except ImportError:
        PRISM_AVAILABLE = False
        print("⚠️ Generic PRISM not available")

# MEPT and LEAP Systems - Use CHRONOS-specific versions
try:
    from src.training_systems.chronos_mept import create_chronos_mept_system, ChronosMEPTLoss
    from src.training_systems.chronos_leap import create_chronos_leap_system, ChronosLEAPTrainer
    CHRONOS_MEPT_LEAP_AVAILABLE = True
except ImportError:
    CHRONOS_MEPT_LEAP_AVAILABLE = False
    print("⚠️ CHRONOS-specific MEPT/LEAP not available")
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
USE_MEPT = True and (CHRONOS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_LEAP = True and (CHRONOS_MEPT_LEAP_AVAILABLE or MEPT_LEAP_AVAILABLE)
USE_PRISM = True and (CHRONOS_PRISM_AVAILABLE or PRISM_AVAILABLE)
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
            len(self.replay_buffer.temporal_memory) > 0):
            # Handle different replay buffer APIs
            if hasattr(self.replay_buffer, 'replay_experiences'):
                # CHRONOS-specific memory system
                experiences = self.replay_buffer.replay_experiences(1)
            else:
                # Generic replay buffer
                experiences = self.replay_buffer.sample(1, exact_ratio=0.7)  # Favor temporal matches
            if experiences:
                exp = experiences[0]
                if hasattr(exp, 'sequence'):
                    # CHRONOS TemporalMemory object
                    sequence = exp.sequence
                    input_tensor = sequence[0] if len(sequence) > 0 else torch.zeros(1, 10, 6, 6)
                    output_tensor = sequence[-1] if len(sequence) > 1 else input_tensor
                else:
                    # Generic replay buffer format - handle key variations
                    input_tensor = exp.get('input', exp.get('inputs'))
                    output_tensor = exp.get('output', exp.get('outputs'))
                
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
    """CHRONOS-specific loss - SIMPLIFIED FOR STABILITY"""
    def __init__(self):
        super().__init__()
        self.weights = {
            'reconstruction': 1.0,
            'transformation': CHRONOS_CONFIG['transform_penalty'],
            'exact_match': CHRONOS_CONFIG['exact_match_bonus']
        }
        
    def forward(self, pred_output, target_output, input_grid, model_outputs=None):
        """SIMPLIFIED CHRONOS loss - based on working IRIS approach"""
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
            'temporal': torch.tensor(0.0, device=total_loss.device),  # Dummy for display
            'movement': torch.tensor(0.0, device=total_loss.device)  # Dummy for display
        }
    
    def _focal_loss(self, pred, target, gamma=2.0):
        """Simple focal loss like IRIS"""
        target_idx = target.argmax(dim=1)
        ce_loss = F.cross_entropy(pred, target_idx, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
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
    
    # MEPT System - Use CHRONOS-specific if available
    if USE_MEPT:
        if CHRONOS_MEPT_LEAP_AVAILABLE:
            mept_components = create_chronos_mept_system(
                memory_capacity=60000,
                transformation_penalty=CHRONOS_CONFIG['transform_penalty'],
                exact_match_bonus=CHRONOS_CONFIG['exact_match_bonus']
            )
            systems['replay_buffer'] = mept_components['memory_system']
            systems['loss_fn'] = mept_components['loss_fn']  # CHRONOS-specific loss
            print("✅ CHRONOS-specific MEPT system initialized")
        elif MEPT_LEAP_AVAILABLE:
            mept_components = create_mept_system(
                capacity=60000,  # Medium for temporal patterns
                pattern_bank_size=12000,
                transformation_penalty=CHRONOS_CONFIG['transform_penalty'],
                exact_match_bonus=CHRONOS_CONFIG['exact_match_bonus']
            )
            systems['replay_buffer'] = mept_components['replay_buffer']
            systems['pattern_bank'] = mept_components['pattern_bank']
            print("✅ Generic MEPT system initialized")
    
    # LEAP System - Use CHRONOS-specific if available 
    if USE_LEAP:
        if CHRONOS_MEPT_LEAP_AVAILABLE:
            leap_components = create_chronos_leap_system(
                grid_size=STAGE_CONFIG[7]['max_grid_size']
            )
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['generator']
            print("✅ CHRONOS-specific LEAP system initialized")
        elif MEPT_LEAP_AVAILABLE:
            leap_components = create_leap_system(device)
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['pattern_generator']
            systems['weak_detector'] = leap_components['detector']
            print("✅ Generic LEAP system initialized")
    
    # PRISM System - Use CHRONOS-specific if available
    if USE_PRISM:
        if CHRONOS_PRISM_AVAILABLE:
            prism_components = create_chronos_prism_system(
                hidden_dim=CHRONOS_CONFIG['hidden_dim']
            )
            systems['prism_synthesizer'] = prism_components['synthesizer']
            systems['prism_evaluator'] = prism_components['evaluator']
            print("✅ CHRONOS-specific PRISM system initialized")
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
        # Use CHRONOS-specific MEPT loss if available
        loss_fn = systems['loss_fn'].to(device)
        print("✅ Using CHRONOS-specific MEPT loss function")
    else:
        loss_fn = ChronosSpecializedLoss().to(device)
        print("✅ Using ChronosSpecializedLoss")
    
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
    start_stage = 0
    
    # Check for existing checkpoint
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    checkpoint_path = f'{models_dir}/chronos_checkpoint.pt'
    best_model_path = f'{models_dir}/chronos_best.pt'
    
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
    
    for stage in range(start_stage, CHRONOS_CONFIG['curriculum_stages']):
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        synthesis_ratio = stage_config['synthesis_ratio']
        
        print(f"\n⏰ CHRONOS Stage {stage}: {grid_size}x{grid_size} Temporal Sequence Analysis")
        print(f"   📏 Grid Size: {grid_size}x{grid_size} | Synthesis: {synthesis_ratio*100:.0f}% | LEAP: {stage_config['leap_complexity']}")
        print("=" * 60)
        
        # Add DSL-generated samples for CHRONOS's temporal patterns
        print(f"🔧 Adding CHRONOS-specific DSL samples for stage {stage}...")
        dsl_samples = CHRONOSDSLTraining.create_chronos_dsl_samples(curriculum_stage=stage)
        print(f"✅ Added {len(dsl_samples)} CHRONOS DSL samples for {grid_size}x{grid_size} temporal grids")
        
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
        
        # DISABLE the specialized dataset wrapper - it's causing DataLoader hanging
        # The ChronosSpecializedDataset replay buffer sampling is the source of the hang
        # if USE_MEPT and 'replay_buffer' in systems:
        #     train_dataset = ChronosSpecializedDataset(
        #         train_dataset, 
        #         systems['replay_buffer'],
        #         replay_ratio=0.3 if stage == 0 else 0.2
        #     )
        
        # Data loaders with stage-specific grid sizes
        # Use 0 workers to avoid hanging issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=CHRONOS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,  # Disable when using 0 workers
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            persistent_workers=False  # Can't use with 0 workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CHRONOS_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False,  # Disable when using 0 workers
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
            
            # Exact match injection training (Stage 0 only, FIRST EPOCH ONLY)
            if exact_dataset and stage == 0 and epoch == 0:  # ONLY FIRST EPOCH
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
                # CHRONOS DSL augmentation
                if batch_idx % 5 == 0 and dsl_samples:  # Every 5th batch
                    try:
                        batch = CHRONOSDSLTraining.augment_batch_with_chronos_dsl(
                            batch, curriculum_stage=stage, dsl_ratio=0.3
                        )
                    except Exception as e:
                        print(f"⚠️ DSL augmentation error at batch {batch_idx}: {e}")
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
                    # CHRONOS forward pass - use simple single frame mode for stability
                    model_outputs = model([input_grids])  # This will trigger _forward_single
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
                if 'temporal' in losses:
                    postfix_dict['temporal'] = f"{losses['temporal'].item():.3f}"
                elif 'focal' in losses:
                    postfix_dict['focal'] = f"{losses['focal'].item():.3f}"
                    
                if 'movement' in losses:
                    postfix_dict['movement'] = f"{losses['movement'].item():.3f}"
                elif 'transform' in losses:
                    postfix_dict['transform'] = f"{losses['transform'].item():.3f}"
                
                pbar.set_postfix(postfix_dict)
                
                # LEAP training integration with stage-specific complexity
                if USE_LEAP and 'leap_trainer' in systems and batch_idx % 3 == 0:
                    # Adjust LEAP complexity based on current stage
                    leap_complexity = stage_config['leap_complexity']
                    leap_grid_size = min(grid_size, 12)  # Cap LEAP at 12x12 for stability
                    
                    # Generate LEAP batch with stage-specific parameters for CHRONOS
                    if CHRONOS_MEPT_LEAP_AVAILABLE and hasattr(systems['leap_trainer'], 'generate_leap_batch'):
                        leap_batch = systems['leap_trainer'].generate_leap_batch(
                            batch_size=max(32, 64 - stage*8),  # Reduce batch size for larger grids
                            stage=stage
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
                            if hasattr(systems['replay_buffer'], 'store_sequence'):
                                # CHRONOS-specific memory system - store as temporal sequence
                                sequence = [input_grids[i], output_grids[i]]
                                pattern_type = 'temporal_transformation'
                                additional_features = {
                                    'accuracy': temporal_accuracy[i].item(),
                                    'exact_match': exact_matches[i].item(),
                                    'loss': losses['total'].item()
                                }
                                systems['replay_buffer'].store_sequence(
                                    sequence, pattern_type, additional_features
                                )
                            else:
                                # Generic replay buffer
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
                    if hasattr(systems['replay_buffer'], 'get_memory_statistics'):
                        # CHRONOS-specific memory system
                        buffer_stats = systems['replay_buffer'].get_memory_statistics()
                        total_sequences = buffer_stats['total_sequences']
                        memory_util = buffer_stats['memory_utilization'] * 100
                        print(f"   📊 MEPT: {total_sequences:,} temporal sequences | {memory_util:.1f}% memory utilization")
                    else:
                        # Generic replay buffer
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
                
                # Create models directory if needed
                models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
                os.makedirs(models_dir, exist_ok=True)
                
                # Save checkpoint every validation
                checkpoint_path = f'{models_dir}/chronos_checkpoint.pt'
                torch.save({
                    'epoch': global_epoch,
                    'stage': stage,
                    'grid_size': grid_size,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_exact': val_exact_pct,
                    'best_exact': best_exact,
                    'val_loss': val_loss,
                    'config': CHRONOS_CONFIG,
                    'stage_config': STAGE_CONFIG
                }, checkpoint_path)
                
                # Save best model
                if val_exact_pct > best_exact:
                    best_exact = val_exact_pct
                    best_model_path = f'{models_dir}/chronos_best.pt'
                    torch.save({
                        'epoch': global_epoch,
                        'stage': stage,
                        'grid_size': grid_size,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': val_exact_pct,
                        'val_loss': val_loss,
                        'config': CHRONOS_CONFIG,
                        'stage_config': STAGE_CONFIG
                    }, best_model_path)
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