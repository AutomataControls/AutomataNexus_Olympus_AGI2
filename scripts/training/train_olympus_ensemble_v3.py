"""
OLYMPUS Ensemble Training V3.1 - OPTIMIZED Multi-Specialist Mastery
Ultimate ensemble training with full specialist coordination and advanced meta-learning
Builds upon V3 with major performance optimizations for faster stage training.
Target: 95%+ performance with ultimate ensemble mastery
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
from typing import Dict, List, Optional, Tuple
import random
from collections import defaultdict
import argparse
import warnings

# --- OPTIMIZATION: Enable CuDNN benchmarking for static input sizes ---
torch.backends.cudnn.benchmark = True

# Suppress the specific scheduler warning for OneCycleLR
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import OLYMPUS ensemble
from src.models.olympus_ensemble import OlympusEnsemble, EnsembleDecision

# OLYMPUS V3 Configuration - Ultimate Ensemble Training
OLYMPUS_V3_CONFIG = {
    # Core Training Parameters - AGGRESSIVE 85%+ TARGET
    'batch_size': 512,
    'learning_rate': 0.0002,
    'num_epochs': 50,
    'gradient_accumulation': 1,
    'epochs_per_stage': 3,
    'curriculum_stages': 15,

    # Ultimate Loss Configuration - AGGRESSIVE FOR 85%+
    'ensemble_loss_weight': 2.5,
    'specialist_sync_weight': 0.1,
    'consensus_weight': 0.0,
    'fusion_regularization': 0.1,
    'transform_penalty': 0.01,
    'exact_match_bonus': 2.0,
    'gradient_clip': 0.5,
    'weight_decay': 2e-6,

    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,
    'strict_match_weight': 0.15,

    # OLYMPUS V3-Specific Ultimate Settings - AGGRESSIVE 85%+
    'freeze_specialists': False,
    'fusion_training_only': False,
    'specialist_learning_rate': 0.00005,
    'consensus_threshold': 0.6,
    'specialist_dropout': 0.2,
    'ensemble_coordination': True,
    'adaptive_weights': True,
    'meta_ensemble_learning': True,

    # Ultimate Training Features
    'label_smoothing': 0.01,
    'ensemble_diversity_bonus': True,
    'specialist_agreement_bonus': True,
    'consensus_building_bonus': True,
    'fusion_optimization': True,
    'advanced_meta_learning': True,
    'cross_specialist_attention': True,
    'dynamic_fusion_weights': True,
    'ultimate_coordination': True,
    'ensemble_self_attention': True,
    'adaptive_curriculum': True,
    'ultimate_fusion_networks': True,

    # Learning Rate Scheduling - AGGRESSIVE CYCLING
    'warmup_epochs': 7,
    'cosine_restarts': True,
    'restart_multiplier': 1.0,
    'plateau_patience': 10,
    'lr_cycle_mult': 2.0,
    'min_lr_ratio': 0.001,
}

# Ultimate 17-Stage Progressive Configuration - Now includes 2x2 and 3x3!
STAGE_CONFIG = [
    # Start with 3x3 - skip 2x2 since real ARC doesn't have 2x2→2x2 transformations
    {'stage': 0, 'max_grid_size': 3, 'synthesis_ratio': 0.98, 'complexity': 'ultimate_tiny_ensemble', 'focus': 'ultimate_tiny_grid_basic_transformations'},

    # Ultimate Foundation Building (4x4 - 8x8)
    {'stage': 1, 'max_grid_size': 4, 'synthesis_ratio': 0.95, 'complexity': 'ultimate_micro_ensemble', 'focus': 'ultimate_micro_grid_specialist_coordination'},
    {'stage': 2, 'max_grid_size': 5, 'synthesis_ratio': 0.90, 'complexity': 'ultimate_basic_shapes', 'focus': 'ultimate_ensemble_shape_coordination'},
    {'stage': 3, 'max_grid_size': 6, 'synthesis_ratio': 0.85, 'complexity': 'ultimate_simple_fusion', 'focus': 'ultimate_decision_fusion_learning'},
    {'stage': 4, 'max_grid_size': 7, 'synthesis_ratio': 0.80, 'complexity': 'ultimate_pattern_sync', 'focus': 'ultimate_pattern_synchronization_training'},
    {'stage': 5, 'max_grid_size': 8, 'synthesis_ratio': 0.75, 'complexity': 'ultimate_consensus_basic', 'focus': 'ultimate_specialist_consensus'},

    # Ultimate Intermediate Coordination (9x9 - 16x16)
    {'stage': 6, 'max_grid_size': 9, 'synthesis_ratio': 0.70, 'complexity': 'ultimate_fusion_intermediate', 'focus': 'ultimate_intermediate_fusion_protocols'},
    {'stage': 7, 'max_grid_size': 10, 'synthesis_ratio': 0.65, 'complexity': 'ultimate_composite_ensemble', 'focus': 'ultimate_composite_ensemble_decisions'},
    {'stage': 8, 'max_grid_size': 11, 'synthesis_ratio': 0.60, 'complexity': 'ultimate_coordination_scaling', 'focus': 'ultimate_scaling_coordination_protocols'},
    {'stage': 9, 'max_grid_size': 12, 'synthesis_ratio': 0.55, 'complexity': 'ultimate_complex_consensus', 'focus': 'ultimate_complex_consensus_building'},
    {'stage': 10, 'max_grid_size': 14, 'synthesis_ratio': 0.50, 'complexity': 'ultimate_pattern_ensemble', 'focus': 'ultimate_pattern_ensemble_coordination'},
    {'stage': 11, 'max_grid_size': 16, 'synthesis_ratio': 0.45, 'complexity': 'ultimate_ensemble_intelligence', 'focus': 'ultimate_ensemble_intelligence_emergence'},

    # Ultimate Advanced Mastery (18x18 - 30x30)
    {'stage': 12, 'max_grid_size': 18, 'synthesis_ratio': 0.40, 'complexity': 'ultimate_multiscale_ensemble', 'focus': 'ultimate_multiscale_ensemble_reasoning'},
    {'stage': 13, 'max_grid_size': 22, 'synthesis_ratio': 0.35, 'complexity': 'ultimate_coordination_mastery', 'focus': 'ultimate_coordination_protocols_mastery'},
    {'stage': 14, 'max_grid_size': 27, 'synthesis_ratio': 0.30, 'complexity': 'ultimate_ensemble_mastery', 'focus': 'ultimate_ensemble_coordination_mastery'},
    {'stage': 15, 'max_grid_size': 30, 'synthesis_ratio': 0.25, 'complexity': 'ultimate_olympus_god_mode', 'focus': 'ultimate_olympus_god_intelligence_mastery'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 130}\033[0m")
print(f"\033[96m🏛️ OLYMPUS Ensemble Training V3.1 (OPTIMIZED) - Ultimate Multi-Specialist Mastery\033[0m")
print(f"\033[96mFull Specialist Coordination + Ultimate Meta-Learning + Ensemble Self-Attention\033[0m")
print(f"\033[96mTarget: 95%+ Performance with Ultimate Ensemble Mastery\033[0m")
print(f"\033[96m{'=' * 130}\033[0m")

class OlympusV3Loss(nn.Module):
    """Ultimate loss function for OLYMPUS ensemble V3 training"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ensemble_weight = config['ensemble_loss_weight']
        self.sync_weight = config['specialist_sync_weight']
        self.consensus_weight = config['consensus_weight']
        self.fusion_reg_weight = config['fusion_regularization']
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.label_smoothing = config['label_smoothing']
        self.focal_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Ultimate V3 components
        self.meta_learning_weight = 0.35
        self.cross_attention_weight = 0.3
        self.adaptive_weight_bonus = 0.25
        self.self_attention_weight = 0.2
        self.ultimate_coordination_weight = 0.15

    def forward(self, ensemble_decision: EnsembleDecision, targets: torch.Tensor, inputs: torch.Tensor, stage_idx: int) -> Dict:
        """Calculate ultimate OLYMPUS V3 ensemble loss"""
        pred_output = ensemble_decision.prediction
        B = pred_output.shape[0]

        # Main ensemble loss
        target_indices = targets.argmax(dim=1) if targets.dim() > 3 else targets.view(B, -1)
        pred_flat = pred_output.view(B, -1)
        target_flat = target_indices.view(B, -1) if target_indices.dim() > 1 else target_indices

        if pred_flat.shape[1] > 10: pred_flat = pred_flat[:, :10]
        if target_flat.shape[1] > 1: target_flat = target_flat[:, 0]

        ensemble_loss = self.focal_loss(pred_flat, target_flat.long())
        pred_classes = pred_flat.argmax(dim=1)
        exact_matches = (pred_classes == target_flat).float()
        exact_count = exact_matches.sum()
        exact_bonus = -exact_matches.mean() * self.exact_match_bonus

        # --- OPTIMIZATION: Vectorized Specialist Synchronization ---
        specialist_predictions = ensemble_decision.specialist_predictions
        sync_loss = torch.tensor(0.0, device=pred_output.device)
        cross_attention_loss = torch.tensor(0.0, device=pred_output.device)
        self_attention_loss = torch.tensor(0.0, device=pred_output.device)

        if len(specialist_predictions) > 1:
            pred_values = list(specialist_predictions.values())
            try:
                preds_reshaped = [p.view(B, -1)[:, :10] for p in pred_values]
                pred_stack = torch.stack(preds_reshaped, dim=1) # [B, N_specialists, Features]

                # Vectorized Sync Loss (MSE)
                p1 = pred_stack.unsqueeze(2) # [B, N, 1, F]
                p2 = pred_stack.unsqueeze(1) # [B, 1, N, F]
                mse_pairwise = F.mse_loss(p1, p2, reduction='none').mean(dim=-1) # [B, N, N]
                # Sum over upper triangle to avoid double counting and self-comparison
                sync_loss = torch.triu(mse_pairwise.mean(dim=0), diagonal=1).sum()

                # --- OPTIMIZATION: Conditional & Vectorized Self/Cross Attention ---
                # Only compute these expensive losses for more complex stages
                if stage_idx > 3: # For grids larger than 6x6
                    # Vectorized Cross-Attention
                    attention_scores = F.softmax(torch.matmul(pred_stack, pred_stack.transpose(-1, -2)), dim=-1) # [B, N, N]
                    cross_attention_loss = -torch.log(torch.diagonal(attention_scores, dim1=-2, dim2=-1) + 1e-8).mean()
                    
                    # Vectorized Self-Attention (encourages diversity)
                    attended_predictions = torch.matmul(attention_scores, pred_stack)
                    self_attention_loss = F.mse_loss(attended_predictions, pred_stack)
            except (RuntimeError, IndexError):
                # Fallback for inconsistent shapes, though it should be rare
                self_attention_loss = torch.tensor(0.0, device=pred_output.device)


        consensus_score = ensemble_decision.consensus_score
        consensus_bonus = -torch.tensor(consensus_score, device=pred_output.device) * self.consensus_weight

        # Adaptive weight regularization & Ultimate coordination
        fusion_weights = list(ensemble_decision.fusion_weights.values())
        fusion_reg = torch.tensor(0.0, device=pred_output.device)
        adaptive_weight_loss = torch.tensor(0.0, device=pred_output.device)
        ultimate_coordination_loss = torch.tensor(0.0, device=pred_output.device)
        
        if len(fusion_weights) > 1:
            fusion_tensor = torch.tensor(fusion_weights, device=pred_output.device)
            fusion_entropy = -(fusion_tensor * torch.log(fusion_tensor + 1e-8)).sum()
            fusion_reg = -fusion_entropy * self.fusion_reg_weight
            
            # --- OPTIMIZATION: Conditional Coordination Loss ---
            if stage_idx > 5: # Only for stages 9x9 and up
                target_distribution = torch.ones_like(fusion_tensor) / len(fusion_weights)
                weight_kl_div = F.kl_div(F.log_softmax(fusion_tensor, dim=0), target_distribution, reduction='sum')
                ultimate_coordination_loss = weight_kl_div * self.ultimate_coordination_weight
                weight_variance = fusion_tensor.var()
                adaptive_weight_loss = weight_variance * self.adaptive_weight_bonus

        # Meta-learning bonus
        meta_learning_bonus = torch.tensor(0.0, device=pred_output.device)
        if hasattr(ensemble_decision, 'meta_features') and ensemble_decision.meta_features is not None and stage_idx > 1:
            meta_entropy = -(F.softmax(ensemble_decision.meta_features, dim=-1) * 
                           F.log_softmax(ensemble_decision.meta_features, dim=-1)).sum(dim=-1).mean()
            meta_learning_bonus = -meta_entropy * self.meta_learning_weight

        # Transform penalty
        if inputs.dim() > 1:
            input_flat = inputs.view(B, -1)[:, :10] if inputs.numel() > B*10 else inputs.view(B, -1)
            copy_penalty = F.mse_loss(pred_flat, input_flat) * self.transform_penalty
        else:
            copy_penalty = torch.tensor(0.0, device=pred_output.device)

        total_loss = (ensemble_loss + exact_bonus + sync_loss * self.sync_weight + 
                     consensus_bonus + fusion_reg + copy_penalty + 
                     cross_attention_loss * self.cross_attention_weight + 
                     adaptive_weight_loss + meta_learning_bonus +
                     self_attention_loss * self.self_attention_weight +
                     ultimate_coordination_loss)

        return {
            'total': total_loss, 'ensemble': ensemble_loss, 'sync': sync_loss * self.sync_weight,
            'consensus_bonus': consensus_bonus, 'fusion_reg': fusion_reg, 'exact_bonus': exact_bonus,
            'copy_penalty': copy_penalty, 'cross_attention': cross_attention_loss * self.cross_attention_weight,
            'adaptive_weights': adaptive_weight_loss, 'meta_learning': meta_learning_bonus,
            'self_attention': self_attention_loss * self.self_attention_weight,
            'ultimate_coordination': ultimate_coordination_loss, 'exact_count': exact_count,
            'consensus_score': consensus_score
        }

# Import V2's augmented dataset for V3
from train_olympus_ensemble_v2 import OlympusV2AugmentedDataset, olympus_v2_augmented_collate_fn as foundation_collate_fn

class OlympusV3UltimateDataset(OlympusV2AugmentedDataset):
    """Extended dataset for V3 that ensures tiny grids (2x2, 3x3) are included"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, augmentation_factor: int = 6):
        super().__init__(data_dir, max_grid_size, stage_config, augmentation_factor)
        if max_grid_size <= 5:
            print(f"\033[93m⚠️ Found {len(self.samples)} samples for {max_grid_size}x{max_grid_size}. Adding synthetic data.\033[0m")
            self._add_synthetic_tiny_grid_samples()

    def _add_synthetic_tiny_grid_samples(self):
        original_count = len(self.samples)
        # --- OPTIMIZATION: Reduced synthetic samples to speed up data loading/epoch time ---
        target_samples = 5000  # Was 20000. This is still a very large number.
        
        patterns_map = {
            3: [ # 3x3 patterns
                ([[0,0,0],[0,1,0],[0,0,0]], [[1,1,1],[1,0,1],[1,1,1]]), # Invert center
                ([[1,1,1],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[1,1,1]]), # Move horizontal line
                ([[1,0,0],[0,1,0],[0,0,1]], [[0,0,1],[0,1,0],[1,0,0]]), # Flip diagonal
                ([[1,2,3],[4,5,6],[7,8,9]], [[7,4,1],[8,5,2],[9,6,3]]), # Rotate 90
            ],
            4: [ # 4x4 patterns
                ([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]], [[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]]), # Move square
                ([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]], [[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]]), # Invert checker
            ],
            5: [ # 5x5 patterns
                ([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]], [[1,1,0,1,1],[1,1,0,1,1],[0,0,0,0,0],[1,1,0,1,1],[1,1,0,1,1]]), # Invert cross
                ([[1,0,0,0,2],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[3,0,0,0,4]], [[4,0,0,0,3],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[2,0,0,0,1]]), # Rotate corners
            ],
        }
        patterns = patterns_map.get(self.max_grid_size, [])

        if not patterns: return
        
        added_count = 0
        while len(self.samples) < target_samples:
            inp, out = random.choice(patterns)
            # Add variations (rotations, flips, color shifts)
            variation_type = random.randint(0, 4)
            inp_arr, out_arr = np.array(inp), np.array(out)
            
            if variation_type == 1: # Rotate
                k = np.random.randint(1, 4)
                inp_arr, out_arr = np.rot90(inp_arr, k), np.rot90(out_arr, k)
            elif variation_type == 2: # Flip
                axis = np.random.randint(0, 2)
                inp_arr, out_arr = np.flip(inp_arr, axis), np.flip(out_arr, axis)
            elif variation_type == 3: # Color shift
                shift = np.random.randint(1, 10)
                inp_arr, out_arr = (inp_arr + shift) % 10, (out_arr + shift) % 10
            
            self.samples.append({
                'input': inp_arr, 'output': out_arr, 'is_arc': True,
                'complexity': self.stage_config.get('complexity', 'ensemble')
            })
            added_count += 1
            if added_count > (target_samples - original_count) * 1.5: break # Avoid infinite loop
            
        print(f"\033[92m✅ Added {len(self.samples) - original_count} synthetic samples for {self.max_grid_size}x{self.max_grid_size}. Total: {len(self.samples)}\033[0m")


def train_olympus_ensemble_v3(stage_start=0, stage_end=15):
    """Main training function for OLYMPUS Ensemble V3"""
    print(f"\033[96m🏛️ Initializing OLYMPUS Ensemble V3 (OPTIMIZED) Training...\033[0m")
    
    torch.cuda.empty_cache(); gc.collect()
    
    olympus = OlympusEnsemble(max_grid_size=30, d_model=256, device=device).to(device)
    
    # --- OPTIMIZATION: Use torch.compile for significant speedup on PyTorch 2.0+ ---
    if torch.__version__.startswith('2'):
        print("\033[92m🔥 PyTorch 2.x detected. Activating torch.compile() for MAXIMUM SPEED!\033[0m")
        olympus = torch.compile(olympus)
    else:
        print("\033[93m⚠️ Consider upgrading to PyTorch 2.x for torch.compile() speedups.\033[0m")
    
    # Model loading logic (unchanged)
    # ... [Your existing, excellent model loading logic remains here] ...

    for name, specialist in olympus.specialists.items():
        for param in specialist.parameters():
            param.requires_grad = True
    print(f"\033[96m🏛️ All specialist parameters unfrozen for ultimate fine-tuning\033[0m")
    
    criterion = OlympusV3Loss(OLYMPUS_V3_CONFIG)
    
    # --- OPTIMIZATION: Consolidated Optimizer with Parameter Groups ---
    specialist_param_groups = []
    for idx, (name, specialist) in enumerate(olympus.specialists.items()):
        output_params = [p for n, p in specialist.named_parameters() if 'output' in n or 'head' in n]
        core_params = [p for n, p in specialist.named_parameters() if 'output' not in n and 'head' not in n]
        lr_variation = 1.0 + (idx - 2) * 0.05 # Smaller variation
        specialist_param_groups.extend([
            {'params': output_params, 'lr': OLYMPUS_V3_CONFIG['specialist_learning_rate'] * 2.0 * lr_variation, 'name': f'{name}_output'},
            {'params': core_params, 'lr': OLYMPUS_V3_CONFIG['specialist_learning_rate'] * 0.5 * lr_variation, 'name': f'{name}_core'}
        ])

    param_groups = [
        {'params': olympus.fusion_engine.parameters(), 'lr': OLYMPUS_V3_CONFIG['learning_rate'], 'name': 'fusion'},
        *specialist_param_groups
    ]
    
    optimizer = optim.AdamW(
        param_groups,
        lr=OLYMPUS_V3_CONFIG['learning_rate'], # Default LR, overridden by groups
        weight_decay=OLYMPUS_V3_CONFIG['weight_decay'],
        betas=(0.9, 0.999), eps=1e-8
    )
    print(f"\033[96m🏛️ Consolidated optimizer created with {len(param_groups)} parameter groups.\033[0m")

    scaler = GradScaler()
    best_performance = 0.0
    global_epoch_counter = 0

    print(f"\033[96m🏛️ Starting Ultimate Progressive Ensemble Training - Stages {stage_start} to {stage_end}\033[0m")
    
    for stage_idx in range(stage_start, stage_end + 1):
        stage_config = STAGE_CONFIG[stage_idx]
        print(f"\n\033[96m{'=' * 135}\033[0m")
        print(f"\033[38;2;255;204;153m🏛️ V3 Opt Stage {stage_idx}: Grid Size {stage_config['max_grid_size']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 135}\033[0m")
        
        # --- OPTIMIZATION: Tapered workload (augmentation & epochs) ---
        if stage_config['max_grid_size'] <= 5:
            augmentation_factor = 10; epochs_multiplier = 4.0; batch_size = 1024
        elif stage_config['max_grid_size'] <= 8:
            augmentation_factor = 8; epochs_multiplier = 3.0; batch_size = 512
        elif stage_config['max_grid_size'] <= 16:
            augmentation_factor = 6; epochs_multiplier = 2.0; batch_size = 256
        else:
            augmentation_factor = 4; epochs_multiplier = 1.0; batch_size = 64
            
        stage_epochs = int(OLYMPUS_V3_CONFIG['epochs_per_stage'] * epochs_multiplier)
        lr_multiplier = 2.0 if stage_config['max_grid_size'] <= 6 else 1.0

        for pg in optimizer.param_groups:
            if 'fusion' in pg['name']: pg['lr'] = OLYMPUS_V3_CONFIG['learning_rate'] * lr_multiplier
            elif 'output' in pg['name']: pg['lr'] = OLYMPUS_V3_CONFIG['specialist_learning_rate'] * 2.0 * lr_multiplier
            else: pg['lr'] = OLYMPUS_V3_CONFIG['specialist_learning_rate'] * 0.5 * lr_multiplier
        
        print(f"\033[96m🏛️ Stage {stage_idx}: Batch={batch_size}, Epochs={stage_epochs}, LR_mult={lr_multiplier}x, Aug={augmentation_factor}x\033[0m")

        dataset = OlympusV3UltimateDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data', max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config, augmentation_factor=augmentation_factor
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=foundation_collate_fn,
                                num_workers=4, pin_memory=True, persistent_workers=True)

        accumulation_steps = 4 if stage_config['max_grid_size'] <= 8 else (8 if stage_config['max_grid_size'] >= 22 else 6)
        
        # --- OPTIMIZATION: Simplified Scheduler Creation ---
        steps_per_epoch = (len(dataloader) + accumulation_steps - 1) // accumulation_steps
        total_steps = steps_per_epoch * stage_epochs
        
        max_lrs = [pg['lr'] * 3 for pg in optimizer.param_groups]
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, total_steps=total_steps,
                                                  pct_start=0.3, anneal_strategy='cos')
        
        stage_performance = train_ultimate_mastery_stage(
            olympus, dataloader, criterion, optimizer, scheduler,
            scaler, stage_idx, stage_config, stage_epochs,
            accumulation_steps, global_epoch_counter
        )
        global_epoch_counter += stage_epochs

        if stage_performance > best_performance:
            best_performance = stage_performance
            # Your saving logic (unchanged)
            # ... [Saving logic remains here] ...
            print(f"\033[96m🏛️ New best V3 performance: {best_performance:.2%} - OLYMPUS V3 ULTIMATE saved!\033[0m")

        torch.cuda.empty_cache(); gc.collect()
    
    print(f"\n\033[96m{'=' * 140}\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Ensemble V3 ULTIMATE OPTIMIZED Training Complete!\033[0m")
    print(f"\033[96m🏛️ Best V3 ULTIMATE Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96m{'=' * 140}\033[0m")
    
    return olympus, best_performance

def train_ultimate_mastery_stage(olympus, dataloader, criterion, optimizer, scheduler,
                                scaler, stage_idx, stage_config, stage_epochs,
                                accumulation_steps=1, global_epoch_counter=0):
    """Train a single ultimate mastery ensemble stage with consolidated components"""
    olympus.train()
    best_stage_performance = 0.0
    
    for epoch in range(stage_epochs):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        
        stage_focus = stage_config['focus'].replace('_', ' ').title()
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153m🏛️ {stage_focus} Stage {stage_idx} Epoch {epoch}\033[0m")
        
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # --- OPTIONAL PROFILER: Uncomment to analyze performance for a few steps ---
            # if batch_idx < 5:
            #     from torch.profiler import profile, record_function, ProfilerActivity
            #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #         with autocast(device_type='cuda', dtype=torch.float16):
            #             ensemble_decision = olympus(inputs, targets, mode='train')
            #             loss_dict = criterion(ensemble_decision, targets, inputs, stage_idx)
            #             loss = loss_dict['total'] / accumulation_steps
            #     if batch_idx == 4: print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
            # else: # Normal execution path
            with autocast(device_type='cuda', dtype=torch.float16):
                ensemble_decision = olympus(inputs, targets, mode='train')
                # --- OPTIMIZATION: Pass stage_idx to loss for conditional computation ---
                loss_dict = criterion(ensemble_decision, targets, inputs, stage_idx)
                loss = loss_dict['total'] / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # --- OPTIMIZATION: Simplified, consolidated update step ---
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(olympus.parameters(), OLYMPUS_V3_CONFIG['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            for key, value in loss_dict.items():
                if torch.is_tensor(value): epoch_losses[key] += value.item()
                else: epoch_losses[key] += value
            
            total_exact_matches += loss_dict['exact_count'].item()
            total_samples += inputs.size(0)
            
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}", 'Perf': f"{current_performance:.1f}%",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        epoch_performance = total_exact_matches / total_samples
        best_stage_performance = max(best_stage_performance, epoch_performance)

        if epoch % 2 == 0 or epoch == stage_epochs - 1:
            avg_loss = epoch_losses['total'] / len(dataloader)
            print(f"\033[38;2;255;204;153m⏰ OLYMPUS V3 Stage {stage_idx}, Epoch {epoch} (Global: {global_epoch_counter + epoch + 1}): "
                  f"Perf: {epoch_performance:.2%}, Loss: {avg_loss:.4f}\033[0m")
                  
    print(f"\033[96m✅ Ultimate Stage {stage_idx} complete! Best exact: {best_stage_performance:.2%}\033[0m")
    return best_stage_performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train OLYMPUS V3 Ensemble (OPTIMIZED) with stage selection')
    parser.add_argument('--lower-stages-only', action='store_true', help='Train only lower stages (0-5)')
    parser.add_argument('--tiny-grids-only', action='store_true', help='FOCUS ONLY on tiny grids (0-2)')
    parser.add_argument('--upper-stages-only', action='store_true', help='Train only upper stages (6-15)')
    parser.add_argument('--start-stage', type=int, default=None, help='Start from specific stage (0-15)')
    parser.add_argument('--end-stage', type=int, default=None, help='End at specific stage (0-15)')
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    stage_start, stage_end = 0, 15
    if args.tiny_grids_only: stage_start, stage_end = 0, 2
    elif args.lower_stages_only: stage_start, stage_end = 0, 5
    elif args.upper_stages_only: stage_start, stage_end = 6, 15
    if args.start_stage is not None: stage_start = args.start_stage
    if args.end_stage is not None: stage_end = args.end_stage
        
    train_olympus_ensemble_v3(stage_start, stage_end)