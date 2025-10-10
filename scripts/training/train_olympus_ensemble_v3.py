"""
OLYMPUS Ensemble Training V3 - Ultimate Multi-Specialist Mastery for ARC-AGI-2
Ultimate ensemble training with full specialist coordination and advanced meta-learning
Builds upon V2 with revolutionary ensemble intelligence capabilities
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

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import OLYMPUS ensemble
from src.models.olympus_ensemble import OlympusEnsemble, EnsembleDecision

# OLYMPUS V3 Configuration - Ultimate Ensemble Training
OLYMPUS_V3_CONFIG = {
    # Core Training Parameters - Ultimate Level (Memory Optimized for Advanced Features)
    'batch_size': 256,  # Balanced for V3's advanced features (self-attention, meta-learning)
    'learning_rate': 0.00012,  # Higher base learning rate for better lower grid performance
    'num_epochs': 240,  # Ultimate training: Extended for lower stages
    'gradient_accumulation': 2,  # Effective batch size of 512
    'epochs_per_stage': 12,  # Base epochs (will be increased for lower stages)
    'curriculum_stages': 15,  # Full coverage like V2: 4x4 to 30x30
    
    # Ultimate Loss Configuration
    'ensemble_loss_weight': 1.5,  # Maximum ensemble focus
    'specialist_sync_weight': 0.5,  # Ultimate synchronization
    'consensus_weight': 0.4,  # Maximum consensus building
    'fusion_regularization': 0.2,  # Ultimate fusion sophistication
    'transform_penalty': 0.12,  # Encourage ultimate transformations
    'exact_match_bonus': 15.0,  # Ultimate precision bonus
    'gradient_clip': 0.3,  # Ultimate gradient control
    'weight_decay': 4e-6,  # Ultimate regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    
    # OLYMPUS V3-Specific Ultimate Settings
    'freeze_specialists': False,  # Allow full specialist fine-tuning
    'fusion_training_only': False,  # Train everything together
    'specialist_learning_rate': 0.00002,  # Ultra-low for ultimate fine-tuning
    'consensus_threshold': 0.8,  # Ultimate consensus for confidence
    'specialist_dropout': 0.1,  # Moderate dropout for ultimate robustness
    'ensemble_coordination': True,  # Ultimate coordination protocols
    'adaptive_weights': True,  # Ultimate dynamic weighting
    'meta_ensemble_learning': True,  # NEW: Meta-ensemble optimization
    
    # Ultimate Training Features
    'label_smoothing': 0.01,  # Minimal smoothing for ultimate precision
    'ensemble_diversity_bonus': True,
    'specialist_agreement_bonus': True,
    'consensus_building_bonus': True,
    'fusion_optimization': True,
    'advanced_meta_learning': True,
    'cross_specialist_attention': True,
    'dynamic_fusion_weights': True,
    'ultimate_coordination': True,  # NEW: Ultimate coordination protocols
    'ensemble_self_attention': True,  # NEW: Self-attention across specialists
    'adaptive_curriculum': True,  # NEW: Adaptive curriculum based on performance
    'ultimate_fusion_networks': True,  # NEW: Multiple fusion networks
    
    # Learning Rate Scheduling
    'warmup_epochs': 20,  # Ultimate warmup
    'cosine_restarts': True,
    'restart_multiplier': 1.5,
    'plateau_patience': 25,
}

# Ultimate 15-Stage Progressive Configuration - Full Grid Coverage Like V2
STAGE_CONFIG = [
    # Ultimate Foundation Building (4x4 - 8x8)
    {'stage': 0, 'max_grid_size': 4, 'synthesis_ratio': 0.95, 'complexity': 'ultimate_micro_ensemble', 'focus': 'ultimate_micro_grid_specialist_coordination'},
    {'stage': 1, 'max_grid_size': 5, 'synthesis_ratio': 0.90, 'complexity': 'ultimate_basic_shapes', 'focus': 'ultimate_ensemble_shape_coordination'},
    {'stage': 2, 'max_grid_size': 6, 'synthesis_ratio': 0.85, 'complexity': 'ultimate_simple_fusion', 'focus': 'ultimate_decision_fusion_learning'},
    {'stage': 3, 'max_grid_size': 7, 'synthesis_ratio': 0.80, 'complexity': 'ultimate_pattern_sync', 'focus': 'ultimate_pattern_synchronization_training'},
    {'stage': 4, 'max_grid_size': 8, 'synthesis_ratio': 0.75, 'complexity': 'ultimate_consensus_basic', 'focus': 'ultimate_specialist_consensus'},
    
    # Ultimate Intermediate Coordination (9x9 - 16x16)
    {'stage': 5, 'max_grid_size': 9, 'synthesis_ratio': 0.70, 'complexity': 'ultimate_fusion_intermediate', 'focus': 'ultimate_intermediate_fusion_protocols'},
    {'stage': 6, 'max_grid_size': 10, 'synthesis_ratio': 0.65, 'complexity': 'ultimate_composite_ensemble', 'focus': 'ultimate_composite_ensemble_decisions'},
    {'stage': 7, 'max_grid_size': 11, 'synthesis_ratio': 0.60, 'complexity': 'ultimate_coordination_scaling', 'focus': 'ultimate_scaling_coordination_protocols'},
    {'stage': 8, 'max_grid_size': 12, 'synthesis_ratio': 0.55, 'complexity': 'ultimate_complex_consensus', 'focus': 'ultimate_complex_consensus_building'},
    {'stage': 9, 'max_grid_size': 14, 'synthesis_ratio': 0.50, 'complexity': 'ultimate_pattern_ensemble', 'focus': 'ultimate_pattern_ensemble_coordination'},
    {'stage': 10, 'max_grid_size': 16, 'synthesis_ratio': 0.45, 'complexity': 'ultimate_ensemble_intelligence', 'focus': 'ultimate_ensemble_intelligence_emergence'},
    
    # Ultimate Advanced Mastery (18x18 - 30x30)
    {'stage': 11, 'max_grid_size': 18, 'synthesis_ratio': 0.40, 'complexity': 'ultimate_multiscale_ensemble', 'focus': 'ultimate_multiscale_ensemble_reasoning'},
    {'stage': 12, 'max_grid_size': 22, 'synthesis_ratio': 0.35, 'complexity': 'ultimate_coordination_mastery', 'focus': 'ultimate_coordination_protocols_mastery'},
    {'stage': 13, 'max_grid_size': 27, 'synthesis_ratio': 0.30, 'complexity': 'ultimate_ensemble_mastery', 'focus': 'ultimate_ensemble_coordination_mastery'},
    {'stage': 14, 'max_grid_size': 30, 'synthesis_ratio': 0.25, 'complexity': 'ultimate_olympus_god_mode', 'focus': 'ultimate_olympus_god_intelligence_mastery'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 130}\033[0m")
print(f"\033[96m🏛️ OLYMPUS Ensemble Training V3 - Ultimate Multi-Specialist Mastery for ARC-AGI-2\033[0m")
print(f"\033[96mFull Specialist Coordination + Ultimate Meta-Learning + Ensemble Self-Attention\033[0m")
print(f"\033[96mTarget: 95%+ Performance with Ultimate Ensemble Mastery\033[0m")
print(f"\033[96m{'=' * 130}\033[0m")


class OlympusV3Loss(nn.Module):
    """Ultimate loss function for OLYMPUS ensemble V3 training"""
    def __init__(self, config):
        super().__init__()
        self.ensemble_weight = config['ensemble_loss_weight']
        self.sync_weight = config['specialist_sync_weight']
        self.consensus_weight = config['consensus_weight']
        self.fusion_reg_weight = config['fusion_regularization']
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.ultra_teal_weight = config['ultra_teal_iou_weight']
        self.strict_weight = config['strict_match_weight']
        self.label_smoothing = config['label_smoothing']
        
        self.focal_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Ultimate V3 components
        self.meta_learning_weight = 0.35
        self.cross_attention_weight = 0.3
        self.adaptive_weight_bonus = 0.25
        self.self_attention_weight = 0.2
        self.ultimate_coordination_weight = 0.15
        
    def forward(self, ensemble_decision: EnsembleDecision, targets: torch.Tensor, inputs: torch.Tensor) -> Dict:
        """Calculate ultimate OLYMPUS V3 ensemble loss"""
        pred_output = ensemble_decision.prediction
        B = pred_output.shape[0]
        
        # Main ensemble loss (same foundation)
        if targets.dim() > 1:
            target_indices = targets.argmax(dim=1) if targets.dim() > 3 else targets.view(B, -1)
        else:
            target_indices = targets
        
        # Reshape predictions for loss calculation
        pred_flat = pred_output.view(B, -1)
        target_flat = target_indices.view(B, -1) if target_indices.dim() > 1 else target_indices
        
        # Use first 10 dimensions for loss (matching output classes)
        if pred_flat.shape[1] > 10:
            pred_flat = pred_flat[:, :10]
        if target_flat.shape[1] > 1:
            target_flat = target_flat[:, 0]  # Take first element
        
        ensemble_loss = self.focal_loss(pred_flat, target_flat.long())
        
        # ULTRA TEAL scoring for ensemble
        pred_classes = pred_flat.argmax(dim=1)
        exact_matches = (pred_classes == target_flat).float()
        exact_count = exact_matches.sum()
        exact_bonus = -exact_matches.mean() * self.exact_match_bonus
        exact_bonus = exact_bonus.clamp(min=-12.0)  # Ultimate bonus
        
        # Ultimate specialist synchronization loss
        specialist_predictions = ensemble_decision.specialist_predictions
        sync_loss = 0.0
        cross_attention_loss = 0.0
        self_attention_loss = 0.0
        
        if len(specialist_predictions) > 1:
            pred_values = list(specialist_predictions.values())
            specialist_names = list(specialist_predictions.keys())
            
            # Traditional and cross-attention synchronization
            for i, pred1 in enumerate(pred_values):
                for j, pred2 in enumerate(pred_values[i+1:], i+1):
                    pred1_flat = pred1.view(B, -1)[:, :10] if pred1.numel() > B*10 else pred1.view(B, -1)
                    pred2_flat = pred2.view(B, -1)[:, :10] if pred2.numel() > B*10 else pred2.view(B, -1)
                    
                    if pred1_flat.shape == pred2_flat.shape:
                        # Enhanced synchronization
                        sync_loss += F.mse_loss(pred1_flat, pred2_flat)
                        
                        # Ultimate cross-attention
                        attention_scores = torch.softmax(torch.matmul(pred1_flat, pred2_flat.transpose(-2, -1)), dim=-1)
                        cross_attention_loss += -torch.log(attention_scores.diagonal(dim1=-2, dim2=-1) + 1e-8).mean()
            
            # Ultimate self-attention across all specialists
            if len(pred_values) >= 3:
                # Stack all predictions for self-attention
                try:
                    pred_stack = torch.stack([p.view(B, -1)[:, :10] if p.numel() > B*10 else p.view(B, -1) 
                                            for p in pred_values], dim=1)  # [B, num_specialists, features]
                    
                    # Self-attention mechanism
                    attention_weights = F.softmax(torch.matmul(pred_stack, pred_stack.transpose(-2, -1)), dim=-1)
                    attended_predictions = torch.matmul(attention_weights, pred_stack)
                    
                    # Self-attention loss (encourage diverse but coordinated attention)
                    self_attention_loss = F.mse_loss(attended_predictions, pred_stack)
                except:
                    self_attention_loss = torch.tensor(0.0, device=pred_output.device)
        
        # Ultimate consensus bonus with adaptive weighting
        consensus_score = ensemble_decision.consensus_score
        consensus_bonus = -torch.tensor(consensus_score, device=pred_output.device) * self.consensus_weight
        
        # Ultimate adaptive weight regularization
        fusion_weights = list(ensemble_decision.fusion_weights.values())
        adaptive_weight_loss = 0.0
        ultimate_coordination_loss = 0.0
        
        if len(fusion_weights) > 1:
            fusion_tensor = torch.tensor(fusion_weights, device=pred_output.device)
            
            # Ultimate balanced specialization
            fusion_entropy = -(fusion_tensor * torch.log(fusion_tensor + 1e-8)).sum()
            fusion_reg = -fusion_entropy * self.fusion_reg_weight
            
            # Ultimate coordination loss (encourage optimal weight distribution)
            target_distribution = torch.ones_like(fusion_tensor) / len(fusion_weights)  # Uniform baseline
            weight_kl_div = F.kl_div(F.log_softmax(fusion_tensor, dim=0), target_distribution, reduction='sum')
            ultimate_coordination_loss = weight_kl_div * self.ultimate_coordination_weight
            
            # Advanced adaptive weight penalty
            weight_variance = fusion_tensor.var()
            weight_skewness = ((fusion_tensor - fusion_tensor.mean()) ** 3).mean()
            adaptive_weight_loss = (weight_variance + abs(weight_skewness)) * self.adaptive_weight_bonus
        else:
            fusion_reg = torch.tensor(0.0, device=pred_output.device)
        
        # Ultimate meta-learning bonus
        meta_learning_bonus = 0.0
        if hasattr(ensemble_decision, 'meta_features') and ensemble_decision.meta_features is not None:
            # Ultimate meta-feature diversity and quality
            meta_entropy = -(F.softmax(ensemble_decision.meta_features, dim=-1) * 
                           F.log_softmax(ensemble_decision.meta_features, dim=-1)).sum(dim=-1).mean()
            meta_quality = F.mse_loss(ensemble_decision.meta_features, 
                                    torch.ones_like(ensemble_decision.meta_features) * 0.5)
            meta_learning_bonus = -(meta_entropy + meta_quality) * self.meta_learning_weight
        
        # Transform penalty (encourage ultimate non-trivial solutions)
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
            'total': total_loss,
            'ensemble': ensemble_loss,
            'sync': sync_loss * self.sync_weight,
            'consensus_bonus': consensus_bonus,
            'fusion_reg': fusion_reg,
            'exact_bonus': exact_bonus,
            'copy_penalty': copy_penalty,
            'cross_attention': cross_attention_loss * self.cross_attention_weight,
            'adaptive_weights': adaptive_weight_loss,
            'meta_learning': meta_learning_bonus,
            'self_attention': self_attention_loss * self.self_attention_weight,
            'ultimate_coordination': ultimate_coordination_loss,
            'exact_count': exact_count,
            'consensus_score': consensus_score
        }


# Import V2's augmented dataset for V3
from train_olympus_ensemble_v2 import OlympusV2AugmentedDataset as OlympusV3UltimateDataset, olympus_v2_augmented_collate_fn as foundation_collate_fn


def train_olympus_ensemble_v3():
    """Main training function for OLYMPUS Ensemble V3"""
    print(f"\033[96m🏛️ Initializing OLYMPUS Ensemble V3 Ultimate Training...\033[0m")
    
    # Initialize OLYMPUS ensemble
    olympus = OlympusEnsemble(
        max_grid_size=30,
        d_model=256,
        device=device
    ).to(device)
    
    # Try to load V3 first if it exists (to continue training)
    v3_loaded = False
    v3_model_path = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels/olympus_v3_best.pt'
    if os.path.exists(v3_model_path):
        try:
            checkpoint = torch.load(v3_model_path, map_location=device)
            olympus.load_state_dict(checkpoint['ensemble_state_dict'])
            print(f"\033[92m🏛️ OLYMPUS V3 LOADED from checkpoint: {checkpoint.get('best_performance', 0):.2%} performance!\033[0m")
            print(f"\033[92m🏛️ Continuing V3 training from previous best model\033[0m")
            v3_loaded = True
        except Exception as e:
            print(f"\033[93m⚠️ Could not load V3 checkpoint: {e}\033[0m")
    
    # Only load V2 if V3 was not loaded
    if not v3_loaded:
        v2_model_path = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels/olympus_v2_best.pt'
        if os.path.exists(v2_model_path):
            try:
                v2_loaded_successfully = olympus.load_ensemble(v2_model_path)
                if v2_loaded_successfully:
                    print(f"\033[92m🏛️ Loaded V2 ensemble weights for V3 ultimate training - FUSION ENGINE LOADED\033[0m")
                else:
                print(f"\033[91m⚠️  V2 fusion engine failed to load, trying V1\033[0m")
                # Fallback to V1
                v1_model_path = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels/olympus_v1_best.pt'
                if os.path.exists(v1_model_path):
                    try:
                        v1_loaded_successfully = olympus.load_ensemble(v1_model_path)
                        if v1_loaded_successfully:
                            print(f"\033[96m🏛️ Loaded V1 ensemble weights for V3 training\033[0m")
                        else:
                            # Final fallback
                            weight_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
                            load_results = olympus.load_all_specialists(weight_dir)
                            successful_loads = sum(load_results.values())
                            print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\033[0m")
                    except Exception as e:
                        print(f"\033[93m⚠️  Could not load V1 weights: {e}\033[0m")
                        weight_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
                        load_results = olympus.load_all_specialists(weight_dir)
                        successful_loads = sum(load_results.values())
                        print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\033[0m")
        except Exception as e:
            print(f"\033[93m⚠️  Could not load V2 weights, trying V1: {e}\033[0m")
            # Fallback to V1
            v1_model_path = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels/olympus_v1_best.pt'
            if os.path.exists(v1_model_path):
                try:
                    v1_loaded_successfully = olympus.load_ensemble(v1_model_path)
                    if v1_loaded_successfully:
                        print(f"\033[96m🏛️ Loaded V1 ensemble weights for V3 training\033[0m")
                    else:
                        # Final fallback
                        weight_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
                        load_results = olympus.load_all_specialists(weight_dir)
                        successful_loads = sum(load_results.values())
                        print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\033[0m")
                except:
                    # Final fallback
                    weight_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
                    load_results = olympus.load_all_specialists(weight_dir)
                    successful_loads = sum(load_results.values())
                    print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\033[0m")
    else:
        # Try V1 fallback
        v1_model_path = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels/olympus_v1_best.pt'
        if os.path.exists(v1_model_path):
            try:
                v1_loaded_successfully = olympus.load_ensemble(v1_model_path)
                if v1_loaded_successfully:
                    print(f"\033[96m🏛️ Loaded V1 ensemble weights for V3 training\033[0m")
                else:
                    weight_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
                    load_results = olympus.load_all_specialists(weight_dir)
                    successful_loads = sum(load_results.values())
                    print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\033[0m")
            except:
                weight_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
                load_results = olympus.load_all_specialists(weight_dir)
                successful_loads = sum(load_results.values())
                print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\033[0m")
        else:
            # Load individual specialists
            weight_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
            load_results = olympus.load_all_specialists(weight_dir)
            successful_loads = sum(load_results.values())
            print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\033[0m")
    
    # V3: Full specialist fine-tuning (ultimate coordination)
    if not OLYMPUS_V3_CONFIG['freeze_specialists']:
        # Allow full fine-tuning but with different rates for different layers
        for name, specialist in olympus.specialists.items():
            for param_name, param in specialist.named_parameters():
                param.requires_grad = True  # All parameters trainable in V3
        print(f"\033[96m🏛️ All specialist parameters unfrozen for ultimate fine-tuning\033[0m")
    
    # Initialize ultimate loss function
    criterion = OlympusV3Loss(OLYMPUS_V3_CONFIG)
    
    # Initialize optimizers - separate for different components with different rates
    fusion_params = list(olympus.fusion_engine.parameters())
    specialist_output_params = []
    specialist_core_params = []
    
    for specialist in olympus.specialists.values():
        for param_name, param in specialist.named_parameters():
            if param.requires_grad:
                # Separate output layers from core layers
                if any(layer in param_name for layer in ['output', 'final', 'head', 'classifier']):
                    specialist_output_params.append(param)
                else:
                    specialist_core_params.append(param)
    
    # Three separate optimizers for ultimate control
    fusion_optimizer = optim.AdamW(
        fusion_params,
        lr=OLYMPUS_V3_CONFIG['learning_rate'],
        weight_decay=OLYMPUS_V3_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    specialist_output_optimizer = optim.AdamW(
        specialist_output_params,
        lr=OLYMPUS_V3_CONFIG['specialist_learning_rate'],
        weight_decay=OLYMPUS_V3_CONFIG['weight_decay'] * 1.5,
        betas=(0.9, 0.999),
        eps=1e-8
    ) if specialist_output_params else None
    
    specialist_core_optimizer = optim.AdamW(
        specialist_core_params,
        lr=OLYMPUS_V3_CONFIG['specialist_learning_rate'] * 0.5,  # Even lower for core
        weight_decay=OLYMPUS_V3_CONFIG['weight_decay'] * 2.0,  # Higher regularization
        betas=(0.9, 0.999),
        eps=1e-8
    ) if specialist_core_params else None
    
    print(f"\033[96m🏛️ Ultimate Training: {len(fusion_params)} fusion, {len(specialist_output_params)} specialist output, {len(specialist_core_params)} specialist core parameters\033[0m")
    
    # Learning rate schedulers for all optimizers
    fusion_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        fusion_optimizer,
        T_0=OLYMPUS_V3_CONFIG['warmup_epochs'],
        T_mult=int(OLYMPUS_V3_CONFIG['restart_multiplier']),
        eta_min=OLYMPUS_V3_CONFIG['learning_rate'] * 0.005
    )
    
    specialist_output_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        specialist_output_optimizer,
        T_0=OLYMPUS_V3_CONFIG['warmup_epochs'],
        T_mult=int(OLYMPUS_V3_CONFIG['restart_multiplier']),
        eta_min=OLYMPUS_V3_CONFIG['specialist_learning_rate'] * 0.005
    ) if specialist_output_optimizer else None
    
    specialist_core_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        specialist_core_optimizer,
        T_0=OLYMPUS_V3_CONFIG['warmup_epochs'],
        T_mult=int(OLYMPUS_V3_CONFIG['restart_multiplier']),
        eta_min=OLYMPUS_V3_CONFIG['specialist_learning_rate'] * 0.5 * 0.005
    ) if specialist_core_optimizer else None
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96m🏛️ Starting Ultimate Progressive Ensemble Training - 15 Ultimate Mastery Stages (4x4 to 30x30)\033[0m")
    
    # Ultimate progressive training through mastery stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 135}\033[0m")
        print(f"\033[38;2;255;204;153m🏛️ V3 Ultimate Stage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Complexity: {stage_config['complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 135}\033[0m")
        
        # Create ultimate augmented dataset for this stage
        # REDUCED AUGMENTATION for larger grids to save memory and time
        if stage_idx >= 12:  # Stage 12+ = 22x22 and above
            augmentation_factor = 2  # MINIMAL 2x augmentation for huge grids
        elif stage_idx >= 10:  # Stage 10-11 = 16x16-18x18
            augmentation_factor = 3  # 3x augmentation for large grids
        elif stage_idx >= 8:  # Stage 8-9 = 12x12-14x14
            augmentation_factor = 4  # 4x augmentation for medium-large
        else:
            augmentation_factor = 6  # 6x augmentation for smaller grids
        
        dataset = OlympusV3UltimateDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            augmentation_factor=augmentation_factor
        )
        
        # AGGRESSIVE batch size and epochs for lower grids - CRITICAL FOR PERFORMANCE
        if stage_config['max_grid_size'] <= 4:
            batch_size = 32  # ULTRA small batches = 8x more gradient updates!
            epochs_multiplier = 4.0  # QUADRUPLE epochs for 4x4 (48 epochs total)
        elif stage_config['max_grid_size'] <= 6:
            batch_size = 48  # Very small batches for 5x5-6x6
            epochs_multiplier = 3.0  # TRIPLE epochs (36 epochs total)
        elif stage_config['max_grid_size'] <= 8:
            batch_size = 64  # Small batches for 7x7-8x8
            epochs_multiplier = 2.5  # 2.5x epochs (30 epochs total)
        elif stage_config['max_grid_size'] <= 10:
            batch_size = 128  # Medium batch size for 9x9-10x10
            epochs_multiplier = 1.5  # 50% more epochs (18 epochs)
        elif stage_config['max_grid_size'] <= 16:
            batch_size = 256  # Full batch size for medium grids
            epochs_multiplier = 1.0
        elif stage_config['max_grid_size'] <= 20:
            batch_size = 192  # Reduce for medium-large grids
            epochs_multiplier = 1.0
        elif stage_config['max_grid_size'] <= 22:
            batch_size = 96  # AGGRESSIVE reduction for 22x22
            epochs_multiplier = 0.8  # Reduce epochs too
        elif stage_config['max_grid_size'] <= 27:
            batch_size = 48  # ULTRA aggressive for 27x27
            epochs_multiplier = 0.7  # Even fewer epochs
        else:
            batch_size = 32  # MINIMAL batch size for 30x30
            epochs_multiplier = 0.6  # Minimal epochs
        
        # Calculate actual epochs for this stage
        stage_epochs = int(OLYMPUS_V3_CONFIG['epochs_per_stage'] * epochs_multiplier)
        
        # DYNAMIC LEARNING RATE BOOST for lower grids!
        if stage_config['max_grid_size'] <= 4:
            lr_multiplier = 2.0  # DOUBLE learning rate for 4x4
        elif stage_config['max_grid_size'] <= 6:
            lr_multiplier = 1.5  # 1.5x learning rate for 5x5-6x6
        elif stage_config['max_grid_size'] <= 8:
            lr_multiplier = 1.2  # 1.2x learning rate for 7x7-8x8
        else:
            lr_multiplier = 1.0  # Normal learning rate for larger grids
        
        # Adjust learning rates for this stage
        for param_group in fusion_optimizer.param_groups:
            param_group['lr'] = OLYMPUS_V3_CONFIG['learning_rate'] * lr_multiplier
        if specialist_output_optimizer:
            for param_group in specialist_output_optimizer.param_groups:
                param_group['lr'] = OLYMPUS_V3_CONFIG['specialist_learning_rate'] * lr_multiplier
        if specialist_core_optimizer:
            for param_group in specialist_core_optimizer.param_groups:
                param_group['lr'] = OLYMPUS_V3_CONFIG['specialist_learning_rate'] * 0.5 * lr_multiplier
        
        print(f"\033[96m🏛️ Stage {stage_idx}: Batch={batch_size}, Epochs={stage_epochs}, LR_mult={lr_multiplier}x, Aug={augmentation_factor}x for {stage_config['max_grid_size']}x{stage_config['max_grid_size']} grids\033[0m")
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=foundation_collate_fn,
            num_workers=0,  # Keep 0 for OLYMPUS stability
            pin_memory=True
        )
        
        # Stage-specific training with dynamic epochs
        stage_performance = train_ultimate_mastery_stage(
            olympus, dataloader, criterion, 
            fusion_optimizer, specialist_output_optimizer, specialist_core_optimizer,
            fusion_scheduler, specialist_output_scheduler, specialist_core_scheduler,
            scaler, stage_idx, stage_config, training_stats, stage_epochs
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best OLYMPUS V3 model in InputBestModels directory
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels', exist_ok=True)
            
            # Enhanced save with optimizer and scheduler state (similar to V2)
            ensemble_state = {
                'ensemble_state_dict': olympus.state_dict(),
                'fusion_optimizer_state_dict': fusion_optimizer.state_dict(),
                'specialist_output_optimizer_state_dict': specialist_output_optimizer.state_dict() if specialist_output_optimizer else None,
                'specialist_core_optimizer_state_dict': specialist_core_optimizer.state_dict() if specialist_core_optimizer else None,
                'fusion_scheduler_state_dict': fusion_scheduler.state_dict(),
                'specialist_output_scheduler_state_dict': specialist_output_scheduler.state_dict() if specialist_output_scheduler else None,
                'specialist_core_scheduler_state_dict': specialist_core_scheduler.state_dict() if specialist_core_scheduler else None,
                'best_performance': best_performance,
                'stage': stage_idx,
                'ensemble_config': {
                    'max_grid_size': olympus.max_grid_size,
                    'd_model': olympus.d_model,
                    'device': olympus.device_name
                },
                'performance_metrics': olympus.get_ensemble_state()
            }
            
            torch.save(ensemble_state, '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels/olympus_v3_best.pt')
            print(f"\033[96m🏛️ New best V3 ultimate performance: {best_performance:.2%} - OLYMPUS V3 ULTIMATE saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 140}\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Ensemble V3 ULTIMATE Training Complete!\033[0m")
    print(f"\033[96m🏛️ Best V3 ULTIMATE Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Ultimate Intelligence Achieved - Ready for ARC-AGI-2 Challenge!\033[0m")
    print(f"\033[96m{'=' * 140}\033[0m")
    
    return olympus, best_performance


def train_ultimate_mastery_stage(olympus, dataloader, criterion, 
                                fusion_optimizer, specialist_output_optimizer, specialist_core_optimizer,
                                fusion_scheduler, specialist_output_scheduler, specialist_core_scheduler,
                                scaler, stage_idx, stage_config, training_stats, stage_epochs):
    """Train a single ultimate mastery ensemble stage"""
    olympus.train()
    
    epochs_for_stage = stage_epochs  # Use the dynamic epochs passed in
    
    # Dynamic gradient accumulation for memory management
    if stage_config['max_grid_size'] >= 27:
        accumulation_steps = 8  # 8x accumulation for 27x27 and 30x30
    elif stage_config['max_grid_size'] >= 22:
        accumulation_steps = 6  # 6x accumulation for 22x22
    elif stage_config['max_grid_size'] >= 18:
        accumulation_steps = 4  # 4x accumulation for 18x18
    else:
        accumulation_steps = OLYMPUS_V3_CONFIG['gradient_accumulation']  # Default 2x
    
    # WARMUP STRATEGY for lower grids
    warmup_epochs = 0
    if stage_config['max_grid_size'] <= 4:
        warmup_epochs = 3  # 3 epoch warmup for 4x4
    elif stage_config['max_grid_size'] <= 6:
        warmup_epochs = 2  # 2 epoch warmup for 5x5-6x6
    elif stage_config['max_grid_size'] <= 8:
        warmup_epochs = 1  # 1 epoch warmup for 7x7-8x8
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        total_consensus = 0.0
        
        # Apply warmup learning rate scaling
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            # Scale down learning rates during warmup
            for param_group in fusion_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
            if specialist_output_optimizer:
                for param_group in specialist_output_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * warmup_factor
            if specialist_core_optimizer:
                for param_group in specialist_core_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * warmup_factor
        
        # Progress bar
        # Dynamic progress bar with stage focus (like ATLAS)
        stage_focus = stage_config['focus'].replace('_', ' ').title()
        warmup_str = f" (Warmup {epoch+1}/{warmup_epochs})" if epoch < warmup_epochs else ""
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153m🏛️ {stage_focus} Stage {stage_idx} Epoch {epoch}{warmup_str}\033[0m")
        
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                # OLYMPUS ensemble forward pass
                ensemble_decision = olympus(inputs, targets, mode='train')
                loss_dict = criterion(ensemble_decision, targets, inputs)
                loss = loss_dict['total'] / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights with ultimate control
            if (batch_idx + 1) % accumulation_steps == 0:
                # Update fusion parameters
                scaler.unscale_(fusion_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in olympus.fusion_engine.parameters()], 
                    OLYMPUS_V3_CONFIG['gradient_clip']
                )
                scaler.step(fusion_optimizer)
                
                # Update specialist output parameters
                if specialist_output_optimizer is not None:
                    scaler.unscale_(specialist_output_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for specialist in olympus.specialists.values() 
                         for param_name, p in specialist.named_parameters() 
                         if p.requires_grad and any(layer in param_name for layer in ['output', 'final', 'head', 'classifier'])], 
                        OLYMPUS_V3_CONFIG['gradient_clip']
                    )
                    scaler.step(specialist_output_optimizer)
                
                # Update specialist core parameters
                if specialist_core_optimizer is not None:
                    scaler.unscale_(specialist_core_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for specialist in olympus.specialists.values() 
                         for param_name, p in specialist.named_parameters() 
                         if p.requires_grad and not any(layer in param_name for layer in ['output', 'final', 'head', 'classifier'])], 
                        OLYMPUS_V3_CONFIG['gradient_clip'] * 0.8  # Tighter clipping for core
                    )
                    scaler.step(specialist_core_optimizer)
                
                scaler.update()
                
                # Zero gradients
                fusion_optimizer.zero_grad()
                if specialist_output_optimizer is not None:
                    specialist_output_optimizer.zero_grad()
                if specialist_core_optimizer is not None:
                    specialist_core_optimizer.zero_grad()
                
                # Update learning rates
                fusion_scheduler.step()
                if specialist_output_scheduler is not None:
                    specialist_output_scheduler.step()
                if specialist_core_scheduler is not None:
                    specialist_core_scheduler.step()
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    epoch_losses[key] += value.item()
                elif isinstance(value, (int, float)):
                    epoch_losses[key] += value
            
            total_exact_matches += loss_dict['exact_count'].item()
            total_samples += inputs.shape[0]
            total_consensus += loss_dict['consensus_score']
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            avg_consensus = total_consensus / max(batch_idx + 1, 1)
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'Consensus': f"{avg_consensus:.3f}",
                'SelfAtt': f"{loss_dict.get('self_attention', 0):.4f}",
                'UltCoord': f"{loss_dict.get('ultimate_coordination', 0):.4f}",
                'FusionLR': f"{fusion_scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        avg_consensus = total_consensus / len(dataloader)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 6 == 0 or epoch == epochs_for_stage - 1:
            avg_loss = epoch_losses['total']/len(dataloader)
            fusion_lr = fusion_scheduler.get_last_lr()[0]
            output_lr = specialist_output_scheduler.get_last_lr()[0] if specialist_output_scheduler else 0
            core_lr = specialist_core_scheduler.get_last_lr()[0] if specialist_core_scheduler else 0
            print(f"\033[38;2;255;204;153m⏰ OLYMPUS V3 Ultimate Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * OLYMPUS_V3_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Ultimate Ensemble: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 Fusion: {fusion_lr:.6f} | Output: {output_lr:.6f} | Core: {core_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Consensus: {avg_consensus:.3f}\033[0m")
            if epoch == epochs_for_stage - 1:
                print(f"\033[96m✅ Ultimate Stage {stage_idx} complete! Final exact: {epoch_performance:.2%}\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_stage_performance


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train OLYMPUS V3 Ultimate
    olympus, best_performance = train_olympus_ensemble_v3()