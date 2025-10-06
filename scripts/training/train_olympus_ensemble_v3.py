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
    # Core Training Parameters - Ultimate Level
    'batch_size': 4,  # Smallest batches for maximum precision
    'learning_rate': 0.00005,  # Ultra-low rate for ultimate fine-tuning
    'num_epochs': 300,  # Ultimate training: 10 stages x 30 epochs
    'gradient_accumulation': 8,  # Effective batch 32 for ultimate stability
    'epochs_per_stage': 30,  # Ultimate epochs per stage
    'curriculum_stages': 10,  # Ultimate curriculum stages
    
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

# Ultimate 10-Stage Progressive Configuration
STAGE_CONFIG = [
    # Ultimate Ensemble Mastery (Building on V2 Advanced)
    {'stage': 0, 'max_grid_size': 12, 'synthesis_ratio': 0.9, 'complexity': 'ultimate_foundation', 'focus': 'full_specialist_coordination_basic'},
    {'stage': 1, 'max_grid_size': 16, 'synthesis_ratio': 0.8, 'complexity': 'meta_ensemble_basic', 'focus': 'meta_ensemble_learning_introduction'},
    {'stage': 2, 'max_grid_size': 20, 'synthesis_ratio': 0.7, 'complexity': 'self_attention_ensemble', 'focus': 'ensemble_self_attention_mastery'},
    {'stage': 3, 'max_grid_size': 22, 'synthesis_ratio': 0.6, 'complexity': 'adaptive_curriculum', 'focus': 'adaptive_curriculum_optimization'},
    {'stage': 4, 'max_grid_size': 24, 'synthesis_ratio': 0.5, 'complexity': 'ultimate_fusion_networks', 'focus': 'multiple_fusion_network_coordination'},
    {'stage': 5, 'max_grid_size': 26, 'synthesis_ratio': 0.4, 'complexity': 'ensemble_intelligence', 'focus': 'ultimate_ensemble_intelligence_emergence'},
    {'stage': 6, 'max_grid_size': 28, 'synthesis_ratio': 0.3, 'complexity': 'meta_coordination_advanced', 'focus': 'advanced_meta_coordination_protocols'},
    {'stage': 7, 'max_grid_size': 30, 'synthesis_ratio': 0.2, 'complexity': 'olympus_mastery_basic', 'focus': 'basic_olympus_ultimate_mastery'},
    {'stage': 8, 'max_grid_size': 30, 'synthesis_ratio': 0.1, 'complexity': 'olympus_mastery_advanced', 'focus': 'advanced_olympus_ultimate_mastery'},
    {'stage': 9, 'max_grid_size': 30, 'synthesis_ratio': 0.05, 'complexity': 'olympus_god_mode', 'focus': 'ultimate_olympus_god_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 130}\\033[0m")
print(f"\033[96m🏛️ OLYMPUS Ensemble Training V3 - Ultimate Multi-Specialist Mastery for ARC-AGI-2\\033[0m")
print(f"\033[96mFull Specialist Coordination + Ultimate Meta-Learning + Ensemble Self-Attention\\033[0m")
print(f"\033[96mTarget: 95%+ Performance with Ultimate Ensemble Mastery\\033[0m")
print(f"\033[96m{'=' * 130}\\033[0m")


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


# Use the same dataset as V1/V2
from train_olympus_ensemble_v1 import FoundationEnsembleDataset, foundation_collate_fn


def train_olympus_ensemble_v3():
    """Main training function for OLYMPUS Ensemble V3"""
    print(f"\033[96m🏛️ Initializing OLYMPUS Ensemble V3 Ultimate Training...\\033[0m")
    
    # Initialize OLYMPUS ensemble
    olympus = OlympusEnsemble(
        max_grid_size=30,
        d_model=256,
        device=device
    ).to(device)
    
    # Load V2 ensemble weights first
    v2_model_path = '/content/AutomataNexus_Olympus_AGI2/models/olympus_v2_best.pt'
    if os.path.exists(v2_model_path):
        try:
            ensemble_state = torch.load(v2_model_path, map_location=device)
            olympus.load_state_dict(ensemble_state['ensemble_state_dict'])
            print(f"\033[96m🏛️ Loaded V2 ensemble weights for V3 ultimate training\\033[0m")
        except Exception as e:
            print(f"\033[93m⚠️  Could not load V2 weights, trying V1: {e}\\033[0m")
            # Fallback to V1
            v1_model_path = '/content/AutomataNexus_Olympus_AGI2/models/olympus_v1_best.pt'
            if os.path.exists(v1_model_path):
                try:
                    ensemble_state = torch.load(v1_model_path, map_location=device)
                    olympus.load_state_dict(ensemble_state['ensemble_state_dict'])
                    print(f"\033[96m🏛️ Loaded V1 ensemble weights for V3 training\\033[0m")
                except:
                    # Final fallback
                    weight_dir = '/content/AutomataNexus_Olympus_AGI2/models'
                    load_results = olympus.load_all_specialists(weight_dir)
                    successful_loads = sum(load_results.values())
                    print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\\033[0m")
    else:
        # Try V1 fallback
        v1_model_path = '/content/AutomataNexus_Olympus_AGI2/models/olympus_v1_best.pt'
        if os.path.exists(v1_model_path):
            try:
                ensemble_state = torch.load(v1_model_path, map_location=device)
                olympus.load_state_dict(ensemble_state['ensemble_state_dict'])
                print(f"\033[96m🏛️ Loaded V1 ensemble weights for V3 training\\033[0m")
            except:
                weight_dir = '/content/AutomataNexus_Olympus_AGI2/models'
                load_results = olympus.load_all_specialists(weight_dir)
                successful_loads = sum(load_results.values())
                print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\\033[0m")
        else:
            # Load individual specialists
            weight_dir = '/content/AutomataNexus_Olympus_AGI2/models'
            load_results = olympus.load_all_specialists(weight_dir)
            successful_loads = sum(load_results.values())
            print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 individual specialist models\\033[0m")
    
    # V3: Full specialist fine-tuning (ultimate coordination)
    if not OLYMPUS_V3_CONFIG['freeze_specialists']:
        # Allow full fine-tuning but with different rates for different layers
        for name, specialist in olympus.specialists.items():
            for param_name, param in specialist.named_parameters():
                param.requires_grad = True  # All parameters trainable in V3
        print(f"\033[96m🏛️ All specialist parameters unfrozen for ultimate fine-tuning\\033[0m")
    
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
    
    print(f"\033[96m🏛️ Ultimate Training: {len(fusion_params)} fusion, {len(specialist_output_params)} specialist output, {len(specialist_core_params)} specialist core parameters\\033[0m")
    
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
    
    print(f"\033[96m🏛️ Starting Ultimate Progressive Ensemble Training - 10 Ultimate Mastery Stages\\033[0m")
    
    # Ultimate progressive training through mastery stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\\033[96m{'=' * 135}\\033[0m")
        print(f"\033[38;2;255;204;153m🏛️ V3 Ultimate Stage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Complexity: {stage_config['complexity']} | Focus: {stage_config['focus']}\\033[0m")
        print(f"\033[96m{'=' * 135}\\033[0m")
        
        # Create ultimate dataset for this stage
        dataset = FoundationEnsembleDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=OLYMPUS_V3_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=foundation_collate_fn,
            num_workers=0,  # Keep 0 for OLYMPUS stability
            pin_memory=True
        )
        
        # Stage-specific training
        stage_performance = train_ultimate_mastery_stage(
            olympus, dataloader, criterion, 
            fusion_optimizer, specialist_output_optimizer, specialist_core_optimizer,
            fusion_scheduler, specialist_output_scheduler, specialist_core_scheduler,
            scaler, stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best OLYMPUS V3 model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            olympus.save_ensemble('/content/AutomataNexus_Olympus_AGI2/models/olympus_v3_best.pt')
            print(f"\033[96m🏛️ New best V3 ultimate performance: {best_performance:.2%} - OLYMPUS V3 ULTIMATE saved!\\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\\033[96m{'=' * 140}\\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Ensemble V3 ULTIMATE Training Complete!\\033[0m")
    print(f"\033[96m🏛️ Best V3 ULTIMATE Performance: {best_performance:.2%}\\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Ultimate Intelligence Achieved - Ready for ARC-AGI-2 Challenge!\\033[0m")
    print(f"\033[96m{'=' * 140}\\033[0m")
    
    return olympus, best_performance


def train_ultimate_mastery_stage(olympus, dataloader, criterion, 
                                fusion_optimizer, specialist_output_optimizer, specialist_core_optimizer,
                                fusion_scheduler, specialist_output_scheduler, specialist_core_scheduler,
                                scaler, stage_idx, stage_config, training_stats):
    """Train a single ultimate mastery ensemble stage"""
    olympus.train()
    
    epochs_for_stage = OLYMPUS_V3_CONFIG['epochs_per_stage']
    accumulation_steps = OLYMPUS_V3_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        total_consensus = 0.0
        
        # Progress bar
        # Dynamic progress bar with stage focus (like ATLAS)
        stage_focus = stage_config['focus'].replace('_', ' ').title()
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153m🏛️ {stage_focus} Stage {stage_idx} Epoch {epoch}\\033[0m")
        
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
            print(f"\033[38;2;255;204;153m⏰ OLYMPUS V3 Ultimate Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * OLYMPUS_V3_CONFIG['epochs_per_stage'] + epoch + 1}):\\033[0m")
            print(f"\033[96m   🎯 Ultimate Ensemble: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\\033[0m")
            print(f"\033[96m   📊 Fusion: {fusion_lr:.6f} | Output: {output_lr:.6f} | Core: {core_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Consensus: {avg_consensus:.3f}\\033[0m")
            if epoch == epochs_for_stage - 1:
                print(f"\033[96m✅ Ultimate Stage {stage_idx} complete! Final exact: {epoch_performance:.2%}\\033[0m")
        
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