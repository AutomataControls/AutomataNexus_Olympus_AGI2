"""
OLYMPUS Ensemble Training V2 - Advanced Multi-Specialist Coordination for ARC-AGI-2
Enhanced ensemble training with partial specialist fine-tuning and advanced fusion
Builds upon V1 foundation with sophisticated coordination protocols
Target: 90%+ performance with advanced ensemble synergy
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

# OLYMPUS V2 Configuration - Advanced Ensemble Training
OLYMPUS_V2_CONFIG = {
    # Core Training Parameters - Advanced Level
    'batch_size': 6,  # Smaller batches for complex ensemble training
    'learning_rate': 0.00008,  # Lower rate for fine-tuning specialists
    'num_epochs': 200,  # Advanced training: 8 stages x 25 epochs
    'gradient_accumulation': 6,  # Effective batch 36 for stability
    'epochs_per_stage': 25,  # Advanced epochs per stage
    'curriculum_stages': 8,  # Advanced curriculum stages
    
    # Enhanced Loss Configuration
    'ensemble_loss_weight': 1.2,  # Increased ensemble focus
    'specialist_sync_weight': 0.4,  # Enhanced synchronization
    'consensus_weight': 0.3,  # Stronger consensus building
    'fusion_regularization': 0.15,  # More sophisticated fusion
    'transform_penalty': 0.08,  # Encourage complex transformations
    'exact_match_bonus': 12.0,  # Higher precision bonus
    'gradient_clip': 0.4,  # Tighter gradient control
    'weight_decay': 3e-6,  # Balanced regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    
    # OLYMPUS V2-Specific Advanced Settings
    'freeze_specialists': False,  # Allow partial specialist fine-tuning
    'fusion_training_only': False,  # Train both fusion and specialists
    'specialist_learning_rate': 0.00003,  # Lower rate for specialist fine-tuning
    'consensus_threshold': 0.7,  # Higher consensus for confidence
    'specialist_dropout': 0.05,  # Light dropout for robustness
    'ensemble_coordination': True,  # Enable advanced coordination
    'adaptive_weights': True,  # Dynamic specialist weighting
    
    # Advanced Training Features
    'label_smoothing': 0.015,  # Refined smoothing for advanced ensemble
    'ensemble_diversity_bonus': True,
    'specialist_agreement_bonus': True,
    'consensus_building_bonus': True,
    'fusion_optimization': True,
    'advanced_meta_learning': True,  # NEW: Meta-learning fusion
    'cross_specialist_attention': True,  # NEW: Inter-specialist attention
    'dynamic_fusion_weights': True,  # NEW: Adaptive fusion weighting
    
    # Learning Rate Scheduling
    'warmup_epochs': 15,  # Advanced warmup
    'cosine_restarts': True,
    'restart_multiplier': 1.4,
    'plateau_patience': 20,
}

# Advanced 8-Stage Progressive Configuration
STAGE_CONFIG = [
    # Advanced Ensemble Coordination (Building on V1 Foundation)
    {'stage': 0, 'max_grid_size': 10, 'synthesis_ratio': 0.85, 'complexity': 'advanced_coordination', 'focus': 'specialist_fine_tuning_basic'},
    {'stage': 1, 'max_grid_size': 14, 'synthesis_ratio': 0.75, 'complexity': 'meta_fusion_basic', 'focus': 'meta_learning_fusion_introduction'},
    {'stage': 2, 'max_grid_size': 18, 'synthesis_ratio': 0.65, 'complexity': 'cross_attention', 'focus': 'inter_specialist_attention_learning'},
    {'stage': 3, 'max_grid_size': 22, 'synthesis_ratio': 0.55, 'complexity': 'dynamic_weighting', 'focus': 'adaptive_fusion_weight_optimization'},
    {'stage': 4, 'max_grid_size': 26, 'synthesis_ratio': 0.45, 'complexity': 'advanced_consensus', 'focus': 'sophisticated_consensus_algorithms'},
    {'stage': 5, 'max_grid_size': 28, 'synthesis_ratio': 0.35, 'complexity': 'ensemble_synergy', 'focus': 'advanced_ensemble_synergy_mastery'},
    {'stage': 6, 'max_grid_size': 30, 'synthesis_ratio': 0.25, 'complexity': 'meta_coordination', 'focus': 'meta_level_coordination_protocols'},
    {'stage': 7, 'max_grid_size': 30, 'synthesis_ratio': 0.15, 'complexity': 'olympus_advanced', 'focus': 'advanced_olympus_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 125}\\033[0m")
print(f"\033[96m🏛️ OLYMPUS Ensemble Training V2 - Advanced Multi-Specialist Coordination for ARC-AGI-2\\033[0m")
print(f"\033[96mPartial Specialist Fine-Tuning + Advanced Fusion + Meta-Learning Coordination\\033[0m")
print(f"\033[96mTarget: 90%+ Performance with Advanced Ensemble Synergy\\033[0m")
print(f"\033[96m{'=' * 125}\\033[0m")


class OlympusV2Loss(nn.Module):
    """Advanced loss function for OLYMPUS ensemble V2 training"""
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
        
        # Advanced V2 components
        self.meta_learning_weight = 0.25
        self.cross_attention_weight = 0.2
        self.adaptive_weight_bonus = 0.15
        
    def forward(self, ensemble_decision: EnsembleDecision, targets: torch.Tensor, inputs: torch.Tensor) -> Dict:
        """Calculate comprehensive OLYMPUS V2 ensemble loss"""
        pred_output = ensemble_decision.prediction
        B = pred_output.shape[0]
        
        # Main ensemble loss (same as V1)
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
        exact_bonus = exact_bonus.clamp(min=-8.0)
        
        # Advanced specialist synchronization loss
        specialist_predictions = ensemble_decision.specialist_predictions
        sync_loss = 0.0
        cross_attention_loss = 0.0
        
        if len(specialist_predictions) > 1:
            pred_values = list(specialist_predictions.values())
            specialist_names = list(specialist_predictions.keys())
            
            for i, pred1 in enumerate(pred_values):
                for j, pred2 in enumerate(pred_values[i+1:], i+1):
                    # Enhanced synchronization with cross-attention
                    pred1_flat = pred1.view(B, -1)[:, :10] if pred1.numel() > B*10 else pred1.view(B, -1)
                    pred2_flat = pred2.view(B, -1)[:, :10] if pred2.numel() > B*10 else pred2.view(B, -1)
                    
                    if pred1_flat.shape == pred2_flat.shape:
                        # Traditional synchronization
                        sync_loss += F.mse_loss(pred1_flat, pred2_flat)
                        
                        # Cross-attention loss (encourage complementary predictions)
                        attention_scores = torch.softmax(torch.matmul(pred1_flat, pred2_flat.transpose(-2, -1)), dim=-1)
                        cross_attention_loss += -torch.log(attention_scores.diagonal(dim1=-2, dim2=-1) + 1e-8).mean()
        
        # Enhanced consensus bonus with adaptive weighting
        consensus_score = ensemble_decision.consensus_score
        consensus_bonus = -torch.tensor(consensus_score, device=pred_output.device) * self.consensus_weight
        
        # Advanced adaptive weight regularization
        fusion_weights = list(ensemble_decision.fusion_weights.values())
        adaptive_weight_loss = 0.0
        
        if len(fusion_weights) > 1:
            fusion_tensor = torch.tensor(fusion_weights, device=pred_output.device)
            
            # Encourage balanced but specialized weights
            fusion_entropy = -(fusion_tensor * torch.log(fusion_tensor + 1e-8)).sum()
            fusion_reg = -fusion_entropy * self.fusion_reg_weight
            
            # Adaptive weight penalty (discourage extreme weights)
            weight_variance = fusion_tensor.var()
            adaptive_weight_loss = weight_variance * self.adaptive_weight_bonus
        else:
            fusion_reg = torch.tensor(0.0, device=pred_output.device)
        
        # Meta-learning bonus (encourage learning from specialist interactions)
        meta_learning_bonus = 0.0
        if hasattr(ensemble_decision, 'meta_features') and ensemble_decision.meta_features is not None:
            # Encourage diverse meta-features
            meta_entropy = -(F.softmax(ensemble_decision.meta_features, dim=-1) * 
                           F.log_softmax(ensemble_decision.meta_features, dim=-1)).sum(dim=-1).mean()
            meta_learning_bonus = -meta_entropy * self.meta_learning_weight
        
        # Transform penalty (encourage non-trivial solutions)
        if inputs.dim() > 1:
            input_flat = inputs.view(B, -1)[:, :10] if inputs.numel() > B*10 else inputs.view(B, -1)
            copy_penalty = F.mse_loss(pred_flat, input_flat) * self.transform_penalty
        else:
            copy_penalty = torch.tensor(0.0, device=pred_output.device)
        
        total_loss = (ensemble_loss + exact_bonus + sync_loss * self.sync_weight + 
                     consensus_bonus + fusion_reg + copy_penalty + 
                     cross_attention_loss * self.cross_attention_weight + 
                     adaptive_weight_loss + meta_learning_bonus)
        
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
            'exact_count': exact_count,
            'consensus_score': consensus_score
        }


# Use the same dataset as V1
from train_olympus_ensemble_v1 import FoundationEnsembleDataset, foundation_collate_fn


def train_olympus_ensemble_v2():
    """Main training function for OLYMPUS Ensemble V2"""
    print(f"\033[96m🏛️ Initializing OLYMPUS Ensemble V2 Training...\\033[0m")
    
    # Initialize OLYMPUS ensemble
    olympus = OlympusEnsemble(
        max_grid_size=30,
        d_model=256,
        device=device
    ).to(device)
    
    # Load V1 ensemble weights first
    v1_model_path = '/content/AutomataNexus_Olympus_AGI2/models/olympus_v1_best.pt'
    if os.path.exists(v1_model_path):
        try:
            ensemble_state = torch.load(v1_model_path, map_location=device)
            olympus.load_state_dict(ensemble_state['ensemble_state_dict'])
            print(f"\033[96m🏛️ Loaded V1 ensemble weights for V2 training\\033[0m")
        except Exception as e:
            print(f"\033[93m⚠️  Could not load V1 weights, loading individual specialists: {e}\\033[0m")
            # Fallback to individual specialist loading
            weight_dir = '/content/AutomataNexus_Olympus_AGI2/models'
            load_results = olympus.load_all_specialists(weight_dir)
            successful_loads = sum(load_results.values())
            print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 specialist models\\033[0m")
    else:
        # Load all specialist weights individually
        weight_dir = '/content/AutomataNexus_Olympus_AGI2/models'
        load_results = olympus.load_all_specialists(weight_dir)
        successful_loads = sum(load_results.values())
        print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 specialist models\\033[0m")
    
    # V2: Partial specialist fine-tuning (not fully frozen)
    if not OLYMPUS_V2_CONFIG['freeze_specialists']:
        # Freeze most specialist layers, allow fine-tuning of top layers
        for name, specialist in olympus.specialists.items():
            for param_name, param in specialist.named_parameters():
                # Only allow fine-tuning of output layers and top transformer layers
                if any(layer in param_name for layer in ['output', 'final', 'head', 'classifier']):
                    param.requires_grad = True
                elif 'layer' in param_name:
                    # Extract layer number and only allow top 2 layers
                    import re
                    layer_match = re.search(r'layer\.(\d+)', param_name)
                    if layer_match and int(layer_match.group(1)) >= 6:  # Top 2 layers (6,7)
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False
        print(f"\033[96m🏛️ Specialist top layers unfrozen for partial fine-tuning\\033[0m")
    
    # Initialize advanced loss function
    criterion = OlympusV2Loss(OLYMPUS_V2_CONFIG)
    
    # Initialize optimizers - separate for specialists and fusion
    fusion_params = list(olympus.fusion_engine.parameters())
    specialist_params = []
    for specialist in olympus.specialists.values():
        specialist_params.extend([p for p in specialist.parameters() if p.requires_grad])
    
    fusion_optimizer = optim.AdamW(
        fusion_params,
        lr=OLYMPUS_V2_CONFIG['learning_rate'],
        weight_decay=OLYMPUS_V2_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    specialist_optimizer = optim.AdamW(
        specialist_params,
        lr=OLYMPUS_V2_CONFIG['specialist_learning_rate'],
        weight_decay=OLYMPUS_V2_CONFIG['weight_decay'] * 2,  # Higher regularization
        betas=(0.9, 0.999),
        eps=1e-8
    ) if specialist_params else None
    
    print(f"\033[96m🏛️ Training {len(fusion_params)} fusion parameters and {len(specialist_params)} specialist parameters\\033[0m")
    
    # Learning rate schedulers
    fusion_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        fusion_optimizer,
        T_0=OLYMPUS_V2_CONFIG['warmup_epochs'],
        T_mult=int(OLYMPUS_V2_CONFIG['restart_multiplier']),
        eta_min=OLYMPUS_V2_CONFIG['learning_rate'] * 0.01
    )
    
    specialist_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        specialist_optimizer,
        T_0=OLYMPUS_V2_CONFIG['warmup_epochs'],
        T_mult=int(OLYMPUS_V2_CONFIG['restart_multiplier']),
        eta_min=OLYMPUS_V2_CONFIG['specialist_learning_rate'] * 0.01
    ) if specialist_optimizer else None
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96m🏛️ Starting Advanced Progressive Ensemble Training - 8 Advanced Coordination Stages\\033[0m")
    
    # Advanced progressive training through coordination stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\\033[96m{'=' * 130}\\033[0m")
        print(f"\033[96m🏛️ V2 Stage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Complexity: {stage_config['complexity']} | Focus: {stage_config['focus']}\\033[0m")
        print(f"\033[96m{'=' * 130}\\033[0m")
        
        # Create advanced dataset for this stage
        dataset = FoundationEnsembleDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=OLYMPUS_V2_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=foundation_collate_fn,
            num_workers=0,  # Keep 0 for OLYMPUS stability
            pin_memory=True
        )
        
        # Stage-specific training
        stage_performance = train_advanced_coordination_stage(
            olympus, dataloader, criterion, 
            fusion_optimizer, specialist_optimizer,
            fusion_scheduler, specialist_scheduler, 
            scaler, stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best OLYMPUS V2 model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            olympus.save_ensemble('/content/AutomataNexus_Olympus_AGI2/models/olympus_v2_best.pt')
            print(f"\033[96m🏛️ New best V2 ensemble performance: {best_performance:.2%} - OLYMPUS V2 saved!\\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\\033[96m{'=' * 135}\\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Ensemble V2 Advanced Training Complete!\\033[0m")
    print(f"\033[96m🏛️ Best V2 Advanced Performance: {best_performance:.2%}\\033[0m")
    print(f"\033[96m🏛️ All 5 Specialists Advanced Coordinated and Ready for V3 Ultimate Training\\033[0m")
    print(f"\033[96m{'=' * 135}\\033[0m")
    
    return olympus, best_performance


def train_advanced_coordination_stage(olympus, dataloader, criterion, 
                                    fusion_optimizer, specialist_optimizer,
                                    fusion_scheduler, specialist_scheduler,
                                    scaler, stage_idx, stage_config, training_stats):
    """Train a single advanced coordination ensemble stage"""
    olympus.train()
    
    epochs_for_stage = OLYMPUS_V2_CONFIG['epochs_per_stage']
    accumulation_steps = OLYMPUS_V2_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        total_consensus = 0.0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153m🏛️ Advanced Stage {stage_idx} Epoch {epoch}\\033[0m")
        
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
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                # Update fusion parameters
                scaler.unscale_(fusion_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in olympus.fusion_engine.parameters()], 
                    OLYMPUS_V2_CONFIG['gradient_clip']
                )
                scaler.step(fusion_optimizer)
                
                # Update specialist parameters (if training)
                if specialist_optimizer is not None:
                    scaler.unscale_(specialist_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for specialist in olympus.specialists.values() for p in specialist.parameters() if p.requires_grad], 
                        OLYMPUS_V2_CONFIG['gradient_clip']
                    )
                    scaler.step(specialist_optimizer)
                
                scaler.update()
                fusion_optimizer.zero_grad()
                if specialist_optimizer is not None:
                    specialist_optimizer.zero_grad()
                
                # Update learning rates
                fusion_scheduler.step()
                if specialist_scheduler is not None:
                    specialist_scheduler.step()
            
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
                'MetaLoss': f"{loss_dict.get('meta_learning', 0):.4f}",
                'CrossAtt': f"{loss_dict.get('cross_attention', 0):.4f}",
                'FusionLR': f"{fusion_scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        avg_consensus = total_consensus / len(dataloader)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = fusion_scheduler.get_last_lr()[0]
            specialist_lr = specialist_scheduler.get_last_lr()[0] if specialist_scheduler else 0
            print(f"\033[38;2;255;204;153m⏰ OLYMPUS V2 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * OLYMPUS_V2_CONFIG['epochs_per_stage'] + epoch + 1}):\\033[0m")
            print(f"\033[96m   🎯 Ensemble: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\\033[0m")
            print(f"\033[96m   📊 Fusion LR: {current_lr:.6f} | Specialist LR: {specialist_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Consensus: {avg_consensus:.3f}\\033[0m")
            if epoch == epochs_for_stage - 1:
                print(f"\033[96m✅ Advanced Stage {stage_idx} complete! Final exact: {epoch_performance:.2%}\\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_stage_performance


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train OLYMPUS V2
    olympus, best_performance = train_olympus_ensemble_v2()