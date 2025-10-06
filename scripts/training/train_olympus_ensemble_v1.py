"""
OLYMPUS Ensemble Training V1 - Foundation Multi-Specialist Coordination for ARC-AGI-2
Basic ensemble training with all 5 specialists working together
Establishes fundamental fusion and coordination protocols
Target: 85%+ performance with basic ensemble synergy
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

# OLYMPUS V1 Configuration - Foundation Ensemble Training
OLYMPUS_V1_CONFIG = {
    # Core Training Parameters - Foundation Level
    'batch_size': 8,  # Small batches for stability with 5 models
    'learning_rate': 0.0001,  # Conservative for ensemble coordination
    'num_epochs': 300,  # Foundation training: 15 stages x 20 epochs
    'gradient_accumulation': 4,  # Effective batch 32 for stability
    'epochs_per_stage': 20,  # Foundation epochs per stage
    'curriculum_stages': 15,  # Comprehensive curriculum stages
    
    # Enhanced Loss Configuration
    'ensemble_loss_weight': 1.0,  # Primary ensemble loss
    'specialist_sync_weight': 0.3,  # Specialist synchronization bonus
    'consensus_weight': 0.2,  # Consensus building bonus
    'fusion_regularization': 0.1,  # Fusion network regularization
    'transform_penalty': 0.05,  # Encourage transformations
    'exact_match_bonus': 8.0,  # Foundation exact match bonus
    'gradient_clip': 0.5,  # Stable gradient clipping
    'weight_decay': 2e-6,  # Light regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    
    # OLYMPUS V1-Specific Settings
    'freeze_specialists': True,  # Freeze specialist weights initially
    'fusion_training_only': True,  # Train only fusion components
    'consensus_threshold': 0.6,  # Minimum consensus for high confidence
    'specialist_dropout': 0.0,  # No dropout in V1
    'ensemble_coordination': True,  # Enable coordination training
    
    # Advanced Training Features
    'label_smoothing': 0.01,  # Light smoothing for ensemble
    'ensemble_diversity_bonus': True,
    'specialist_agreement_bonus': True,
    'consensus_building_bonus': True,
    'fusion_optimization': True,
    
    # Learning Rate Scheduling
    'warmup_epochs': 10,  # Foundation warmup
    'cosine_restarts': True,
    'restart_multiplier': 1.2,
    'plateau_patience': 15,
}

# Comprehensive 15-Stage Progressive Configuration - Matching Specialist Training
STAGE_CONFIG = [
    # Foundation Ensemble Coordination (4x4 - 8x8) 
    {'stage': 0, 'max_grid_size': 4,  'synthesis_ratio': 0.95, 'complexity': 'micro_ensemble', 'focus': 'micro_grid_specialist_coordination'},
    {'stage': 1, 'max_grid_size': 5,  'synthesis_ratio': 0.9,  'complexity': 'basic_shapes', 'focus': 'basic_ensemble_shape_coordination'},
    {'stage': 2, 'max_grid_size': 6,  'synthesis_ratio': 0.85, 'complexity': 'simple_fusion', 'focus': 'simple_decision_fusion_learning'},
    {'stage': 3, 'max_grid_size': 7,  'synthesis_ratio': 0.8,  'complexity': 'pattern_sync', 'focus': 'pattern_synchronization_training'},
    {'stage': 4, 'max_grid_size': 8,  'synthesis_ratio': 0.75, 'complexity': 'consensus_basic', 'focus': 'basic_specialist_consensus'},
    
    # Intermediate Ensemble Coordination (9x9 - 16x16)
    {'stage': 5, 'max_grid_size': 9,  'synthesis_ratio': 0.7,  'complexity': 'fusion_intermediate', 'focus': 'intermediate_fusion_protocols'},
    {'stage': 6, 'max_grid_size': 10, 'synthesis_ratio': 0.65, 'complexity': 'composite_ensemble', 'focus': 'composite_ensemble_decisions'},
    {'stage': 7, 'max_grid_size': 11, 'synthesis_ratio': 0.6,  'complexity': 'coordination_scaling', 'focus': 'scaling_coordination_protocols'},
    {'stage': 8, 'max_grid_size': 12, 'synthesis_ratio': 0.55, 'complexity': 'complex_consensus', 'focus': 'complex_consensus_building'},
    {'stage': 9, 'max_grid_size': 14, 'synthesis_ratio': 0.5,  'complexity': 'pattern_ensemble', 'focus': 'pattern_ensemble_coordination'},
    {'stage': 10, 'max_grid_size': 16, 'synthesis_ratio': 0.45, 'complexity': 'ensemble_intelligence', 'focus': 'ensemble_intelligence_emergence'},
    
    # Advanced Ensemble Mastery (18x18 - 30x30)
    {'stage': 11, 'max_grid_size': 18, 'synthesis_ratio': 0.4,  'complexity': 'multiscale_ensemble', 'focus': 'multiscale_ensemble_reasoning'},
    {'stage': 12, 'max_grid_size': 22, 'synthesis_ratio': 0.35, 'complexity': 'advanced_coordination', 'focus': 'advanced_coordination_protocols'},
    {'stage': 13, 'max_grid_size': 27, 'synthesis_ratio': 0.3,  'complexity': 'ensemble_mastery', 'focus': 'ensemble_coordination_mastery'},
    {'stage': 14, 'max_grid_size': 30, 'synthesis_ratio': 0.25, 'complexity': 'olympus_foundation', 'focus': 'foundation_olympus_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 120}\033[0m")
print(f"\033[96m🏛️ OLYMPUS Ensemble Training V1 - Foundation Multi-Specialist Coordination for ARC-AGI-2\033[0m")
print(f"\033[96mAll 5 Specialists + Decision Fusion + Ensemble Coordination Training\033[0m")
print(f"\033[96mTarget: 85%+ Performance with Foundation Ensemble Synergy\033[0m")
print(f"\033[96m{'=' * 120}\033[0m")


class OlympusV1Loss(nn.Module):
    """Foundation loss function for OLYMPUS ensemble training"""
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
        
    def forward(self, ensemble_decision: EnsembleDecision, targets: torch.Tensor, inputs: torch.Tensor) -> Dict:
        """Calculate comprehensive OLYMPUS ensemble loss"""
        pred_output = ensemble_decision.prediction
        B = pred_output.shape[0]
        
        # Main ensemble loss
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
        exact_bonus = exact_bonus.clamp(min=-6.0)
        
        # Specialist synchronization loss
        specialist_predictions = ensemble_decision.specialist_predictions
        sync_loss = 0.0
        if len(specialist_predictions) > 1:
            pred_values = list(specialist_predictions.values())
            for i, pred1 in enumerate(pred_values):
                for j, pred2 in enumerate(pred_values[i+1:], i+1):
                    # Encourage similarity between specialist predictions
                    pred1_flat = pred1.view(B, -1)[:, :10] if pred1.numel() > B*10 else pred1.view(B, -1)
                    pred2_flat = pred2.view(B, -1)[:, :10] if pred2.numel() > B*10 else pred2.view(B, -1)
                    
                    if pred1_flat.shape == pred2_flat.shape:
                        sync_loss += F.mse_loss(pred1_flat, pred2_flat)
        
        # Consensus bonus
        consensus_score = ensemble_decision.consensus_score
        consensus_bonus = -torch.tensor(consensus_score, device=pred_output.device) * self.consensus_weight
        
        # Fusion regularization (encourage diverse but coordinated fusion weights)
        fusion_weights = list(ensemble_decision.fusion_weights.values())
        if len(fusion_weights) > 1:
            fusion_tensor = torch.tensor(fusion_weights, device=pred_output.device)
            # Encourage balanced but not uniform weights
            fusion_entropy = -(fusion_tensor * torch.log(fusion_tensor + 1e-8)).sum()
            fusion_reg = -fusion_entropy * self.fusion_reg_weight  # Negative to encourage diversity
        else:
            fusion_reg = torch.tensor(0.0, device=pred_output.device)
        
        # Transform penalty (encourage non-trivial solutions)
        if inputs.dim() > 1:
            input_flat = inputs.view(B, -1)[:, :10] if inputs.numel() > B*10 else inputs.view(B, -1)
            copy_penalty = F.mse_loss(pred_flat, input_flat) * self.transform_penalty
        else:
            copy_penalty = torch.tensor(0.0, device=pred_output.device)
        
        total_loss = (ensemble_loss + exact_bonus + sync_loss * self.sync_weight + 
                     consensus_bonus + fusion_reg + copy_penalty)
        
        return {
            'total': total_loss,
            'ensemble': ensemble_loss,
            'sync': sync_loss * self.sync_weight,
            'consensus_bonus': consensus_bonus,
            'fusion_reg': fusion_reg,
            'exact_bonus': exact_bonus,
            'copy_penalty': copy_penalty,
            'exact_count': exact_count,
            'consensus_score': consensus_score
        }


class FoundationEnsembleDataset(Dataset):
    """Foundation dataset for OLYMPUS ensemble training"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        
        # Load data for ensemble training
        self.samples = []
        self._load_foundation_data()
        
        print(f"\033[96m🏛️ Loaded {len(self.samples)} foundation samples for OLYMPUS V1 training\033[0m")
    
    def _load_foundation_data(self):
        """Load ARC data for foundation ensemble training"""
        # Load training data (challenges + solutions)
        challenges_path = os.path.join(self.data_dir, 'arc-agi_training_challenges.json')
        solutions_path = os.path.join(self.data_dir, 'arc-agi_training_solutions.json')
        
        if os.path.exists(challenges_path) and os.path.exists(solutions_path):
            with open(challenges_path, 'r') as f:
                challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
            
            for task_id, task_data in challenges.items():
                if task_id in solutions:
                    combined_task = {
                        'train': task_data['train'],
                        'test': []
                    }
                    for i, test_input in enumerate(task_data['test']):
                        if i < len(solutions[task_id]):
                            combined_task['test'].append({
                                'input': test_input['input'],
                                'output': solutions[task_id][i]
                            })
                    self._process_foundation_task(combined_task)
        
        # Load evaluation data
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                eval_task = {'train': task_data['train'], 'test': []}
                self._process_foundation_task(eval_task)
    
    def _process_foundation_task(self, task: Dict):
        """Process task for foundation ensemble training"""
        # Process all examples for ensemble learning
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_foundation_sample(example)
            if sample:
                self.samples.append(sample)
    
    def _create_foundation_sample(self, example: Dict) -> Optional[Dict]:
        """Create sample for foundation ensemble training"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': True,
            'complexity': self.stage_config.get('complexity', 'foundation')
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        
        # Convert to tensors
        input_tensor = torch.tensor(sample['input'], dtype=torch.long)
        output_tensor = torch.tensor(sample['output'], dtype=torch.long)
        
        # Pad to consistent size
        target_h = min(self.max_grid_size, max(input_tensor.shape[0], output_tensor.shape[0]))
        target_w = min(self.max_grid_size, max(input_tensor.shape[1], output_tensor.shape[1]))
        
        input_padded = F.pad(input_tensor, (0, target_w - input_tensor.shape[1], 
                                          0, target_h - input_tensor.shape[0]))
        output_padded = F.pad(output_tensor, (0, target_w - output_tensor.shape[1], 
                                            0, target_h - output_tensor.shape[0]))
        
        # Convert to one-hot
        input_final = F.one_hot(input_padded, num_classes=10).float().permute(2, 0, 1)
        output_final = F.one_hot(output_padded, num_classes=10).float().permute(2, 0, 1)
        
        metadata = {
            'is_arc': sample['is_arc'],
            'complexity': sample['complexity']
        }
        
        return input_final, output_final, metadata


def foundation_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Collate function for foundation ensemble training"""
    inputs, outputs, metadata = zip(*batch)
    
    # Find maximum dimensions for padding
    max_h = max(t.shape[1] for t in inputs + outputs)
    max_w = max(t.shape[2] for t in inputs + outputs)
    
    # Pad all tensors to same size
    inputs_padded = []
    outputs_padded = []
    
    for inp, out in zip(inputs, outputs):
        inp_padded = F.pad(inp, (0, max_w - inp.shape[2], 0, max_h - inp.shape[1]))
        out_padded = F.pad(out, (0, max_w - out.shape[2], 0, max_h - out.shape[1]))
        
        inputs_padded.append(inp_padded)
        outputs_padded.append(out_padded)
    
    return torch.stack(inputs_padded), torch.stack(outputs_padded), list(metadata)


def train_olympus_ensemble_v1():
    """Main training function for OLYMPUS Ensemble V1"""
    print(f"\033[96m🏛️ Initializing OLYMPUS Ensemble V1 Training...\033[0m")
    
    # Initialize OLYMPUS ensemble
    olympus = OlympusEnsemble(
        max_grid_size=30,
        d_model=256,
        device=device
    ).to(device)
    
    # Load all specialist weights from InputBestModels directory
    weight_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
    load_results = olympus.load_all_specialists(weight_dir)
    successful_loads = sum(load_results.values())
    print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 specialist models\033[0m")
    
    # Freeze specialist parameters for V1 (train only fusion components)
    if OLYMPUS_V1_CONFIG['freeze_specialists']:
        for name, specialist in olympus.specialists.items():
            for param in specialist.parameters():
                param.requires_grad = False
        print(f"\033[96m🏛️ Specialist weights frozen - training fusion components only\033[0m")
    
    # Initialize loss function
    criterion = OlympusV1Loss(OLYMPUS_V1_CONFIG)
    
    # Initialize optimizer (only fusion parameters if specialists are frozen)
    if OLYMPUS_V1_CONFIG['fusion_training_only']:
        fusion_params = list(olympus.fusion_engine.parameters())
        trainable_params = fusion_params
        print(f"\033[96m🏛️ Training {len(fusion_params)} fusion parameters\033[0m")
    else:
        trainable_params = list(olympus.parameters())
        print(f"\033[96m🏛️ Training all {len(trainable_params)} parameters\033[0m")
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=OLYMPUS_V1_CONFIG['learning_rate'],
        weight_decay=OLYMPUS_V1_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=OLYMPUS_V1_CONFIG['warmup_epochs'],
        T_mult=int(OLYMPUS_V1_CONFIG['restart_multiplier']),
        eta_min=OLYMPUS_V1_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96m🏛️ Starting Foundation Progressive Ensemble Training - 6 Foundation Stages\033[0m")
    
    # Progressive training through foundation stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 125}\033[0m")
        print(f"\033[38;2;255;204;153m🏛️ Stage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Complexity: {stage_config['complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 125}\033[0m")
        
        # Create foundation dataset for this stage
        dataset = FoundationEnsembleDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=OLYMPUS_V1_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=foundation_collate_fn,
            num_workers=0,  # Keep 0 for OLYMPUS stability
            pin_memory=True
        )
        
        # Stage-specific training
        stage_performance = train_foundation_stage(
            olympus, dataloader, criterion, optimizer, scheduler, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best OLYMPUS model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            olympus.save_ensemble('/content/AutomataNexus_Olympus_AGI2/models/olympus_v1_best.pt')
            print(f"\033[96m🏛️ New best V1 ensemble performance: {best_performance:.2%} - OLYMPUS saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 130}\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Ensemble V1 Foundation Training Complete!\033[0m")
    print(f"\033[96m🏛️ Best V1 Foundation Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96m🏛️ All 5 Specialists Coordinated and Ready for Advanced Training\033[0m")
    print(f"\033[96m{'=' * 130}\033[0m")
    
    return olympus, best_performance


def train_foundation_stage(olympus, dataloader, criterion, optimizer, scheduler, scaler,
                          stage_idx, stage_config, training_stats):
    """Train a single foundation ensemble stage"""
    olympus.train()
    
    epochs_for_stage = OLYMPUS_V1_CONFIG['epochs_per_stage']
    accumulation_steps = OLYMPUS_V1_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        total_consensus = 0.0
        
        # Dynamic progress bar with stage focus (like ATLAS)
        stage_focus = stage_config['focus'].replace('_', ' ').title()
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153m🏛️ {stage_focus} Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in olympus.parameters() if p.requires_grad], 
                    OLYMPUS_V1_CONFIG['gradient_clip']
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update learning rate
                scheduler.step()
            
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
                'Specialists': f"{len(ensemble_decision.specialist_predictions)}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        avg_consensus = total_consensus / len(dataloader)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[38;2;255;204;153m⏰ OLYMPUS V1 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * OLYMPUS_V1_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Ensemble: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Consensus: {avg_consensus:.3f}\033[0m")
            if epoch == epochs_for_stage - 1:
                print(f"\033[96m✅ Foundation Stage {stage_idx} complete! Final exact: {epoch_performance:.2%}\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_stage_performance


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train OLYMPUS V1
    olympus, best_performance = train_olympus_ensemble_v1()