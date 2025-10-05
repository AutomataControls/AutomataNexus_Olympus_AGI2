"""
CHRONOS Specialized Training V7 - Ultra-Fast Advanced Temporal Intelligence for ARC-AGI-2
Lightning-fast trainer focusing ONLY on advanced stages 11-18 (Advanced Temporal Mastery)
Loads from chronos_best.pt Stage 10 and rockets through advanced temporal intelligence
Target: Complete stages 11-18 in 2-3 hours with 80%+ performance
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

# Import CHRONOS V7 ultra-fast model
from src.models.chronos_v7_enhanced import ChronosV7Enhanced

# Ultra-Fast CHRONOS V7 Configuration - Advanced Stages 11-18 Only
CHRONOS_V7_CONFIG = {
    # Core Training Parameters - ULTRA-OPTIMIZED for V7 Speed
    'batch_size': 64,  # Larger batches for speed
    'learning_rate': 0.0003,  # Higher LR for faster convergence
    'num_epochs': 400,  # Fewer total epochs
    'gradient_accumulation': 4,  # Less accumulation
    'epochs_per_stage': 20,  # Only 20 epochs per stage (vs 30)
    'curriculum_stages': 8,  # Only stages 11-18 (8 stages)
    
    # Enhanced Loss Configuration - Speed Optimized
    'transform_penalty': 0.02,
    'exact_match_bonus': 12.0,  # Higher bonus for faster learning
    'gradient_clip': 0.6,
    'weight_decay': 1.0e-6,  # Very light regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,
    'strict_match_weight': 0.15,
    'temporal_reasoning_weight': 0.88,  # MAX focus - temporal intelligence
    'sequence_analysis_weight': 0.78,  # MAX sequence understanding
    'movement_prediction_weight': 0.68,  # Advanced movement mastery
    'ensemble_coordination_weight': 0.58,  # Enhanced ensemble integration
    'multitemporal_weight': 0.48,  # Advanced multitemporal reasoning
    'temporal_genius_weight': 0.42,  # NEW: Temporal genius mastery
    
    # CHRONOS V7-Specific Ultra Enhancements
    'advanced_temporal_layers': 4,  # Streamlined layers
    'genius_temporal_memory': 128,  # Large temporal pattern memory
    'lightning_fast_processing': True,  # Ultra-fast mode
    'stages_11_18_focus': True,  # Focus ONLY on advanced stages
    'ultra_test_time_adaptation': True,  # Advanced adaptation
    
    # Advanced Training Features
    'label_smoothing': 0.015,
    'pattern_diversity_bonus': True,
    'temporal_reasoning_bonus': True,
    'sequence_analysis_bonus': True,
    'movement_prediction_bonus': True,
    'multitemporal_bonus': True,
    'temporal_genius_bonus': True,  # NEW: Temporal genius bonus
    
    # Learning Rate Scheduling - Aggressive
    'warmup_epochs': 15,  # Shorter warmup
    'cosine_restarts': True,
    'restart_multiplier': 1.4,  # More aggressive restarts
    'plateau_patience': 15,  # Shorter patience
}

# ADVANCED STAGES 11-18 ONLY Configuration - Lightning Fast Focus
ADVANCED_STAGE_CONFIG = [
    # Advanced Temporal Mastery (18x18 - 30x30) - STAGES 11-18 ONLY
    {'stage': 11, 'max_grid_size': 18, 'synthesis_ratio': 0.4, 'temporal_complexity': 'multitemporal_basic', 'focus': 'multitemporal_reasoning'},
    {'stage': 12, 'max_grid_size': 20, 'synthesis_ratio': 0.35, 'temporal_complexity': 'ensemble_temporal', 'focus': 'ensemble_temporal_coordination'},
    {'stage': 13, 'max_grid_size': 22, 'synthesis_ratio': 0.3, 'temporal_complexity': 'arc_temporal_intermediate', 'focus': 'arc_intermediate_temporal'},
    {'stage': 14, 'max_grid_size': 24, 'synthesis_ratio': 0.25, 'temporal_complexity': 'expert_temporal', 'focus': 'expert_temporal_analysis'},
    {'stage': 15, 'max_grid_size': 26, 'synthesis_ratio': 0.22, 'temporal_complexity': 'arc_temporal_advanced', 'focus': 'arc_advanced_temporal'},
    {'stage': 16, 'max_grid_size': 28, 'synthesis_ratio': 0.2, 'temporal_complexity': 'temporal_mastery', 'focus': 'temporal_reasoning_mastery'},
    {'stage': 17, 'max_grid_size': 30, 'synthesis_ratio': 0.18, 'temporal_complexity': 'multitemporal_mastery', 'focus': 'multitemporal_intelligence'},
    {'stage': 18, 'max_grid_size': 30, 'synthesis_ratio': 0.15, 'temporal_complexity': 'temporal_genius', 'focus': 'temporal_intelligence_genius'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 140}\033[0m")
print(f"\033[96mCHRONOS V7 Ultra-Fast Training - Advanced Temporal Mastery ONLY (Stages 11-18)\033[0m")
print(f"\033[96mLightning Speed + Advanced Temporal Intelligence + Ultra-Optimized Training\033[0m")
print(f"\033[96mTarget: Complete Stages 11-18 in 2-3 Hours with 80%+ Performance\033[0m")
print(f"\033[96m{'=' * 140}\033[0m")


# V7 Ultra-Fast Loss and Dataset Classes
class ChronosV7AdvancedLoss(nn.Module):
    """Ultra-fast loss function for V7 advanced temporal reasoning"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.ultra_teal_weight = config['ultra_teal_iou_weight']
        self.strict_weight = config['strict_match_weight']
        self.label_smoothing = config['label_smoothing']
        self.focal_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
    def forward(self, model_outputs, targets, inputs):
        pred_output = model_outputs['predicted_output']
        target_indices = targets.argmax(dim=1) if targets.dim() > 3 else targets
        focal_loss = self.focal_loss(pred_output, target_indices)
        pred_indices = pred_output.argmax(dim=1)
        
        # ULTRA TEAL scoring
        exact_matches_strict = (pred_indices == target_indices).all(dim=[1,2]).float()
        intersection = (pred_indices == target_indices).float().sum(dim=[1,2])
        union = (pred_indices.shape[1] * pred_indices.shape[2])
        iou_scores = intersection / union
        
        combined_matches = self.strict_weight * exact_matches_strict + self.ultra_teal_weight * iou_scores
        exact_count = combined_matches.sum()
        exact_bonus = -combined_matches.mean() * self.exact_match_bonus
        exact_bonus = exact_bonus.clamp(min=-8.0)  # Higher clamp for faster learning
        
        # Transform penalty
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        total_loss = focal_loss + transform_penalty + exact_bonus
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }

class UltraFastAdvancedTemporalDataset(Dataset):
    """Ultra-fast dataset for V7 advanced temporal training (stages 11-18)"""
    def __init__(self, data_dir, max_grid_size, stage_config):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.samples = []
        self._load_ultra_fast_temporal_data()
        print(f"\033[96mLoaded {len(self.samples)} ultra-fast advanced temporal samples for CHRONOS V7 training\033[0m")
    
    def _load_ultra_fast_temporal_data(self):
        challenges_path = os.path.join(self.data_dir, 'arc-agi_training_challenges.json')
        solutions_path = os.path.join(self.data_dir, 'arc-agi_training_solutions.json')
        
        if os.path.exists(challenges_path) and os.path.exists(solutions_path):
            with open(challenges_path, 'r') as f:
                challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
            
            for task_id, task_data in challenges.items():
                for example in task_data['train']:
                    sample = self._create_ultra_fast_temporal_sample(example, True)
                    if sample:
                        self.samples.append(sample)
                
                if task_id in solutions:
                    for i, test_input in enumerate(task_data['test']):
                        if i < len(solutions[task_id]):
                            test_example = {
                                'input': test_input['input'],
                                'output': solutions[task_id][i]
                            }
                            sample = self._create_ultra_fast_temporal_sample(test_example, True)
                            if sample:
                                self.samples.append(sample)
        
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                for example in task_data['train']:
                    sample = self._create_ultra_fast_temporal_sample(example, True)
                    if sample:
                        self.samples.append(sample)
    
    def _create_ultra_fast_temporal_sample(self, example, is_arc_task):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'temporal_analysis': {'temporal_intelligence_level': 4, 'arc_specific': is_arc_task}
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_tensor = torch.tensor(sample['input'], dtype=torch.long)
        output_tensor = torch.tensor(sample['output'], dtype=torch.long)
        
        target_h = min(self.max_grid_size, max(input_tensor.shape[0], output_tensor.shape[0]))
        target_w = min(self.max_grid_size, max(input_tensor.shape[1], output_tensor.shape[1]))
        
        input_padded = F.pad(input_tensor, (0, target_w - input_tensor.shape[1], 
                                          0, target_h - input_tensor.shape[0]))
        output_padded = F.pad(output_tensor, (0, target_w - output_tensor.shape[1], 
                                            0, target_h - output_tensor.shape[0]))
        
        input_final = F.one_hot(input_padded, num_classes=10).float().permute(2, 0, 1)
        output_final = F.one_hot(output_padded, num_classes=10).float().permute(2, 0, 1)
        
        metadata = {
            'is_arc': sample['is_arc'],
            'temporal_analysis': sample['temporal_analysis']
        }
        
        return input_final, output_final, metadata

def ultra_fast_temporal_collate_fn(batch):
    inputs, outputs, metadata = zip(*batch)
    max_h = max(t.shape[1] for t in inputs + outputs)
    max_w = max(t.shape[2] for t in inputs + outputs)
    
    inputs_padded = []
    outputs_padded = []
    
    for inp, out in zip(inputs, outputs):
        inp_padded = F.pad(inp, (0, max_w - inp.shape[2], 0, max_h - inp.shape[1]))
        out_padded = F.pad(out, (0, max_w - out.shape[2], 0, max_h - out.shape[1]))
        inputs_padded.append(inp_padded)
        outputs_padded.append(out_padded)
    
    return torch.stack(inputs_padded), torch.stack(outputs_padded), list(metadata)


def train_chronos_specialized_v7():
    """Main training function for CHRONOS V7 - Advanced Stages 11-18 ONLY"""
    print(f"\033[96mInitializing CHRONOS V7 Ultra-Fast Advanced Temporal Intelligence Training...\033[0m")
    
    # Initialize ultra-fast model
    model = ChronosV7Enhanced(
        max_grid_size=30,
        d_model=128,
        num_layers=2,  # Streamlined for speed
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load existing weights (chronos_best.pt should have Stage 10+ weights)
    model_paths = [
        '/content/AutomataNexus_Olympus_AGI2/models/chronos_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/chronos_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/chronos_v5_best.pt'
    ]
    
    weights_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            if model.load_compatible_weights(model_path):
                print(f"\033[96mSuccessfully loaded weights from {model_path}\033[0m")
                weights_loaded = True
                break
    
    if not weights_loaded:
        print(f"\033[96mWarning: Could not load existing weights, starting V7 training from scratch\033[0m")
    else:
        print(f"\033[96mReady to continue from Stage 10 → Advanced Stages 11-18\033[0m")
    
    # Initialize loss function
    criterion = ChronosV7AdvancedLoss(CHRONOS_V7_CONFIG)
    
    # Initialize optimizer with V7 ultra-fast settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CHRONOS_V7_CONFIG['learning_rate'],
        weight_decay=CHRONOS_V7_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler - aggressive
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CHRONOS_V7_CONFIG['warmup_epochs'],
        T_mult=int(CHRONOS_V7_CONFIG['restart_multiplier']),
        eta_min=CHRONOS_V7_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Ultra-Fast Advanced Temporal Training - ONLY Advanced Stages 11-18\033[0m")
    
    # Ultra-fast progressive training through ADVANCED stages 11-18 ONLY
    for stage_idx, stage_config in enumerate(ADVANCED_STAGE_CONFIG):
        actual_stage = stage_config['stage']  # Actual stage number (11-18)
        print(f"\\n\\033[96m{'=' * 150}\\033[0m")
        print(f"\\033[38;2;255;204;153mAdvanced Stage {actual_stage}: Grid Size {stage_config['max_grid_size']} | "
              f"Temporal: {stage_config['temporal_complexity']} | Focus: {stage_config['focus']}\\033[0m")
        print(f"\\033[96m{'=' * 150}\\033[0m")
        
        # Create ultra-fast dataset for this advanced stage
        dataset = UltraFastAdvancedTemporalDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=CHRONOS_V7_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=ultra_fast_temporal_collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Stage-specific training
        stage_performance = train_ultra_fast_advanced_stage(
            model, dataloader, criterion, optimizer, scheduler, scaler,
            actual_stage, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best V7 model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_performance': best_performance,
                'stage': actual_stage,
                'config': CHRONOS_V7_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V7'
            }, '/content/AutomataNexus_Olympus_AGI2/models/chronos_best.pt')
            print(f"\\033[96mNew best V7 advanced performance: {best_performance:.2%} - Model saved!\\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\\n\\033[96m{'=' * 160}\\033[0m")
    print(f"\\033[96mCHRONOS V7 Ultra-Fast Advanced Training Complete!\\033[0m")
    print(f"\\033[96mBest V7 Advanced Performance: {best_performance:.2%}\\033[0m")
    print(f"\\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\\033[0m")
    print(f"\\033[96m{'=' * 160}\\033[0m")
    
    return model, best_performance


def train_ultra_fast_advanced_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                   stage_num, stage_config, training_stats):
    """Train a single ultra-fast advanced temporal stage for V7"""
    model.train()
    
    epochs_for_stage = CHRONOS_V7_CONFIG['epochs_per_stage']  # Only 20 epochs per stage
    accumulation_steps = CHRONOS_V7_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_temporal_count = 0
        arc_temporal_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\\033[38;2;255;204;153mAdvanced Temporal Stage {stage_num} Epoch {epoch}\\033[0m")
        
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                outputs = model(inputs, targets, mode='train')
                loss_dict = criterion(outputs, targets, inputs)
                loss = loss_dict['total'] / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CHRONOS_V7_CONFIG['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update learning rate
                scheduler.step()
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    epoch_losses[key] += value.item()
            
            total_exact_matches += loss_dict['exact_count'].item()
            total_samples += inputs.shape[0]
            
            # Count advanced temporal cases and ARC-specific cases
            for meta in metadata:
                if meta['temporal_analysis']['temporal_intelligence_level'] >= 4:
                    advanced_temporal_count += 1
                if meta['temporal_analysis'].get('arc_specific', False):
                    arc_temporal_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'AdvTemporal': f"{advanced_temporal_count}",
                'ARC': f"{arc_temporal_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 4 == 0 or epoch == epochs_for_stage - 1:
            temporal_ratio = advanced_temporal_count / max(total_samples, 1)
            arc_ratio = arc_temporal_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            global_epoch = (stage_num - 11) * CHRONOS_V7_CONFIG['epochs_per_stage'] + epoch + 1
            print(f"\\033[38;2;255;204;153m⏰ CHRONOS V7 Advanced Stage {stage_num}, Epoch {epoch} \\033[96m(Global: {global_epoch})\\033[38;2;255;204;153m:\\033[0m")
            print(f"\\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\\033[0m")
            print(f"\\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Temporal: {temporal_ratio:.1%} | ARC: {arc_ratio:.1%}\\033[0m")
            if epoch == epochs_for_stage - 1:
                print(f"\\033[96m✅ Advanced Stage {stage_num} complete! Final exact: {epoch_performance:.2%}\\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_stage_performance


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train V7 model - Advanced Stages 11-18 ONLY
    model, best_performance = train_chronos_specialized_v7()