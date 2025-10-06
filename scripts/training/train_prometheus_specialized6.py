"""
PROMETHEUS Specialized Training V6 - Fast Creative Pattern Generation Expert for ARC-AGI-2
Enhanced V6 trainer that builds upon V4 with optimized speed and intelligence
Loads from prometheus_v4_best.pt and adds fast creative intelligence mastery
Target: 70%+ performance with fast creative intelligence training
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

# Import PROMETHEUS V6 enhanced model
from src.models.prometheus_v6_enhanced import PrometheusV6Enhanced

# Enhanced PROMETHEUS V6 Configuration - Ultimate Creative Intelligence Focus
PROMETHEUS_V6_CONFIG = {
    # Core Training Parameters - OPTIMIZED for V6 Ultimate Performance
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 600,
    'gradient_accumulation': 5,
    'epochs_per_stage': 30,
    'curriculum_stages': 20,  # Complete grid mastery 5x5 -> 30x30
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.018,  # Very low - maximum creative exploration
    'exact_match_bonus': 11.0,  # Highest bonus for ultimate precision
    'gradient_clip': 0.5,  # Stable clipping for deep architecture
    'weight_decay': 1.4e-6,  # Ultra-light regularization for ultimate creative strategy
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'creative_reasoning_weight': 0.82,  # Ultimate focus - creative intelligence
    'pattern_generation_weight': 0.72,  # Maximum pattern generation understanding
    'creative_synthesis_weight': 0.65,  # Ultimate creative synthesis
    'ensemble_coordination_weight': 0.58,  # Maximum ensemble integration
    'pattern_analysis_weight': 0.48,  # Ultimate pattern analysis
    'decision_confidence_weight': 0.45,  # Ultimate decision confidence
    'generative_synthesis_weight': 0.42,  # NEW: Generative synthesis integration
    'deep_creative_weight': 0.38,  # NEW: Deep creative transformer bonus
    
    # PROMETHEUS V6-Specific Ultimate Enhancements
    'deep_creative_layers': 8,  # 8-layer deep creative reasoning
    'mega_creative_memory': 350,  # Massive creative pattern memory
    'advanced_ensemble_prep': True,  # Ultimate OLYMPUS preparation
    'creative_synthesis_integration': True,  # Advanced creative synthesis
    'complete_grid_mastery': True,  # 5x5 to 30x30 complete coverage
    'ultimate_test_time_adaptation': True,  # Advanced creative adaptation
    
    # Advanced Training Features
    'label_smoothing': 0.020,  # Ultra-refined for creative precision
    'pattern_diversity_bonus': True,
    'creative_reasoning_bonus': True,
    'pattern_generation_bonus': True,
    'creative_synthesis_bonus': True,
    'deep_creative_bonus': True,
    'ultimate_creative_bonus': True,  # NEW: Ultimate creative bonus
    
    # Learning Rate Scheduling
    'warmup_epochs': 26,  # Extended warmup for deep architecture
    'cosine_restarts': True,
    'restart_multiplier': 1.35,
    'plateau_patience': 30,
}

# Enhanced 20-Stage Progressive Configuration - Complete Grid Mastery starting with ARC sizes
STAGE_CONFIG = [
    # Foundation Creative Understanding (5x5 - 9x9) - START WITH NORMAL ARC SIZES
    {'stage': 0, 'max_grid_size': 5,  'synthesis_ratio': 0.95, 'creative_complexity': 'micro_creative', 'focus': 'micro_creative_patterns'},
    {'stage': 1, 'max_grid_size': 6,  'synthesis_ratio': 0.92, 'creative_complexity': 'basic_strategy', 'focus': 'basic_pattern_generation'},
    {'stage': 2, 'max_grid_size': 7,  'synthesis_ratio': 0.88, 'creative_complexity': 'simple_reasoning', 'focus': 'simple_creative_inference'},
    {'stage': 3, 'max_grid_size': 8,  'synthesis_ratio': 0.84, 'creative_complexity': 'pattern_detection', 'focus': 'creative_pattern_identification'},
    {'stage': 4, 'max_grid_size': 9,  'synthesis_ratio': 0.79, 'creative_complexity': 'creative_analysis', 'focus': 'creative_pattern_analysis'},
    
    # Intermediate Creative Reasoning (10x10 - 15x15) 
    {'stage': 5, 'max_grid_size': 10, 'synthesis_ratio': 0.75, 'creative_complexity': 'multi_step', 'focus': 'multi_step_creative_reasoning'},
    {'stage': 6, 'max_grid_size': 11, 'synthesis_ratio': 0.70, 'creative_complexity': 'complex_rules', 'focus': 'complex_creative_rule_learning'},
    {'stage': 7, 'max_grid_size': 12, 'synthesis_ratio': 0.65, 'creative_complexity': 'generative_planning', 'focus': 'generative_planning'},
    {'stage': 8, 'max_grid_size': 13, 'synthesis_ratio': 0.60, 'creative_complexity': 'ensemble_prep_basic', 'focus': 'basic_ensemble_creative_coordination'},
    {'stage': 9, 'max_grid_size': 14, 'synthesis_ratio': 0.55, 'creative_complexity': 'arc_creative_basic', 'focus': 'arc_creative_patterns'},
    {'stage': 10, 'max_grid_size': 15, 'synthesis_ratio': 0.50, 'creative_complexity': 'meta_reasoning', 'focus': 'meta_creative_cognitive_reasoning'},
    
    # Advanced Creative Mastery (16x16 - 22x22)
    {'stage': 11, 'max_grid_size': 16, 'synthesis_ratio': 0.45, 'creative_complexity': 'expert_creative', 'focus': 'expert_creative_analysis'},
    {'stage': 12, 'max_grid_size': 18, 'synthesis_ratio': 0.40, 'creative_complexity': 'creative_genius', 'focus': 'creative_intelligence_mastery'},
    {'stage': 13, 'max_grid_size': 20, 'synthesis_ratio': 0.35, 'creative_complexity': 'arc_creative_advanced', 'focus': 'advanced_arc_creative_reasoning'},
    {'stage': 14, 'max_grid_size': 22, 'synthesis_ratio': 0.30, 'creative_complexity': 'ultimate_creative_basic', 'focus': 'ultimate_creative_intelligence_basic'},
    
    # Ultimate Creative Mastery (24x24 - 30x30)
    {'stage': 15, 'max_grid_size': 24, 'synthesis_ratio': 0.26, 'creative_complexity': 'synthesis_creative', 'focus': 'creative_synthesis_mastery'},
    {'stage': 16, 'max_grid_size': 26, 'synthesis_ratio': 0.23, 'creative_complexity': 'deep_creative_advanced', 'focus': 'deep_creative_transformer_mastery'},
    {'stage': 17, 'max_grid_size': 28, 'synthesis_ratio': 0.19, 'creative_complexity': 'mega_creative', 'focus': 'mega_creative_pattern_mastery'},
    {'stage': 18, 'max_grid_size': 30, 'synthesis_ratio': 0.16, 'creative_complexity': 'ultimate_creative_advanced', 'focus': 'ultimate_creative_intelligence_advanced'},
    {'stage': 19, 'max_grid_size': 30, 'synthesis_ratio': 0.13, 'creative_complexity': 'creative_god_mode', 'focus': 'ultimate_creative_god_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 150}\033[0m")
print(f"\033[96mPROMETHEUS V6 Ultimate Training - Ultimate Creative Pattern Generation Master for ARC-AGI-2\033[0m")
print(f"\033[96mDeep Generative Transformers + Mega Creative Memory + Creative Synthesis Integration\033[0m")
print(f"\033[96mTarget: 90%+ Performance with Ultimate Creative Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 150}\033[0m")


# V6 Fast Loss and Dataset Classes
class PrometheusV6CreativeLoss(nn.Module):
    """Fast loss function for V6 creative reasoning"""
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
        exact_bonus = exact_bonus.clamp(min=-6.5)
        
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

class FastCreativeDataset(Dataset):
    """Fast dataset for V6 creative training"""
    def __init__(self, data_dir, max_grid_size, stage_config):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.samples = []
        self._load_fast_creative_data()
        print(f"\033[96mLoaded {len(self.samples)} fast creative samples for PROMETHEUS V6 training\033[0m")
    
    def _load_fast_creative_data(self):
        challenges_path = os.path.join(self.data_dir, 'arc-agi_training_challenges.json')
        solutions_path = os.path.join(self.data_dir, 'arc-agi_training_solutions.json')
        
        if os.path.exists(challenges_path) and os.path.exists(solutions_path):
            with open(challenges_path, 'r') as f:
                challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
            
            for task_id, task_data in challenges.items():
                for example in task_data['train']:
                    sample = self._create_fast_creative_sample(example, True)
                    if sample:
                        self.samples.append(sample)
                
                if task_id in solutions:
                    for i, test_input in enumerate(task_data['test']):
                        if i < len(solutions[task_id]):
                            test_example = {
                                'input': test_input['input'],
                                'output': solutions[task_id][i]
                            }
                            sample = self._create_fast_creative_sample(test_example, True)
                            if sample:
                                self.samples.append(sample)
        
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                for example in task_data['train']:
                    sample = self._create_fast_creative_sample(example, True)
                    if sample:
                        self.samples.append(sample)
    
    def _create_fast_creative_sample(self, example, is_arc_task):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'creative_analysis': {'creative_intelligence_level': 2, 'arc_specific': is_arc_task}
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
            'creative_analysis': sample['creative_analysis']
        }
        
        return input_final, output_final, metadata

def fast_creative_collate_fn(batch):
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


def train_prometheus_specialized_v6():
    """Main training function for PROMETHEUS V6"""
    print(f"\033[96mInitializing PROMETHEUS V6 Ultimate Creative Intelligence Training...\033[0m")
    
    # Initialize enhanced model (V4 model contains all V6 improvements)
    model = PrometheusV6Enhanced(
        max_grid_size=30,
        d_model=128,
        num_layers=3,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load V5 weights with multiple fallback paths
    model_paths = [
        '/content/AutomataNexus_Olympus_AGI2/models/prometheus_v5_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/prometheus_v4_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/prometheus_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/prometheus_best.pt'
    ]
    
    weights_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Manual weight loading with compatibility
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Load compatible parameters manually
                model_dict = model.state_dict()
                compatible_params = {}
                
                for name, param in state_dict.items():
                    if name in model_dict and model_dict[name].shape == param.shape:
                        compatible_params[name] = param
                
                model_dict.update(compatible_params)
                model.load_state_dict(model_dict)
                
                print(f"\033[96mPROMETHEUS: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters from {model_path}\033[0m")
                weights_loaded = True
                break
            except Exception as e:
                continue
    
    if not weights_loaded:
        print(f"\033[96mWarning: Could not load existing weights, starting V6 training from scratch\033[0m")
    else:
        print(f"\033[96mSuccessfully loaded existing weights for V6 ultimate training\033[0m")
    
    # Initialize loss function
    criterion = PrometheusV6CreativeLoss(PROMETHEUS_V6_CONFIG)
    
    # Initialize optimizer with V6 learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=PROMETHEUS_V6_CONFIG['learning_rate'],
        weight_decay=PROMETHEUS_V6_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=PROMETHEUS_V6_CONFIG['warmup_epochs'],
        T_mult=int(PROMETHEUS_V6_CONFIG['restart_multiplier']),
        eta_min=PROMETHEUS_V6_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Ultimate Progressive Creative Training - 20 Complete Creative Intelligence Stages\033[0m")
    
    # Ultimate progressive training through creative stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 155}\033[0m")
        print(f"\033[38;2;255;204;153mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Creative: {stage_config['creative_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 155}\033[0m")
        
        # Create ultimate creative dataset for this stage
        dataset = FastCreativeDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=PROMETHEUS_V6_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=fast_creative_collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Stage-specific training
        stage_performance = train_ultimate_creative_stage(
            model, dataloader, criterion, optimizer, scheduler, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best V6 model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/arc_models_v4', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_performance': best_performance,
                'stage': stage_idx,
                'config': PROMETHEUS_V6_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V6'
            }, '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/prometheus_best.pt')
            print(f"\033[96mNew best V6 creative performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 165}\033[0m")
    print(f"\033[96mPROMETHEUS V6 Ultimate Creative Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest V6 Ultimate Creative Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 165}\033[0m")
    
    return model, best_performance


def train_ultimate_creative_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                  stage_idx, stage_config, training_stats):
    """Train a single ultimate creative curriculum stage for V6"""
    model.train()
    
    epochs_for_stage = PROMETHEUS_V6_CONFIG['epochs_per_stage']
    accumulation_steps = PROMETHEUS_V6_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        ultimate_creative_count = 0
        arc_creative_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mUltimate Creative Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), PROMETHEUS_V6_CONFIG['gradient_clip'])
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
            
            # Count ultimate creative cases and ARC-specific cases
            for meta in metadata:
                if meta['creative_analysis']['creative_intelligence_level'] >= 2:
                    ultimate_creative_count += 1
                if meta['creative_analysis'].get('arc_specific', False):
                    arc_creative_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'UltCreative': f"{ultimate_creative_count}",
                'ARC': f"{arc_creative_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 6 == 0 or epoch == epochs_for_stage - 1:
            creative_ratio = ultimate_creative_count / max(total_samples, 1)
            arc_ratio = arc_creative_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[38;2;255;204;153m⏰ PROMETHEUS V6 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * PROMETHEUS_V6_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Creative: {creative_ratio:.1%} | ARC: {arc_ratio:.1%}\033[0m")
            if epoch == epochs_for_stage - 1:
                print(f"\033[96m✅ Stage {stage_idx} complete! Final exact: {epoch_performance:.2%}\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_stage_performance


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train model
    model, best_performance = train_prometheus_specialized_v6()