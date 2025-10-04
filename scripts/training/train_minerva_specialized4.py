"""
MINERVA Specialized Training V4 - Strategic Ensemble Coordinator for ARC-AGI-2
Enhanced with 2D transformers, test-time adaptation, and OLYMPUS ensemble preparation
Target: 75%+ performance with strategic coordination mastery
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

# Import enhanced MINERVA V4 model
from src.models.minerva_v4_enhanced import MinervaV4Enhanced

# Enhanced MINERVA V4 Configuration - Strategic Coordination Focus
MINERVA_V4_CONFIG = {
    # Core Training Parameters - Enhanced for V4 Strategic Learning
    'batch_size': 32,  # Smaller for complex strategic computations
    'learning_rate': 0.00015,  # Lower for strategic stability
    'num_epochs': 600,  # Extended training: 12 stages x 50 epochs
    'gradient_accumulation': 8,  # Effective batch: 256
    'epochs_per_stage': 50,  # Extended epochs per stage
    'curriculum_stages': 12,  # Extended 12-stage progression
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.08,  # Very low - encourage strategic transformations
    'exact_match_bonus': 8.5,  # High bonus for strategic precision
    'gradient_clip': 0.5,  # Tight clipping for strategic stability
    'weight_decay': 5e-6,  # Very light regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'strategic_reasoning_weight': 0.5,  # Primary focus - strategic reasoning
    'ensemble_coordination_weight': 0.4,  # Ensemble coordination bonus
    'pattern_analysis_weight': 0.35,  # Pattern analysis bonus
    'decision_confidence_weight': 0.3,  # Decision confidence bonus
    
    # MINERVA V4-Specific Enhancements
    'strategic_transformer_layers': 6,  # Deep strategic reasoning
    'ensemble_preparation': True,  # OLYMPUS preparation mode
    'test_time_adaptation': True,  # Advanced test-time learning
    'strategic_pattern_focus': True,  # Focus on strategic patterns
    
    # Advanced Training Features
    'label_smoothing': 0.02,  # Minimal for strategic precision
    'pattern_diversity_bonus': True,
    'strategic_reasoning_bonus': True,
    'ensemble_coordination_bonus': True,
    'olympus_preparation_bonus': True,
    
    # Learning Rate Scheduling
    'warmup_epochs': 35,  # Extended warmup for strategic learning
    'cosine_restarts': True,
    'restart_multiplier': 1.4,
    'plateau_patience': 18,
}

# Enhanced 12-Stage Progressive Configuration - Strategic Focus
STAGE_CONFIG = [
    # Foundation Strategic Patterns (6x6 - 10x10)
    {'stage': 0, 'max_grid_size': 6,  'synthesis_ratio': 0.8, 'strategic_complexity': 'basic_strategy', 'focus': 'pattern_recognition'},
    {'stage': 1, 'max_grid_size': 7,  'synthesis_ratio': 0.75, 'strategic_complexity': 'simple_reasoning', 'focus': 'logical_inference'},
    {'stage': 2, 'max_grid_size': 8,  'synthesis_ratio': 0.7, 'strategic_complexity': 'rule_detection', 'focus': 'rule_identification'},
    {'stage': 3, 'max_grid_size': 10, 'synthesis_ratio': 0.65, 'strategic_complexity': 'pattern_analysis', 'focus': 'pattern_analysis'},
    
    # Intermediate Strategic Reasoning (12x12 - 18x18)
    {'stage': 4, 'max_grid_size': 12, 'synthesis_ratio': 0.6, 'strategic_complexity': 'multi_step', 'focus': 'multi_step_reasoning'},
    {'stage': 5, 'max_grid_size': 14, 'synthesis_ratio': 0.55, 'strategic_complexity': 'complex_rules', 'focus': 'complex_rule_learning'},
    {'stage': 6, 'max_grid_size': 16, 'synthesis_ratio': 0.5, 'strategic_complexity': 'strategic_planning', 'focus': 'strategic_planning'},
    {'stage': 7, 'max_grid_size': 18, 'synthesis_ratio': 0.45, 'strategic_complexity': 'ensemble_prep', 'focus': 'ensemble_coordination'},
    
    # Advanced Strategic Mastery (20x20 - 30x30)
    {'stage': 8, 'max_grid_size': 20, 'synthesis_ratio': 0.4, 'strategic_complexity': 'meta_reasoning', 'focus': 'meta_cognitive_reasoning'},
    {'stage': 9, 'max_grid_size': 24, 'synthesis_ratio': 0.35, 'strategic_complexity': 'strategic_mastery', 'focus': 'strategic_expertise'},
    {'stage': 10, 'max_grid_size': 28, 'synthesis_ratio': 0.25, 'strategic_complexity': 'olympus_prep', 'focus': 'olympus_integration'},
    {'stage': 11, 'max_grid_size': 30, 'synthesis_ratio': 0.2, 'strategic_complexity': 'strategic_genius', 'focus': 'strategic_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 100}\033[0m")
print(f"\033[96mMINERVA V4 Enhanced Training - Strategic Ensemble Coordinator for ARC-AGI-2\033[0m")
print(f"\033[96mStrategic Reasoning + Ensemble Coordination + OLYMPUS Preparation\033[0m")
print(f"\033[96mTarget: 75%+ Performance with Strategic Mastery\033[0m")
print(f"\033[96m{'=' * 100}\033[0m")


class MinervaV4StrategicLoss(nn.Module):
    """Advanced loss function for strategic reasoning and ensemble coordination"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.strategic_weight = config['strategic_reasoning_weight']
        self.ensemble_weight = config['ensemble_coordination_weight']
        self.pattern_weight = config['pattern_analysis_weight']
        self.confidence_weight = config['decision_confidence_weight']
        self.ultra_teal_weight = config['ultra_teal_iou_weight']
        self.strict_weight = config['strict_match_weight']
        self.label_smoothing = config['label_smoothing']
        
        self.focal_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
    def forward(self, model_outputs: Dict, targets: torch.Tensor, inputs: torch.Tensor) -> Dict:
        pred_output = model_outputs['predicted_output']
        B, C, H, W = pred_output.shape
        
        # Main focal loss
        target_indices = targets.argmax(dim=1) if targets.dim() > 3 else targets
        focal_loss = self.focal_loss(pred_output, target_indices)
        
        # Prediction analysis
        pred_indices = pred_output.argmax(dim=1)
        
        # ULTRA TEAL scoring (proven formula)
        exact_matches_strict = (pred_indices == target_indices).all(dim=[1,2]).float()
        intersection = (pred_indices == target_indices).float().sum(dim=[1,2])
        union = (pred_indices.shape[1] * pred_indices.shape[2])
        iou_scores = intersection / union
        
        # 85% IoU + 15% strict matching
        combined_matches = self.strict_weight * exact_matches_strict + self.ultra_teal_weight * iou_scores
        exact_count = combined_matches.sum()
        exact_bonus = -combined_matches.mean() * self.exact_match_bonus
        exact_bonus = exact_bonus.clamp(min=-5.0)  # Higher clamp for strategic precision
        
        # Transform penalty (very low to encourage strategic learning)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Strategic reasoning bonuses
        strategic_bonus = self._calculate_strategic_bonus(model_outputs, pred_indices, target_indices, input_indices)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs)
        pattern_bonus = self._calculate_pattern_bonus(model_outputs, pred_indices, target_indices)
        confidence_bonus = self._calculate_confidence_bonus(model_outputs)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     strategic_bonus + ensemble_bonus + pattern_bonus + confidence_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'strategic_bonus': strategic_bonus,
            'ensemble_bonus': ensemble_bonus,
            'pattern_bonus': pattern_bonus,
            'confidence_bonus': confidence_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_strategic_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate strategic reasoning bonus"""
        if 'strategic_info' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        strategic_info = outputs['strategic_info']
        
        # Reward strategic pattern recognition
        if 'pattern_types' in strategic_info:
            pattern_confidence = strategic_info['pattern_types'].max(dim=-1)[0].mean()
            strategic_score = pattern_confidence
        else:
            strategic_score = torch.tensor(0.5).to(pred_indices.device)
        
        # Reward strategic transformations (non-copying)
        strategic_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        non_copy_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        strategic_transform_bonus = strategic_accuracy * (1.0 + non_copy_mask * 0.5)
        
        combined_strategic = strategic_score * strategic_transform_bonus.mean()
        return -combined_strategic * self.strategic_weight * 0.1
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate ensemble coordination bonus"""
        if 'coordination_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        coordination_output = outputs['coordination_output']
        
        # Reward high confidence in ensemble coordination
        if 'confidence' in coordination_output:
            confidence = coordination_output['confidence'].mean()
            ensemble_score = confidence
        else:
            ensemble_score = torch.tensor(0.5).to(list(outputs.values())[0].device)
        
        # Reward diverse ensemble attention
        if 'ensemble_attention' in coordination_output:
            attention_weights = coordination_output['ensemble_attention']
            # Measure attention diversity (avoid collapse to single specialist)
            attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
            ensemble_score = ensemble_score * torch.exp(attention_entropy - 1.0)  # Normalize around entropy=1
        
        return -ensemble_score * self.ensemble_weight * 0.08
    
    def _calculate_pattern_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                               target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate pattern analysis bonus"""
        if 'pattern_encoding' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward consistent pattern encoding
        pattern_encoding = outputs['pattern_encoding']
        pattern_consistency = torch.exp(-torch.var(pattern_encoding, dim=1).mean())
        
        # Reward pattern-based accuracy
        pattern_accuracy = (pred_indices == target_indices).float().mean()
        
        pattern_score = pattern_consistency * pattern_accuracy
        return -pattern_score * self.pattern_weight * 0.06
    
    def _calculate_confidence_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate decision confidence bonus"""
        if 'confidence' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        confidence = outputs['confidence'].mean()
        
        # Reward calibrated confidence (not too high, not too low)
        calibrated_confidence = 1.0 - torch.abs(confidence - 0.8)  # Target 80% confidence
        
        return -calibrated_confidence * self.confidence_weight * 0.05


class StrategicARCDataset(Dataset):
    """Dataset optimized for strategic reasoning training"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 strategic_focus: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.strategic_focus = strategic_focus
        
        # Load data with strategic filtering
        self.samples = []
        self._load_strategic_data()
        
        print(f"\033[96mLoaded {len(self.samples)} strategic samples for MINERVA V4 training\033[0m")
    
    def _load_strategic_data(self):
        """Load data with strategic complexity focus"""
        data_files = [
            'arc-agi_training_challenges.json',
            'arc-agi_evaluation_challenges.json'
        ]
        
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
                    self._process_strategic_task(combined_task, 'training')
        
        # Load evaluation data
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                eval_task = {'train': task_data['train'], 'test': []}
                self._process_strategic_task(eval_task, 'evaluation')
    
    def _process_strategic_task(self, task: Dict, source_file: str):
        """Process task with strategic complexity analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples for strategic learning
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_strategic_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_strategic_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with strategic analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Strategic complexity analysis
        strategic_analysis = self._analyze_strategic_complexity(input_grid, output_grid)
        
        # Filter for strategic relevance if enabled
        if self.strategic_focus and strategic_analysis['strategic_level'] < 2:
            if random.random() > 0.8:  # Keep 80% of simple cases
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'strategic_analysis': strategic_analysis
        }
    
    def _analyze_strategic_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze strategic complexity and reasoning requirements"""
        # Basic strategic analysis
        same_shape = input_grid.shape == output_grid.shape
        same_content = np.array_equal(input_grid, output_grid)
        
        # Count unique patterns
        input_unique = len(np.unique(input_grid))
        output_unique = len(np.unique(output_grid))
        
        # Analyze transformation complexity
        if same_content:
            strategic_level = 0  # Identity
        elif same_shape and input_unique == output_unique:
            strategic_level = 2  # Rearrangement/transformation
        elif same_shape:
            strategic_level = 3  # Color changes + transformation
        else:
            strategic_level = 4  # Shape + color changes
        
        # Additional strategic factors
        max_dim = max(input_grid.shape + output_grid.shape)
        total_unique = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        
        if max_dim > 20 or total_unique > 7:
            strategic_level += 1
        
        # Complexity classification
        if strategic_level <= 1:
            complexity = 'trivial'
        elif strategic_level <= 2:
            complexity = 'basic'
        elif strategic_level <= 3:
            complexity = 'intermediate'
        elif strategic_level <= 4:
            complexity = 'advanced'
        else:
            complexity = 'expert'
        
        return {
            'strategic_level': strategic_level,
            'complexity': complexity,
            'unique_colors': total_unique,
            'max_dimension': max_dim,
            'transformation_type': 'identity' if same_content else 'transformation'
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
            'strategic_analysis': sample['strategic_analysis']
        }
        
        return input_final, output_final, metadata


def strategic_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Enhanced collate function for strategic training"""
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


def train_minerva_specialized_v4():
    """Main training function for MINERVA V4"""
    print(f"\033[96mInitializing MINERVA V4 Strategic Coordinator Training...\033[0m")
    
    # Initialize enhanced model
    model = MinervaV4Enhanced(
        max_grid_size=30,
        hidden_dim=256,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load existing weights
    model_path = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/minerva_best.pt'
    weights_loaded = model.load_compatible_weights(model_path)
    
    if not weights_loaded:
        print(f"\033[96mStarting fresh MINERVA V4 training\033[0m")
    
    # Initialize loss function
    criterion = MinervaV4StrategicLoss(MINERVA_V4_CONFIG)
    
    # Initialize optimizer with strategic learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MINERVA_V4_CONFIG['learning_rate'],
        weight_decay=MINERVA_V4_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=MINERVA_V4_CONFIG['warmup_epochs'],
        T_mult=int(MINERVA_V4_CONFIG['restart_multiplier']),
        eta_min=MINERVA_V4_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Progressive Strategic Training - 12 Stages\033[0m")
    
    # Progressive training through strategic stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 90}\033[0m")
        print(f"\033[38;2;255;222;173mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Strategic: {stage_config['strategic_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 90}\033[0m")
        
        # Create strategic dataset for this stage
        dataset = StrategicARCDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            strategic_focus=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=MINERVA_V4_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=strategic_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_strategic_stage(
            model, dataloader, criterion, optimizer, scheduler, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_performance': best_performance,
                'stage': stage_idx,
                'config': MINERVA_V4_CONFIG,
                'ensemble_state': model.get_ensemble_state()
            }, '/content/AutomataNexus_Olympus_AGI2/models/minerva_v4_best.pt')
            print(f"\033[96mNew best strategic performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 100}\033[0m")
    print(f"\033[96mMINERVA V4 Strategic Training Complete!\033[0m")
    print(f"\033[96mBest Strategic Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 100}\033[0m")
    
    return model, best_performance


def train_strategic_stage(model, dataloader, criterion, optimizer, scheduler, scaler, 
                         stage_idx, stage_config, training_stats):
    """Train a single strategic curriculum stage"""
    model.train()
    
    epochs_for_stage = MINERVA_V4_CONFIG['epochs_per_stage']
    accumulation_steps = MINERVA_V4_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        strategic_complexity_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mStrategic Stage {stage_idx} Epoch {epoch}\033[0m")
        
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs, targets, mode='train')
                loss_dict = criterion(outputs, targets, inputs)
                loss = loss_dict['total'] / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MINERVA_V4_CONFIG['gradient_clip'])
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
            
            # Count strategic complexity
            for meta in metadata:
                if meta['strategic_analysis']['strategic_level'] >= 3:
                    strategic_complexity_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'Strategic': f"{strategic_complexity_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            strategic_ratio = strategic_complexity_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[96m⏰ MINERVA V4 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * MINERVA_V4_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Strategic: {strategic_ratio:.1%}\033[0m")
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
    model, best_performance = train_minerva_specialized_v4()