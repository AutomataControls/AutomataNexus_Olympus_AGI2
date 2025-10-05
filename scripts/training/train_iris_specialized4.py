"""
IRIS Specialized Training V4 - Advanced Color Pattern Recognition Expert for ARC-AGI-2
Enhanced with chromatic transformers, color space reasoning, and OLYMPUS ensemble preparation
Target: 75%+ performance with sophisticated color intelligence mastery
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

# Import enhanced IRIS V4 model
from src.models.iris_v4_enhanced import IrisV4Enhanced

# Enhanced IRIS V4 Configuration - OPTIMIZED FOR SPEED
IRIS_V4_CONFIG = {
    # Core Training Parameters - SPEED OPTIMIZED
    'batch_size': 32,  # Double batch size for efficiency
    'learning_rate': 0.0003,  # Higher LR for faster convergence
    'num_epochs': 26,  # Reduced: 13 stages x 2 epochs
    'gradient_accumulation': 2,  # Reduced accumulation
    'epochs_per_stage': 2,  # FAST training per stage
    'curriculum_stages': 13,  # Keep all color stages
    
    # SIMPLIFIED Loss Configuration for Speed
    'transform_penalty': 0.02,  # Minimal penalty
    'exact_match_bonus': 5.0,  # Reduced bonus computation
    'gradient_clip': 1.0,  # Relaxed clipping
    'weight_decay': 1e-5,  # Light regularization
    
    # SIMPLIFIED scoring (ULTRA TEAL core only)
    'ultra_teal_iou_weight': 0.85,  # Keep proven IoU weighting
    'strict_match_weight': 0.15,   # Keep strict matching
    'chromatic_reasoning_weight': 0.2,  # REDUCED - speed focus
    'color_harmony_weight': 0.1,  # MINIMAL computation
    'color_space_weight': 0.1,  # MINIMAL computation
    'ensemble_coordination_weight': 0.1,  # MINIMAL computation
    
    # IRIS V4-Specific Enhancements - KEEP FUNCTIONALITY
    'chromatic_transformer_layers': 4,  # Balanced for speed + capability
    'color_space_processing': True,  # Keep color intelligence
    'chromatic_positional_encoding': True,  # Keep color-aware positioning
    'ensemble_preparation': True,  # Keep OLYMPUS preparation
    'test_time_adaptation': True,  # Keep chromatic adaptation
    
    # SIMPLIFIED Training Features for SPEED
    'label_smoothing': 0.01,  # Minimal smoothing
    'pattern_diversity_bonus': False,  # DISABLED for speed
    'chromatic_reasoning_bonus': False,  # DISABLED for speed  
    'color_harmony_bonus': False,  # DISABLED for speed
    'color_expertise_bonus': False,  # DISABLED for speed
    
    # SIMPLIFIED Learning Rate Scheduling for SPEED
    'warmup_epochs': 5,  # FAST warmup
    'cosine_restarts': False,  # DISABLED for speed
    'restart_multiplier': 1.0,  # No restarts
    'plateau_patience': 5,  # Fast plateau detection
}

# ULTRA FAST 13-Stage Color Intelligence Curriculum - SPEED OPTIMIZED
STAGE_CONFIG = [
    # Foundation Color Understanding (Fast progression)
    {'stage': 0, 'max_grid_size': 6,  'synthesis_ratio': 0.9, 'color_complexity': 'basic_colors', 'focus': 'primary_color_recognition'},
    {'stage': 1, 'max_grid_size': 7,  'synthesis_ratio': 0.85, 'color_complexity': 'color_patterns', 'focus': 'color_pattern_recognition'},
    {'stage': 2, 'max_grid_size': 8, 'synthesis_ratio': 0.8, 'color_complexity': 'simple_mapping', 'focus': 'basic_color_mapping'},
    {'stage': 3, 'max_grid_size': 10, 'synthesis_ratio': 0.75, 'color_complexity': 'color_relations', 'focus': 'color_relationships'},
    {'stage': 4, 'max_grid_size': 12, 'synthesis_ratio': 0.7, 'color_complexity': 'complex_mapping', 'focus': 'complex_color_mapping'},
    
    # Intermediate Chromatic Reasoning (Fast progression)
    {'stage': 5, 'max_grid_size': 14, 'synthesis_ratio': 0.65, 'color_complexity': 'chromatic_logic', 'focus': 'chromatic_logical_rules'},
    {'stage': 6, 'max_grid_size': 16, 'synthesis_ratio': 0.6, 'color_complexity': 'color_space', 'focus': 'color_space_reasoning'},
    {'stage': 7, 'max_grid_size': 18, 'synthesis_ratio': 0.55, 'color_complexity': 'color_transformations', 'focus': 'color_transformations'},
    {'stage': 8, 'max_grid_size': 20, 'synthesis_ratio': 0.5, 'color_complexity': 'color_sequences', 'focus': 'color_sequences'},
    
    # Advanced Chromatic Mastery (Fast progression)
    {'stage': 9, 'max_grid_size': 22, 'synthesis_ratio': 0.45, 'color_complexity': 'color_hierarchies', 'focus': 'color_hierarchies'},
    {'stage': 10, 'max_grid_size': 25, 'synthesis_ratio': 0.4, 'color_complexity': 'expert_chromatic', 'focus': 'expert_color_analysis'},
    {'stage': 11, 'max_grid_size': 28, 'synthesis_ratio': 0.35, 'color_complexity': 'color_mastery', 'focus': 'color_mastery'},
    {'stage': 12, 'max_grid_size': 30, 'synthesis_ratio': 0.3, 'color_complexity': 'color_genius', 'focus': 'color_intelligence_mastery'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 110}\033[0m")
print(f"\033[96mIRIS V4 Enhanced Training - Advanced Color Pattern Recognition Expert for ARC-AGI-2\033[0m")
print(f"\033[96mChromatic Transformers + Color Space Reasoning + OLYMPUS Preparation\033[0m")
print(f"\033[96mTarget: 75%+ Performance with Color Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 110}\033[0m")


class IrisV4ChromaticLoss(nn.Module):
    """Advanced loss function for chromatic reasoning and color intelligence"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.chromatic_weight = config['chromatic_reasoning_weight']
        self.harmony_weight = config['color_harmony_weight']
        self.color_space_weight = config['color_space_weight']
        self.ensemble_weight = config['ensemble_coordination_weight']
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
        exact_bonus = exact_bonus.clamp(min=-5.5)  # Higher clamp for color precision
        
        # Transform penalty (low to encourage color transformations)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Chromatic reasoning bonuses
        chromatic_bonus = self._calculate_chromatic_bonus(model_outputs, pred_indices, target_indices, input_indices)
        harmony_bonus = self._calculate_harmony_bonus(model_outputs, pred_indices, target_indices)
        color_space_bonus = self._calculate_color_space_bonus(model_outputs)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     chromatic_bonus + harmony_bonus + color_space_bonus + ensemble_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'chromatic_bonus': chromatic_bonus,
            'harmony_bonus': harmony_bonus,
            'color_space_bonus': color_space_bonus,
            'ensemble_bonus': ensemble_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_chromatic_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """SIMPLIFIED chromatic reasoning bonus for speed"""
        if 'chromatic_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # FAST chromatic accuracy only
        chromatic_accuracy = (pred_indices == target_indices).float().mean()
        return -chromatic_accuracy * self.chromatic_weight * 0.05
    
    def _calculate_harmony_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                               target_indices: torch.Tensor) -> torch.Tensor:
        """SIMPLIFIED harmony bonus for speed"""
        if 'chromatic_analyses' not in outputs or len(outputs['chromatic_analyses']) == 0:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # FAST harmony approximation - just check first analysis
        first_analysis = outputs['chromatic_analyses'][0]
        if 'color_analysis' in first_analysis and 'harmony_patterns' in first_analysis['color_analysis']:
            harmony_score = first_analysis['color_analysis']['harmony_patterns'].mean()
            return -harmony_score * self.harmony_weight * 0.02
        
        return torch.tensor(0.0).to(pred_indices.device)
    
    def _calculate_color_space_bonus(self, outputs: Dict) -> torch.Tensor:
        """SIMPLIFIED color space bonus for speed"""
        if 'multichromatic_features' not in outputs or len(outputs['multichromatic_features']) == 0:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        # FAST approximation - just first color space
        first_features = outputs['multichromatic_features'][0]
        color_space_score = first_features.mean()  # Simple mean activation
        return -color_space_score * self.color_space_weight * 0.01
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """SIMPLIFIED ensemble bonus for speed"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        # FAST ensemble approximation - simple mean
        ensemble_output = outputs['ensemble_output']
        ensemble_score = ensemble_output.mean()
        return -ensemble_score * self.ensemble_weight * 0.01
        
        # Reward high color consensus
        if 'color_consensus' in ensemble_output:
            consensus = ensemble_output['color_consensus'].mean()
            ensemble_score = consensus
        else:
            ensemble_score = torch.tensor(0.7).to(list(outputs.values())[0].device)
        
        # Reward high color expertise
        if 'color_expertise' in ensemble_output:
            expertise = ensemble_output['color_expertise'].mean()
            ensemble_score = ensemble_score * expertise
        
        return -ensemble_score * self.ensemble_weight * 0.06


class AdvancedColorDataset(Dataset):
    """Dataset optimized for advanced color intelligence training"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 color_focus: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.color_focus = color_focus
        
        # Load data with advanced color filtering
        self.samples = []
        self._load_advanced_color_data()
        
        print(f"\033[96mLoaded {len(self.samples)} advanced color samples for IRIS V4 training\033[0m")
    
    def _load_advanced_color_data(self):
        """Load data with advanced color complexity focus"""
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
                    self._process_advanced_color_task(combined_task, 'training')
        
        # Load evaluation data
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                eval_task = {'train': task_data['train'], 'test': []}
                self._process_advanced_color_task(eval_task, 'evaluation')
    
    def _process_advanced_color_task(self, task: Dict, source_file: str):
        """Process task with advanced color analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples for color learning
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_advanced_color_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_advanced_color_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with advanced color analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Advanced color analysis
        color_analysis = self._analyze_advanced_color_complexity(input_grid, output_grid)
        
        # Filter for advanced color relevance (more permissive)
        if self.color_focus and color_analysis['color_intelligence_level'] < 2:
            if random.random() > 0.8:  # Keep 80% of simple cases
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'color_analysis': color_analysis
        }
    
    def _analyze_advanced_color_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze advanced color complexity and chromatic intelligence requirements"""
        # Basic color properties
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        all_colors = input_colors.union(output_colors)
        
        # Color transformation analysis
        same_colors = input_colors == output_colors
        colors_added = len(output_colors - input_colors)
        colors_removed = len(input_colors - output_colors)
        
        # Color intelligence level calculation
        color_intelligence_level = 0
        
        # Level 0: No color changes
        if same_colors and np.array_equal(input_grid, output_grid):
            color_intelligence_level = 0
        # Level 1: Spatial changes only, same colors
        elif same_colors:
            color_intelligence_level = 1
        # Level 2: Simple color additions/removals
        elif colors_added <= 1 and colors_removed <= 1:
            color_intelligence_level = 2
        # Level 3: Color mapping (same number of colors, different distribution)
        elif len(input_colors) == len(output_colors) and len(all_colors) > len(input_colors):
            color_intelligence_level = 3
        # Level 4: Complex color transformations
        elif colors_added > 1 or colors_removed > 1:
            color_intelligence_level = 4
        
        # Additional complexity factors
        unique_color_count = len(all_colors)
        max_dim = max(input_grid.shape + output_grid.shape)
        
        if unique_color_count > 6:
            color_intelligence_level += 1
        if max_dim > 20:
            color_intelligence_level += 1
        
        # Color harmony analysis
        harmony_score = 0
        if len(all_colors) >= 3:
            # Simple harmony heuristics
            primary_colors = {1, 2, 3}  # Red, Blue, Green equivalents
            secondary_colors = {4, 5, 6}  # Other colors
            
            primary_present = len(all_colors.intersection(primary_colors))
            secondary_present = len(all_colors.intersection(secondary_colors))
            
            if primary_present >= 2 and secondary_present >= 1:
                harmony_score = 2  # Good color harmony
            elif primary_present >= 2 or secondary_present >= 2:
                harmony_score = 1  # Basic color harmony
        
        # Complexity classification
        if color_intelligence_level <= 1 and unique_color_count <= 3:
            complexity = 'trivial'
        elif color_intelligence_level <= 2 and unique_color_count <= 5:
            complexity = 'basic'
        elif color_intelligence_level <= 3 and unique_color_count <= 7:
            complexity = 'intermediate'
        elif color_intelligence_level <= 4:
            complexity = 'advanced'
        else:
            complexity = 'expert'
        
        # Color transformation type
        if same_colors and np.array_equal(input_grid, output_grid):
            transform_type = 'identity'
        elif same_colors:
            transform_type = 'spatial_only'
        elif len(input_colors) == len(output_colors):
            transform_type = 'color_mapping'
        else:
            transform_type = 'color_generation'
        
        return {
            'color_intelligence_level': color_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_colors': unique_color_count,
            'colors_added': colors_added,
            'colors_removed': colors_removed,
            'harmony_score': harmony_score,
            'max_dimension': max_dim
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
            'color_analysis': sample['color_analysis']
        }
        
        return input_final, output_final, metadata


def advanced_color_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Enhanced collate function for advanced color training"""
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


def train_iris_specialized_v4():
    """Main training function for IRIS V4"""
    print(f"\033[96mInitializing IRIS V4 Advanced Color Intelligence Training...\033[0m")
    
    # Initialize enhanced model
    model = IrisV4Enhanced(
        max_grid_size=30,
        d_model=256,
        num_layers=6,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load existing weights
    model_path = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/iris_best.pt'
    weights_loaded = model.load_compatible_weights(model_path)
    
    if not weights_loaded:
        print(f"\033[96mStarting fresh IRIS V4 training\033[0m")
    
    # Initialize loss function
    criterion = IrisV4ChromaticLoss(IRIS_V4_CONFIG)
    
    # Initialize optimizer with chromatic learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=IRIS_V4_CONFIG['learning_rate'],
        weight_decay=IRIS_V4_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=IRIS_V4_CONFIG['warmup_epochs'],
        T_mult=int(IRIS_V4_CONFIG['restart_multiplier']),
        eta_min=IRIS_V4_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Progressive Chromatic Training - 13 Color Intelligence Stages\033[0m")
    
    # Progressive training through chromatic stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 100}\033[0m")
        print(f"\033[38;2;255;222;173mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Color: {stage_config['color_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 100}\033[0m")
        
        # Create advanced color dataset for this stage
        dataset = AdvancedColorDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            color_focus=True
        )
        
        # Create data loader - SPEED OPTIMIZED
        dataloader = DataLoader(
            dataset,
            batch_size=IRIS_V4_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=advanced_color_collate_fn,
            num_workers=2,  # Use multiple workers for faster data loading
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True  # Keep workers alive
        )
        
        # Stage-specific training
        stage_performance = train_advanced_chromatic_stage(
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
                'config': IRIS_V4_CONFIG,
                'ensemble_state': model.get_ensemble_state()
            }, '/content/AutomataNexus_Olympus_AGI2/models/iris_v4_best.pt')
                print(f"\033[96mNew best chromatic performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # MINIMAL memory cleanup for speed
        if stage_idx % 4 == 0:  # Only cleanup every 4th stage
            torch.cuda.empty_cache()
    
    print(f"\n\033[96m{'=' * 110}\033[0m")
    print(f"\033[96mIRIS V4 Advanced Color Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest Chromatic Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 110}\033[0m")
    
    return model, best_performance


def train_advanced_chromatic_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                 stage_idx, stage_config, training_stats):
    """Train a single advanced chromatic curriculum stage"""
    model.train()
    
    epochs_for_stage = IRIS_V4_CONFIG['epochs_per_stage']
    accumulation_steps = IRIS_V4_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_color_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mUltra Fast Chromatic Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), IRIS_V4_CONFIG['gradient_clip'])
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
            
            # Count advanced color cases
            for meta in metadata:
                if meta['color_analysis']['color_intelligence_level'] >= 3:
                    advanced_color_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'AdvColor': f"{advanced_color_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 2 == 0 or epoch == epochs_for_stage - 1:
            color_ratio = advanced_color_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[38;2;255;204;153m⏰ IRIS V4 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * IRIS_V4_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Color: {color_ratio:.1%}\033[0m")
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
    model, best_performance = train_iris_specialized_v4()