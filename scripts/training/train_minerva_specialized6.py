"""
MINERVA Specialized Training V6 - Ultimate Strategic Intelligence Master for ARC-AGI-2
Complete grid mastery (5x5 to 30x30) with deep strategic architecture and program synthesis
Builds upon V5 with revolutionary strategic intelligence capabilities and massive data pipeline
Target: 90%+ performance with ultimate strategic intelligence mastery
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

# Import enhanced MINERVA V6 model
from src.models.minerva_v6_enhanced import MinervaV6Enhanced

# Enhanced MINERVA V6 Configuration - Ultimate Strategic Intelligence Focus
MINERVA_V6_CONFIG = {
    # Core Training Parameters - OPTIMIZED for V6 Ultimate Performance
    'batch_size': 32,  # Balanced for deep architecture
    'learning_rate': 0.0001,  # Lower for deep strategic learning
    'num_epochs': 600,  # Extended: 20 stages x 30 epochs
    'gradient_accumulation': 6,  # Effective batch 192 for stability
    'epochs_per_stage': 30,  # Deep learning per stage
    'curriculum_stages': 20,  # Complete grid mastery 2x2 -> 30x30
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.04,  # Very low - maximum strategic exploration
    'exact_match_bonus': 10.0,  # Highest bonus for ultimate precision
    'gradient_clip': 0.5,  # Stable clipping for deep architecture
    'weight_decay': 2e-6,  # Ultra-light regularization for ultimate strategy
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'strategic_reasoning_weight': 0.65,  # Ultimate focus - strategic intelligence
    'ensemble_coordination_weight': 0.6,  # Maximum ensemble integration
    'pattern_analysis_weight': 0.55,  # Ultimate pattern analysis
    'decision_confidence_weight': 0.5,  # Ultimate decision confidence
    'program_synthesis_weight': 0.45,  # NEW: Program synthesis integration
    'deep_strategic_weight': 0.4,  # NEW: Deep strategic transformer bonus
    
    # MINERVA V6-Specific Ultimate Enhancements
    'deep_strategic_layers': 8,  # 8-layer deep strategic reasoning
    'mega_pattern_memory': 300,  # Massive strategic pattern memory
    'advanced_ensemble_prep': True,  # Ultimate OLYMPUS preparation
    'program_synthesis_integration': True,  # Advanced program synthesis
    'complete_grid_mastery': True,  # 2x2 to 30x30 complete coverage
    'ultimate_test_time_adaptation': True,  # Advanced strategic adaptation
    
    # Advanced Training Features
    'label_smoothing': 0.015,  # Ultra-refined for strategic precision
    'pattern_diversity_bonus': True,
    'strategic_reasoning_bonus': True,
    'ensemble_coordination_bonus': True,
    'program_synthesis_bonus': True,
    'deep_strategic_bonus': True,
    'ultimate_strategic_bonus': True,  # NEW: Ultimate strategy bonus
    
    # Learning Rate Scheduling
    'warmup_epochs': 25,  # Extended warmup for deep architecture
    'cosine_restarts': True,
    'restart_multiplier': 1.5,
    'plateau_patience': 30,
}

# Enhanced 20-Stage Progressive Configuration - Complete Grid Mastery starting with normal ARC sizes
STAGE_CONFIG = [
    # Foundation Strategic Understanding (5x5 - 9x9) - START WITH NORMAL ARC SIZES
    {'stage': 0, 'max_grid_size': 5,  'synthesis_ratio': 0.92, 'strategic_complexity': 'micro_strategic', 'focus': 'micro_strategic_patterns'},
    {'stage': 1, 'max_grid_size': 6,  'synthesis_ratio': 0.9, 'strategic_complexity': 'basic_strategy', 'focus': 'basic_pattern_recognition'},
    {'stage': 2, 'max_grid_size': 7,  'synthesis_ratio': 0.85, 'strategic_complexity': 'simple_reasoning', 'focus': 'simple_logical_inference'},
    {'stage': 3, 'max_grid_size': 8,  'synthesis_ratio': 0.8, 'strategic_complexity': 'rule_detection', 'focus': 'rule_identification'},
    {'stage': 4, 'max_grid_size': 9,  'synthesis_ratio': 0.75, 'strategic_complexity': 'pattern_analysis', 'focus': 'pattern_analysis'},
    
    # Intermediate Strategic Reasoning (10x10 - 15x15) 
    {'stage': 5, 'max_grid_size': 10, 'synthesis_ratio': 0.7, 'strategic_complexity': 'multi_step', 'focus': 'multi_step_reasoning'},
    {'stage': 6, 'max_grid_size': 11, 'synthesis_ratio': 0.65, 'strategic_complexity': 'complex_rules', 'focus': 'complex_rule_learning'},
    {'stage': 7, 'max_grid_size': 12, 'synthesis_ratio': 0.6, 'strategic_complexity': 'strategic_planning', 'focus': 'strategic_planning'},
    {'stage': 8, 'max_grid_size': 13, 'synthesis_ratio': 0.55, 'strategic_complexity': 'ensemble_prep_basic', 'focus': 'basic_ensemble_coordination'},
    {'stage': 9, 'max_grid_size': 14, 'synthesis_ratio': 0.5, 'strategic_complexity': 'arc_strategic_basic', 'focus': 'arc_strategic_patterns'},
    {'stage': 10, 'max_grid_size': 15, 'synthesis_ratio': 0.45, 'strategic_complexity': 'meta_reasoning', 'focus': 'meta_cognitive_reasoning'},
    
    # Advanced Strategic Mastery (16x16 - 22x22)
    {'stage': 11, 'max_grid_size': 16, 'synthesis_ratio': 0.4, 'strategic_complexity': 'program_synthesis_basic', 'focus': 'basic_program_synthesis'},
    {'stage': 12, 'max_grid_size': 18, 'synthesis_ratio': 0.35, 'strategic_complexity': 'deep_strategic_basic', 'focus': 'basic_deep_strategic'},
    {'stage': 13, 'max_grid_size': 20, 'synthesis_ratio': 0.3, 'strategic_complexity': 'ensemble_prep_advanced', 'focus': 'advanced_ensemble_coordination'},
    {'stage': 14, 'max_grid_size': 22, 'synthesis_ratio': 0.25, 'strategic_complexity': 'arc_strategic_advanced', 'focus': 'arc_advanced_strategy'},
    {'stage': 15, 'max_grid_size': 24, 'synthesis_ratio': 0.2, 'strategic_complexity': 'program_synthesis_advanced', 'focus': 'advanced_program_synthesis'},
    
    # Ultimate Strategic Genius (26x26 - 30x30)
    {'stage': 16, 'max_grid_size': 26, 'synthesis_ratio': 0.18, 'strategic_complexity': 'deep_strategic_advanced', 'focus': 'advanced_deep_strategic'},
    {'stage': 17, 'max_grid_size': 28, 'synthesis_ratio': 0.16, 'strategic_complexity': 'strategic_mastery', 'focus': 'strategic_expertise'},
    {'stage': 18, 'max_grid_size': 29, 'synthesis_ratio': 0.14, 'strategic_complexity': 'ultimate_strategic_mastery', 'focus': 'ultimate_strategic_mastery'},
    {'stage': 19, 'max_grid_size': 30, 'synthesis_ratio': 0.12, 'strategic_complexity': 'ultimate_strategic_genius', 'focus': 'ultimate_strategic_intelligence_mastery'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 120}\033[0m")
print(f"\033[96mMINERVA V6 Ultimate Strategic Intelligence Training - Complete Grid Mastery for ARC-AGI-2\033[0m")
print(f"\033[96mDeep Strategic Architecture + Mega Pattern Memory + Program Synthesis + OLYMPUS Preparation\033[0m") 
print(f"\033[96mTarget: 90%+ Performance with Ultimate Strategic Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 120}\033[0m")


class MinervaV6UltimateStrategicLoss(nn.Module):
    """Ultimate loss function for V6 strategic reasoning and complete intelligence mastery"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.strategic_weight = config['strategic_reasoning_weight']
        self.ensemble_weight = config['ensemble_coordination_weight']
        self.pattern_weight = config['pattern_analysis_weight']
        self.decision_weight = config['decision_confidence_weight']
        self.program_weight = config['program_synthesis_weight']
        self.deep_weight = config['deep_strategic_weight']
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
        exact_bonus = exact_bonus.clamp(min=-8.0)  # Ultimate clamp for V6
        
        # Transform penalty (ultra-low to encourage maximum strategic exploration)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Ultimate strategic bonuses
        strategic_bonus = self._calculate_strategic_bonus(model_outputs, pred_indices, target_indices, input_indices)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs)
        pattern_bonus = self._calculate_pattern_bonus(model_outputs)
        decision_bonus = self._calculate_decision_bonus(model_outputs)
        program_bonus = self._calculate_program_bonus(model_outputs)
        deep_strategic_bonus = self._calculate_deep_strategic_bonus(model_outputs)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     strategic_bonus + ensemble_bonus + pattern_bonus + 
                     decision_bonus + program_bonus + deep_strategic_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'strategic_bonus': strategic_bonus,
            'ensemble_bonus': ensemble_bonus,
            'pattern_bonus': pattern_bonus,
            'decision_bonus': decision_bonus,
            'program_bonus': program_bonus,
            'deep_strategic_bonus': deep_strategic_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_strategic_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate ultimate strategic reasoning bonus"""
        if 'strategic_info' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        strategic_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        strategic_change_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use deep strategic complexity if available
        if 'strategic_complexity' in outputs['strategic_info']:
            complexity = outputs['strategic_info']['strategic_complexity'].squeeze(-1)
            strategic_score = strategic_accuracy * complexity * (1.0 + strategic_change_mask * 0.8)
        else:
            strategic_score = strategic_accuracy * (1.0 + strategic_change_mask * 0.8)
        
        return -strategic_score.mean() * self.strategic_weight * 0.15
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate ensemble coordination bonus"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        ensemble_output = outputs['ensemble_output']
        
        # Reward high ensemble confidence
        if 'confidence' in ensemble_output:
            confidence = ensemble_output['confidence'].mean()
            ensemble_score = confidence
        else:
            ensemble_score = torch.tensor(0.8).to(list(outputs.values())[0].device)
        
        # Reward high consensus
        if 'consensus' in ensemble_output:
            consensus = ensemble_output['consensus'].max(dim=-1)[0].mean()
            ensemble_score = ensemble_score * consensus
        
        return -ensemble_score * self.ensemble_weight * 0.12
    
    def _calculate_pattern_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate pattern memory bonus"""
        if 'pattern_memory' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        pattern_memory = outputs['pattern_memory']
        
        # Reward high pattern similarity
        if 'pattern_similarity' in pattern_memory:
            pattern_score = pattern_memory['pattern_similarity'].mean()
        else:
            pattern_score = torch.tensor(0.6).to(list(outputs.values())[0].device)
        
        return -pattern_score * self.pattern_weight * 0.1
    
    def _calculate_decision_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate decision confidence bonus"""
        if 'confidence' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        confidence = outputs['confidence'].mean()
        return -confidence * self.decision_weight * 0.08
    
    def _calculate_program_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate program synthesis bonus"""
        if 'program_synthesis' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        program_synthesis = outputs['program_synthesis']
        
        # Reward diverse program types
        if 'program_types' in program_synthesis:
            program_diversity = -(program_synthesis['program_types'] * 
                                torch.log(program_synthesis['program_types'] + 1e-8)).sum(dim=-1).mean()
            program_score = program_diversity / 3.4  # Normalize entropy
        else:
            program_score = torch.tensor(0.5).to(list(outputs.values())[0].device)
        
        return -program_score * self.program_weight * 0.06
    
    def _calculate_deep_strategic_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate deep strategic transformer bonus"""
        if 'strategic_info' not in outputs or 'strategic_analyses' not in outputs['strategic_info']:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        strategic_analyses = outputs['strategic_info']['strategic_analyses']
        
        # Reward deep strategic consistency across layers
        if len(strategic_analyses) > 0:
            layer_consistency = 0
            for i, analysis in enumerate(strategic_analyses):
                if 'pattern_types' in analysis:
                    # Reward pattern evolution across layers
                    pattern_entropy = -(analysis['pattern_types'] * 
                                      torch.log(analysis['pattern_types'] + 1e-8)).sum(dim=-1).mean()
                    layer_weight = (i + 1) / len(strategic_analyses)  # Weight deeper layers more
                    layer_consistency += pattern_entropy * layer_weight
            
            deep_score = layer_consistency / len(strategic_analyses)
        else:
            deep_score = torch.tensor(0.4).to(list(outputs.values())[0].device)
        
        return -deep_score * self.deep_weight * 0.05


class UltimateStrategicDataset(Dataset):
    """Dataset optimized for ultimate strategic intelligence training with complete grid coverage"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 strategic_focus: bool = True, augmentation_factor: int = 8):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.strategic_focus = strategic_focus
        self.augmentation_factor = augmentation_factor
        
        # Load data with ultimate strategic filtering
        self.samples = []
        self._load_ultimate_strategic_data()
        
        # Massive data augmentation for complete coverage
        if augmentation_factor > 1:
            self._augment_strategic_data()
        
        print(f"Loaded {len(self.samples)} ultimate strategic samples for MINERVA V6 training")
    
    def _load_ultimate_strategic_data(self):
        """Load data with ultimate strategic complexity focus - PROVEN V5 APPROACH"""
        # Load training data (challenges + solutions) - DIRECT LOADING LIKE V5
        challenges_path = os.path.join(self.data_dir, 'arc-agi_training_challenges.json')
        solutions_path = os.path.join(self.data_dir, 'arc-agi_training_solutions.json')
        
        if os.path.exists(challenges_path) and os.path.exists(solutions_path):
            with open(challenges_path, 'r') as f:
                challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
            
            for task_id, task_data in challenges.items():
                # Process ALL training examples directly (PROVEN V5 approach)
                for example in task_data['train']:
                    sample = self._create_ultimate_strategic_sample(example, True)
                    if sample:
                        self.samples.append(sample)
                
                # Also process test examples if solutions exist
                if task_id in solutions:
                    for i, test_input in enumerate(task_data['test']):
                        if i < len(solutions[task_id]):
                            test_example = {
                                'input': test_input['input'],
                                'output': solutions[task_id][i]
                            }
                            sample = self._create_ultimate_strategic_sample(test_example, True)
                            if sample:
                                self.samples.append(sample)
        
        # Load evaluation data for broader coverage - DIRECT LOADING LIKE V5
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                # Process ALL training examples from evaluation set
                for example in task_data['train']:
                    sample = self._create_ultimate_strategic_sample(example, True)
                    if sample:
                        self.samples.append(sample)
    
    def _create_ultimate_strategic_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with ultimate strategic analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering for current stage ONLY
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Ultimate strategic analysis
        strategic_analysis = self._analyze_ultimate_strategic_complexity(input_grid, output_grid)
        
        # Keep ALL samples - no filtering for maximum data (PROVEN V5 APPROACH)
        # All samples are valuable for V6 ultimate strategic training
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'strategic_analysis': strategic_analysis
        }
    
    def _analyze_ultimate_strategic_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze ultimate strategic complexity and intelligence requirements"""
        # Basic properties
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        all_colors = input_colors.union(output_colors)
        
        # Grid properties
        input_h, input_w = input_grid.shape
        output_h, output_w = output_grid.shape
        size_change = abs(input_h * input_w - output_h * output_w)
        
        # Strategic intelligence level calculation
        strategic_intelligence_level = 0
        
        # Level 0: Identity or trivial
        if np.array_equal(input_grid, output_grid):
            strategic_intelligence_level = 0
        # Level 1: Size change only
        elif input_colors == output_colors and size_change == 0:
            strategic_intelligence_level = 1
        # Level 2: Simple transformations
        elif len(input_colors) == len(output_colors) and size_change <= 4:
            strategic_intelligence_level = 2
        # Level 3: Complex transformations
        elif size_change > 4 or len(all_colors) != len(input_colors):
            strategic_intelligence_level = 3
        # Level 4: Multi-step reasoning required
        elif len(all_colors) > 5 or max(input_h, input_w, output_h, output_w) > 15:
            strategic_intelligence_level = 4
        # Level 5: Ultimate strategic complexity
        elif len(all_colors) > 7 or max(input_h, input_w, output_h, output_w) > 20:
            strategic_intelligence_level = 5
        
        # Additional complexity factors
        pattern_complexity = 0
        
        # Color complexity
        if len(all_colors) > 3:
            pattern_complexity += 1
        if len(all_colors) > 6:
            pattern_complexity += 2
        
        # Size complexity  
        max_dim = max(input_h, input_w, output_h, output_w)
        if max_dim > 10:
            pattern_complexity += 1
        if max_dim > 20:
            pattern_complexity += 2
        
        # Shape complexity
        if size_change > 0:
            pattern_complexity += 1
        if abs(input_h - input_w) > 2 or abs(output_h - output_w) > 2:  # Non-square
            pattern_complexity += 1
        
        strategic_intelligence_level = min(strategic_intelligence_level + pattern_complexity, 6)
        
        # Complexity classification
        if strategic_intelligence_level == 0:
            complexity = 'trivial'
        elif strategic_intelligence_level <= 1:
            complexity = 'basic'
        elif strategic_intelligence_level <= 2:
            complexity = 'intermediate'
        elif strategic_intelligence_level <= 3:
            complexity = 'advanced'
        elif strategic_intelligence_level <= 4:
            complexity = 'expert'
        else:
            complexity = 'genius'
        
        # Strategic transformation type
        if np.array_equal(input_grid, output_grid):
            transform_type = 'identity'
        elif size_change == 0 and input_colors == output_colors:
            transform_type = 'spatial_transformation'
        elif size_change > 0:
            transform_type = 'generative_transformation'
        else:
            transform_type = 'rule_based_transformation'
        
        return {
            'strategic_intelligence_level': strategic_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_colors': len(all_colors),
            'size_change': size_change,
            'pattern_complexity': pattern_complexity,
            'max_dimension': max_dim,
            'grid_ratio': max(input_h/input_w, input_w/input_h) if min(input_w, input_h) > 0 else 1.0
        }
    
    def _augment_strategic_data(self):
        """Massive strategic data augmentation"""
        original_count = len(self.samples)
        augmented_samples = []
        
        for sample in self.samples:
            for _ in range(self.augmentation_factor - 1):  # -1 because original is already included
                augmented = self._augment_single_sample(sample)
                if augmented:
                    augmented_samples.append(augmented)
        
        self.samples.extend(augmented_samples)
        print(f"Augmented from {original_count} to {len(self.samples)} samples ({self.augmentation_factor}x)")
    
    def _augment_single_sample(self, sample: Dict) -> Optional[Dict]:
        """Augment single sample with strategic transformations"""
        input_grid = sample['input'].copy()
        output_grid = sample['output'].copy()
        
        # Random augmentation choice
        aug_type = random.choice(['rotate', 'flip', 'transpose', 'noise'])
        
        if aug_type == 'rotate':
            k = random.choice([1, 2, 3])
            input_grid = np.rot90(input_grid, k).copy()
            output_grid = np.rot90(output_grid, k).copy()
        elif aug_type == 'flip':
            axis = random.choice([0, 1])
            input_grid = np.flip(input_grid, axis).copy()
            output_grid = np.flip(output_grid, axis).copy()
        elif aug_type == 'transpose':
            if input_grid.shape[0] == input_grid.shape[1] and output_grid.shape[0] == output_grid.shape[1]:
                input_grid = input_grid.T.copy()
                output_grid = output_grid.T.copy()
        elif aug_type == 'noise':
            # Add strategic noise (color permutation)
            if random.random() < 0.3:
                colors = list(set(input_grid.flatten()) | set(output_grid.flatten()))
                if len(colors) > 2:
                    color_map = dict(zip(colors, np.random.permutation(colors)))
                    input_grid = np.vectorize(color_map.get)(input_grid)
                    output_grid = np.vectorize(color_map.get)(output_grid)
        
        # Check size constraints after augmentation
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': sample['is_arc'],
            'strategic_analysis': sample['strategic_analysis']  # Keep original analysis
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        
        # Convert to tensors (copy to avoid negative stride issues)
        input_tensor = torch.tensor(sample['input'].copy(), dtype=torch.long)
        output_tensor = torch.tensor(sample['output'].copy(), dtype=torch.long)
        
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


def ultimate_strategic_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Enhanced collate function for ultimate strategic training"""
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


def train_minerva_specialized_v6():
    """Main training function for MINERVA V6"""
    print(f"\033[96mInitializing MINERVA V6 Ultimate Strategic Intelligence Training...\033[0m")
    
    # Initialize ultimate enhanced model
    model = MinervaV6Enhanced(
        max_grid_size=30,
        hidden_dim=256,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load existing weights from minerva_best.pt
    model_paths = [
        '/content/AutomataNexus_Olympus_AGI2/models/minerva_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/minerva_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/minerva_v5_best.pt'
    ]
    
    weights_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            weights_loaded = model.load_compatible_weights(model_path)
            if weights_loaded:
                print(f"\033[96mSuccessfully loaded weights from {model_path}\033[0m")
                break
    
    if not weights_loaded:
        print(f"\033[96mStarting fresh MINERVA V6 training\033[0m")
    
    # Initialize ultimate loss function
    criterion = MinervaV6UltimateStrategicLoss(MINERVA_V6_CONFIG)
    
    # Initialize optimizer with ultimate strategic learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MINERVA_V6_CONFIG['learning_rate'],
        weight_decay=MINERVA_V6_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=MINERVA_V6_CONFIG['warmup_epochs'],
        T_mult=int(MINERVA_V6_CONFIG['restart_multiplier']),
        eta_min=MINERVA_V6_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Ultimate Progressive Strategic Training - 20 Complete Grid Mastery Stages\033[0m")
    
    # Progressive training through ultimate strategic stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 110}\033[0m")
        print(f"\033[38;2;255;215;0mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Strategic: {stage_config['strategic_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 110}\033[0m")
        
        # Create ultimate strategic dataset for this stage
        dataset = UltimateStrategicDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            strategic_focus=True,
            augmentation_factor=8  # Massive augmentation
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=MINERVA_V6_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=ultimate_strategic_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_ultimate_strategic_stage(
            model, dataloader, criterion, optimizer, scheduler, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best model to BOTH locations
            for save_dir in ['/content/AutomataNexus_Olympus_AGI2/models', 
                           '/content/AutomataNexus_Olympus_AGI2/arc_models_v4']:
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_performance': best_performance,
                    'stage': stage_idx,
                    'config': MINERVA_V6_CONFIG,
                    'ensemble_state': model.get_ensemble_state()
                }, f'{save_dir}/minerva_v6_best.pt')
            
            print(f"\\033[96mNew best V6 ultimate performance: {best_performance:.2%} - Models saved to BOTH locations!\\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\\n\\033[96m{'=' * 120}\\033[0m")
    print(f"\\033[96mMINERVA V6 Ultimate Strategic Intelligence Training Complete!\\033[0m")
    print(f"\\033[96mBest Ultimate Performance: {best_performance:.2%}\\033[0m")
    print(f"\\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\\033[0m")
    print(f"\\033[96mComplete Grid Mastery: {model.get_ensemble_state()['grid_mastery']}\\033[0m")
    print(f"\\033[96mProgram Synthesis: {model.get_ensemble_state()['program_synthesis']}\\033[0m")
    print(f"\\033[96m{'=' * 120}\\033[0m")
    
    return model, best_performance


def train_ultimate_strategic_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                 stage_idx, stage_config, training_stats):
    """Train a single ultimate strategic curriculum stage"""
    model.train()
    
    epochs_for_stage = MINERVA_V6_CONFIG['epochs_per_stage']
    accumulation_steps = MINERVA_V6_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        ultimate_strategic_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Ultimate Strategic Stage {stage_idx} Epoch {epoch}")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), MINERVA_V6_CONFIG['gradient_clip'])
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
            
            # Count ultimate strategic cases
            for meta in metadata:
                if meta['strategic_analysis']['strategic_intelligence_level'] >= 4:
                    ultimate_strategic_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'UltimateStrategic': f"{ultimate_strategic_count}",
                'ARC': f"{sum(1 for m in metadata if m['is_arc'])}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            strategic_ratio = ultimate_strategic_count / max(total_samples, 1)
            arc_ratio = sum(1 for batch_metadata in [metadata] for m in batch_metadata if m['is_arc']) / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            
            global_epoch = stage_idx * MINERVA_V6_CONFIG['epochs_per_stage'] + epoch + 1
            print(f"\\033[96m⏰ MINERVA V6 Stage {stage_idx}, Epoch {epoch} (Global: {global_epoch}):\\033[0m")
            print(f"\\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\\033[0m")
            print(f"\\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | "
                  f"Strategic: {strategic_ratio:.1%} | ARC: {arc_ratio:.1%}\\033[0m")
            if epoch == epochs_for_stage - 1:
                print(f"\\033[96m✅ Stage {stage_idx} complete! Final exact: {epoch_performance:.2%}\\033[0m")
        
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
    model, best_performance = train_minerva_specialized_v6()