"""
CHRONOS Specialized Training V5 - Advanced Temporal Sequence Reasoning Expert for ARC-AGI-2
Enhanced V5 trainer that builds upon V4 with more ARC-specific training, stages, and epochs
Loads from chronos_v4_best.pt and adds sophisticated temporal intelligence mastery
Target: 75%+ performance with extended temporal intelligence training
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

# Import enhanced CHRONOS V4 model
from src.models.chronos_v4_enhanced import ChronosV4Enhanced

# Enhanced CHRONOS V5 Configuration - Extended Temporal Intelligence Focus
CHRONOS_V5_CONFIG = {
    # Core Training Parameters - Enhanced for V5 Extended Training
    'batch_size': 22,  # Optimal for extended temporal transformer computations
    'learning_rate': 0.00011,  # Lower for fine-tuning from V4
    'num_epochs': 950,  # Extended training: 19 stages x 50 epochs
    'gradient_accumulation': 10,  # Effective batch: 220
    'epochs_per_stage': 50,  # Extended epochs per stage
    'curriculum_stages': 19,  # Extended 19-stage temporal progression
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.025,  # Very low - encourage temporal transformations
    'exact_match_bonus': 10.5,  # Higher bonus for temporal precision
    'gradient_clip': 0.45,  # Refined clipping for V5
    'weight_decay': 2.2e-6,  # Even lighter regularization for temporal learning
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'temporal_reasoning_weight': 0.65,  # Enhanced focus - temporal intelligence
    'sequence_analysis_weight': 0.56,  # Enhanced sequence understanding mastery
    'movement_prediction_weight': 0.52,  # Enhanced movement/change prediction
    'ensemble_coordination_weight': 0.48,  # Enhanced ensemble integration
    'arc_temporal_weight': 0.42,  # NEW: ARC-specific temporal reasoning
    
    # CHRONOS V5-Specific Enhancements
    'temporal_transformer_layers': 6,  # Deep temporal reasoning
    'sequence_memory_size': 220,  # Larger temporal pattern memory
    'temporal_positional_encoding': True,  # Temporal-aware positioning
    'multitemporal_processing': True,  # Multi-scale temporal processing
    'ensemble_preparation': True,  # OLYMPUS preparation mode
    'test_time_adaptation': True,  # Advanced temporal adaptation
    'arc_temporal_training': True,  # NEW: ARC-specific temporal training mode
    
    # Advanced Training Features
    'label_smoothing': 0.015,  # Refined for temporal precision
    'pattern_diversity_bonus': True,
    'temporal_reasoning_bonus': True,
    'sequence_continuity_bonus': True,
    'movement_prediction_bonus': True,
    'arc_temporal_bonus': True,  # NEW: ARC-specific temporal bonus
    
    # Learning Rate Scheduling
    'warmup_epochs': 35,  # Extended warmup for temporal transformers
    'cosine_restarts': True,
    'restart_multiplier': 1.2,
    'plateau_patience': 25,
}

# Enhanced 19-Stage Progressive Configuration - Extended Temporal Intelligence Focus
STAGE_CONFIG = [
    # Foundation Temporal Understanding (5x5 - 9x9)
    {'stage': 0, 'max_grid_size': 5,  'synthesis_ratio': 0.95, 'temporal_complexity': 'micro_temporal', 'focus': 'micro_temporal_patterns'},
    {'stage': 1, 'max_grid_size': 6,  'synthesis_ratio': 0.9, 'temporal_complexity': 'static_patterns', 'focus': 'static_pattern_recognition'},
    {'stage': 2, 'max_grid_size': 7,  'synthesis_ratio': 0.85, 'temporal_complexity': 'simple_movement', 'focus': 'simple_movement_detection'},
    {'stage': 3, 'max_grid_size': 8,  'synthesis_ratio': 0.8, 'temporal_complexity': 'basic_sequence', 'focus': 'basic_sequence_learning'},
    {'stage': 4, 'max_grid_size': 9,  'synthesis_ratio': 0.75, 'temporal_complexity': 'pattern_evolution', 'focus': 'pattern_evolution_understanding'},
    
    # Intermediate Temporal Reasoning (10x10 - 16x16)
    {'stage': 5, 'max_grid_size': 10, 'synthesis_ratio': 0.7, 'temporal_complexity': 'temporal_rules', 'focus': 'temporal_rule_learning'},
    {'stage': 6, 'max_grid_size': 11, 'synthesis_ratio': 0.65, 'temporal_complexity': 'sequence_prediction', 'focus': 'sequence_prediction'},
    {'stage': 7, 'max_grid_size': 12, 'synthesis_ratio': 0.6, 'temporal_complexity': 'movement_complex', 'focus': 'complex_movement_patterns'},
    {'stage': 8, 'max_grid_size': 14, 'synthesis_ratio': 0.55, 'temporal_complexity': 'temporal_logic', 'focus': 'temporal_logical_reasoning'},
    {'stage': 9, 'max_grid_size': 15, 'synthesis_ratio': 0.5, 'temporal_complexity': 'continuity_advanced', 'focus': 'advanced_continuity_analysis'},
    {'stage': 10, 'max_grid_size': 16, 'synthesis_ratio': 0.45, 'temporal_complexity': 'arc_temporal_basic', 'focus': 'arc_temporal_patterns'},
    
    # Advanced Temporal Mastery (18x18 - 30x30)
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

print(f"\033[96m{'=' * 130}\033[0m")
print(f"\033[96mCHRONOS V5 Enhanced Training - Extended Temporal Sequence Reasoning Expert for ARC-AGI-2\033[0m")
print(f"\033[96mBuilds on V4 with Extended Training: 19 Stages + ARC-Specific Temporal Intelligence\033[0m")
print(f"\033[96mTarget: 75%+ Performance with Extended Temporal Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 130}\033[0m")


class ChronosV5TemporalLoss(nn.Module):
    """Extended loss function for V5 temporal reasoning and ARC-specific temporal intelligence"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.temporal_weight = config['temporal_reasoning_weight']
        self.sequence_weight = config['sequence_analysis_weight']
        self.movement_weight = config['movement_prediction_weight']
        self.ensemble_weight = config['ensemble_coordination_weight']
        self.arc_temporal_weight = config['arc_temporal_weight']
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
        exact_bonus = exact_bonus.clamp(min=-7.2)  # Higher clamp for V5 temporal precision
        
        # Transform penalty (very low to encourage temporal transformations)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # V5 Enhanced temporal reasoning bonuses
        temporal_bonus = self._calculate_temporal_bonus(model_outputs, pred_indices, target_indices, input_indices)
        sequence_bonus = self._calculate_sequence_bonus(model_outputs, pred_indices, target_indices)
        movement_bonus = self._calculate_movement_bonus(model_outputs)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs)
        arc_temporal_bonus = self._calculate_arc_temporal_bonus(model_outputs, pred_indices, target_indices)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     temporal_bonus + sequence_bonus + movement_bonus + ensemble_bonus + arc_temporal_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'temporal_bonus': temporal_bonus,
            'sequence_bonus': sequence_bonus,
            'movement_bonus': movement_bonus,
            'ensemble_bonus': ensemble_bonus,
            'arc_temporal_bonus': arc_temporal_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_temporal_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced temporal reasoning bonus for V5"""
        if 'temporal_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward temporal transformations
        temporal_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        change_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use temporal expertise if available
        if 'temporal_expertise' in outputs:
            temporal_confidence = outputs['temporal_expertise'].squeeze(-1)
            temporal_score = temporal_accuracy * temporal_confidence * (1.0 + change_mask * 0.8)
        else:
            temporal_score = temporal_accuracy * (1.0 + change_mask * 0.8)
        
        return -temporal_score.mean() * self.temporal_weight * 0.16
    
    def _calculate_sequence_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced sequence analysis bonus for V5"""
        if 'temporal_analyses' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        sequence_score = 0
        temporal_analyses = outputs['temporal_analyses']
        
        for analysis in temporal_analyses:
            if 'temporal_analysis' in analysis:
                temp_analysis = analysis['temporal_analysis']
                
                # Reward sequence pattern confidence
                if 'sequence_patterns' in temp_analysis:
                    pattern_confidence = temp_analysis['sequence_patterns'].max(dim=-1)[0].mean()
                    sequence_score += pattern_confidence
                
                # Reward continuity analysis
                if 'continuity_analysis' in temp_analysis:
                    continuity_confidence = temp_analysis['continuity_analysis'].mean()
                    sequence_score += continuity_confidence * 0.8
                
                # Reward temporal confidence
                if 'temporal_confidence' in temp_analysis:
                    temporal_confidence = temp_analysis['temporal_confidence'].mean()
                    sequence_score += temporal_confidence * 0.6
        
        # Normalize by number of analyses
        if len(temporal_analyses) > 0:
            sequence_score = sequence_score / len(temporal_analyses)
        
        return -sequence_score * self.sequence_weight * 0.14
    
    def _calculate_movement_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced movement prediction bonus for V5"""
        if 'multitemporal_features' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        multitemporal_features = outputs['multitemporal_features']
        
        # Encourage diverse temporal scale representations
        movement_score = 0
        for i, temporal_features in enumerate(multitemporal_features):
            # Measure temporal diversity at each scale
            temporal_diversity = temporal_features.std(dim=[2, 3]).mean()
            movement_score += temporal_diversity * (1.0 / (i + 1))  # Weight by scale importance
        
        # Normalize
        movement_score = movement_score / len(multitemporal_features)
        
        return -movement_score * self.movement_weight * 0.12
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced ensemble coordination bonus for V5"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        ensemble_output = outputs['ensemble_output']
        
        # Reward high temporal consensus
        if 'temporal_consensus' in ensemble_output:
            consensus = ensemble_output['temporal_consensus'].mean()
            ensemble_score = consensus
        else:
            ensemble_score = torch.tensor(0.68).to(list(outputs.values())[0].device)
        
        # Reward high temporal expertise
        if 'temporal_expertise' in ensemble_output:
            expertise = ensemble_output['temporal_expertise'].mean()
            ensemble_score = ensemble_score * expertise
        
        # Reward effective cross-attention
        if 'cross_attention_weights' in ensemble_output and ensemble_output['cross_attention_weights'] is not None:
            attention_weights = ensemble_output['cross_attention_weights']
            # Measure attention diversity (avoid collapse)
            attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
            ensemble_score = ensemble_score * torch.sigmoid(attention_entropy)
        
        return -ensemble_score * self.ensemble_weight * 0.11
    
    def _calculate_arc_temporal_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                    target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate NEW ARC-specific temporal bonus for V5"""
        # ARC-specific temporal patterns bonus
        arc_temporal_score = 0
        
        # Reward complex temporal transformations typical in ARC
        temporal_complexity = (pred_indices != target_indices).float().sum(dim=[1,2]) / (pred_indices.shape[1] * pred_indices.shape[2])
        arc_temporal_score = temporal_complexity.mean()
        
        # Bonus for temporal memory utilization
        if 'temporal_memory_similarity' in outputs:
            memory_usage = outputs['temporal_memory_similarity'].mean()
            arc_temporal_score = arc_temporal_score * (1.0 + memory_usage)
        
        return -arc_temporal_score * self.arc_temporal_weight * 0.1


class ExtendedTemporalDataset(Dataset):
    """Extended dataset optimized for V5 temporal intelligence with ARC-specific focus"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 temporal_focus: bool = True, arc_specific: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.temporal_focus = temporal_focus
        self.arc_specific = arc_specific
        
        # Load data with extended temporal filtering
        self.samples = []
        self._load_extended_temporal_data()
        
        print(f"\033[96mLoaded {len(self.samples)} extended temporal samples for CHRONOS V5 training\033[0m")
    
    def _load_extended_temporal_data(self):
        """Load data with extended temporal complexity focus and ARC specificity"""
        # Load training data (challenges + solutions) - WORKING V4 APPROACH
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
                    
                    self._process_extended_temporal_task(combined_task, 'arc_training')
        
        # Load evaluation data for broader coverage - ONLY TRAINING EXAMPLES
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                # Only process training examples from evaluation data (test has no outputs)
                combined_task = {
                    'train': task_data['train'],
                    'test': []  # Empty test since evaluation test has no solutions
                }
                self._process_extended_temporal_task(combined_task, 'arc_evaluation')
    
    def _process_extended_temporal_task(self, task: Dict, source_file: str):
        """Process task with extended temporal analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples for temporal learning
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_extended_temporal_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_extended_temporal_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with extended temporal analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Extended temporal analysis
        temporal_analysis = self._analyze_extended_temporal_complexity(input_grid, output_grid, is_arc_task)
        
        # Filter for extended temporal relevance (more inclusive for V5)
        if self.temporal_focus and temporal_analysis['temporal_intelligence_level'] < 1:
            if random.random() > 0.7:  # Keep 70% of simple cases for V5
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'temporal_analysis': temporal_analysis
        }
    
    def _analyze_extended_temporal_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray, is_arc_task: bool) -> Dict:
        """Analyze extended temporal complexity with ARC-specific considerations"""
        # Basic temporal properties
        same_shape = input_grid.shape == output_grid.shape
        same_content = np.array_equal(input_grid, output_grid)
        
        # Pattern analysis
        input_unique = len(np.unique(input_grid))
        output_unique = len(np.unique(output_grid))
        total_unique = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        
        # Extended temporal intelligence level calculation
        temporal_intelligence_level = 0
        
        # Level 0: Identity (no temporal change)
        if same_content:
            temporal_intelligence_level = 0
        # Level 1: Simple modifications (basic temporal change)
        elif same_shape and input_unique == output_unique:
            # Check for simple transformations
            if np.sum(input_grid != output_grid) < (input_grid.size * 0.3):
                temporal_intelligence_level = 1
            else:
                temporal_intelligence_level = 2
        # Level 2: Pattern modifications with temporal logic
        elif same_shape:
            temporal_intelligence_level = 2
            # Complex pattern changes
            if abs(input_unique - output_unique) > 1:
                temporal_intelligence_level += 1
        # Level 3+: Shape changes (advanced temporal reasoning)
        else:
            temporal_intelligence_level = 3
            # Scale changes indicate sequence prediction
            scale_factor = (output_grid.shape[0] * output_grid.shape[1]) / (input_grid.shape[0] * input_grid.shape[1])
            if scale_factor > 1.5 or scale_factor < 0.5:
                temporal_intelligence_level += 1
            
            # Complex temporal patterns
            if total_unique > 6 or max(output_grid.shape) > 22:
                temporal_intelligence_level += 1
        
        # Additional temporal complexity factors
        max_dim = max(input_grid.shape + output_grid.shape)
        
        # Enhanced movement detection (improved heuristic for V5)
        movement_score = 0
        if same_shape:
            # Check for potential object movement patterns
            for color in range(1, 10):  # Skip background (0)
                input_mask = (input_grid == color)
                output_mask = (output_grid == color)
                if input_mask.sum() > 0 and output_mask.sum() > 0:
                    input_positions = np.argwhere(input_mask)
                    output_positions = np.argwhere(output_mask)
                    if len(input_positions) > 0 and len(output_positions) > 0:
                        input_center = np.mean(input_positions, axis=0)
                        output_center = np.mean(output_positions, axis=0)
                        movement_dist = np.linalg.norm(input_center - output_center)
                        movement_score = max(movement_score, movement_dist)
        
        if movement_score > 2.0:
            temporal_intelligence_level += 0.7  # Higher bonus for V5
        
        # ARC-specific bonus
        if is_arc_task:
            temporal_intelligence_level += 0.9  # Higher boost for ARC temporal patterns
        
        # Complexity classification (extended for V5)
        if temporal_intelligence_level <= 0.5 and max_dim <= 8:
            complexity = 'micro'
        elif temporal_intelligence_level <= 1.5 and max_dim <= 12:
            complexity = 'trivial'
        elif temporal_intelligence_level <= 2.5 and max_dim <= 18:
            complexity = 'basic'
        elif temporal_intelligence_level <= 3.5 and max_dim <= 24:
            complexity = 'intermediate'
        elif temporal_intelligence_level <= 4.5:
            complexity = 'advanced'
        else:
            complexity = 'expert'
        
        # Temporal transformation type (enhanced for V5)
        if same_content:
            transform_type = 'identity'
        elif same_shape and movement_score > 1.0:
            transform_type = 'movement_based'
        elif same_shape and abs(input_unique - output_unique) <= 1:
            transform_type = 'pattern_evolution'
        elif same_shape:
            transform_type = 'color_temporal'
        else:
            transform_type = 'sequence_generation'
        
        # Temporal potential (enhanced for V5)
        temporal_potential = temporal_intelligence_level + (total_unique - 2) * 0.8 + max_dim * 0.12 + movement_score * 0.3
        if is_arc_task:
            temporal_potential += 1.8  # ARC bonus
        
        return {
            'temporal_intelligence_level': temporal_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_patterns': total_unique,
            'movement_score': movement_score,
            'temporal_potential': temporal_potential,
            'max_dimension': max_dim,
            'temporal_density': (input_unique + output_unique) / (max_dim ** 2),
            'arc_specific': is_arc_task
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
            'temporal_analysis': sample['temporal_analysis']
        }
        
        return input_final, output_final, metadata


def extended_temporal_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Extended collate function for V5 temporal training"""
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


def train_chronos_specialized_v5():
    """Main training function for CHRONOS V5"""
    print(f"\033[96mInitializing CHRONOS V5 Extended Temporal Intelligence Training...\033[0m")
    
    # Initialize enhanced model
    model = ChronosV4Enhanced(
        max_grid_size=30,
        d_model=256,
        num_layers=6,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load V4 weights
    model_path = '/content/AutomataNexus_Olympus_AGI2/models/chronos_v4_best.pt'
    weights_loaded = model.load_compatible_weights(model_path)
    
    if not weights_loaded:
        print(f"\033[96mWarning: Could not load V4 weights, starting V5 training from scratch\033[0m")
    else:
        print(f"\033[96mSuccessfully loaded V4 weights for V5 extended training\033[0m")
    
    # Initialize loss function
    criterion = ChronosV5TemporalLoss(CHRONOS_V5_CONFIG)
    
    # Initialize optimizer with V5 learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CHRONOS_V5_CONFIG['learning_rate'],
        weight_decay=CHRONOS_V5_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CHRONOS_V5_CONFIG['warmup_epochs'],
        T_mult=int(CHRONOS_V5_CONFIG['restart_multiplier']),
        eta_min=CHRONOS_V5_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Extended Progressive Temporal Training - 19 Enhanced Temporal Intelligence Stages\033[0m")
    
    # Extended progressive training through temporal stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 120}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Temporal: {stage_config['temporal_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 120}\033[0m")
        
        # Create extended temporal dataset for this stage
        dataset = ExtendedTemporalDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            temporal_focus=True,
            arc_specific=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=CHRONOS_V5_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=extended_temporal_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_extended_temporal_stage(
            model, dataloader, criterion, optimizer, scheduler, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best V5 model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_performance': best_performance,
                'stage': stage_idx,
                'config': CHRONOS_V5_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V5'
            }, '/content/AutomataNexus_Olympus_AGI2/models/chronos_v5_best.pt')
            print(f"\033[96mNew best V5 temporal performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 130}\033[0m")
    print(f"\033[96mCHRONOS V5 Extended Temporal Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest V5 Temporal Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 130}\033[0m")
    
    return model, best_performance


def train_extended_temporal_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                stage_idx, stage_config, training_stats):
    """Train a single extended temporal curriculum stage for V5"""
    model.train()
    
    epochs_for_stage = CHRONOS_V5_CONFIG['epochs_per_stage']
    accumulation_steps = CHRONOS_V5_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_temporal_count = 0
        arc_temporal_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[96mExtended Temporal Stage {stage_idx} Epoch {epoch}\033[0m")
        
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                # CHRONOS expects sequence input, convert single frame to sequence
                outputs = model([inputs], targets, mode='train')
                loss_dict = criterion(outputs, targets, inputs)
                loss = loss_dict['total'] / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CHRONOS_V5_CONFIG['gradient_clip'])
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
                if meta['temporal_analysis']['arc_specific']:
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
        
        # Log detailed progress
        if epoch % 4 == 0 or epoch == epochs_for_stage - 1:
            temporal_ratio = advanced_temporal_count / max(total_samples, 1)
            arc_ratio = arc_temporal_count / max(total_samples, 1)
            print(f"\033[96mExtended Temporal Stage {stage_idx} Epoch {epoch}: "
                  f"Performance = {epoch_performance:.1%}, "
                  f"Advanced Temporal = {temporal_ratio:.1%}, "
                  f"ARC Temporal = {arc_ratio:.1%}, "
                  f"Loss = {epoch_losses['total']/len(dataloader):.4f}\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    return best_stage_performance


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train V5 model
    model, best_performance = train_chronos_specialized_v5()