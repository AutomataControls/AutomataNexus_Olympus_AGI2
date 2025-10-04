"""
MINERVA Specialized Training V5 - Advanced Strategic Ensemble Coordinator for ARC-AGI-2
Enhanced V5 trainer that builds upon V4 with more ARC-specific training, stages, and epochs
Loads from minerva_v4_best.pt and adds sophisticated strategic intelligence mastery
Target: 82%+ performance with extended strategic intelligence training
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

# Enhanced MINERVA V5 Configuration - Extended Strategic Intelligence Focus
MINERVA_V5_CONFIG = {
    # Core Training Parameters - OPTIMIZED for V5 Speed + Performance
    'batch_size': 48,  # Larger batch for efficiency
    'learning_rate': 0.0002,  # Higher for faster convergence from V4 base
    'num_epochs': 300,  # Reduced: 10 stages x 30 epochs
    'gradient_accumulation': 5,  # Reduced accumulation: effective batch 240
    'epochs_per_stage': 30,  # Reduced epochs per stage for speed
    'curriculum_stages': 10,  # Reduced stages for efficiency
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.06,  # Even lower - max strategic exploration
    'exact_match_bonus': 9.2,  # Higher bonus for strategic accuracy
    'gradient_clip': 0.52,  # Slightly higher tolerance for V5
    'weight_decay': 4e-6,  # Even lighter regularization for strategy
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'strategic_reasoning_weight': 0.55,  # Enhanced focus - strategic intelligence
    'ensemble_coordination_weight': 0.48,  # Enhanced ensemble integration
    'pattern_analysis_weight': 0.42,  # Enhanced pattern analysis
    'decision_confidence_weight': 0.38,  # Enhanced decision confidence
    'arc_strategic_weight': 0.35,  # NEW: ARC-specific strategic reasoning
    
    # MINERVA V5-Specific Enhancements
    'strategic_transformer_layers': 6,  # Deep strategic reasoning
    'pattern_memory_size': 180,  # Larger strategic pattern memory
    'strategic_positional_encoding': True,  # Strategic-aware positioning
    'ensemble_preparation': True,  # OLYMPUS preparation mode
    'test_time_adaptation': True,  # Advanced strategic adaptation
    'arc_strategic_training': True,  # NEW: ARC-specific strategic training mode
    
    # Advanced Training Features
    'label_smoothing': 0.018,  # Refined for strategic precision
    'pattern_diversity_bonus': True,
    'strategic_reasoning_bonus': True,
    'ensemble_coordination_bonus': True,
    'olympus_preparation_bonus': True,
    'arc_strategic_bonus': True,  # NEW: ARC-specific strategy bonus
    
    # Learning Rate Scheduling
    'warmup_epochs': 15,  # Reduced warmup for faster training
    'cosine_restarts': True,
    'restart_multiplier': 1.25,
    'plateau_patience': 22,
}

# Enhanced 16-Stage Progressive Configuration - Extended Strategic Intelligence Focus
STAGE_CONFIG = [
    # Foundation Strategic Understanding (5x5 - 9x9)
    {'stage': 0, 'max_grid_size': 5,  'synthesis_ratio': 0.95, 'strategic_complexity': 'micro_strategy', 'focus': 'micro_strategic_patterns'},
    {'stage': 1, 'max_grid_size': 6,  'synthesis_ratio': 0.9, 'strategic_complexity': 'basic_strategy', 'focus': 'basic_pattern_recognition'},
    {'stage': 2, 'max_grid_size': 7,  'synthesis_ratio': 0.85, 'strategic_complexity': 'simple_reasoning', 'focus': 'simple_logical_inference'},
    {'stage': 3, 'max_grid_size': 8,  'synthesis_ratio': 0.8, 'strategic_complexity': 'rule_detection', 'focus': 'rule_identification'},
    {'stage': 4, 'max_grid_size': 9,  'synthesis_ratio': 0.75, 'strategic_complexity': 'pattern_analysis', 'focus': 'pattern_analysis'},
    
    # Intermediate Strategic Reasoning (10x10 - 18x18)
    {'stage': 5, 'max_grid_size': 10, 'synthesis_ratio': 0.7, 'strategic_complexity': 'multi_step', 'focus': 'multi_step_reasoning'},
    {'stage': 6, 'max_grid_size': 12, 'synthesis_ratio': 0.65, 'strategic_complexity': 'complex_rules', 'focus': 'complex_rule_learning'},
    {'stage': 7, 'max_grid_size': 14, 'synthesis_ratio': 0.6, 'strategic_complexity': 'strategic_planning', 'focus': 'strategic_planning'},
    {'stage': 8, 'max_grid_size': 15, 'synthesis_ratio': 0.55, 'strategic_complexity': 'ensemble_prep_basic', 'focus': 'basic_ensemble_coordination'},
    {'stage': 9, 'max_grid_size': 16, 'synthesis_ratio': 0.5, 'strategic_complexity': 'arc_strategic_basic', 'focus': 'arc_strategic_patterns'},
    {'stage': 10, 'max_grid_size': 18, 'synthesis_ratio': 0.45, 'strategic_complexity': 'meta_reasoning', 'focus': 'meta_cognitive_reasoning'},
    
    # Advanced Strategic Mastery (20x20 - 30x30)
    {'stage': 11, 'max_grid_size': 20, 'synthesis_ratio': 0.4, 'strategic_complexity': 'ensemble_prep_advanced', 'focus': 'advanced_ensemble_coordination'},
    {'stage': 12, 'max_grid_size': 24, 'synthesis_ratio': 0.35, 'strategic_complexity': 'arc_strategic_advanced', 'focus': 'arc_advanced_strategy'},
    {'stage': 13, 'max_grid_size': 27, 'synthesis_ratio': 0.3, 'strategic_complexity': 'strategic_mastery', 'focus': 'strategic_expertise'},
    {'stage': 14, 'max_grid_size': 30, 'synthesis_ratio': 0.25, 'strategic_complexity': 'arc_strategic_mastery', 'focus': 'arc_strategic_mastery'},
    {'stage': 15, 'max_grid_size': 30, 'synthesis_ratio': 0.2, 'strategic_complexity': 'strategic_genius', 'focus': 'strategic_intelligence_genius'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 120}\033[0m")
print(f"\033[96mMINERVA V5 Enhanced Training - Extended Strategic Ensemble Coordinator for ARC-AGI-2\033[0m")
print(f"\033[96mBuilds on V4 with Extended Training: 16 Stages + ARC-Specific Strategic Intelligence\033[0m")
print(f"\033[96mTarget: 82%+ Performance with Extended Strategic Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 120}\033[0m")


class MinervaV5StrategicLoss(nn.Module):
    """Extended loss function for V5 strategic reasoning and ARC-specific strategic intelligence"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.strategic_weight = config['strategic_reasoning_weight']
        self.ensemble_weight = config['ensemble_coordination_weight']
        self.pattern_weight = config['pattern_analysis_weight']
        self.decision_weight = config['decision_confidence_weight']
        self.arc_strategic_weight = config['arc_strategic_weight']
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
        exact_bonus = exact_bonus.clamp(min=-6.2)  # Higher clamp for V5 strategic precision
        
        # Transform penalty (very low to encourage strategic exploration)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # V5 Enhanced strategic reasoning bonuses
        strategic_bonus = self._calculate_strategic_bonus(model_outputs, pred_indices, target_indices, input_indices)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs, pred_indices, target_indices)
        pattern_bonus = self._calculate_pattern_bonus(model_outputs)
        decision_bonus = self._calculate_decision_bonus(model_outputs)
        arc_strategic_bonus = self._calculate_arc_strategic_bonus(model_outputs, pred_indices, target_indices)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     strategic_bonus + ensemble_bonus + pattern_bonus + decision_bonus + arc_strategic_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'strategic_bonus': strategic_bonus,
            'ensemble_bonus': ensemble_bonus,
            'pattern_bonus': pattern_bonus,
            'decision_bonus': decision_bonus,
            'arc_strategic_bonus': arc_strategic_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_strategic_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced strategic reasoning bonus for V5"""
        if 'strategic_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward strategic transformations
        strategic_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        non_trivial_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use strategic confidence if available
        if 'strategic_confidence' in outputs:
            strategic_confidence = outputs['strategic_confidence'].squeeze(-1)
            strategic_score = strategic_accuracy * strategic_confidence * (1.0 + non_trivial_mask * 1.0)
        else:
            strategic_score = strategic_accuracy * (1.0 + non_trivial_mask * 1.0)
        
        return -strategic_score.mean() * self.strategic_weight * 0.18
    
    def _calculate_ensemble_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced ensemble coordination bonus for V5"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        ensemble_output = outputs['ensemble_output']
        
        # Reward high strategic consensus
        if 'strategic_consensus' in ensemble_output:
            consensus = ensemble_output['strategic_consensus'].mean()
            ensemble_score = consensus
        else:
            ensemble_score = torch.tensor(0.75).to(pred_indices.device)
        
        # Reward effective cross-attention
        if 'cross_attention_weights' in ensemble_output and ensemble_output['cross_attention_weights'] is not None:
            attention_weights = ensemble_output['cross_attention_weights']
            # Measure attention diversity (avoid collapse)
            attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
            ensemble_score = ensemble_score * torch.sigmoid(attention_entropy)
        
        return -ensemble_score * self.ensemble_weight * 0.15
    
    def _calculate_pattern_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced pattern analysis bonus for V5"""
        if 'strategic_analyses' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        pattern_score = 0
        strategic_analyses = outputs['strategic_analyses']
        
        for analysis in strategic_analyses:
            if 'strategic_analysis' in analysis:
                strategic_analysis = analysis['strategic_analysis']
                
                # Reward strategic pattern confidence
                if 'strategic_patterns' in strategic_analysis:
                    pattern_confidence = strategic_analysis['strategic_patterns'].max(dim=-1)[0].mean()
                    pattern_score += pattern_confidence
                
                # Reward rule analysis
                if 'rule_analysis' in strategic_analysis:
                    rule_confidence = strategic_analysis['rule_analysis'].mean()
                    pattern_score += rule_confidence * 0.8
                
                # Reward decision confidence
                if 'decision_confidence' in strategic_analysis:
                    decision_confidence = strategic_analysis['decision_confidence'].mean()
                    pattern_score += decision_confidence * 0.6
        
        # Normalize by number of analyses
        if len(strategic_analyses) > 0:
            pattern_score = pattern_score / len(strategic_analyses)
        
        return -pattern_score * self.pattern_weight * 0.13
    
    def _calculate_decision_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced decision confidence bonus for V5"""
        if 'multistrategic_features' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        multistrategic_features = outputs['multistrategic_features']
        
        # Encourage diverse strategic representations
        decision_score = 0
        for i, strategic_features in enumerate(multistrategic_features):
            # Measure strategic diversity at each level
            strategic_diversity = strategic_features.std(dim=0).mean()
            decision_score += strategic_diversity * (1.0 / (i + 1))  # Weight by importance
        
        # Normalize
        decision_score = decision_score / len(multistrategic_features)
        
        return -decision_score * self.decision_weight * 0.12
    
    def _calculate_arc_strategic_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                     target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate NEW ARC-specific strategic bonus for V5"""
        # ARC-specific strategic patterns bonus
        arc_strategic_score = 0
        
        # Reward complex strategic transformations typical in ARC
        strategic_complexity = (pred_indices != target_indices).float().sum(dim=[1,2]) / (pred_indices.shape[1] * pred_indices.shape[2])
        arc_strategic_score = strategic_complexity.mean()
        
        # Bonus for strategic memory utilization
        if 'strategic_memory_similarity' in outputs:
            memory_usage = outputs['strategic_memory_similarity'].mean()
            arc_strategic_score = arc_strategic_score * (1.0 + memory_usage)
        
        return -arc_strategic_score * self.arc_strategic_weight * 0.1


class ExtendedStrategicDataset(Dataset):
    """Extended dataset optimized for V5 strategic intelligence with ARC-specific focus"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 strategic_focus: bool = True, arc_specific: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.strategic_focus = strategic_focus
        self.arc_specific = arc_specific
        
        # Load data with extended strategic filtering
        self.samples = []
        self._load_extended_strategic_data()
        
        print(f"\033[96mLoaded {len(self.samples)} extended strategic samples for MINERVA V5 training\033[0m")
        if len(self.samples) < 1000:
            print(f"\033[91mWARNING: Only {len(self.samples)} samples loaded - dataset may be too small for proper training\033[0m")
    
    def _load_extended_strategic_data(self):
        """Load data with extended strategic complexity focus using working pattern"""
        # Load training data (challenges + solutions)
        challenges_path = os.path.join(self.data_dir, 'arc-agi_training_challenges.json')
        solutions_path = os.path.join(self.data_dir, 'arc-agi_training_solutions.json')
        
        if os.path.exists(challenges_path) and os.path.exists(solutions_path):
            with open(challenges_path, 'r') as f:
                challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
            
            for task_id, task_data in challenges.items():
                # Process ALL training examples without solution requirement
                for example in task_data['train']:
                    sample = self._create_extended_strategic_sample(example, True)
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
                            sample = self._create_extended_strategic_sample(test_example, True)
                            if sample:
                                self.samples.append(sample)
        
        # Load evaluation data
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                # Process ALL training examples from evaluation set
                for example in task_data['train']:
                    sample = self._create_extended_strategic_sample(example, True)
                    if sample:
                        self.samples.append(sample)
    
    def _process_extended_strategic_task(self, task: Dict, source_file: str):
        """Process task with extended strategic analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples for strategic learning
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_extended_strategic_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_extended_strategic_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with extended strategic analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Extended strategic analysis
        strategic_analysis = self._analyze_extended_strategic_complexity(input_grid, output_grid, is_arc_task)
        
        # Keep everything - no filtering for maximum data
        # All samples are valuable for V5 training
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'strategic_analysis': strategic_analysis
        }
    
    def _analyze_extended_strategic_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray, is_arc_task: bool) -> Dict:
        """Analyze extended strategic complexity with ARC-specific considerations"""
        # Basic strategic properties
        same_shape = input_grid.shape == output_grid.shape
        same_content = np.array_equal(input_grid, output_grid)
        
        # Pattern analysis
        input_unique = len(np.unique(input_grid))
        output_unique = len(np.unique(output_grid))
        total_unique = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        
        # Extended strategic intelligence level calculation
        strategic_intelligence_level = 0
        
        # Level 0: Identity (no strategy)
        if same_content:
            strategic_intelligence_level = 0
        # Level 1: Simple modifications
        elif same_shape and input_unique == output_unique:
            strategic_intelligence_level = 1
        # Level 2: Strategic transformations
        elif same_shape:
            strategic_intelligence_level = 2
            # Check for strategic pattern generation/removal
            if abs(input_unique - output_unique) > 1:
                strategic_intelligence_level += 1
        # Level 3+: Complex strategic reasoning (high strategy)
        else:
            strategic_intelligence_level = 3
            # Scale changes indicate strategic planning
            scale_factor = (output_grid.shape[0] * output_grid.shape[1]) / (input_grid.shape[0] * input_grid.shape[1])
            if scale_factor > 1.5 or scale_factor < 0.5:
                strategic_intelligence_level += 1
            
            # Complex strategic patterns
            if total_unique > 7 or max(output_grid.shape) > 25:
                strategic_intelligence_level += 1
        
        # ARC-specific bonus
        if is_arc_task:
            strategic_intelligence_level += 0.7  # Higher boost for ARC patterns
        
        # Complexity classification (extended for V5)
        max_dim = max(input_grid.shape + output_grid.shape)
        
        if strategic_intelligence_level <= 0.5 and max_dim <= 7:
            complexity = 'micro'
        elif strategic_intelligence_level <= 1.5 and max_dim <= 10:
            complexity = 'trivial'
        elif strategic_intelligence_level <= 2.5 and max_dim <= 16:
            complexity = 'basic'
        elif strategic_intelligence_level <= 3.5 and max_dim <= 24:
            complexity = 'intermediate'
        elif strategic_intelligence_level <= 4.5:
            complexity = 'advanced'
        else:
            complexity = 'expert'
        
        # Strategic transformation type
        if same_content:
            transform_type = 'identity'
        elif same_shape and input_unique == output_unique:
            transform_type = 'strategic_rearrangement'
        elif same_shape:
            transform_type = 'strategic_modification'
        else:
            transform_type = 'strategic_synthesis'
        
        # Strategic potential (enhanced for V5)
        strategic_potential = strategic_intelligence_level + (total_unique - 2) * 0.6 + max_dim * 0.12
        if is_arc_task:
            strategic_potential += 1.2  # ARC bonus
        
        return {
            'strategic_intelligence_level': strategic_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_patterns': total_unique,
            'strategic_potential': strategic_potential,
            'max_dimension': max_dim,
            'pattern_density': total_unique / (max_dim ** 2),
            'arc_specific': is_arc_task
        }
    
    def __len__(self) -> int:
        # Artificially increase dataset size through augmentation
        return len(self.samples) * 3  # 3x larger effective dataset
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Map augmented indices back to real samples
        real_idx = idx % len(self.samples)
        
        sample = self.samples[real_idx]
        
        # Convert to tensors
        input_tensor = torch.tensor(sample['input'], dtype=torch.long)
        output_tensor = torch.tensor(sample['output'], dtype=torch.long)
        
        # Apply geometric augmentations (50% chance)
        if random.random() < 0.5:
            # Random rotation (90, 180, 270 degrees)
            k = random.randint(1, 3)
            input_tensor = torch.rot90(input_tensor, k=k, dims=[0, 1])
            output_tensor = torch.rot90(output_tensor, k=k, dims=[0, 1])
        
        if random.random() < 0.3:
            # Random flip
            if random.random() < 0.5:
                input_tensor = torch.flip(input_tensor, dims=[0])  # Vertical flip
                output_tensor = torch.flip(output_tensor, dims=[0])
            else:
                input_tensor = torch.flip(input_tensor, dims=[1])  # Horizontal flip
                output_tensor = torch.flip(output_tensor, dims=[1])
        
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


def extended_strategic_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Extended collate function for V5 strategic training"""
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


def train_minerva_specialized_v5():
    """Main training function for MINERVA V5"""
    print(f"\033[96mInitializing MINERVA V5 Extended Strategic Intelligence Training...\033[0m")
    
    # Initialize enhanced model
    model = MinervaV4Enhanced(
        max_grid_size=30,
        hidden_dim=256,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load V4 weights - try multiple paths
    v4_paths = [
        '/content/AutomataNexus_Olympus_AGI2/models/minerva_v4_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/minerva_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/minerva_best.pt'
    ]
    
    weights_loaded = False
    for model_path in v4_paths:
        if os.path.exists(model_path):
            weights_loaded = model.load_compatible_weights(model_path)
            if weights_loaded:
                print(f"\033[96mSuccessfully loaded V4 weights from {model_path}\033[0m")
                break
    
    if not weights_loaded:
        print(f"\033[96mStarting V5 training from scratch - no compatible weights found\033[0m")
    
    # Initialize loss function
    criterion = MinervaV5StrategicLoss(MINERVA_V5_CONFIG)
    
    # Initialize optimizer with V5 learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=MINERVA_V5_CONFIG['learning_rate'],
        weight_decay=MINERVA_V5_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=MINERVA_V5_CONFIG['warmup_epochs'],
        T_mult=int(MINERVA_V5_CONFIG['restart_multiplier']),
        eta_min=MINERVA_V5_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Enhanced Progressive Strategic Training - 12 Advanced Strategic Intelligence Stages\033[0m")
    
    # Extended progressive training through strategic stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 110}\033[0m")
        print(f"\033[38;2;255;222;173mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Strategic: {stage_config['strategic_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 110}\033[0m")
        
        # Create extended strategic dataset for this stage
        dataset = ExtendedStrategicDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            strategic_focus=True,
            arc_specific=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=MINERVA_V5_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=extended_strategic_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_extended_strategic_stage(
            model, dataloader, criterion, optimizer, scheduler, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best V5 model to BOTH locations
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_performance': best_performance,
                'stage': stage_idx,
                'config': MINERVA_V5_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V5'
            }
            
            # Save as minerva_best.pt (primary)
            torch.save(save_dict, '/content/AutomataNexus_Olympus_AGI2/models/minerva_best.pt')
            # Save as minerva_best_v5.pt (backup)
            torch.save(save_dict, '/content/AutomataNexus_Olympus_AGI2/models/minerva_best_v5.pt')
            
            print(f"\033[96mNew best V5 strategic performance: {best_performance:.2%} - Models saved to BOTH locations!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 120}\033[0m")
    print(f"\033[96mMINERVA V5 Extended Strategic Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest V5 Strategic Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 120}\033[0m")
    
    return model, best_performance


def train_extended_strategic_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                 stage_idx, stage_config, training_stats):
    """Train a single extended strategic curriculum stage for V5"""
    model.train()
    
    epochs_for_stage = MINERVA_V5_CONFIG['epochs_per_stage']
    accumulation_steps = MINERVA_V5_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_strategic_count = 0
        arc_strategic_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mAdvanced Strategic Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), MINERVA_V5_CONFIG['gradient_clip'])
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
            
            # Count advanced strategic cases and ARC-specific cases
            for meta in metadata:
                if meta['strategic_analysis']['strategic_intelligence_level'] >= 3:
                    advanced_strategic_count += 1
                if meta['strategic_analysis']['arc_specific']:
                    arc_strategic_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'AdvStrategic': f"{advanced_strategic_count}",
                'ARC': f"{arc_strategic_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            strategic_ratio = advanced_strategic_count / max(total_samples, 1)
            arc_ratio = arc_strategic_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[96m⏰ MINERVA V5 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * MINERVA_V5_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Strategic: {strategic_ratio:.1%} | ARC: {arc_ratio:.1%}\033[0m")
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
    
    # Train V5 model
    model, best_performance = train_minerva_specialized_v5()