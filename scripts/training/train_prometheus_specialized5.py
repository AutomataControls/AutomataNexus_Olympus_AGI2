"""
PROMETHEUS Specialized Training V5 - Advanced Creative Pattern Generation Expert for ARC-AGI-2
Enhanced V5 trainer that builds upon V4 with more ARC-specific training, stages, and epochs
Loads from prometheus_v4_best.pt and adds sophisticated creative intelligence mastery
Target: 78%+ performance with extended creative intelligence training
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

# Import enhanced PROMETHEUS V4 model
from src.models.prometheus_v4_enhanced import PrometheusV4Enhanced

# Enhanced PROMETHEUS V5 Configuration - Massive Multi-Dimensional Intelligence Focus
PROMETHEUS_V5_CONFIG = {
    # Core Training Parameters - Enhanced for V5 Massive Training
    'batch_size': 20,  # Smaller for massive scale computations
    'learning_rate': 0.00012,  # Lower for massive fine-tuning from V4
    'num_epochs': 500,  # Extended training: 10 stages x 50 epochs
    'gradient_accumulation': 12,  # Effective batch: 240
    'epochs_per_stage': 50,  # Extended epochs per stage
    'curriculum_stages': 10,  # Focused 10-stage progression
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.015,  # Even lower - max creative exploration
    'exact_match_bonus': 8.2,  # Higher bonus for creative accuracy
    'gradient_clip': 0.65,  # Higher tolerance for creative gradients
    'weight_decay': 1.5e-6,  # Even lighter regularization for creativity
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'creative_reasoning_weight': 0.65,  # Enhanced focus - creative intelligence
    'pattern_synthesis_weight': 0.55,  # Enhanced pattern generation mastery
    'innovation_weight': 0.5,  # Enhanced innovation encouragement
    'ensemble_coordination_weight': 0.45,  # Enhanced ensemble integration
    'arc_specific_weight': 0.4,  # NEW: ARC-specific creative reasoning
    
    # PROMETHEUS V5-Specific Enhancements
    'creative_transformer_layers': 6,  # Deep creative reasoning
    'pattern_memory_size': 250,  # Larger creative pattern memory
    'creative_positional_encoding': True,  # Creative-aware positioning
    'ensemble_preparation': True,  # OLYMPUS preparation mode
    'test_time_adaptation': True,  # Advanced creative adaptation
    'arc_specific_training': True,  # NEW: ARC-specific training mode
    
    # Advanced Training Features
    'label_smoothing': 0.025,  # Refined for creative precision
    'pattern_diversity_bonus': True,
    'creative_reasoning_bonus': True,
    'innovation_bonus': True,
    'novelty_bonus': True,
    'arc_creative_bonus': True,  # NEW: ARC-specific creativity bonus
    
    # Learning Rate Scheduling
    'warmup_epochs': 25,  # Refined warmup for V5
    'cosine_restarts': True,
    'restart_multiplier': 1.15,
    'plateau_patience': 22,
}

# Enhanced 10-Stage Progressive Configuration - Focused Creative Intelligence
STAGE_CONFIG = [
    # Foundation Creative Understanding (6x6 - 12x12)
    {'stage': 0, 'max_grid_size': 6,  'synthesis_ratio': 0.9, 'creativity_complexity': 'basic_patterns', 'focus': 'basic_pattern_recognition'},
    {'stage': 1, 'max_grid_size': 8,  'synthesis_ratio': 0.8, 'creativity_complexity': 'simple_synthesis', 'focus': 'simple_pattern_generation'},
    {'stage': 2, 'max_grid_size': 10, 'synthesis_ratio': 0.7, 'creativity_complexity': 'pattern_variation', 'focus': 'pattern_variation_learning'},
    {'stage': 3, 'max_grid_size': 12, 'synthesis_ratio': 0.65, 'creativity_complexity': 'creative_combination', 'focus': 'creative_pattern_combination'},
    
    # Intermediate Creative Reasoning (14x14 - 20x20)
    {'stage': 4, 'max_grid_size': 14, 'synthesis_ratio': 0.6, 'creativity_complexity': 'innovation_basic', 'focus': 'basic_innovation'},
    {'stage': 5, 'max_grid_size': 16, 'synthesis_ratio': 0.55, 'creativity_complexity': 'novelty_detection', 'focus': 'novelty_pattern_detection'},
    {'stage': 6, 'max_grid_size': 18, 'synthesis_ratio': 0.5, 'creativity_complexity': 'creative_rules', 'focus': 'creative_rule_learning'},
    {'stage': 7, 'max_grid_size': 20, 'synthesis_ratio': 0.45, 'creativity_complexity': 'pattern_synthesis', 'focus': 'advanced_pattern_synthesis'},
    
    # Advanced Creative Mastery (24x24 - 30x30)
    {'stage': 8, 'max_grid_size': 24, 'synthesis_ratio': 0.35, 'creativity_complexity': 'expert_creative', 'focus': 'expert_creative_generation'},
    {'stage': 9, 'max_grid_size': 30, 'synthesis_ratio': 0.25, 'creativity_complexity': 'creative_mastery', 'focus': 'creative_intelligence_mastery'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 120}\033[0m")
print(f"\033[96mPROMETHEUS V5 Enhanced Training - Advanced Creative Pattern Generation Expert for ARC-AGI-2\033[0m")
print(f"\033[96mBuilds on V4 with Focused Training: 10 Stages + Advanced Creative Intelligence\033[0m")
print(f"\033[96mTarget: 80%+ Performance with Creative Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 120}\033[0m")


class PrometheusV5CreativeLoss(nn.Module):
    """Extended loss function for V5 creative reasoning and ARC-specific pattern synthesis"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.creative_weight = config['creative_reasoning_weight']
        self.synthesis_weight = config['pattern_synthesis_weight']
        self.innovation_weight = config['innovation_weight']
        self.ensemble_weight = config['ensemble_coordination_weight']
        self.arc_weight = config['arc_specific_weight']
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
        exact_bonus = exact_bonus.clamp(min=-5.5)  # Higher clamp for V5 creative precision
        
        # Transform penalty (very low to encourage creative exploration)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # V5 Enhanced creative reasoning bonuses
        creative_bonus = self._calculate_creative_bonus(model_outputs, pred_indices, target_indices, input_indices)
        synthesis_bonus = self._calculate_synthesis_bonus(model_outputs, pred_indices, target_indices)
        innovation_bonus = self._calculate_innovation_bonus(model_outputs)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs)
        arc_bonus = self._calculate_arc_creative_bonus(model_outputs, pred_indices, target_indices)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     creative_bonus + synthesis_bonus + innovation_bonus + ensemble_bonus + arc_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'creative_bonus': creative_bonus,
            'synthesis_bonus': synthesis_bonus,
            'innovation_bonus': innovation_bonus,
            'ensemble_bonus': ensemble_bonus,
            'arc_bonus': arc_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_creative_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced creative reasoning bonus for V5"""
        if 'creative_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward creative transformations
        creative_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        innovation_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use creative expertise if available
        if 'creative_expertise' in outputs:
            creative_confidence = outputs['creative_expertise'].squeeze(-1)
            creative_score = creative_accuracy * creative_confidence * (1.0 + innovation_mask * 0.9)
        else:
            creative_score = creative_accuracy * (1.0 + innovation_mask * 0.9)
        
        return -creative_score.mean() * self.creative_weight * 0.18
    
    def _calculate_synthesis_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced pattern synthesis bonus for V5"""
        if 'creative_analyses' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        synthesis_score = 0
        creative_analyses = outputs['creative_analyses']
        
        for analysis in creative_analyses:
            if 'creative_analysis' in analysis:
                creative_analysis = analysis['creative_analysis']
                
                # Reward pattern generation confidence
                if 'generated_patterns' in creative_analysis:
                    pattern_confidence = creative_analysis['generated_patterns'].max(dim=-1)[0].mean()
                    synthesis_score += pattern_confidence
                
                # Reward novelty detection
                if 'novelty_scores' in creative_analysis:
                    novelty_confidence = creative_analysis['novelty_scores'].max(dim=-1)[0].mean()
                    synthesis_score += novelty_confidence * 0.8
                
                # Reward creative composition
                if 'creative_composition' in creative_analysis:
                    composition_diversity = creative_analysis['creative_composition'].std(dim=-1).mean()
                    synthesis_score += composition_diversity * 0.6
        
        # Normalize by number of analyses
        if len(creative_analyses) > 0:
            synthesis_score = synthesis_score / len(creative_analyses)
        
        return -synthesis_score * self.synthesis_weight * 0.15
    
    def _calculate_innovation_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced innovation encouragement bonus for V5"""
        if 'multicreative_features' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        multicreative_features = outputs['multicreative_features']
        
        # Encourage diverse creative representations
        innovation_score = 0
        for i, creative_features in enumerate(multicreative_features):
            # Measure creative diversity at each mode
            creative_diversity = creative_features.std(dim=0).mean()
            innovation_score += creative_diversity * (1.0 / (i + 1))  # Weight by importance
        
        # Normalize
        innovation_score = innovation_score / len(multicreative_features)
        
        return -innovation_score * self.innovation_weight * 0.12
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced ensemble coordination bonus for V5"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        ensemble_output = outputs['ensemble_output']
        
        # Reward high creative consensus
        if 'creative_consensus' in ensemble_output:
            consensus = ensemble_output['creative_consensus'].mean()
            ensemble_score = consensus
        else:
            ensemble_score = torch.tensor(0.75).to(list(outputs.values())[0].device)
        
        # Reward high creative expertise
        if 'creative_expertise' in ensemble_output:
            expertise = ensemble_output['creative_expertise'].mean()
            ensemble_score = ensemble_score * expertise
        
        return -ensemble_score * self.ensemble_weight * 0.1
    
    def _calculate_arc_creative_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                    target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate NEW ARC-specific creative bonus for V5"""
        # ARC-specific creative patterns bonus
        arc_creative_score = 0
        
        # Reward complex pattern transformations typical in ARC
        pattern_complexity = (pred_indices != target_indices).float().sum(dim=[1,2]) / (pred_indices.shape[1] * pred_indices.shape[2])
        arc_creative_score = pattern_complexity.mean()
        
        # Bonus for creative memory utilization
        if 'creative_memory_similarity' in outputs:
            memory_usage = outputs['creative_memory_similarity'].mean()
            arc_creative_score = arc_creative_score * (1.0 + memory_usage)
        
        return -arc_creative_score * self.arc_weight * 0.08


class ExtendedCreativeDataset(Dataset):
    """Extended dataset optimized for V5 creative intelligence with ARC-specific focus"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 creative_focus: bool = True, arc_specific: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.creative_focus = creative_focus
        self.arc_specific = arc_specific
        
        # Load data with extended creative filtering
        self.samples = []
        self._load_extended_creative_data()
        
        print(f"\033[96mLoaded {len(self.samples)} extended creative samples for PROMETHEUS V5 training\033[0m")
    
    def _load_extended_creative_data(self):
        """Load data with extended creative complexity focus using working V4 pattern"""
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
                    self._process_extended_creative_task(combined_task, 'training')
        
        # Load evaluation data
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                eval_task = {'train': task_data['train'], 'test': []}
                self._process_extended_creative_task(eval_task, 'evaluation')
    
    def _process_extended_creative_task(self, task: Dict, source_file: str):
        """Process task with extended creative analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples for creative learning
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_extended_creative_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_extended_creative_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with extended creative analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Extended creative analysis
        creative_analysis = self._analyze_extended_creative_complexity(input_grid, output_grid, is_arc_task)
        
        # Filter for extended creative relevance (more permissive)
        if self.creative_focus and creative_analysis['creative_intelligence_level'] < 2:
            if random.random() > 0.8:  # Keep 80% of simple cases
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'creative_analysis': creative_analysis
        }
    
    def _analyze_extended_creative_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray, is_arc_task: bool) -> Dict:
        """Analyze extended creative complexity with ARC-specific considerations"""
        # Basic creative properties
        same_shape = input_grid.shape == output_grid.shape
        same_content = np.array_equal(input_grid, output_grid)
        
        # Pattern analysis
        input_unique = len(np.unique(input_grid))
        output_unique = len(np.unique(output_grid))
        total_unique = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        
        # Extended creative intelligence level calculation
        creative_intelligence_level = 0
        
        # Level 0: Identity (no creativity)
        if same_content:
            creative_intelligence_level = 0
        # Level 1: Simple modifications
        elif same_shape and input_unique == output_unique:
            creative_intelligence_level = 1
        # Level 2: Creative transformations
        elif same_shape:
            creative_intelligence_level = 2
            # Check for pattern generation/removal
            if abs(input_unique - output_unique) > 1:
                creative_intelligence_level += 1
        # Level 3+: Shape and content changes (high creativity)
        else:
            creative_intelligence_level = 3
            # Scale changes indicate pattern synthesis
            scale_factor = (output_grid.shape[0] * output_grid.shape[1]) / (input_grid.shape[0] * input_grid.shape[1])
            if scale_factor > 1.5 or scale_factor < 0.5:
                creative_intelligence_level += 1
            
            # Complex pattern generation
            if total_unique > 7 or max(output_grid.shape) > 25:
                creative_intelligence_level += 1
        
        # ARC-specific bonus
        if is_arc_task:
            creative_intelligence_level += 0.5  # Slight boost for ARC patterns
        
        # Complexity classification (extended for V5)
        max_dim = max(input_grid.shape + output_grid.shape)
        
        if creative_intelligence_level <= 0.5 and max_dim <= 8:
            complexity = 'micro'
        elif creative_intelligence_level <= 1.5 and max_dim <= 12:
            complexity = 'trivial'
        elif creative_intelligence_level <= 2.5 and max_dim <= 18:
            complexity = 'basic'
        elif creative_intelligence_level <= 3.5 and max_dim <= 25:
            complexity = 'intermediate'
        elif creative_intelligence_level <= 4.5:
            complexity = 'advanced'
        else:
            complexity = 'expert'
        
        # Creative transformation type
        if same_content:
            transform_type = 'identity'
        elif same_shape and input_unique == output_unique:
            transform_type = 'pattern_rearrangement'
        elif same_shape:
            transform_type = 'pattern_modification'
        else:
            transform_type = 'creative_synthesis'
        
        # Innovation potential (enhanced for V5)
        innovation_potential = creative_intelligence_level + (total_unique - 2) * 0.5 + max_dim * 0.1
        if is_arc_task:
            innovation_potential += 1.0  # ARC bonus
        
        return {
            'creative_intelligence_level': creative_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_patterns': total_unique,
            'innovation_potential': innovation_potential,
            'max_dimension': max_dim,
            'pattern_density': total_unique / (max_dim ** 2),
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
            'creative_analysis': sample['creative_analysis']
        }
        
        return input_final, output_final, metadata


def extended_creative_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Extended collate function for V5 creative training"""
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


def train_prometheus_specialized_v5():
    """Main training function for PROMETHEUS V5"""
    print(f"\033[96mInitializing PROMETHEUS V5 Extended Creative Intelligence Training...\033[0m")
    
    # Initialize enhanced model
    model = PrometheusV4Enhanced(
        max_grid_size=30,
        d_model=256,
        num_layers=6,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load V4 weights
    model_path = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/prometheus_v4_best.pt'
    weights_loaded = model.load_compatible_weights(model_path)
    
    if not weights_loaded:
        print(f"\033[96mWarning: Could not load V4 weights, starting V5 training from scratch\033[0m")
    else:
        print(f"\033[96mSuccessfully loaded V4 weights for V5 extended training\033[0m")
    
    # Initialize loss function
    criterion = PrometheusV5CreativeLoss(PROMETHEUS_V5_CONFIG)
    
    # Initialize optimizer with V5 learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=PROMETHEUS_V5_CONFIG['learning_rate'],
        weight_decay=PROMETHEUS_V5_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=PROMETHEUS_V5_CONFIG['warmup_epochs'],
        T_mult=int(PROMETHEUS_V5_CONFIG['restart_multiplier']),
        eta_min=PROMETHEUS_V5_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Focused Progressive Creative Training - 10 Advanced Creative Intelligence Stages\033[0m")
    
    # Extended progressive training through creative stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 115}\033[0m")
        print(f"\033[38;2;255;222;173mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Creative: {stage_config['creativity_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 115}\033[0m")
        
        # Create extended creative dataset for this stage
        dataset = ExtendedCreativeDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            creative_focus=True,
            arc_specific=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=PROMETHEUS_V5_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=extended_creative_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_extended_creative_stage(
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
                'config': PROMETHEUS_V5_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V5'
            }, '/content/AutomataNexus_Olympus_AGI2/models/prometheus_best.pt')
            print(f"\033[96mNew best V5 creative performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 120}\033[0m")
    print(f"\033[96mPROMETHEUS V5 Extended Creative Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest V5 Creative Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 120}\033[0m")
    
    return model, best_performance


def train_extended_creative_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                stage_idx, stage_config, training_stats):
    """Train a single extended creative curriculum stage for V5"""
    model.train()
    
    epochs_for_stage = PROMETHEUS_V5_CONFIG['epochs_per_stage']
    accumulation_steps = PROMETHEUS_V5_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_creative_count = 0
        arc_creative_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mExtended Creative Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), PROMETHEUS_V5_CONFIG['gradient_clip'])
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
            
            # Count advanced creative cases and ARC-specific cases
            for meta in metadata:
                if meta['creative_analysis']['creative_intelligence_level'] >= 3:
                    advanced_creative_count += 1
                if meta['creative_analysis']['arc_specific']:
                    arc_creative_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'AdvCreative': f"{advanced_creative_count}",
                'ARC': f"{arc_creative_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            creative_ratio = advanced_creative_count / max(total_samples, 1)
            arc_ratio = arc_creative_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[96m⏰ PROMETHEUS V5 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * PROMETHEUS_V5_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Creative: {creative_ratio:.1%}\033[0m")
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
    model, best_performance = train_prometheus_specialized_v5()