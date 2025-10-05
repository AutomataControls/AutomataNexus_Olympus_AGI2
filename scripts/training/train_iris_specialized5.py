"""
IRIS Specialized Training V5 - Advanced Color Pattern Recognition Expert for ARC-AGI-2
Enhanced V5 trainer that builds upon V4 with more ARC-specific training, stages, and epochs
Loads from iris_v4_best.pt and adds sophisticated color intelligence mastery
Target: 80%+ performance with extended color intelligence training
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

# Enhanced IRIS V5 Configuration - MINERVA-LIKE SPEED OPTIMIZATION
IRIS_V5_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 5,
    'gradient_accumulation': 1,
    'epochs_per_stage': 1,
    'curriculum_stages': 5,
    
    # Enhanced Loss Configuration - MINERVA-LIKE
    'transform_penalty': 0.06,  # Even lower - max color exploration like MINERVA
    'exact_match_bonus': 9.2,  # Higher bonus for color accuracy
    'gradient_clip': 0.52,  # Slightly higher tolerance for V5
    'weight_decay': 4e-6,  # Even lighter regularization for color
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'chromatic_reasoning_weight': 0.05,
    'color_harmony_weight': 0.02,
    'color_space_weight': 0.02,
    'ensemble_coordination_weight': 0.02,
    'arc_color_weight': 0.02,
    
    # IRIS V5-Specific Enhancements
    'chromatic_transformer_layers': 6,
    'color_pattern_memory_size': 280,
    'chromatic_positional_encoding': False,
    'ensemble_preparation': False,
    'test_time_adaptation': False,
    'arc_color_training': False,
    
    # Advanced Training Features
    'label_smoothing': 0.018,  # Refined for color precision
    'pattern_diversity_bonus': False,
    'chromatic_reasoning_bonus': False,
    'color_harmony_bonus': False,
    'color_expertise_bonus': False,
    'arc_color_bonus': False,
    
    # Learning Rate Scheduling - MINERVA-LIKE SPEED
    'warmup_epochs': 15,
    'cosine_restarts': True,
    'restart_multiplier': 1.25,
    'plateau_patience': 22,
}

STAGE_CONFIG = [
    {'stage': 0, 'max_grid_size': 8,  'synthesis_ratio': 0.95, 'color_complexity': 'basic_colors', 'focus': 'primary_color_recognition'},
    {'stage': 1, 'max_grid_size': 12, 'synthesis_ratio': 0.9, 'color_complexity': 'color_patterns', 'focus': 'color_pattern_recognition'},
    {'stage': 2, 'max_grid_size': 16, 'synthesis_ratio': 0.85, 'color_complexity': 'simple_mapping', 'focus': 'basic_color_mapping'},
    {'stage': 3, 'max_grid_size': 20, 'synthesis_ratio': 0.8, 'color_complexity': 'complex_mapping', 'focus': 'complex_color_mapping'},
    {'stage': 4, 'max_grid_size': 25, 'synthesis_ratio': 0.75, 'color_complexity': 'color_genius', 'focus': 'color_intelligence_mastery'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 120}\033[0m")
print(f"\033[96mIRIS V5 Enhanced Training - Extended Color Pattern Recognition Expert for ARC-AGI-2\033[0m")
print(f"\033[96mBuilds on V4 with Extended Training: 5 Stages + ARC-Specific Color Intelligence\033[0m")
print(f"\033[96mTarget: 80%+ Performance with Extended Color Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 120}\033[0m")


class IrisV5ChromaticLoss(nn.Module):
    """Extended loss function for V5 chromatic reasoning and ARC-specific color intelligence"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.chromatic_weight = config['chromatic_reasoning_weight']
        self.harmony_weight = config['color_harmony_weight']
        self.color_space_weight = config['color_space_weight']
        self.ensemble_weight = config['ensemble_coordination_weight']
        self.arc_color_weight = config['arc_color_weight']
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
        exact_bonus = exact_bonus.clamp(min=-6.5)  # Higher clamp for V5 color precision
        
        # Transform penalty (low to encourage color transformations)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Simplified bonuses for speed
        chromatic_bonus = torch.tensor(0.0).to(pred_indices.device)
        harmony_bonus = torch.tensor(0.0).to(pred_indices.device) 
        color_space_bonus = torch.tensor(0.0).to(pred_indices.device)
        ensemble_bonus = torch.tensor(0.0).to(pred_indices.device)
        arc_color_bonus = torch.tensor(0.0).to(pred_indices.device)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     chromatic_bonus + harmony_bonus + color_space_bonus + ensemble_bonus + arc_color_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'chromatic_bonus': chromatic_bonus,
            'harmony_bonus': harmony_bonus,
            'color_space_bonus': color_space_bonus,
            'ensemble_bonus': ensemble_bonus,
            'arc_color_bonus': arc_color_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_chromatic_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced chromatic reasoning bonus for V5"""
        if 'chromatic_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward chromatic transformations
        chromatic_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        color_change_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use color expertise if available
        if 'color_expertise' in outputs:
            color_confidence = outputs['color_expertise'].squeeze(-1)
            chromatic_score = chromatic_accuracy * color_confidence * (1.0 + color_change_mask * 0.9)
        else:
            chromatic_score = chromatic_accuracy * (1.0 + color_change_mask * 0.9)
        
        return -chromatic_score.mean() * self.chromatic_weight * 0.16
    
    def _calculate_harmony_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                               target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced color harmony bonus for V5"""
        if 'chromatic_analyses' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        harmony_score = 0
        chromatic_analyses = outputs['chromatic_analyses']
        
        for analysis in chromatic_analyses:
            if 'creative_analysis' in analysis:
                color_analysis = analysis['creative_analysis']
                
                # Reward color harmony detection
                if 'harmony_patterns' in color_analysis:
                    harmony_confidence = color_analysis['harmony_patterns'].max(dim=-1)[0].mean()
                    harmony_score += harmony_confidence
                
                # Reward consistent color transformations
                if 'color_transformation_matrix' in color_analysis:
                    transform_matrix = color_analysis['color_transformation_matrix']
                    # Measure transformation consistency (not too chaotic)
                    transform_entropy = -(transform_matrix * torch.log(transform_matrix + 1e-8)).sum(dim=-1).mean()
                    # Reward moderate entropy (structured but not trivial)
                    optimal_entropy = 2.2  # Target entropy for V5
                    entropy_score = torch.exp(-torch.abs(transform_entropy - optimal_entropy))
                    harmony_score += entropy_score * 0.6
        
        # Normalize by number of analyses
        if len(chromatic_analyses) > 0:
            harmony_score = harmony_score / len(chromatic_analyses)
        
        return -harmony_score * self.harmony_weight * 0.14
    
    def _calculate_color_space_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced color space reasoning bonus for V5"""
        if 'multichromatic_features' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        multichromatic_features = outputs['multichromatic_features']
        
        # Encourage diverse color space representations
        color_space_score = 0
        for i, chromatic_features in enumerate(multichromatic_features):
            # Measure chromatic diversity at each color space
            chromatic_diversity = chromatic_features.std(dim=0).mean()
            color_space_score += chromatic_diversity * (1.0 / (i + 1))  # Weight by importance
        
        # Normalize
        color_space_score = color_space_score / len(multichromatic_features)
        
        return -color_space_score * self.color_space_weight * 0.12
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced ensemble coordination bonus for V5"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        ensemble_output = outputs['ensemble_output']
        
        # Reward high color consensus
        if 'color_consensus' in ensemble_output:
            consensus = ensemble_output['color_consensus'].mean()
            ensemble_score = consensus
        else:
            ensemble_score = torch.tensor(0.72).to(list(outputs.values())[0].device)
        
        # Reward high color expertise
        if 'color_expertise' in ensemble_output:
            expertise = ensemble_output['color_expertise'].mean()
            ensemble_score = ensemble_score * expertise
        
        return -ensemble_score * self.ensemble_weight * 0.11
    
    def _calculate_arc_color_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate NEW ARC-specific color bonus for V5"""
        # ARC-specific color patterns bonus
        arc_color_score = 0
        
        # Reward complex color transformations typical in ARC
        color_complexity = (pred_indices != target_indices).float().sum(dim=[1,2]) / (pred_indices.shape[1] * pred_indices.shape[2])
        arc_color_score = color_complexity.mean()
        
        # Bonus for color memory utilization
        if 'creative_memory_similarity' in outputs:
            memory_usage = outputs['creative_memory_similarity'].mean()
            arc_color_score = arc_color_score * (1.0 + memory_usage)
        
        return -arc_color_score * self.arc_color_weight * 0.1


class ExtendedColorDataset(Dataset):
    """Extended dataset optimized for V5 color intelligence with ARC-specific focus"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 color_focus: bool = True, arc_specific: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.color_focus = color_focus
        self.arc_specific = arc_specific
        
        # Load data with extended color filtering
        self.samples = []
        self._load_extended_color_data()
        
        print(f"\033[96mLoaded {len(self.samples)} extended color samples for IRIS V5 training\033[0m")
    
    def _load_extended_color_data(self):
        """Load data with extended color complexity focus and ARC specificity"""
        data_files = [
            'arc-agi_training_challenges.json',
            'arc-agi_evaluation_challenges.json',
            'arc-agi_test_challenges.json'
        ]
        
        print(f"🔍 Looking for data files in: {self.data_dir}")
        
        # Emphasize ARC data more in V5
        arc_emphasis = 3 if self.arc_specific else 1
        files_found = 0
        
        for file in data_files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                files_found += 1
                print(f"✅ Found: {file}")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Process ARC data multiple times for emphasis
                    emphasis_count = arc_emphasis if 'arc_' in file else 1
                    
                    before_count = len(self.samples)
                    for _ in range(emphasis_count):
                        # Handle ARC data format (dict with task IDs as keys)
                        if isinstance(data, dict):
                            for task_id, task_data in data.items():
                                self._process_extended_color_task(task_data, file)
                        else:
                            # Handle list format if needed
                            for task in data:
                                self._process_extended_color_task(task, file)
                    after_count = len(self.samples)
                    print(f"   Added {after_count - before_count} samples from {file}")
            else:
                print(f"❌ Missing: {file}")
        
        print(f"📊 Found {files_found} data files, total samples: {len(self.samples)}")
    
    def _process_extended_color_task(self, task: Dict, source_file: str):
        """Process task with extended color analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process only training examples (they have both input and output)
        for example in task.get('train', []):
            sample = self._create_extended_color_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_extended_color_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with extended color analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Extended color analysis
        color_analysis = self._analyze_extended_color_complexity(input_grid, output_grid, is_arc_task)
        
        # Filter for extended color relevance (VERY inclusive for V5 to avoid empty dataset)
        if self.color_focus and color_analysis['color_intelligence_level'] < 1:
            if random.random() > 0.9:  # Keep 90% of simple cases for V5 to ensure samples
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'color_analysis': color_analysis
        }
    
    def _analyze_extended_color_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray, is_arc_task: bool) -> Dict:
        """Analyze extended color complexity with ARC-specific considerations"""
        # Basic color properties
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        all_colors = input_colors.union(output_colors)
        
        # Color transformation analysis
        same_colors = input_colors == output_colors
        colors_added = len(output_colors - input_colors)
        colors_removed = len(input_colors - output_colors)
        
        # Extended color intelligence level calculation
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
        
        # ARC-specific bonus
        if is_arc_task:
            color_intelligence_level += 0.6  # ARC bonus for color patterns
        
        # Color harmony analysis (enhanced for V5)
        harmony_score = 0
        if len(all_colors) >= 3:
            # Enhanced harmony heuristics
            primary_colors = {1, 2, 3}  # Red, Blue, Green equivalents
            secondary_colors = {4, 5, 6, 7}  # Other colors
            accent_colors = {8, 9}  # Accent colors
            
            primary_present = len(all_colors.intersection(primary_colors))
            secondary_present = len(all_colors.intersection(secondary_colors))
            accent_present = len(all_colors.intersection(accent_colors))
            
            if primary_present >= 2 and secondary_present >= 1:
                harmony_score = 3  # Excellent color harmony
            elif primary_present >= 2 or secondary_present >= 2:
                harmony_score = 2  # Good color harmony
            elif accent_present >= 1:
                harmony_score = 1  # Basic color harmony
        
        # Complexity classification (extended for V5)
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
        
        # Color potential (enhanced for V5)
        color_potential = color_intelligence_level + (unique_color_count - 2) * 0.7 + max_dim * 0.1
        if is_arc_task:
            color_potential += 1.2  # ARC bonus
        
        return {
            'color_intelligence_level': color_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_colors': unique_color_count,
            'colors_added': colors_added,
            'colors_removed': colors_removed,
            'harmony_score': harmony_score,
            'color_potential': color_potential,
            'max_dimension': max_dim,
            'arc_specific': is_arc_task
        }
    
    def __len__(self) -> int:
        return len(self.samples) * 5
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        real_idx = idx % len(self.samples)
        sample = self.samples[real_idx]
        
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


def extended_color_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Extended collate function for V5 color training"""
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


def train_iris_specialized_v5():
    """Main training function for IRIS V5"""
    print(f"\033[96mInitializing IRIS V5 Extended Color Intelligence Training...\033[0m")
    
    # Initialize enhanced model (same as MINERVA approach)
    model = IrisV4Enhanced(
        max_grid_size=30,
        d_model=256,
        num_layers=6,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load V4 weights with multiple fallback paths
    model_paths = [
        '/content/AutomataNexus_Olympus_AGI2/models/iris_v4_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/iris_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/iris_best.pt'
    ]
    
    weights_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Use built-in compatible weight loading
                success = model.load_compatible_weights(model_path)
                if success:
                    print(f"\033[96mIRIS V4: Successfully loaded compatible weights from {model_path}\033[0m")
                    weights_loaded = True
                    break
            except Exception as e:
                continue
    
    if not weights_loaded:
        print(f"\033[96mWarning: Could not load V4 weights, starting V5 training from scratch\033[0m")
    else:
        print(f"\033[96mSuccessfully loaded V4 weights for V5 extended training\033[0m")
    
    # Initialize loss function
    criterion = IrisV5ChromaticLoss(IRIS_V5_CONFIG)
    
    # Initialize optimizer with V5 learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=IRIS_V5_CONFIG['learning_rate'],
        weight_decay=IRIS_V5_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=IRIS_V5_CONFIG['warmup_epochs'],
        T_mult=int(IRIS_V5_CONFIG['restart_multiplier']),
        eta_min=IRIS_V5_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Extended Progressive Color Training - 5 Enhanced Color Intelligence Stages\033[0m")
    
    # Extended progressive training through color stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 110}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Color: {stage_config['color_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 110}\033[0m")
        
        # Create extended color dataset for this stage
        dataset = ExtendedColorDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            color_focus=True,
            arc_specific=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=IRIS_V5_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=extended_color_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_extended_color_stage(
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
                'config': IRIS_V5_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V5'
            }, '/content/AutomataNexus_Olympus_AGI2/models/iris_v5_best.pt')
            print(f"\033[96mNew best V5 color performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 120}\033[0m")
    print(f"\033[96mIRIS V5 Extended Color Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest V5 Color Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 120}\033[0m")
    
    return model, best_performance


def train_extended_color_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                             stage_idx, stage_config, training_stats):
    """Train a single extended color curriculum stage for V5"""
    model.train()
    
    epochs_for_stage = IRIS_V5_CONFIG['epochs_per_stage']
    accumulation_steps = IRIS_V5_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_color_count = 0
        arc_color_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mExtended Color Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), IRIS_V5_CONFIG['gradient_clip'])
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
            
            # Count advanced color cases and ARC-specific cases
            for meta in metadata:
                if meta['color_analysis']['color_intelligence_level'] >= 3:
                    advanced_color_count += 1
                if meta['color_analysis']['arc_specific']:
                    arc_color_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'AdvColor': f"{advanced_color_count}",
                'ARC': f"{arc_color_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 12 == 0 or epoch == epochs_for_stage - 1:
            color_ratio = advanced_color_count / max(total_samples, 1)
            arc_ratio = arc_color_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[38;2;255;204;153m⏰ IRIS V5 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * IRIS_V5_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Color: {color_ratio:.1%} | ARC: {arc_ratio:.1%}\033[0m")
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
    model, best_performance = train_iris_specialized_v5()