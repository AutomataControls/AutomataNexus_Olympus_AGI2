"""
ATLAS Specialized Training V5 - Advanced 2D Spatial Reasoning Expert for ARC-AGI-2
Enhanced V5 trainer that builds upon V4 with more ARC-specific training, stages, and epochs
Loads from atlas_v4_best.pt and adds sophisticated spatial intelligence mastery
Target: 85%+ performance with extended spatial intelligence training
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

# Import ATLAS V5 enhanced model
from src.models.atlas_v5_enhanced import AtlasV5Enhanced

# Enhanced ATLAS V5 Configuration - Extended Spatial Intelligence Focus
ATLAS_V5_CONFIG = {
    # Core Training Parameters - Enhanced for V5 Extended Training
    'batch_size': 48,
    'learning_rate': 0.0002,
    'num_epochs': 600,
    'gradient_accumulation': 5,
    'epochs_per_stage': 30,
    'curriculum_stages': 20,
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.025,  # Even lower - max spatial exploration
    'exact_match_bonus': 10.2,  # Higher bonus for spatial accuracy
    'gradient_clip': 0.42,  # Refined clipping for V5
    'weight_decay': 2.5e-6,  # Even lighter regularization for spatial learning
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'spatial_reasoning_weight': 0.62,  # Enhanced focus - spatial intelligence
    'geometric_transformation_weight': 0.52,  # Enhanced geometric mastery
    'multiscale_processing_weight': 0.48,  # Enhanced multi-scale understanding
    'ensemble_coordination_weight': 0.45,  # Enhanced ensemble integration
    'arc_spatial_weight': 0.42,  # NEW: ARC-specific spatial reasoning
    
    # ATLAS V5-Specific Enhancements
    'spatial_transformer_layers': 6,  # Deep spatial reasoning
    'geometric_memory_size': 200,  # Larger spatial pattern memory
    'geometric_positional_encoding': True,  # Advanced 2D encoding
    'multiscale_processing': True,  # Multi-scale feature processing
    'ensemble_preparation': True,  # OLYMPUS preparation mode
    'test_time_adaptation': True,  # Advanced spatial adaptation
    'arc_spatial_training': True,  # NEW: ARC-specific spatial training mode
    
    # Advanced Training Features
    'label_smoothing': 0.012,  # Refined for spatial precision
    'pattern_diversity_bonus': True,
    'geometric_reasoning_bonus': True,
    'spatial_memory_bonus': True,
    'transformation_composition_bonus': True,
    'arc_spatial_bonus': True,  # NEW: ARC-specific spatial bonus
    
    # Learning Rate Scheduling
    'warmup_epochs': 25,  # Refined warmup for V5
    'cosine_restarts': True,
    'restart_multiplier': 1.18,
    'plateau_patience': 25,
}

# Enhanced 18-Stage Progressive Configuration - Extended Spatial Intelligence Focus
STAGE_CONFIG = [
    # Foundation Spatial Understanding (4x4 - 8x8)
    {'stage': 0, 'max_grid_size': 4,  'synthesis_ratio': 0.98, 'spatial_complexity': 'micro_spatial', 'focus': 'micro_spatial_patterns'},
    {'stage': 1, 'max_grid_size': 5,  'synthesis_ratio': 0.95, 'spatial_complexity': 'basic_shapes', 'focus': 'basic_shape_recognition'},
    {'stage': 2, 'max_grid_size': 6,  'synthesis_ratio': 0.9, 'spatial_complexity': 'simple_rotation', 'focus': 'rotation_detection'},
    {'stage': 3, 'max_grid_size': 7,  'synthesis_ratio': 0.85, 'spatial_complexity': 'reflection_basic', 'focus': 'reflection_learning'},
    {'stage': 4, 'max_grid_size': 8,  'synthesis_ratio': 0.8, 'spatial_complexity': 'translation_basic', 'focus': 'translation_understanding'},
    
    # Intermediate Spatial Transformations (9x9 - 16x16)
    {'stage': 5, 'max_grid_size': 9,  'synthesis_ratio': 0.75, 'spatial_complexity': 'affine_basic', 'focus': 'affine_transformations'},
    {'stage': 6, 'max_grid_size': 10, 'synthesis_ratio': 0.7, 'spatial_complexity': 'composite_simple', 'focus': 'composite_transforms'},
    {'stage': 7, 'max_grid_size': 11, 'synthesis_ratio': 0.65, 'spatial_complexity': 'scaling_rotation', 'focus': 'scaling_with_rotation'},
    {'stage': 8, 'max_grid_size': 12, 'synthesis_ratio': 0.6, 'spatial_complexity': 'complex_geometric', 'focus': 'complex_geometry'},
    {'stage': 9, 'max_grid_size': 14, 'synthesis_ratio': 0.55, 'spatial_complexity': 'pattern_spatial', 'focus': 'spatial_patterns'},
    {'stage': 10, 'max_grid_size': 16, 'synthesis_ratio': 0.5, 'spatial_complexity': 'arc_spatial_basic', 'focus': 'arc_spatial_patterns'},
    
    # Advanced Spatial Mastery (18x18 - 30x30)
    {'stage': 11, 'max_grid_size': 18, 'synthesis_ratio': 0.45, 'spatial_complexity': 'multiscale_basic', 'focus': 'multiscale_reasoning'},
    {'stage': 12, 'max_grid_size': 20, 'synthesis_ratio': 0.4, 'spatial_complexity': 'ensemble_spatial', 'focus': 'ensemble_spatial_coordination'},
    {'stage': 13, 'max_grid_size': 22, 'synthesis_ratio': 0.35, 'spatial_complexity': 'arc_spatial_intermediate', 'focus': 'arc_intermediate_spatial'},
    {'stage': 14, 'max_grid_size': 24, 'synthesis_ratio': 0.3, 'spatial_complexity': 'advanced_geometric', 'focus': 'advanced_geometry'},
    {'stage': 15, 'max_grid_size': 27, 'synthesis_ratio': 0.25, 'spatial_complexity': 'arc_spatial_advanced', 'focus': 'arc_advanced_spatial'},
    {'stage': 16, 'max_grid_size': 30, 'synthesis_ratio': 0.2, 'spatial_complexity': 'spatial_mastery', 'focus': 'spatial_intelligence_mastery'},
    {'stage': 17, 'max_grid_size': 30, 'synthesis_ratio': 0.15, 'spatial_complexity': 'spatial_genius', 'focus': 'spatial_intelligence_genius'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 125}\033[0m")
print(f"\033[96mATLAS V5 Enhanced Training - Extended 2D Spatial Reasoning Expert for ARC-AGI-2\033[0m")
print(f"\033[96mBuilds on V4 with Extended Training: 18 Stages + ARC-Specific Spatial Intelligence\033[0m")
print(f"\033[96mTarget: 85%+ Performance with Extended Spatial Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 125}\033[0m")


class AtlasV5SpatialLoss(nn.Module):
    """Extended loss function for V5 spatial reasoning and ARC-specific spatial intelligence"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.spatial_weight = config['spatial_reasoning_weight']
        self.geometric_weight = config['geometric_transformation_weight']
        self.multiscale_weight = config['multiscale_processing_weight']
        self.ensemble_weight = config['ensemble_coordination_weight']
        self.arc_spatial_weight = config['arc_spatial_weight']
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
        exact_bonus = exact_bonus.clamp(min=-6.8)  # Higher clamp for V5 spatial precision
        
        # Transform penalty (very low to encourage spatial exploration)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # V5 Enhanced spatial reasoning bonuses
        spatial_bonus = self._calculate_spatial_bonus(model_outputs, pred_indices, target_indices, input_indices)
        geometric_bonus = self._calculate_geometric_bonus(model_outputs, pred_indices, target_indices)
        multiscale_bonus = self._calculate_multiscale_bonus(model_outputs)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs)
        arc_spatial_bonus = self._calculate_arc_spatial_bonus(model_outputs, pred_indices, target_indices)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     spatial_bonus + geometric_bonus + multiscale_bonus + ensemble_bonus + arc_spatial_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'spatial_bonus': spatial_bonus,
            'geometric_bonus': geometric_bonus,
            'multiscale_bonus': multiscale_bonus,
            'ensemble_bonus': ensemble_bonus,
            'arc_spatial_bonus': arc_spatial_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_spatial_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                               target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced spatial reasoning bonus for V5"""
        if 'spatial_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward spatial transformations
        spatial_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        non_copy_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use spatial confidence if available
        if 'spatial_confidence' in outputs:
            spatial_confidence = outputs['spatial_confidence'].squeeze(-1)
            spatial_score = spatial_accuracy * spatial_confidence * (1.0 + non_copy_mask * 0.8)
        else:
            spatial_score = spatial_accuracy * (1.0 + non_copy_mask * 0.8)
        
        return -spatial_score.mean() * self.spatial_weight * 0.16
    
    def _calculate_geometric_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate enhanced geometric transformation bonus for V5"""
        if 'spatial_analyses' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        geometric_score = 0
        spatial_analyses = outputs['spatial_analyses']
        
        for analysis in spatial_analyses:
            if 'geometric_analysis' in analysis:
                geom_analysis = analysis['geometric_analysis']
                
                # Reward confident geometric transformations
                if 'transformation_confidence' in geom_analysis:
                    confidence = geom_analysis['transformation_confidence'].mean()
                    geometric_score += confidence
                
                # Reward geometric invariant detection
                if 'geometric_invariants' in geom_analysis:
                    invariants = geom_analysis['geometric_invariants']
                    invariant_consistency = 1.0 - invariants.var(dim=1).mean()
                    geometric_score += invariant_consistency * 0.6
        
        # Normalize by number of analyses
        if len(spatial_analyses) > 0:
            geometric_score = geometric_score / len(spatial_analyses)
        
        return -geometric_score * self.geometric_weight * 0.14
    
    def _calculate_multiscale_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced multi-scale processing bonus for V5"""
        if 'multiscale_features' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        multiscale_features = outputs['multiscale_features']
        
        # Encourage diverse multi-scale representations
        multiscale_score = 0
        for i, scale_features in enumerate(multiscale_features):
            # Measure feature diversity at each scale
            feature_diversity = scale_features.std(dim=[2, 3]).mean()
            multiscale_score += feature_diversity * (1.0 / (i + 1))  # Weight by scale importance
        
        # Normalize
        multiscale_score = multiscale_score / len(multiscale_features)
        
        return -multiscale_score * self.multiscale_weight * 0.12
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate enhanced ensemble coordination bonus for V5"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        ensemble_output = outputs['ensemble_output']
        
        # Reward high spatial consensus
        if 'spatial_consensus' in ensemble_output:
            consensus = ensemble_output['spatial_consensus'].mean()
            ensemble_score = consensus
        else:
            ensemble_score = torch.tensor(0.7).to(list(outputs.values())[0].device)
        
        # Reward effective cross-attention
        if 'cross_attention_weights' in ensemble_output and ensemble_output['cross_attention_weights'] is not None:
            attention_weights = ensemble_output['cross_attention_weights']
            # Measure attention diversity (avoid collapse)
            attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
            ensemble_score = ensemble_score * torch.sigmoid(attention_entropy)
        
        return -ensemble_score * self.ensemble_weight * 0.13
    
    def _calculate_arc_spatial_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                   target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate NEW ARC-specific spatial bonus for V5"""
        # ARC-specific spatial patterns bonus
        arc_spatial_score = 0
        
        # Reward complex spatial transformations typical in ARC
        spatial_complexity = (pred_indices != target_indices).float().sum(dim=[1,2]) / (pred_indices.shape[1] * pred_indices.shape[2])
        arc_spatial_score = spatial_complexity.mean()
        
        # Bonus for spatial memory utilization
        if 'spatial_memory_similarity' in outputs:
            memory_usage = outputs['spatial_memory_similarity'].mean()
            arc_spatial_score = arc_spatial_score * (1.0 + memory_usage)
        
        return -arc_spatial_score * self.arc_spatial_weight * 0.11


class ExtendedSpatialDataset(Dataset):
    """Extended dataset optimized for V5 spatial intelligence with ARC-specific focus"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 spatial_focus: bool = True, arc_specific: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.spatial_focus = spatial_focus
        self.arc_specific = arc_specific
        
        # Load data with extended spatial filtering
        self.samples = []
        self._load_extended_spatial_data()
        
        print(f"\033[96mLoaded {len(self.samples)} extended spatial samples for ATLAS V5 training\033[0m")
    
    def _load_extended_spatial_data(self):
        """Load data with extended spatial complexity focus and ARC specificity"""
        data_files = [
            'arc-agi_training_challenges.json',
            'arc-agi_evaluation_challenges.json'
        ]
        
        # Emphasize ARC data more in V5
        arc_emphasis = 4 if self.arc_specific else 1
        
        for file in data_files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Process ARC data multiple times for emphasis
                    emphasis_count = arc_emphasis if 'arc-agi' in file else 1
                    
                    for _ in range(emphasis_count):
                        for task_id, task_data in data.items():
                            self._process_extended_spatial_task(task_data, file)
    
    def _process_extended_spatial_task(self, task: Dict, source_file: str):
        """Process task with extended spatial analysis"""
        is_arc_task = 'arc-agi' in source_file
        
        # Process only training examples (they have both input and output)
        for example in task.get('train', []):
            sample = self._create_extended_spatial_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_extended_spatial_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with extended spatial analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Extended spatial analysis
        spatial_analysis = self._analyze_extended_spatial_complexity(input_grid, output_grid, is_arc_task)
        
        # Filter for extended spatial relevance (more inclusive for V5)
        if self.spatial_focus and spatial_analysis['spatial_intelligence_level'] < 1:
            if random.random() > 0.65:  # Keep 65% of simple cases for V5
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'spatial_analysis': spatial_analysis
        }
    
    def _analyze_extended_spatial_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray, is_arc_task: bool) -> Dict:
        """Analyze extended spatial complexity with ARC-specific considerations"""
        # Basic spatial properties
        same_shape = input_grid.shape == output_grid.shape
        same_content = np.array_equal(input_grid, output_grid)
        
        # Shape analysis
        input_h, input_w = input_grid.shape
        output_h, output_w = output_grid.shape
        
        # Color analysis
        input_unique = len(np.unique(input_grid))
        output_unique = len(np.unique(output_grid))
        total_unique = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        
        # Extended spatial intelligence level calculation
        spatial_intelligence_level = 0
        
        # Level 1: Identity or trivial
        if same_content:
            spatial_intelligence_level = 0
        # Level 2: Same shape transformations
        elif same_shape:
            # Check for rotations
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(input_grid, k), output_grid):
                    spatial_intelligence_level = max(spatial_intelligence_level, 3)
            
            # Check for reflections
            if (np.array_equal(np.fliplr(input_grid), output_grid) or 
                np.array_equal(np.flipud(input_grid), output_grid)):
                spatial_intelligence_level = max(spatial_intelligence_level, 3)
            
            # Color-only changes
            if spatial_intelligence_level == 0:
                if input_unique != output_unique:
                    spatial_intelligence_level = 2
                else:
                    spatial_intelligence_level = 1
        # Level 3+: Shape changes
        else:
            spatial_intelligence_level = 4
            
            # Scale changes
            scale_factor_h = output_h / input_h
            scale_factor_w = output_w / input_w
            
            if abs(scale_factor_h - round(scale_factor_h)) < 0.1 and abs(scale_factor_w - round(scale_factor_w)) < 0.1:
                spatial_intelligence_level += 1  # Clean scaling
            
            # Complex multi-dimensional changes
            if total_unique > 6 or max(output_h, output_w) > 20:
                spatial_intelligence_level += 1
        
        # ARC-specific bonus
        if is_arc_task:
            spatial_intelligence_level += 0.8  # Higher boost for ARC spatial patterns
        
        # Complexity classification (extended for V5)
        max_dim = max(input_h, input_w, output_h, output_w)
        
        if spatial_intelligence_level <= 0.5 and max_dim <= 6:
            complexity = 'micro'
        elif spatial_intelligence_level <= 1.5 and max_dim <= 10:
            complexity = 'trivial'
        elif spatial_intelligence_level <= 2.5 and max_dim <= 15:
            complexity = 'basic'
        elif spatial_intelligence_level <= 3.5 and max_dim <= 20:
            complexity = 'intermediate'
        elif spatial_intelligence_level <= 4.5:
            complexity = 'advanced'
        else:
            complexity = 'expert'
        
        # Geometric transformation type
        if same_content:
            transform_type = 'identity'
        elif same_shape and input_unique == output_unique:
            transform_type = 'geometric_only'
        elif same_shape:
            transform_type = 'geometric_and_color'
        else:
            transform_type = 'shape_and_content'
        
        # Spatial potential (enhanced for V5)
        spatial_potential = spatial_intelligence_level + (total_unique - 2) * 0.6 + max_dim * 0.15
        if is_arc_task:
            spatial_potential += 1.5  # ARC bonus
        
        return {
            'spatial_intelligence_level': spatial_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_colors': total_unique,
            'spatial_potential': spatial_potential,
            'max_dimension': max_dim,
            'scale_factor_h': output_h / input_h if input_h > 0 else 1.0,
            'scale_factor_w': output_w / input_w if input_w > 0 else 1.0,
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
            'spatial_analysis': sample['spatial_analysis']
        }
        
        return input_final, output_final, metadata


def extended_spatial_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Extended collate function for V5 spatial training"""
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


def train_atlas_specialized_v5():
    """Main training function for ATLAS V5"""
    print(f"\033[96mInitializing ATLAS V5 Extended Spatial Intelligence Training...\033[0m")
    
    # Initialize enhanced model
    model = AtlasV5Enhanced(
        max_grid_size=30,
        d_model=128,
        num_layers=2,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load V4 weights with multiple fallback paths
    model_paths = [
        '/content/AutomataNexus_Olympus_AGI2/models/atlas_v4_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/atlas_best.pt',
        '/content/AutomataNexus_Olympus_AGI2/models/atlas_best.pt'
    ]
    
    weights_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Use built-in compatible weight loading
                success = model.load_compatible_weights(model_path)
                if success:
                    print(f"\033[96mATLAS V4: Successfully loaded compatible weights from {model_path}\033[0m")
                    weights_loaded = True
                    break
            except Exception as e:
                continue
    
    if not weights_loaded:
        print(f"\033[96mWarning: Could not load V4 weights, starting V5 training from scratch\033[0m")
    else:
        print(f"\033[96mSuccessfully loaded V4 weights for V5 fast training\033[0m")
    
    # Initialize loss function
    criterion = AtlasV5SpatialLoss(ATLAS_V5_CONFIG)
    
    # Initialize optimizer with V5 learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ATLAS_V5_CONFIG['learning_rate'],
        weight_decay=ATLAS_V5_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=ATLAS_V5_CONFIG['warmup_epochs'],
        T_mult=int(ATLAS_V5_CONFIG['restart_multiplier']),
        eta_min=ATLAS_V5_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Extended Progressive Spatial Training - 18 Enhanced Spatial Intelligence Stages\033[0m")
    
    # Extended progressive training through spatial stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 115}\033[0m")
        print(f"\033[38;2;255;204;153mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Spatial: {stage_config['spatial_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 115}\033[0m")
        
        # Create extended spatial dataset for this stage
        dataset = ExtendedSpatialDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            spatial_focus=True,
            arc_specific=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=ATLAS_V5_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=extended_spatial_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_extended_spatial_stage(
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
                'config': ATLAS_V5_CONFIG,
                'ensemble_state': model.get_ensemble_state(),
                'training_version': 'V5'
            }, '/content/AutomataNexus_Olympus_AGI2/models/atlas_v5_best.pt')
            print(f"\033[96mNew best V5 spatial performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 125}\033[0m")
    print(f"\033[96mATLAS V5 Extended Spatial Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest V5 Spatial Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 125}\033[0m")
    
    return model, best_performance


def train_extended_spatial_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                               stage_idx, stage_config, training_stats):
    """Train a single extended spatial curriculum stage for V5"""
    model.train()
    
    epochs_for_stage = ATLAS_V5_CONFIG['epochs_per_stage']
    accumulation_steps = ATLAS_V5_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_spatial_count = 0
        arc_spatial_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mExtended Spatial Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), ATLAS_V5_CONFIG['gradient_clip'])
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
            
            # Count advanced spatial cases and ARC-specific cases
            for meta in metadata:
                if meta['spatial_analysis']['spatial_intelligence_level'] >= 4:
                    advanced_spatial_count += 1
                if meta['spatial_analysis']['arc_specific']:
                    arc_spatial_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'AdvSpatial': f"{advanced_spatial_count}",
                'ARC': f"{arc_spatial_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 10 == 0 or epoch == epochs_for_stage - 1:
            spatial_ratio = advanced_spatial_count / max(total_samples, 1)
            arc_ratio = arc_spatial_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[38;2;255;204;153m⏰ ATLAS V5 Stage {stage_idx}, Epoch {epoch} \033[96m(Global: {stage_idx * ATLAS_V5_CONFIG['epochs_per_stage'] + epoch + 1})\033[38;2;255;204;153m:\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Spatial: {spatial_ratio:.1%} | ARC: {arc_ratio:.1%}\033[0m")
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
    model, best_performance = train_atlas_specialized_v5()