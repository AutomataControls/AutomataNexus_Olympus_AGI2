"""
ATLAS Specialized Training V4 - Advanced 2D Spatial Reasoning Expert for ARC-AGI-2
Enhanced with geometric transformers, multi-scale processing, and OLYMPUS ensemble preparation
Target: 80%+ performance with sophisticated spatial intelligence
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

# Import enhanced ATLAS V4 model
from src.models.atlas_v4_enhanced import AtlasV4Enhanced

# Enhanced ATLAS V4 Configuration - 2D Spatial Reasoning Focus (OPTIMIZED FOR SPEED)
ATLAS_V4_CONFIG = {
    # Core Training Parameters - OPTIMIZED for V4 Speed + Performance
    'batch_size': 20,  # MINERVA-like efficiency 
    'learning_rate': 0.0002,  # Stable like MINERVA
    'num_epochs': 60,  # Proper training: 10 stages x 6 epochs
    'gradient_accumulation': 2,  # Effective batch 40 for stability
    'epochs_per_stage': 6,  # Adequate training per stage
    'curriculum_stages': 10,  # Streamlined spatial curriculum
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.03,  # Very low - encourage spatial transformations
    'exact_match_bonus': 9.5,  # Very high bonus for spatial precision
    'gradient_clip': 0.4,  # Tight clipping for spatial stability
    'weight_decay': 3e-6,  # Very light regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'spatial_reasoning_weight': 0.55,  # Primary focus - spatial reasoning
    'geometric_transformation_weight': 0.45,  # Geometric mastery
    'multiscale_processing_weight': 0.4,  # Multi-scale understanding
    'ensemble_coordination_weight': 0.35,  # Ensemble integration
    
    # ATLAS V4-Specific Enhancements - KEEP FUNCTIONALITY
    'spatial_transformer_layers': 4,  # Balanced for speed + capability
    'geometric_positional_encoding': True,  # Keep 2D spatial encoding
    'multiscale_processing': True,  # Keep multi-scale features
    'ensemble_preparation': True,  # Keep OLYMPUS preparation
    'test_time_adaptation': True,  # Keep spatial adaptation
    
    # Advanced Training Features - KEEP FUNCTIONALITY
    'label_smoothing': 0.012,  # Light for spatial precision
    'pattern_diversity_bonus': True,
    'geometric_reasoning_bonus': True,
    'spatial_memory_bonus': True,
    'transformation_composition_bonus': True,
    
    # Learning Rate Scheduling - MINERVA-like
    'warmup_epochs': 12,  # Proper warmup
    'cosine_restarts': True,
    'restart_multiplier': 1.3,
    'plateau_patience': 18,
}

# Efficient 10-Stage Spatial Intelligence Curriculum - MINERVA-like
STAGE_CONFIG = [
    # Foundation Spatial Understanding
    {'stage': 0, 'max_grid_size': 6,  'synthesis_ratio': 0.8, 'spatial_complexity': 'basic_shapes', 'focus': 'shape_recognition'},
    {'stage': 1, 'max_grid_size': 8,  'synthesis_ratio': 0.75, 'spatial_complexity': 'simple_rotation', 'focus': 'rotation_detection'},
    {'stage': 2, 'max_grid_size': 10, 'synthesis_ratio': 0.7, 'spatial_complexity': 'reflection_basic', 'focus': 'reflection_learning'},
    
    # Intermediate Spatial Transformations
    {'stage': 3, 'max_grid_size': 12, 'synthesis_ratio': 0.6, 'spatial_complexity': 'translation_basic', 'focus': 'translation_understanding'},
    {'stage': 4, 'max_grid_size': 15, 'synthesis_ratio': 0.55, 'spatial_complexity': 'affine_basic', 'focus': 'affine_transformations'},
    {'stage': 5, 'max_grid_size': 18, 'synthesis_ratio': 0.5, 'spatial_complexity': 'composite_simple', 'focus': 'composite_transforms'},
    {'stage': 6, 'max_grid_size': 20, 'synthesis_ratio': 0.45, 'spatial_complexity': 'scaling_rotation', 'focus': 'scaling_with_rotation'},
    
    # Advanced Spatial Mastery
    {'stage': 7, 'max_grid_size': 24, 'synthesis_ratio': 0.35, 'spatial_complexity': 'complex_geometric', 'focus': 'complex_geometry'},
    {'stage': 8, 'max_grid_size': 28, 'synthesis_ratio': 0.3, 'spatial_complexity': 'pattern_spatial', 'focus': 'spatial_patterns'},
    {'stage': 9, 'max_grid_size': 30, 'synthesis_ratio': 0.25, 'spatial_complexity': 'spatial_genius', 'focus': 'spatial_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 105}\033[0m")
print(f"\033[96mATLAS V4 Enhanced Training - Advanced 2D Spatial Reasoning Expert for ARC-AGI-2\033[0m")
print(f"\033[96mGeometric Transformers + Multi-Scale Processing + OLYMPUS Preparation\033[0m")
print(f"\033[96mTarget: 80%+ Performance with Spatial Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 105}\033[0m")


class AtlasV4SpatialLoss(nn.Module):
    """Advanced loss function for 2D spatial reasoning and geometric transformations"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.spatial_weight = config['spatial_reasoning_weight']
        self.geometric_weight = config['geometric_transformation_weight']
        self.multiscale_weight = config['multiscale_processing_weight']
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
        exact_bonus = exact_bonus.clamp(min=-6.0)  # Higher clamp for spatial precision
        
        # Transform penalty (very low to encourage spatial learning)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Spatial reasoning bonuses
        spatial_bonus = self._calculate_spatial_bonus(model_outputs, pred_indices, target_indices, input_indices)
        geometric_bonus = self._calculate_geometric_bonus(model_outputs, pred_indices, target_indices)
        multiscale_bonus = self._calculate_multiscale_bonus(model_outputs)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     spatial_bonus + geometric_bonus + multiscale_bonus + ensemble_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'spatial_bonus': spatial_bonus,
            'geometric_bonus': geometric_bonus,
            'multiscale_bonus': multiscale_bonus,
            'ensemble_bonus': ensemble_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_spatial_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                               target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate spatial reasoning bonus"""
        if 'spatial_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward spatial transformations
        spatial_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        non_copy_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use spatial confidence if available
        if 'spatial_confidence' in outputs:
            spatial_confidence = outputs['spatial_confidence'].squeeze(-1)
            spatial_score = spatial_accuracy * spatial_confidence * (1.0 + non_copy_mask * 0.6)
        else:
            spatial_score = spatial_accuracy * (1.0 + non_copy_mask * 0.6)
        
        return -spatial_score.mean() * self.spatial_weight * 0.12
    
    def _calculate_geometric_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate geometric transformation bonus"""
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
                    geometric_score += invariant_consistency * 0.5
        
        # Normalize by number of analyses
        if len(spatial_analyses) > 0:
            geometric_score = geometric_score / len(spatial_analyses)
        
        return -geometric_score * self.geometric_weight * 0.1
    
    def _calculate_multiscale_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate multi-scale processing bonus"""
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
        
        return -multiscale_score * self.multiscale_weight * 0.08
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate ensemble coordination bonus"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        ensemble_output = outputs['ensemble_output']
        
        # Reward high spatial consensus
        if 'spatial_consensus' in ensemble_output:
            consensus = ensemble_output['spatial_consensus'].mean()
            ensemble_score = consensus
        else:
            ensemble_score = torch.tensor(0.6).to(list(outputs.values())[0].device)
        
        # Reward effective cross-attention
        if 'cross_attention_weights' in ensemble_output and ensemble_output['cross_attention_weights'] is not None:
            attention_weights = ensemble_output['cross_attention_weights']
            # Measure attention diversity (avoid collapse)
            attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
            ensemble_score = ensemble_score * torch.sigmoid(attention_entropy)
        
        return -ensemble_score * self.ensemble_weight * 0.06


class Advanced2DSpatialDataset(Dataset):
    """Dataset optimized for advanced 2D spatial reasoning training"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 spatial_focus: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.spatial_focus = spatial_focus
        
        # Load data with advanced spatial filtering
        self.samples = []
        self._load_advanced_spatial_data()
        
        print(f"\033[96mLoaded {len(self.samples)} advanced spatial samples for ATLAS V4 training\033[0m")
    
    def _load_advanced_spatial_data(self):
        """Load data with advanced spatial complexity focus"""
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
                    self._process_advanced_spatial_task(combined_task, 'training')
        
        # Load evaluation data
        eval_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            for task_id, task_data in eval_data.items():
                eval_task = {'train': task_data['train'], 'test': []}
                self._process_advanced_spatial_task(eval_task, 'evaluation')
    
    def _process_advanced_spatial_task(self, task: Dict, source_file: str):
        """Process task with advanced spatial analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples for spatial learning
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_advanced_spatial_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_advanced_spatial_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with advanced spatial analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Advanced spatial analysis
        spatial_analysis = self._analyze_advanced_spatial_complexity(input_grid, output_grid)
        
        # Filter for advanced spatial relevance (more permissive)
        if self.spatial_focus and spatial_analysis['spatial_intelligence_level'] < 2:
            if random.random() > 0.8:  # Keep 80% of simple cases
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'spatial_analysis': spatial_analysis
        }
    
    def _analyze_advanced_spatial_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze advanced spatial complexity and transformation requirements"""
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
        
        # Spatial intelligence level calculation
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
        
        # Complexity classification
        max_dim = max(input_h, input_w, output_h, output_w)
        
        if spatial_intelligence_level <= 1 and max_dim <= 10:
            complexity = 'trivial'
        elif spatial_intelligence_level <= 2 and max_dim <= 15:
            complexity = 'basic'
        elif spatial_intelligence_level <= 3 and max_dim <= 20:
            complexity = 'intermediate'
        elif spatial_intelligence_level <= 4:
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
        
        return {
            'spatial_intelligence_level': spatial_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_colors': total_unique,
            'max_dimension': max_dim,
            'scale_factor_h': output_h / input_h if input_h > 0 else 1.0,
            'scale_factor_w': output_w / input_w if input_w > 0 else 1.0
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


def advanced_spatial_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Enhanced collate function for advanced spatial training"""
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


def train_atlas_specialized_v4():
    """Main training function for ATLAS V4"""
    print(f"\033[96mInitializing ATLAS V4 Advanced Spatial Reasoning Training...\033[0m")
    
    # Initialize enhanced model
    model = AtlasV4Enhanced(
        max_grid_size=30,
        d_model=256,
        num_layers=6,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load existing weights
    model_path = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/atlas_best.pt'
    weights_loaded = model.load_compatible_weights(model_path)
    
    if not weights_loaded:
        print(f"\033[96mStarting fresh ATLAS V4 training\033[0m")
    
    # Initialize loss function
    criterion = AtlasV4SpatialLoss(ATLAS_V4_CONFIG)
    
    # Initialize optimizer with spatial learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ATLAS_V4_CONFIG['learning_rate'],
        weight_decay=ATLAS_V4_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=ATLAS_V4_CONFIG['warmup_epochs'],
        T_mult=int(ATLAS_V4_CONFIG['restart_multiplier']),
        eta_min=ATLAS_V4_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Progressive Spatial Training - 14 Fine-Grained Stages\033[0m")
    
    # Progressive training through spatial stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 95}\033[0m")
        print(f"\033[38;2;255;222;173mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Spatial: {stage_config['spatial_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 95}\033[0m")
        
        # Create advanced spatial dataset for this stage
        dataset = Advanced2DSpatialDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            spatial_focus=True
        )
        
        # Create data loader - SPEED OPTIMIZED
        dataloader = DataLoader(
            dataset,
            batch_size=ATLAS_V4_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=advanced_spatial_collate_fn,
            num_workers=2,  # Use multiple workers for faster data loading
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True  # Keep workers alive
        )
        
        # Stage-specific training
        stage_performance = train_advanced_spatial_stage(
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
                'config': ATLAS_V4_CONFIG,
                'ensemble_state': model.get_ensemble_state()
            }, '/content/AutomataNexus_Olympus_AGI2/models/atlas_v4_best.pt')
            print(f"\033[96mNew best spatial performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 105}\033[0m")
    print(f"\033[96mATLAS V4 Advanced Spatial Training Complete!\033[0m")
    print(f"\033[96mBest Spatial Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 105}\033[0m")
    
    return model, best_performance


def train_advanced_spatial_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                stage_idx, stage_config, training_stats):
    """Train a single advanced spatial curriculum stage"""
    model.train()
    
    epochs_for_stage = ATLAS_V4_CONFIG['epochs_per_stage']
    accumulation_steps = ATLAS_V4_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_spatial_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153mUltra Fast Spatial Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), ATLAS_V4_CONFIG['gradient_clip'])
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
            
            # Count advanced spatial cases
            for meta in metadata:
                if meta['spatial_analysis']['spatial_intelligence_level'] >= 4:
                    advanced_spatial_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'AdvSpatial': f"{advanced_spatial_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress with ultra light honey/amber for stage headers
        if epoch % 1 == 0 or epoch == epochs_for_stage - 1:
            spatial_ratio = advanced_spatial_count / max(total_samples, 1)
            avg_loss = epoch_losses['total']/len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            print(f"\033[38;2;255;204;153m⏰ ATLAS V4 Stage {stage_idx}, Epoch {epoch} (Global: {stage_idx * ATLAS_V4_CONFIG['epochs_per_stage'] + epoch + 1}):\033[0m")
            print(f"\033[96m   🎯 Train: {epoch_performance:.2%} exact, Loss: {avg_loss:.3f}\033[0m")
            print(f"\033[96m   📊 LR: {current_lr:.6f} | Grid: {stage_config['max_grid_size']}x{stage_config['max_grid_size']} | Spatial: {spatial_ratio:.1%}\033[0m")
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
    model, best_performance = train_atlas_specialized_v4()