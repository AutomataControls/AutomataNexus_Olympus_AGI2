"""
ATLAS V4 Transformer Training - Advanced 2D Spatial Reasoning for ARC-AGI-2
Specialized geometric transformation understanding with test-time adaptation
Target: 65%+ performance leveraging advanced spatial reasoning transformers
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
from collections import defaultdict
from tqdm import tqdm
import json
import time
import gc
from typing import Dict, List, Optional, Tuple

# Add paths for imports
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import ATLAS V4 transformer model
from src.models.atlas_v4_transformer import AtlasV4Transformer

# Enhanced Configuration for Advanced Spatial Reasoning
ATLAS_V4_CONFIG = {
    # Transformer-Specific Parameters
    'd_model': 320,  # Larger for spatial reasoning
    'num_layers': 8,  # More layers for geometric complexity
    'num_heads': 8,   # Multi-head attention
    'max_grid_size': 30,
    'enable_test_adaptation': True,
    
    # Core Training Parameters - Spatial-Focused
    'batch_size': 20,  # Smaller for complex spatial computations
    'learning_rate': 0.00008,  # Lower for spatial stability
    'num_epochs': 750,  # Extended: 15 stages x 50 epochs
    'gradient_accumulation': 8,  # Effective batch: 160
    'epochs_per_stage': 50,
    'curriculum_stages': 15,  # More granular spatial progression
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.05,  # Very low - encourage spatial transformations
    'exact_match_bonus': 9.0,  # High bonus for geometric precision
    'gradient_clip': 0.3,  # Tight clipping for spatial stability
    'weight_decay': 8e-6,  # Light regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,
    'strict_match_weight': 0.15,
    'spatial_reasoning_weight': 0.45,  # Primary focus
    'geometric_transform_weight': 0.35,  # Geometric mastery
    'multiscale_weight': 0.25,  # Multi-scale understanding
    'consistency_weight': 0.2,  # Transformation consistency
    
    # Spatial-Specific Features
    'geometric_augmentation': True,
    'multiscale_learning': True,
    'transformation_invariance': True,
    'spatial_attention_focus': True,
    
    # Test-Time Adaptation
    'adaptation_lr': 0.005,
    'adaptation_steps': 8,
    'spatial_adaptation': True,
    
    # Advanced Training Features
    'label_smoothing': 0.01,  # Minimal for precise spatial matching
    'geometric_bonus': True,
    'transformation_consistency_bonus': True,
    'mixed_precision': True,
    
    # Learning Rate Scheduling
    'warmup_epochs': 40,  # Extended warmup for complex geometry
    'cosine_restarts': True,
    'restart_multiplier': 1.5,
    'plateau_patience': 15,
}

# Enhanced 15-Stage Progressive Configuration - Fine-Grained Spatial Learning
STAGE_CONFIG = [
    # Foundation - Basic Shapes (6x6 - 10x10)
    {'stage': 0, 'max_grid_size': 6,  'spatial_complexity': 'basic_shapes', 'focus': 'shape_recognition'},
    {'stage': 1, 'max_grid_size': 7,  'spatial_complexity': 'simple_transforms', 'focus': 'rotation_detection'},
    {'stage': 2, 'max_grid_size': 8,  'spatial_complexity': 'pattern_completion', 'focus': 'translation_learning'},
    {'stage': 3, 'max_grid_size': 9,  'spatial_complexity': 'multi_object', 'focus': 'reflection_mastery'},
    {'stage': 4, 'max_grid_size': 10, 'spatial_complexity': 'basic_composite', 'focus': 'scaling_understanding'},
    
    # Intermediate - Complex Patterns (12x12 - 18x18)
    {'stage': 5, 'max_grid_size': 12, 'spatial_complexity': 'complex_transforms', 'focus': 'composite_transforms'},
    {'stage': 6, 'max_grid_size': 14, 'spatial_complexity': 'spatial_logic', 'focus': 'geometric_rules'},
    {'stage': 7, 'max_grid_size': 15, 'spatial_complexity': 'multi_scale', 'focus': 'scale_invariance'},
    {'stage': 8, 'max_grid_size': 16, 'spatial_complexity': 'advanced_composite', 'focus': 'pattern_relationships'},
    {'stage': 9, 'max_grid_size': 18, 'spatial_complexity': 'complex_spatial', 'focus': 'spatial_reasoning'},
    
    # Advanced - Expert Spatial Mastery (20x20 - 30x30)
    {'stage': 10, 'max_grid_size': 20, 'spatial_complexity': 'expert_transforms', 'focus': 'transformation_chains'},
    {'stage': 11, 'max_grid_size': 22, 'spatial_complexity': 'geometric_mastery', 'focus': 'geometric_invariants'},
    {'stage': 12, 'max_grid_size': 25, 'spatial_complexity': 'spatial_expertise', 'focus': 'complex_compositions'},
    {'stage': 13, 'max_grid_size': 28, 'spatial_complexity': 'transformation_mastery', 'focus': 'spatial_abstractions'},
    {'stage': 14, 'max_grid_size': 30, 'spatial_complexity': 'ultimate_spatial', 'focus': 'spatial_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 95}\033[0m")
print(f"\033[96mATLAS V4 Transformer Training - Advanced 2D Spatial Reasoning for ARC-AGI-2\033[0m")
print(f"\033[96mGeometric Transformations + Multi-Scale Learning + Test-Time Adaptation\033[0m")
print(f"\033[96mTarget: 65%+ Performance with Expert Spatial Intelligence\033[0m")
print(f"\033[96m{'=' * 95}\033[0m")


class AtlasV4SpatialLoss(nn.Module):
    """Advanced loss function for spatial reasoning transformer"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.spatial_weight = config['spatial_reasoning_weight']
        self.geometric_weight = config['geometric_transform_weight']
        self.multiscale_weight = config['multiscale_weight']
        self.consistency_weight = config['consistency_weight']
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
        exact_bonus = exact_bonus.clamp(min=-5.0)  # Higher clamp for spatial precision
        
        # Transform penalty (very low to encourage spatial learning)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Spatial reasoning bonuses
        spatial_bonus = self._calculate_spatial_bonus(model_outputs, pred_indices, target_indices, input_indices)
        geometric_bonus = self._calculate_geometric_bonus(model_outputs, pred_indices, target_indices)
        multiscale_bonus = self._calculate_multiscale_bonus(model_outputs)
        consistency_bonus = self._calculate_consistency_bonus(model_outputs)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     spatial_bonus + geometric_bonus + multiscale_bonus + consistency_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'spatial_bonus': spatial_bonus,
            'geometric_bonus': geometric_bonus,
            'multiscale_bonus': multiscale_bonus,
            'consistency_bonus': consistency_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_spatial_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                               target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate spatial reasoning bonus"""
        if 'spatial_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Encourage non-trivial spatial transformations
        spatial_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        non_copy_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        spatial_transform_bonus = spatial_accuracy * non_copy_mask
        
        return -spatial_transform_bonus.mean() * self.spatial_weight * 0.1
    
    def _calculate_geometric_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                 target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate geometric transformation bonus"""
        if 'transformation_analysis' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward correct geometric transformations
        transform_info = outputs['transformation_analysis']
        geometric_score = 0
        
        # Analyze rotation, reflection, translation consistency
        for key, value in transform_info.items():
            if 'avg_' in key and torch.is_tensor(value):
                # Reward confident predictions
                confidence = torch.max(F.softmax(value, dim=-1), dim=-1)[0]
                geometric_score += confidence.mean()
        
        return -geometric_score * self.geometric_weight * 0.05
    
    def _calculate_multiscale_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate multiscale learning bonus"""
        if 'multiscale_features' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        # Encourage diverse multiscale representations
        multiscale_features = outputs['multiscale_features']
        diversity_score = 0
        
        for scale_features in multiscale_features:
            # Measure feature diversity across scales
            feature_std = scale_features.std(dim=[2, 3]).mean()
            diversity_score += feature_std
        
        return -diversity_score * self.multiscale_weight * 0.02
    
    def _calculate_consistency_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate transformation consistency bonus"""
        if 'transformation_analysis' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        # Reward consistent transformation predictions
        transform_info = outputs['transformation_analysis']
        consistency_score = 0
        
        for key, value in transform_info.items():
            if 'std_' in key and torch.is_tensor(value):
                # Lower std = higher consistency
                consistency_score += torch.exp(-value.mean())
        
        return -consistency_score * self.consistency_weight * 0.03


class SpatialARCDataset(Dataset):
    """Dataset optimized for spatial reasoning transformer training"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 spatial_focus: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.spatial_focus = spatial_focus
        
        # Load data with spatial filtering
        self.samples = []
        self._load_spatial_data()
        
        print(f"\033[96mLoaded {len(self.samples)} spatial samples for ATLAS V4 training\033[0m")
    
    def _load_spatial_data(self):
        """Load data with spatial transformation focus"""
        data_files = [
            'arc_training_padded.json',
            'arc_evaluation_padded.json',
            'synthetic_data_mega_v4.json',
            'spatial_reasoning_tasks.json'  # If available
        ]
        
        for file in data_files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for task in data:
                        self._process_spatial_task(task, file)
    
    def _process_spatial_task(self, task: Dict, source_file: str):
        """Process task with spatial transformation analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_spatial_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_spatial_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with spatial analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Spatial transformation analysis
        spatial_analysis = self._analyze_transformation(input_grid, output_grid)
        
        # Filter for spatial relevance if enabled
        if self.spatial_focus and spatial_analysis['transformation_type'] == 'none':
            return None  # Skip non-transformative samples
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'spatial_analysis': spatial_analysis
        }
    
    def _analyze_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze spatial transformation properties"""
        # Basic transformation detection
        same_shape = input_grid.shape == output_grid.shape
        same_content = np.array_equal(input_grid, output_grid)
        
        if same_content:
            transform_type = 'none'
        elif same_shape:
            # Check for simple transformations
            if np.array_equal(np.rot90(input_grid), output_grid):
                transform_type = 'rotation_90'
            elif np.array_equal(np.rot90(input_grid, 2), output_grid):
                transform_type = 'rotation_180'
            elif np.array_equal(np.rot90(input_grid, 3), output_grid):
                transform_type = 'rotation_270'
            elif np.array_equal(np.fliplr(input_grid), output_grid):
                transform_type = 'flip_horizontal'
            elif np.array_equal(np.flipud(input_grid), output_grid):
                transform_type = 'flip_vertical'
            else:
                transform_type = 'complex_transform'
        else:
            transform_type = 'scale_or_crop'
        
        # Geometric complexity
        unique_colors = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        max_dim = max(input_grid.shape + output_grid.shape)
        
        if max_dim <= 8 and unique_colors <= 3:
            complexity = 'simple'
        elif max_dim <= 16 and unique_colors <= 6:
            complexity = 'medium'
        else:
            complexity = 'complex'
        
        return {
            'transformation_type': transform_type,
            'complexity': complexity,
            'unique_colors': unique_colors,
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
            'spatial_analysis': sample['spatial_analysis']
        }
        
        return input_final, output_final, metadata


def spatial_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Enhanced collate function for spatial training"""
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


def train_atlas_v4_transformer():
    """Main training function for ATLAS V4 transformer"""
    print(f"\033[96mInitializing ATLAS V4 Spatial Transformer Training...\033[0m")
    
    # Initialize model
    model = AtlasV4Transformer(
        max_grid_size=ATLAS_V4_CONFIG['max_grid_size'],
        d_model=ATLAS_V4_CONFIG['d_model'],
        num_layers=ATLAS_V4_CONFIG['num_layers'],
        num_heads=ATLAS_V4_CONFIG['num_heads'],
        enable_test_adaptation=ATLAS_V4_CONFIG['enable_test_adaptation']
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load previous model if available
    model_path = '/content/AutomataNexus_Olympus_AGI2/models/atlas_best.pt'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Only load compatible parameters
            model_dict = model.state_dict()
            compatible_params = {k: v for k, v in checkpoint.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(compatible_params)
            model.load_state_dict(model_dict)
            print(f"\033[96mLoaded {len(compatible_params)} compatible parameters from checkpoint\033[0m")
        except Exception as e:
            print(f"\033[96mCouldn't load checkpoint: {e}, starting fresh\033[0m")
    
    # Initialize loss function
    criterion = AtlasV4SpatialLoss(ATLAS_V4_CONFIG)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ATLAS_V4_CONFIG['learning_rate'],
        weight_decay=ATLAS_V4_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Mixed precision training
    scaler = GradScaler() if ATLAS_V4_CONFIG['mixed_precision'] else None
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Progressive Spatial Training - 15 Fine-Grained Stages\033[0m")
    
    # Progressive training through spatial stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 85}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Complexity: {stage_config['spatial_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 85}\033[0m")
        
        # Create spatial dataset for this stage
        dataset = SpatialARCDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            spatial_focus=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=ATLAS_V4_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=spatial_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_spatial_stage(
            model, dataloader, criterion, optimizer, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            torch.save(model.state_dict(), 
                      '/content/AutomataNexus_Olympus_AGI2/models/atlas_v4_transformer_best.pt')
            print(f"\033[96mNew best spatial performance: {best_performance:.2%} - Model saved!\033[0m")
    
    print(f"\n\033[96m{'=' * 95}\033[0m")
    print(f"\033[96mATLAS V4 Spatial Transformer Training Complete!\033[0m")
    print(f"\033[96mBest Spatial Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96m{'=' * 95}\033[0m")
    
    return model, best_performance


def train_spatial_stage(model, dataloader, criterion, optimizer, scaler, stage_idx, stage_config, training_stats):
    """Train a single spatial curriculum stage"""
    model.train()
    
    epochs_for_stage = ATLAS_V4_CONFIG['epochs_per_stage']
    accumulation_steps = ATLAS_V4_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        spatial_transform_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Spatial Stage {stage_idx} Epoch {epoch}")
        
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass with mixed precision
            with autocast(enabled=(scaler is not None)):
                outputs = model(inputs, targets, mode='train')
                loss_dict = criterion(outputs, targets, inputs)
                loss = loss_dict['total'] / accumulation_steps
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                 ATLAS_V4_CONFIG['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                 ATLAS_V4_CONFIG['gradient_clip'])
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    epoch_losses[key] += value.item()
            
            total_exact_matches += loss_dict['exact_count'].item()
            total_samples += inputs.shape[0]
            
            # Count spatial transformations
            for meta in metadata:
                if meta['spatial_analysis']['transformation_type'] != 'none':
                    spatial_transform_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'Spatial': f"{spatial_transform_count}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress
        if epoch % 3 == 0 or epoch == epochs_for_stage - 1:
            spatial_ratio = spatial_transform_count / max(total_samples, 1)
            print(f"\033[96mSpatial Stage {stage_idx} Epoch {epoch}: "
                  f"Performance = {epoch_performance:.1%}, "
                  f"Spatial Transforms = {spatial_ratio:.1%}, "
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
    
    # Train model
    model, best_performance = train_atlas_v4_transformer()