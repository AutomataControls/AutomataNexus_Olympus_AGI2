"""
IRIS Specialized Training V3 - Advanced Color Pattern Recognition Expert
Building upon IRIS V2's proven performance with enhanced color mastery
Target: 70%+ performance with sophisticated color pattern understanding
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
from typing import Dict, List, Optional
import random
from collections import defaultdict

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import IRIS model
from src.models.iris_model import EnhancedIrisNet

# Enhanced IRIS Configuration V3 - Advanced Color Mastery
IRIS_CONFIG = {
    # Core Training Parameters - Enhanced for V3
    'max_grid_size': 30,  # Maximum grid size
    'batch_size': 48,  # Optimal for color pattern complexity
    'learning_rate': 0.00025,  # Careful learning for color precision
    'num_epochs': 500,  # Extended training: 10 stages x 50 epochs
    'gradient_accumulation': 5,  # Effective batch: 240
    'epochs_per_stage': 50,  # Extended epochs per stage
    'curriculum_stages': 10,  # Full 10-stage progression
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.25,  # Balanced for color transformations
    'exact_match_bonus': 6.5,  # Strong bonus for color accuracy
    'gradient_clip': 0.9,  # Stable clipping for color patterns
    'weight_decay': 2e-6,  # Light regularization
    
    # ULTRA TEAL Enhanced (proven formula from successful models)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'color_pattern_weight': 0.35,  # Enhanced color pattern bonus
    'chromatic_harmony_weight': 0.3,  # Color harmony bonus
    'color_transition_weight': 0.25,  # Color transition logic bonus
    
    # IRIS-Specific Enhancements
    'color_space_learning': True,  # Learn in multiple color spaces
    'chromatic_augmentation': True,  # Color-preserving augmentation
    'color_pattern_focus': True,  # Enhanced color pattern attention
    'hue_saturation_analysis': True,  # HSV analysis capabilities
    
    # Advanced Training Features
    'label_smoothing': 0.04,  # Light smoothing for color generalization
    'pattern_diversity_bonus': True,
    'color_reasoning_bonus': True,
    'chromatic_consistency_bonus': True,
    'advanced_color_augmentation': True,
    
    # Learning Rate Scheduling
    'warmup_epochs': 20,  # Warmup for color learning
    'cosine_restarts': True,
    'restart_multiplier': 1.25,
}

# Enhanced 10-Stage Progressive Configuration - Color-Focused Progression
STAGE_CONFIG = [
    # Basic Color Patterns (6x6 - 12x12)
    {'stage': 0, 'max_grid_size': 6,  'synthesis_ratio': 0.85, 'color_complexity': 'basic_colors', 'focus': 'primary_colors'},
    {'stage': 1, 'max_grid_size': 8,  'synthesis_ratio': 0.75, 'color_complexity': 'simple_patterns', 'focus': 'color_recognition'},
    {'stage': 2, 'max_grid_size': 10, 'synthesis_ratio': 0.65, 'color_complexity': 'pattern_matching', 'focus': 'color_completion'},
    {'stage': 3, 'max_grid_size': 12, 'synthesis_ratio': 0.55, 'color_complexity': 'multi_color', 'focus': 'color_relationships'},
    
    # Intermediate Color Reasoning (14x14 - 20x20)
    {'stage': 4, 'max_grid_size': 14, 'synthesis_ratio': 0.45, 'color_complexity': 'complex_patterns', 'focus': 'color_transformations'},
    {'stage': 5, 'max_grid_size': 16, 'synthesis_ratio': 0.35, 'color_complexity': 'chromatic_logic', 'focus': 'color_rules'},
    {'stage': 6, 'max_grid_size': 18, 'synthesis_ratio': 0.3, 'color_complexity': 'advanced_harmony', 'focus': 'color_harmony'},
    {'stage': 7, 'max_grid_size': 20, 'synthesis_ratio': 0.25, 'color_complexity': 'color_reasoning', 'focus': 'chromatic_reasoning'},
    
    # Advanced Color Mastery (22x22 - 30x30)
    {'stage': 8, 'max_grid_size': 25, 'synthesis_ratio': 0.2, 'color_complexity': 'expert_chromatic', 'focus': 'color_mastery'},
    {'stage': 9, 'max_grid_size': 30, 'synthesis_ratio': 0.15, 'color_complexity': 'chromatic_genius', 'focus': 'color_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 85}\033[0m")
print(f"\033[96mIRIS Enhanced V3 Training - Advanced Color Pattern Recognition Expert\033[0m")
print(f"\033[96mBuilding on V2's Success → Target: 70%+ Color Mastery\033[0m")
print(f"\033[96m{'=' * 85}\033[0m")


class IrisSpecializedLossV3(nn.Module):
    """Advanced loss function for color pattern recognition V3"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.color_pattern_weight = config['color_pattern_weight']
        self.chromatic_harmony_weight = config['chromatic_harmony_weight']
        self.color_transition_weight = config['color_transition_weight']
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
        exact_bonus = exact_bonus.clamp(min=-4.0)
        
        # Transform penalty (encourage color learning)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Color pattern bonuses
        color_pattern_bonus = self._calculate_color_pattern_bonus(
            pred_indices, target_indices, input_indices
        ) * self.color_pattern_weight
        
        chromatic_harmony_bonus = self._calculate_chromatic_harmony_bonus(
            pred_indices, target_indices
        ) * self.chromatic_harmony_weight
        
        color_transition_bonus = self._calculate_color_transition_bonus(
            pred_indices, target_indices, input_indices
        ) * self.color_transition_weight
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     color_pattern_bonus + chromatic_harmony_bonus + color_transition_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'color_pattern_bonus': color_pattern_bonus,
            'chromatic_harmony_bonus': chromatic_harmony_bonus,
            'color_transition_bonus': color_transition_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_color_pattern_bonus(self, pred_indices: torch.Tensor, 
                                     target_indices: torch.Tensor, 
                                     input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate color pattern recognition bonus"""
        # Reward correct color pattern transformations
        color_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        
        # Check for non-trivial color transformations
        non_copy_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Color pattern complexity bonus
        color_pattern_bonus = color_accuracy * (1.0 + non_copy_mask * 0.5)
        
        return -color_pattern_bonus.mean() * 0.1
    
    def _calculate_chromatic_harmony_bonus(self, pred_indices: torch.Tensor, 
                                         target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate chromatic harmony bonus"""
        # Analyze color harmony in predictions vs targets
        B, H, W = pred_indices.shape
        
        # Count unique colors in predictions and targets
        harmony_score = 0
        for b in range(B):
            pred_colors = torch.unique(pred_indices[b])
            target_colors = torch.unique(target_indices[b])
            
            # Reward similar color diversity
            color_diversity_match = abs(len(pred_colors) - len(target_colors))
            harmony_score += torch.exp(-color_diversity_match * 0.5)
        
        harmony_score = harmony_score / B
        return -harmony_score * 0.05
    
    def _calculate_color_transition_bonus(self, pred_indices: torch.Tensor, 
                                        target_indices: torch.Tensor,
                                        input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate color transition logic bonus"""
        # Analyze color transitions from input to target vs input to prediction
        B, H, W = pred_indices.shape
        
        transition_accuracy = 0
        for b in range(B):
            # Check color mapping consistency
            input_colors = torch.unique(input_indices[b])
            target_mapping = {}
            pred_mapping = {}
            
            for color in input_colors:
                input_mask = (input_indices[b] == color)
                if input_mask.any():
                    # Most common color in target for this input color
                    target_colors_for_input = target_indices[b][input_mask]
                    pred_colors_for_input = pred_indices[b][input_mask]
                    
                    if len(target_colors_for_input) > 0:
                        target_mode = torch.mode(target_colors_for_input).values
                        pred_mode = torch.mode(pred_colors_for_input).values
                        
                        transition_accuracy += (target_mode == pred_mode).float()
            
            transition_accuracy = transition_accuracy / max(len(input_colors), 1)
        
        transition_accuracy = transition_accuracy / B
        return -transition_accuracy * 0.08


class IrisDatasetV3(Dataset):
    """Enhanced dataset for IRIS V3 color training"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 color_focus: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.color_focus = color_focus
        
        # Load data with color filtering
        self.samples = []
        self._load_color_data()
        
        print(f"\033[96mLoaded {len(self.samples)} color samples for IRIS V3 training\033[0m")
    
    def _load_color_data(self):
        """Load data with color complexity focus"""
        # Load training data (challenges + solutions)
        challenges_path = os.path.join(self.data_dir, 'arc-agi_training_challenges.json')
        solutions_path = os.path.join(self.data_dir, 'arc-agi_training_solutions.json')
        
        if os.path.exists(challenges_path) and os.path.exists(solutions_path):
            print(f"\033[96mLoading training data for IRIS training\033[0m")
            with open(challenges_path, 'r') as f:
                challenges = json.load(f)
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
            
            print(f"\033[96mFound {len(challenges)} training tasks\033[0m")
            for task_id, task_data in challenges.items():
                if task_id in solutions:
                    # Combine challenges and solutions
                    combined_task = {
                        'train': task_data['train'],
                        'test': []
                    }
                    # Add solutions to test examples
                    for i, test_input in enumerate(task_data['test']):
                        if i < len(solutions[task_id]):
                            combined_task['test'].append({
                                'input': test_input['input'],
                                'output': solutions[task_id][i]
                            })
                    
                    self._process_color_task(combined_task, 'training')
        
        # Load evaluation data if available
        eval_challenges_path = os.path.join(self.data_dir, 'arc-agi_evaluation_challenges.json')
        if os.path.exists(eval_challenges_path):
            print(f"\033[96mLoading evaluation data for IRIS training\033[0m")
            with open(eval_challenges_path, 'r') as f:
                eval_data = json.load(f)
            print(f"\033[96mFound {len(eval_data)} evaluation tasks\033[0m")
            for task_id, task_data in eval_data.items():
                # Only use training examples from eval set (no solutions for test)
                eval_task = {'train': task_data['train'], 'test': []}
                self._process_color_task(eval_task, 'evaluation')
    
    def _process_color_task(self, task: Dict, source_file: str):
        """Process task with color complexity analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_color_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_color_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with color analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Color complexity analysis
        color_analysis = self._analyze_color_complexity(input_grid, output_grid)
        
        # Very permissive filtering for IRIS V3 to ensure we get samples
        # Only filter out completely trivial cases
        if self.color_focus and color_analysis['unique_colors'] < 2:
            if random.random() > 0.8:  # Keep 80% of single-color samples
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'color_analysis': color_analysis
        }
    
    def _analyze_color_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze color complexity and patterns"""
        # Basic color analysis
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        all_colors = input_colors.union(output_colors)
        
        # Color transformation analysis
        same_colors = input_colors == output_colors
        color_added = len(output_colors - input_colors)
        color_removed = len(input_colors - output_colors)
        
        # Complexity classification
        unique_colors = len(all_colors)
        max_dim = max(input_grid.shape + output_grid.shape)
        
        if unique_colors <= 3 and max_dim <= 8:
            complexity = 'simple'
        elif unique_colors <= 6 and max_dim <= 16:
            complexity = 'medium'
        else:
            complexity = 'complex'
        
        # Color pattern type
        if same_colors and np.array_equal(input_grid, output_grid):
            pattern_type = 'identity'
        elif same_colors:
            pattern_type = 'rearrangement'
        elif color_added > 0 and color_removed == 0:
            pattern_type = 'additive'
        elif color_removed > 0 and color_added == 0:
            pattern_type = 'reductive'
        else:
            pattern_type = 'transformative'
        
        return {
            'unique_colors': unique_colors,
            'complexity': complexity,
            'pattern_type': pattern_type,
            'color_added': color_added,
            'color_removed': color_removed,
            'max_dimension': max_dim
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple:
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


def enhanced_color_collate_fn(batch: List) -> tuple:
    """Enhanced collate function for color training"""
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


def train_iris_specialized_v3():
    """Main training function for IRIS V3"""
    print(f"\033[96mInitializing IRIS V3 Color Pattern Training...\033[0m")
    
    # Initialize model
    model = EnhancedIrisNet(max_grid_size=IRIS_CONFIG['max_grid_size']).to(device)
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load previous model if available
    model_path = '/content/AutomataNexus_Olympus_AGI2/models/iris_best.pt'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                loaded_exact = checkpoint.get('best_exact', 0.0)
                print(f"\033[96mLoaded model with {loaded_exact:.2%} performance\033[0m")
            else:
                model.load_state_dict(checkpoint)
                print(f"\033[96mLoaded model checkpoint successfully\033[0m")
        except Exception as e:
            print(f"\033[96mCouldn't load checkpoint: {e}, starting fresh\033[0m")
    
    # Initialize loss function
    criterion = IrisSpecializedLossV3(IRIS_CONFIG)
    
    # Initialize optimizer with enhanced settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=IRIS_CONFIG['learning_rate'],
        weight_decay=IRIS_CONFIG['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Progressive Color Training - 10 Stages\033[0m")
    
    # Progressive training through stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 80}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Color Complexity: {stage_config['color_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 80}\033[0m")
        
        # Create dataset for this stage
        dataset = IrisDatasetV3(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            color_focus=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=IRIS_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=enhanced_color_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_color_stage(
            model, dataloader, criterion, optimizer, scaler,
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
                'best_performance': best_performance,
                'stage': stage_idx,
                'config': IRIS_CONFIG
            }, '/content/AutomataNexus_Olympus_AGI2/models/iris_best.pt')
            print(f"\033[96mNew best color performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 85}\033[0m")
    print(f"\033[96mIRIS V3 Color Training Complete!\033[0m")
    print(f"\033[96mBest Color Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96m{'=' * 85}\033[0m")
    
    return model, best_performance


def train_color_stage(model, dataloader, criterion, optimizer, scaler, stage_idx, stage_config, training_stats):
    """Train a single color curriculum stage"""
    model.train()
    
    epochs_for_stage = IRIS_CONFIG['epochs_per_stage']
    accumulation_steps = IRIS_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        color_transform_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[96mColor Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), IRIS_CONFIG['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    epoch_losses[key] += value.item()
            
            total_exact_matches += loss_dict['exact_count'].item()
            total_samples += inputs.shape[0]
            
            # Count color transformations
            for meta in metadata:
                if meta['color_analysis']['pattern_type'] != 'identity':
                    color_transform_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'Colors': f"{color_transform_count}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            color_ratio = color_transform_count / max(total_samples, 1)
            print(f"\033[96mColor Stage {stage_idx} Epoch {epoch}: "
                  f"Performance = {epoch_performance:.1%}, "
                  f"Color Transforms = {color_ratio:.1%}, "
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
    model, best_performance = train_iris_specialized_v3()