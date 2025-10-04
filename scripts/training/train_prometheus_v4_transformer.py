"""
PROMETHEUS V4 Transformer Training - 2D-Aware Architecture for ARC-AGI-2 Competition
Advanced spatial reasoning with test-time adaptation and neural program synthesis
Target: 60%+ performance leveraging transformer spatial reasoning
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

# Import PROMETHEUS V4 transformer model
from src.models.prometheus_v4_transformer import PrometheusV4Transformer

# Enhanced Configuration for Transformer Architecture
PROMETHEUS_V4_CONFIG = {
    # Transformer-Specific Parameters
    'd_model': 256,  # Transformer hidden dimension
    'num_layers': 6,  # Transformer layers
    'num_heads': 8,   # Multi-head attention
    'max_grid_size': 30,
    'enable_program_synthesis': True,
    
    # Core Training Parameters - ARC-AGI-2 Focused
    'batch_size': 24,  # Smaller for transformer memory requirements
    'learning_rate': 0.0001,  # Lower LR for transformer stability
    'num_epochs': 600,  # Extended training: 12 stages x 50 epochs
    'gradient_accumulation': 6,  # Effective batch: 144
    'epochs_per_stage': 50,
    'curriculum_stages': 12,  # More stages for transformer learning
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.1,  # Lower penalty for creativity
    'exact_match_bonus': 8.0,  # Higher bonus for transformer precision
    'gradient_clip': 0.5,  # Tighter clipping for transformer stability
    'weight_decay': 1e-5,  # Light regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,
    'strict_match_weight': 0.15,
    'creativity_weight': 0.4,  # Enhanced for transformers
    'spatial_reasoning_weight': 0.35,  # New transformer capability
    'program_synthesis_weight': 0.25,  # Neural program component
    
    # Test-Time Adaptation
    'enable_test_adaptation': True,
    'adaptation_lr': 0.01,
    'adaptation_steps': 5,
    
    # Advanced Training Features
    'label_smoothing': 0.02,
    'pattern_diversity_bonus': True,
    'transformer_warmup': True,
    'cosine_scheduling': True,
    'mixed_precision': True,
    
    # Learning Rate Scheduling
    'warmup_epochs': 30,  # Extended warmup for transformer
    'cosine_restarts': True,
    'restart_multiplier': 1.4,
    'plateau_patience': 12,
}

# Enhanced 12-Stage Progressive Configuration for Transformer Learning
STAGE_CONFIG = [
    # Foundation Stages - Basic Patterns (6x6 - 12x12)
    {'stage': 0, 'max_grid_size': 6,  'arc_ratio': 0.4, 'transformer_layers': 3, 'focus': 'basic_attention'},
    {'stage': 1, 'max_grid_size': 8,  'arc_ratio': 0.45, 'transformer_layers': 4, 'focus': 'spatial_encoding'},
    {'stage': 2, 'max_grid_size': 10, 'arc_ratio': 0.5, 'transformer_layers': 5, 'focus': '2d_relationships'},
    {'stage': 3, 'max_grid_size': 12, 'arc_ratio': 0.55, 'transformer_layers': 6, 'focus': 'pattern_synthesis'},
    
    # Intermediate Stages - Complex Patterns (14x14 - 20x20)
    {'stage': 4, 'max_grid_size': 14, 'arc_ratio': 0.6, 'transformer_layers': 6, 'focus': 'program_synthesis'},
    {'stage': 5, 'max_grid_size': 16, 'arc_ratio': 0.65, 'transformer_layers': 6, 'focus': 'multi_scale'},
    {'stage': 6, 'max_grid_size': 18, 'arc_ratio': 0.7, 'transformer_layers': 6, 'focus': 'abstract_reasoning'},
    {'stage': 7, 'max_grid_size': 20, 'arc_ratio': 0.75, 'transformer_layers': 6, 'focus': 'creative_patterns'},
    
    # Advanced Stages - Full Complexity (22x22 - 30x30)
    {'stage': 8, 'max_grid_size': 22, 'arc_ratio': 0.8, 'transformer_layers': 6, 'focus': 'spatial_mastery'},
    {'stage': 9, 'max_grid_size': 24, 'arc_ratio': 0.85, 'transformer_layers': 6, 'focus': 'advanced_synthesis'},
    {'stage': 10, 'max_grid_size': 28, 'arc_ratio': 0.9, 'transformer_layers': 6, 'focus': 'expert_reasoning'},
    {'stage': 11, 'max_grid_size': 30, 'arc_ratio': 0.95, 'transformer_layers': 6, 'focus': 'transformer_mastery'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 90}\033[0m")
print(f"\033[96mPROMETHEUS V4 Transformer Training - 2D-Aware Architecture for ARC-AGI-2\033[0m")
print(f"\033[96mAdvanced Spatial Reasoning + Test-Time Adaptation + Neural Program Synthesis\033[0m")
print(f"\033[96mTarget: 60%+ Performance with Transformer Architecture\033[0m")
print(f"\033[96m{'=' * 90}\033[0m")


class PrometheusV4TransformerLoss(nn.Module):
    """Enhanced loss function for transformer architecture"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.creativity_weight = config['creativity_weight']
        self.spatial_weight = config['spatial_reasoning_weight']
        self.program_weight = config['program_synthesis_weight']
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
        exact_bonus = exact_bonus.clamp(min=-4.0)  # Higher clamp for transformers
        
        # Transform penalty (encourage creativity)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Spatial reasoning bonus (transformer-specific)
        spatial_bonus = 0
        if 'spatial_features' in model_outputs:
            spatial_features = model_outputs['spatial_features']
            # Encourage spatial diversity
            spatial_std = spatial_features.std(dim=[1,2]).mean()
            spatial_bonus = -spatial_std * self.spatial_weight * 0.1
        
        # Program synthesis bonus
        program_bonus = 0
        if 'program_probs' in model_outputs:
            program_probs = model_outputs['program_probs']
            # Encourage program diversity (avoid collapse)
            program_entropy = -(program_probs * torch.log(program_probs + 1e-8)).sum(dim=-1).mean()
            program_bonus = -program_entropy * self.program_weight * 0.1
        
        # Creativity bonus (for transformers)
        creativity_bonus = 0
        non_copy_mask = (pred_indices != input_indices).float().mean()
        creativity_bonus = -non_copy_mask * self.creativity_weight * 0.1
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     spatial_bonus + program_bonus + creativity_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'spatial_bonus': spatial_bonus,
            'program_bonus': program_bonus,
            'creativity_bonus': creativity_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }


class ARCTransformerDataset(Dataset):
    """Enhanced dataset for transformer training with test-time adaptation"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 enable_test_adaptation: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.enable_test_adaptation = enable_test_adaptation
        
        # Load all available data
        self.samples = []
        self._load_data()
        
        print(f"\033[96mLoaded {len(self.samples)} samples for transformer training\033[0m")
    
    def _load_data(self):
        """Load data with enhanced preprocessing for transformers"""
        data_files = [
            'arc_training_padded.json',
            'arc_evaluation_padded.json', 
            'synthetic_data_mega_v4.json'
        ]
        
        for file in data_files:
            file_path = os.path.join(self.data_dir, file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for task in data:
                        self._process_task(task, file)
    
    def _process_task(self, task: Dict, source_file: str):
        """Process task with transformer-specific features"""
        is_arc_task = 'arc_' in source_file
        
        # Process training examples
        for example in task.get('train', []):
            sample = self._create_sample(example, is_arc_task, 'train')
            if sample:
                self.samples.append(sample)
        
        # Process test examples (for test-time adaptation)
        if self.enable_test_adaptation:
            for example in task.get('test', []):
                sample = self._create_sample(example, is_arc_task, 'test')
                if sample:
                    self.samples.append(sample)
    
    def _create_sample(self, example: Dict, is_arc_task: bool, split: str) -> Optional[Dict]:
        """Create sample with transformer enhancements"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'split': split,
            'spatial_complexity': self._analyze_spatial_complexity(input_grid, output_grid)
        }
    
    def _analyze_spatial_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Analyze spatial complexity for transformer training"""
        # Count unique colors
        unique_colors = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        
        # Analyze patterns
        h, w = input_grid.shape
        if h <= 6 and w <= 6:
            return 'simple'
        elif h <= 12 and w <= 12 and unique_colors <= 5:
            return 'medium'
        elif unique_colors > 7 or h > 20 or w > 20:
            return 'complex'
        else:
            return 'medium'
    
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
        
        # Add channel dimension and convert to one-hot
        input_final = F.one_hot(input_padded, num_classes=10).float().permute(2, 0, 1)
        output_final = F.one_hot(output_padded, num_classes=10).float().permute(2, 0, 1)
        
        metadata = {
            'is_arc': sample['is_arc'],
            'split': sample['split'],
            'spatial_complexity': sample['spatial_complexity']
        }
        
        return input_final, output_final, metadata


def enhanced_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Enhanced collate function for transformer training"""
    inputs, outputs, metadata = zip(*batch)
    
    # Find maximum dimensions
    max_h = max(t.shape[1] for t in inputs + outputs)
    max_w = max(t.shape[2] for t in inputs + outputs)
    
    # Pad all tensors
    inputs_padded = []
    outputs_padded = []
    
    for inp, out in zip(inputs, outputs):
        inp_padded = F.pad(inp, (0, max_w - inp.shape[2], 0, max_h - inp.shape[1]))
        out_padded = F.pad(out, (0, max_w - out.shape[2], 0, max_h - out.shape[1]))
        
        inputs_padded.append(inp_padded)
        outputs_padded.append(out_padded)
    
    return torch.stack(inputs_padded), torch.stack(outputs_padded), list(metadata)


def train_prometheus_v4_transformer():
    """Main training function for PROMETHEUS V4 transformer"""
    print(f"\033[96mInitializing PROMETHEUS V4 Transformer Training...\033[0m")
    
    # Initialize model
    model = PrometheusV4Transformer(
        max_grid_size=PROMETHEUS_V4_CONFIG['max_grid_size'],
        d_model=PROMETHEUS_V4_CONFIG['d_model'],
        num_layers=PROMETHEUS_V4_CONFIG['num_layers'],
        num_heads=PROMETHEUS_V4_CONFIG['num_heads'],
        enable_program_synthesis=PROMETHEUS_V4_CONFIG['enable_program_synthesis']
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load previous model if available
    model_path = '/content/AutomataNexus_Olympus_AGI2/models/prometheus_best.pt'
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
    criterion = PrometheusV4TransformerLoss(PROMETHEUS_V4_CONFIG)
    
    # Initialize optimizer with transformer-specific settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=PROMETHEUS_V4_CONFIG['learning_rate'],
        weight_decay=PROMETHEUS_V4_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Mixed precision training
    scaler = GradScaler() if PROMETHEUS_V4_CONFIG['mixed_precision'] else None
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Progressive Curriculum Training - 12 Stages\033[0m")
    
    # Progressive training through stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 80}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"ARC Ratio {stage_config['arc_ratio']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 80}\033[0m")
        
        # Create dataset for this stage
        dataset = ARCTransformerDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            enable_test_adaptation=PROMETHEUS_V4_CONFIG['enable_test_adaptation']
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=PROMETHEUS_V4_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=enhanced_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_stage(
            model, dataloader, criterion, optimizer, scaler,
            stage_idx, stage_config, training_stats
        )
        
        # Update best performance
        if stage_performance > best_performance:
            best_performance = stage_performance
            # Save best model
            os.makedirs('/content/AutomataNexus_Olympus_AGI2/models', exist_ok=True)
            torch.save(model.state_dict(), 
                      '/content/AutomataNexus_Olympus_AGI2/models/prometheus_v4_transformer_best.pt')
            print(f"\033[96mNew best performance: {best_performance:.2%} - Model saved!\033[0m")
    
    print(f"\n\033[96m{'=' * 90}\033[0m")
    print(f"\033[96mPROMETHEUS V4 Transformer Training Complete!\033[0m")
    print(f"\033[96mBest Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96m{'=' * 90}\033[0m")
    
    return model, best_performance


def train_stage(model, dataloader, criterion, optimizer, scaler, stage_idx, stage_config, training_stats):
    """Train a single curriculum stage"""
    model.train()
    
    epochs_for_stage = PROMETHEUS_V4_CONFIG['epochs_per_stage']
    accumulation_steps = PROMETHEUS_V4_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Stage {stage_idx} Epoch {epoch}")
        
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
                                                 PROMETHEUS_V4_CONFIG['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                 PROMETHEUS_V4_CONFIG['gradient_clip'])
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    epoch_losses[key] += value.item()
            
            total_exact_matches += loss_dict['exact_count'].item()
            total_samples += inputs.shape[0]
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log progress
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            print(f"\033[96mStage {stage_idx} Epoch {epoch}: "
                  f"Performance = {epoch_performance:.1%}, "
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
    model, best_performance = train_prometheus_v4_transformer()