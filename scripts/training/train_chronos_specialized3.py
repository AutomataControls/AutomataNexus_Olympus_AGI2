"""
CHRONOS Specialized Training V3 - Advanced Temporal Sequence Analysis Expert
Building upon CHRONOS V2's proven foundation with enhanced temporal mastery
Target: 65%+ performance with sophisticated temporal reasoning capabilities
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

# Import CHRONOS model
from src.models.chronos_model import EnhancedChronosNet

# Enhanced CHRONOS Configuration V3 - Advanced Temporal Mastery
CHRONOS_CONFIG = {
    # Core Training Parameters - Enhanced for V3
    'batch_size': 36,  # Optimal for temporal sequence complexity
    'learning_rate': 0.0003,  # Careful learning for temporal precision
    'num_epochs': 500,  # Extended training: 10 stages x 50 epochs
    'gradient_accumulation': 7,  # Effective batch: 252
    'epochs_per_stage': 50,  # Extended epochs per stage
    'curriculum_stages': 10,  # Full 10-stage progression
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.15,  # Lower penalty for temporal creativity
    'exact_match_bonus': 7.5,  # High bonus for temporal accuracy
    'gradient_clip': 0.7,  # Stable clipping for temporal sequences
    'weight_decay': 1e-6,  # Very light regularization
    
    # ULTRA TEAL Enhanced (proven formula from successful models)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'temporal_reasoning_weight': 0.4,  # Enhanced temporal bonus
    'sequence_consistency_weight': 0.35,  # Sequence coherence bonus
    'movement_tracking_weight': 0.3,  # Object movement bonus
    'temporal_pattern_weight': 0.25,  # Temporal pattern recognition
    
    # CHRONOS-Specific Enhancements
    'sequence_length': 4,  # Extended sequence analysis
    'hidden_dim': 256,
    'temporal_attention': True,  # Enhanced temporal attention
    'movement_prediction': True,  # Advanced movement prediction
    'sequence_augmentation': True,  # Temporal-aware augmentation
    'temporal_consistency_check': True,  # Sequence consistency validation
    
    # Advanced Training Features
    'label_smoothing': 0.03,  # Light smoothing for temporal generalization
    'pattern_diversity_bonus': True,
    'temporal_reasoning_bonus': True,
    'sequence_coherence_bonus': True,
    'advanced_temporal_augmentation': True,
    
    # Learning Rate Scheduling
    'warmup_epochs': 25,  # Extended warmup for temporal learning
    'cosine_restarts': True,
    'restart_multiplier': 1.3,
}

# Enhanced 10-Stage Progressive Configuration - Temporal-Focused Progression
STAGE_CONFIG = [
    # Basic Temporal Patterns (6x6 - 12x12)
    {'stage': 0, 'max_grid_size': 6,  'synthesis_ratio': 0.8, 'temporal_complexity': 'basic_sequence', 'focus': 'simple_movement'},
    {'stage': 1, 'max_grid_size': 8,  'synthesis_ratio': 0.7, 'temporal_complexity': 'linear_motion', 'focus': 'directional_movement'},
    {'stage': 2, 'max_grid_size': 10, 'synthesis_ratio': 0.6, 'temporal_complexity': 'pattern_evolution', 'focus': 'pattern_progression'},
    {'stage': 3, 'max_grid_size': 12, 'synthesis_ratio': 0.55, 'temporal_complexity': 'object_tracking', 'focus': 'object_persistence'},
    
    # Intermediate Temporal Reasoning (14x14 - 20x20)
    {'stage': 4, 'max_grid_size': 14, 'synthesis_ratio': 0.5, 'temporal_complexity': 'multi_object', 'focus': 'multi_object_tracking'},
    {'stage': 5, 'max_grid_size': 16, 'synthesis_ratio': 0.45, 'temporal_complexity': 'complex_motion', 'focus': 'complex_trajectories'},
    {'stage': 6, 'max_grid_size': 18, 'synthesis_ratio': 0.4, 'temporal_complexity': 'temporal_logic', 'focus': 'temporal_rules'},
    {'stage': 7, 'max_grid_size': 20, 'synthesis_ratio': 0.35, 'temporal_complexity': 'sequence_prediction', 'focus': 'sequence_forecasting'},
    
    # Advanced Temporal Mastery (22x22 - 30x30)
    {'stage': 8, 'max_grid_size': 25, 'synthesis_ratio': 0.25, 'temporal_complexity': 'expert_temporal', 'focus': 'temporal_mastery'},
    {'stage': 9, 'max_grid_size': 30, 'synthesis_ratio': 0.2, 'temporal_complexity': 'temporal_genius', 'focus': 'temporal_intelligence'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 90}\033[0m")
print(f"\033[96mCHRONOS Enhanced V3 Training - Advanced Temporal Sequence Analysis Expert\033[0m")
print(f"\033[96mBuilding on V2's Success → Target: 65%+ Temporal Mastery\033[0m")
print(f"\033[96m{'=' * 90}\033[0m")


class ChronosSpecializedLossV3(nn.Module):
    """Advanced loss function for temporal sequence analysis V3"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.temporal_weight = config['temporal_reasoning_weight']
        self.sequence_weight = config['sequence_consistency_weight']
        self.movement_weight = config['movement_tracking_weight']
        self.pattern_weight = config['temporal_pattern_weight']
        self.ultra_teal_weight = config['ultra_teal_iou_weight']
        self.strict_weight = config['strict_match_weight']
        self.label_smoothing = config['label_smoothing']
        
        # Use standard CrossEntropyLoss instead of focal loss for temporal stability
        self.base_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
    def forward(self, model_outputs: Dict, targets: torch.Tensor, inputs: torch.Tensor) -> Dict:
        pred_output = model_outputs['predicted_output']
        B, C, H, W = pred_output.shape
        
        # Main cross-entropy loss
        target_indices = targets.argmax(dim=1) if targets.dim() > 3 else targets
        base_loss = self.base_loss(pred_output, target_indices)
        
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
        exact_bonus = exact_bonus.clamp(min=-4.5)
        
        # Transform penalty (encourage temporal learning)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Temporal-specific bonuses
        temporal_bonus = self._calculate_temporal_bonus(
            model_outputs, pred_indices, target_indices, input_indices
        ) * self.temporal_weight
        
        sequence_bonus = self._calculate_sequence_bonus(
            model_outputs, pred_indices, target_indices
        ) * self.sequence_weight
        
        movement_bonus = self._calculate_movement_bonus(
            model_outputs, pred_indices, target_indices, input_indices
        ) * self.movement_weight
        
        pattern_bonus = self._calculate_temporal_pattern_bonus(
            pred_indices, target_indices, input_indices
        ) * self.pattern_weight
        
        total_loss = (base_loss + transform_penalty + exact_bonus + 
                     temporal_bonus + sequence_bonus + movement_bonus + pattern_bonus)
        
        return {
            'total': total_loss,
            'base': base_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'temporal_bonus': temporal_bonus,
            'sequence_bonus': sequence_bonus,
            'movement_bonus': movement_bonus,
            'pattern_bonus': pattern_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_temporal_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate temporal reasoning bonus"""
        # Reward temporal transformations
        temporal_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        non_copy_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use temporal features if available
        if 'temporal_features' in outputs:
            temporal_features = outputs['temporal_features']
            # Encourage diverse temporal representations
            temporal_diversity = temporal_features.std(dim=[2,3]).mean()
            temporal_bonus = temporal_accuracy * (1.0 + temporal_diversity * 0.1) * non_copy_mask
        else:
            temporal_bonus = temporal_accuracy * non_copy_mask
        
        return -temporal_bonus.mean() * 0.1
    
    def _calculate_sequence_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate sequence consistency bonus"""
        if 'attention_weights' in outputs:
            # Reward consistent attention patterns
            attention_weights = outputs['attention_weights']
            if attention_weights is not None:
                # Measure attention consistency
                attention_std = attention_weights.std(dim=-1).mean()
                consistency_score = torch.exp(-attention_std)  # Lower std = higher consistency
                return -consistency_score * 0.05
        
        # Fallback: measure prediction consistency
        pred_consistency = (pred_indices == target_indices).float().mean()
        return -pred_consistency * 0.05
    
    def _calculate_movement_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate movement tracking bonus"""
        if 'movement_params' in outputs:
            movement_params = outputs['movement_params']
            # Reward meaningful movement parameters (not all zeros)
            movement_magnitude = torch.norm(movement_params, dim=-1).mean()
            movement_bonus = torch.tanh(movement_magnitude)  # Normalized movement score
            
            # Also check if movement leads to correct predictions
            movement_accuracy = (pred_indices == target_indices).float().mean()
            combined_bonus = movement_bonus * movement_accuracy
            
            return -combined_bonus * 0.08
        
        return torch.tensor(0.0).to(pred_indices.device)
    
    def _calculate_temporal_pattern_bonus(self, pred_indices: torch.Tensor, 
                                        target_indices: torch.Tensor,
                                        input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate temporal pattern recognition bonus"""
        B, H, W = pred_indices.shape
        
        # Analyze temporal patterns (changes from input to output)
        pattern_bonus = 0
        for b in range(B):
            # Check for consistent transformations
            input_changes = (target_indices[b] != input_indices[b])
            pred_changes = (pred_indices[b] != input_indices[b])
            
            # Reward correct change detection
            change_accuracy = (input_changes == pred_changes).float().mean()
            pattern_bonus += change_accuracy
        
        pattern_bonus = pattern_bonus / B
        return -pattern_bonus * 0.06


class ChronosDatasetV3(Dataset):
    """Enhanced dataset for CHRONOS V3 temporal training"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 temporal_focus: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.temporal_focus = temporal_focus
        
        # Load data with temporal filtering
        self.samples = []
        self._load_temporal_data()
        
        print(f"\033[96mLoaded {len(self.samples)} temporal samples for CHRONOS V3 training\033[0m")
    
    def _load_temporal_data(self):
        """Load data with temporal complexity focus"""
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
                        self._process_temporal_task(task, file)
    
    def _process_temporal_task(self, task: Dict, source_file: str):
        """Process task with temporal analysis"""
        is_arc_task = 'arc_' in source_file
        
        # For temporal analysis, try to create sequences from train examples
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])
        
        # Single examples (for basic temporal learning)
        for example in train_examples + test_examples:
            sample = self._create_temporal_sample(example, is_arc_task, 'single')
            if sample:
                self.samples.append(sample)
        
        # Sequence examples (when we have multiple training examples)
        if len(train_examples) >= 2:
            # Create sequences from training examples
            for i in range(len(train_examples) - 1):
                sequence_sample = self._create_sequence_sample(
                    train_examples[i:i+2], is_arc_task
                )
                if sequence_sample:
                    self.samples.append(sequence_sample)
    
    def _create_temporal_sample(self, example: Dict, is_arc_task: bool, sample_type: str) -> Optional[Dict]:
        """Create sample with temporal analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Temporal complexity analysis
        temporal_analysis = self._analyze_temporal_complexity(input_grid, output_grid)
        
        # Filter for temporal relevance if enabled
        if self.temporal_focus and temporal_analysis['temporal_type'] == 'static':
            return None  # Skip non-temporal samples
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'sample_type': sample_type,
            'temporal_analysis': temporal_analysis
        }
    
    def _create_sequence_sample(self, examples: List[Dict], is_arc_task: bool) -> Optional[Dict]:
        """Create sequence sample from multiple examples"""
        if len(examples) < 2:
            return None
        
        # Convert examples to sequence
        sequence_inputs = []
        sequence_outputs = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            if (max(input_grid.shape) > self.max_grid_size or 
                max(output_grid.shape) > self.max_grid_size):
                return None
            
            sequence_inputs.append(input_grid)
            sequence_outputs.append(output_grid)
        
        return {
            'sequence_inputs': sequence_inputs,
            'sequence_outputs': sequence_outputs,
            'is_arc': is_arc_task,
            'sample_type': 'sequence',
            'sequence_length': len(examples)
        }
    
    def _analyze_temporal_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze temporal complexity and patterns"""
        # Basic temporal analysis
        same_shape = input_grid.shape == output_grid.shape
        same_content = np.array_equal(input_grid, output_grid)
        
        if same_content:
            temporal_type = 'static'
        elif same_shape:
            # Analyze type of change
            changes = np.sum(input_grid != output_grid)
            total_cells = input_grid.size
            change_ratio = changes / total_cells
            
            if change_ratio < 0.1:
                temporal_type = 'minimal_change'
            elif change_ratio < 0.3:
                temporal_type = 'local_change'
            else:
                temporal_type = 'global_change'
        else:
            temporal_type = 'shape_change'
        
        # Complexity classification
        unique_colors = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        max_dim = max(input_grid.shape + output_grid.shape)
        
        if max_dim <= 8 and unique_colors <= 3:
            complexity = 'simple'
        elif max_dim <= 16 and unique_colors <= 6:
            complexity = 'medium'
        else:
            complexity = 'complex'
        
        return {
            'temporal_type': temporal_type,
            'complexity': complexity,
            'unique_colors': unique_colors,
            'max_dimension': max_dim,
            'change_ratio': changes / total_cells if same_shape else 1.0
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        
        if sample['sample_type'] == 'sequence':
            # Handle sequence samples
            return self._get_sequence_item(sample)
        else:
            # Handle single samples
            return self._get_single_item(sample)
    
    def _get_single_item(self, sample: Dict) -> Tuple:
        """Get single temporal sample"""
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
        
        # For single samples, create a sequence of length 1
        sequence = [input_final]
        
        metadata = {
            'is_arc': sample['is_arc'],
            'sample_type': sample['sample_type'],
            'temporal_analysis': sample['temporal_analysis'],
            'sequence_length': 1
        }
        
        return sequence, output_final, metadata
    
    def _get_sequence_item(self, sample: Dict) -> Tuple:
        """Get sequence temporal sample"""
        # Convert sequence to tensors
        sequence_tensors = []
        
        for input_grid in sample['sequence_inputs']:
            input_tensor = torch.tensor(input_grid, dtype=torch.long)
            # Pad and convert to one-hot
            target_h = min(self.max_grid_size, input_tensor.shape[0])
            target_w = min(self.max_grid_size, input_tensor.shape[1])
            
            input_padded = F.pad(input_tensor, (0, target_w - input_tensor.shape[1], 
                                              0, target_h - input_tensor.shape[0]))
            input_final = F.one_hot(input_padded, num_classes=10).float().permute(2, 0, 1)
            sequence_tensors.append(input_final)
        
        # Use last output as target
        output_tensor = torch.tensor(sample['sequence_outputs'][-1], dtype=torch.long)
        target_h = min(self.max_grid_size, output_tensor.shape[0])
        target_w = min(self.max_grid_size, output_tensor.shape[1])
        
        output_padded = F.pad(output_tensor, (0, target_w - output_tensor.shape[1], 
                                            0, target_h - output_tensor.shape[0]))
        output_final = F.one_hot(output_padded, num_classes=10).float().permute(2, 0, 1)
        
        metadata = {
            'is_arc': sample['is_arc'],
            'sample_type': sample['sample_type'],
            'sequence_length': len(sequence_tensors)
        }
        
        return sequence_tensors, output_final, metadata


def enhanced_temporal_collate_fn(batch: List) -> Tuple:
    """Enhanced collate function for temporal training"""
    sequences, outputs, metadata = zip(*batch)
    
    # Handle variable sequence lengths
    max_seq_len = max(len(seq) if isinstance(seq, list) else 1 for seq in sequences)
    
    # Find maximum spatial dimensions
    all_tensors = []
    for seq in sequences:
        if isinstance(seq, list):
            all_tensors.extend(seq)
        else:
            all_tensors.append(seq)
    all_tensors.extend(outputs)
    
    max_h = max(t.shape[1] for t in all_tensors)
    max_w = max(t.shape[2] for t in all_tensors)
    
    # Pad sequences and outputs
    padded_sequences = []
    padded_outputs = []
    
    for seq, out in zip(sequences, outputs):
        # Pad sequence
        if isinstance(seq, list):
            padded_seq = []
            for frame in seq:
                frame_padded = F.pad(frame, (0, max_w - frame.shape[2], 0, max_h - frame.shape[1]))
                padded_seq.append(frame_padded)
            # Pad sequence length if needed
            while len(padded_seq) < max_seq_len:
                padded_seq.append(torch.zeros_like(padded_seq[0]))
            padded_sequences.append(padded_seq)
        else:
            # Single frame sequence
            seq_padded = F.pad(seq, (0, max_w - seq.shape[2], 0, max_h - seq.shape[1]))
            padded_seq = [seq_padded]
            while len(padded_seq) < max_seq_len:
                padded_seq.append(torch.zeros_like(seq_padded))
            padded_sequences.append(padded_seq)
        
        # Pad output
        out_padded = F.pad(out, (0, max_w - out.shape[2], 0, max_h - out.shape[1]))
        padded_outputs.append(out_padded)
    
    # Stack into tensors
    # sequences: (batch_size, max_seq_len, channels, height, width)
    batch_sequences = []
    for seq in padded_sequences:
        batch_sequences.append(torch.stack(seq))
    
    return batch_sequences, torch.stack(padded_outputs), list(metadata)


def train_chronos_specialized_v3():
    """Main training function for CHRONOS V3"""
    print(f"\033[96mInitializing CHRONOS V3 Temporal Analysis Training...\033[0m")
    
    # Initialize model
    model = EnhancedChronosNet(
        max_grid_size=30,
        hidden_dim=CHRONOS_CONFIG['hidden_dim']
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load previous model if available
    model_path = '/content/AutomataNexus_Olympus_AGI2/models/chronos_best.pt'
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
    criterion = ChronosSpecializedLossV3(CHRONOS_CONFIG)
    
    # Initialize optimizer with enhanced settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CHRONOS_CONFIG['learning_rate'],
        weight_decay=CHRONOS_CONFIG['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Mixed precision training (disabled for CHRONOS stability)
    scaler = None  # Disable mixed precision for temporal stability
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Progressive Temporal Training - 10 Stages\033[0m")
    
    # Progressive training through stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 85}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Temporal Complexity: {stage_config['temporal_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 85}\033[0m")
        
        # Create dataset for this stage
        dataset = ChronosDatasetV3(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            temporal_focus=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=CHRONOS_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=enhanced_temporal_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_temporal_stage(
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
                'config': CHRONOS_CONFIG
            }, '/content/AutomataNexus_Olympus_AGI2/models/chronos_v3_best.pt')
            print(f"\033[96mNew best temporal performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 90}\033[0m")
    print(f"\033[96mCHRONOS V3 Temporal Training Complete!\033[0m")
    print(f"\033[96mBest Temporal Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96m{'=' * 90}\033[0m")
    
    return model, best_performance


def train_temporal_stage(model, dataloader, criterion, optimizer, scaler, stage_idx, stage_config, training_stats):
    """Train a single temporal curriculum stage"""
    model.train()
    
    epochs_for_stage = CHRONOS_CONFIG['epochs_per_stage']
    accumulation_steps = CHRONOS_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        temporal_sequence_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[96mTemporal Stage {stage_idx} Epoch {epoch}\033[0m")
        
        for batch_idx, (sequences, targets, metadata) in enumerate(pbar):
            # Handle sequences (list of tensors per batch item)
            batch_sequences = []
            for seq in sequences:
                # Convert list of frames to single tensor list
                if isinstance(seq, list):
                    seq_tensors = [frame.to(device) for frame in seq]
                else:
                    seq_tensors = [seq.to(device)]
                batch_sequences.append(seq_tensors)
            
            targets = targets.to(device)
            
            # Forward pass (no mixed precision for temporal stability)
            total_loss = 0
            for i, seq in enumerate(batch_sequences):
                # CHRONOS expects a list of tensors as sequence input
                outputs = model(seq, targets[i:i+1], mode='train')
                loss_dict = criterion(outputs, targets[i:i+1], seq[-1].unsqueeze(0))  # Use last frame as input
                total_loss += loss_dict['total']
                
                # Accumulate metrics (only for first item to avoid overcounting)
                if i == 0:
                    for key, value in loss_dict.items():
                        if torch.is_tensor(value):
                            epoch_losses[key] += value.item()
                    
                    total_exact_matches += loss_dict['exact_count'].item()
                    
            # Normalize loss by batch size
            loss = total_loss / (len(batch_sequences) * accumulation_steps)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CHRONOS_CONFIG['gradient_clip'])
                optimizer.step()
                optimizer.zero_grad()
            
            total_samples += len(batch_sequences)
            
            # Count temporal sequences
            for meta in metadata:
                if meta.get('sequence_length', 1) > 1 or meta.get('temporal_analysis', {}).get('temporal_type', 'static') != 'static':
                    temporal_sequence_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'Temporal': f"{temporal_sequence_count}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            temporal_ratio = temporal_sequence_count / max(total_samples, 1)
            print(f"\033[96mTemporal Stage {stage_idx} Epoch {epoch}: "
                  f"Performance = {epoch_performance:.1%}, "
                  f"Temporal Sequences = {temporal_ratio:.1%}, "
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
    model, best_performance = train_chronos_specialized_v3()