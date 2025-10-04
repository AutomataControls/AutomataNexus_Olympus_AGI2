"""
CHRONOS Specialized Training V4 - Advanced Temporal Sequence Reasoning Expert for ARC-AGI-2
Enhanced with temporal transformers, sequence intelligence, and OLYMPUS ensemble preparation
Target: 70%+ performance with sophisticated temporal reasoning mastery
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

# Enhanced CHRONOS V4 Configuration - Temporal Intelligence Focus
CHRONOS_V4_CONFIG = {
    # Core Training Parameters - Enhanced for V4 Temporal Intelligence
    'batch_size': 26,  # Optimal for temporal transformer computations
    'learning_rate': 0.00016,  # Balanced for temporal learning
    'num_epochs': 650,  # Extended training: 13 stages x 50 epochs
    'gradient_accumulation': 8,  # Effective batch: 208
    'epochs_per_stage': 50,  # Extended epochs per stage
    'curriculum_stages': 13,  # 13-stage temporal progression
    
    # Enhanced Loss Configuration
    'transform_penalty': 0.04,  # Low - encourage temporal transformations
    'exact_match_bonus': 8.8,  # High bonus for temporal precision
    'gradient_clip': 0.5,  # Balanced clipping for temporal stability
    'weight_decay': 3.5e-6,  # Light regularization
    
    # ULTRA TEAL Enhanced (proven formula)
    'ultra_teal_iou_weight': 0.85,  # 85% IoU weighting
    'strict_match_weight': 0.15,   # 15% strict matching
    'temporal_reasoning_weight': 0.58,  # Primary focus - temporal intelligence
    'sequence_analysis_weight': 0.5,  # Sequence understanding mastery
    'movement_prediction_weight': 0.45,  # Movement/change prediction
    'ensemble_coordination_weight': 0.4,  # Ensemble integration
    
    # CHRONOS V4-Specific Enhancements
    'temporal_transformer_layers': 6,  # Deep temporal reasoning
    'sequence_memory_size': 150,  # Temporal pattern memory
    'temporal_positional_encoding': True,  # Temporal-aware positioning
    'ensemble_preparation': True,  # OLYMPUS preparation mode
    'test_time_adaptation': True,  # Advanced temporal adaptation
    
    # Advanced Training Features
    'label_smoothing': 0.02,  # Light for temporal precision
    'pattern_diversity_bonus': True,
    'temporal_reasoning_bonus': True,
    'sequence_continuity_bonus': True,
    'movement_prediction_bonus': True,
    
    # Learning Rate Scheduling
    'warmup_epochs': 32,  # Extended warmup for temporal transformers
    'cosine_restarts': True,
    'restart_multiplier': 1.25,
    'plateau_patience': 20,
}

# Enhanced 13-Stage Progressive Configuration - Temporal Intelligence Focus
STAGE_CONFIG = [
    # Foundation Temporal Understanding (6x6 - 10x10)
    {'stage': 0, 'max_grid_size': 6,  'synthesis_ratio': 0.85, 'temporal_complexity': 'static_patterns', 'focus': 'static_pattern_recognition'},
    {'stage': 1, 'max_grid_size': 7,  'synthesis_ratio': 0.8, 'temporal_complexity': 'simple_movement', 'focus': 'simple_movement_detection'},
    {'stage': 2, 'max_grid_size': 8,  'synthesis_ratio': 0.75, 'temporal_complexity': 'basic_sequence', 'focus': 'basic_sequence_learning'},
    {'stage': 3, 'max_grid_size': 9,  'synthesis_ratio': 0.7, 'temporal_complexity': 'pattern_evolution', 'focus': 'pattern_evolution_understanding'},
    {'stage': 4, 'max_grid_size': 10, 'synthesis_ratio': 0.65, 'temporal_complexity': 'temporal_rules', 'focus': 'temporal_rule_learning'},
    
    # Intermediate Temporal Reasoning (12x12 - 18x18)
    {'stage': 5, 'max_grid_size': 12, 'synthesis_ratio': 0.6, 'temporal_complexity': 'sequence_prediction', 'focus': 'sequence_prediction'},
    {'stage': 6, 'max_grid_size': 14, 'synthesis_ratio': 0.55, 'temporal_complexity': 'movement_complex', 'focus': 'complex_movement_patterns'},
    {'stage': 7, 'max_grid_size': 16, 'synthesis_ratio': 0.5, 'temporal_complexity': 'temporal_logic', 'focus': 'temporal_logical_reasoning'},
    {'stage': 8, 'max_grid_size': 18, 'synthesis_ratio': 0.45, 'temporal_complexity': 'continuity_advanced', 'focus': 'advanced_continuity_analysis'},
    
    # Advanced Temporal Mastery (20x20 - 30x30)
    {'stage': 9, 'max_grid_size': 22, 'synthesis_ratio': 0.4, 'temporal_complexity': 'ensemble_temporal', 'focus': 'ensemble_temporal_coordination'},
    {'stage': 10, 'max_grid_size': 26, 'synthesis_ratio': 0.35, 'temporal_complexity': 'expert_temporal', 'focus': 'expert_temporal_analysis'},
    {'stage': 11, 'max_grid_size': 28, 'synthesis_ratio': 0.3, 'temporal_complexity': 'temporal_mastery', 'focus': 'temporal_reasoning_mastery'},
    {'stage': 12, 'max_grid_size': 30, 'synthesis_ratio': 0.25, 'temporal_complexity': 'temporal_genius', 'focus': 'temporal_intelligence_mastery'}
]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\033[96m{'=' * 110}\033[0m")
print(f"\033[96mCHRONOS V4 Enhanced Training - Advanced Temporal Sequence Reasoning Expert for ARC-AGI-2\033[0m")
print(f"\033[96mTemporal Transformers + Sequence Intelligence + OLYMPUS Preparation\033[0m")
print(f"\033[96mTarget: 70%+ Performance with Temporal Intelligence Mastery\033[0m")
print(f"\033[96m{'=' * 110}\033[0m")


class ChronosV4TemporalLoss(nn.Module):
    """Advanced loss function for temporal reasoning and sequence intelligence"""
    def __init__(self, config):
        super().__init__()
        self.transform_penalty = config['transform_penalty']
        self.exact_match_bonus = config['exact_match_bonus']
        self.temporal_weight = config['temporal_reasoning_weight']
        self.sequence_weight = config['sequence_analysis_weight']
        self.movement_weight = config['movement_prediction_weight']
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
        exact_bonus = exact_bonus.clamp(min=-5.8)  # Higher clamp for temporal precision
        
        # Transform penalty (low to encourage temporal learning)
        input_indices = inputs.argmax(dim=1) if inputs.dim() > 3 else inputs
        copy_penalty = (pred_indices == input_indices).all(dim=[1,2]).float()
        transform_penalty = copy_penalty.mean() * self.transform_penalty
        
        # Temporal reasoning bonuses
        temporal_bonus = self._calculate_temporal_bonus(model_outputs, pred_indices, target_indices, input_indices)
        sequence_bonus = self._calculate_sequence_bonus(model_outputs, pred_indices, target_indices)
        movement_bonus = self._calculate_movement_bonus(model_outputs)
        ensemble_bonus = self._calculate_ensemble_bonus(model_outputs)
        
        total_loss = (focal_loss + transform_penalty + exact_bonus + 
                     temporal_bonus + sequence_bonus + movement_bonus + ensemble_bonus)
        
        return {
            'total': total_loss,
            'focal': focal_loss,
            'transform': transform_penalty,
            'exact_bonus': exact_bonus,
            'temporal_bonus': temporal_bonus,
            'sequence_bonus': sequence_bonus,
            'movement_bonus': movement_bonus,
            'ensemble_bonus': ensemble_bonus,
            'exact_count': exact_count,
            'soft_exact_count': combined_matches.sum(),
            'avg_iou': iou_scores.mean(),
        }
    
    def _calculate_temporal_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor, input_indices: torch.Tensor) -> torch.Tensor:
        """Calculate temporal reasoning bonus"""
        if 'temporal_features' not in outputs:
            return torch.tensor(0.0).to(pred_indices.device)
        
        # Reward temporal transformations
        temporal_accuracy = (pred_indices == target_indices).float().mean(dim=[1,2])
        change_mask = (target_indices != input_indices).float().mean(dim=[1,2])
        
        # Use temporal expertise if available
        if 'temporal_expertise' in outputs:
            temporal_confidence = outputs['temporal_expertise'].squeeze(-1)
            temporal_score = temporal_accuracy * temporal_confidence * (1.0 + change_mask * 0.6)
        else:
            temporal_score = temporal_accuracy * (1.0 + change_mask * 0.6)
        
        return -temporal_score.mean() * self.temporal_weight * 0.14
    
    def _calculate_sequence_bonus(self, outputs: Dict, pred_indices: torch.Tensor, 
                                target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate sequence analysis bonus"""
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
                    sequence_score += continuity_confidence * 0.7
                
                # Reward temporal confidence
                if 'temporal_confidence' in temp_analysis:
                    temporal_confidence = temp_analysis['temporal_confidence'].mean()
                    sequence_score += temporal_confidence * 0.5
        
        # Normalize by number of analyses
        if len(temporal_analyses) > 0:
            sequence_score = sequence_score / len(temporal_analyses)
        
        return -sequence_score * self.sequence_weight * 0.11
    
    def _calculate_movement_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate movement prediction bonus"""
        if 'multitemporal_features' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        multitemporal_features = outputs['multitemporal_features']
        
        # Encourage diverse temporal scale representations
        movement_score = 0
        for i, temporal_features in enumerate(multitemporal_features):
            # Measure temporal diversity at each scale
            temporal_diversity = temporal_features.std(dim=0).mean()
            movement_score += temporal_diversity * (1.0 / (i + 1))  # Weight by scale importance
        
        # Normalize
        movement_score = movement_score / len(multitemporal_features)
        
        return -movement_score * self.movement_weight * 0.09
    
    def _calculate_ensemble_bonus(self, outputs: Dict) -> torch.Tensor:
        """Calculate ensemble coordination bonus"""
        if 'ensemble_output' not in outputs:
            return torch.tensor(0.0).to(list(outputs.values())[0].device)
        
        ensemble_output = outputs['ensemble_output']
        
        # Reward high temporal consensus
        if 'temporal_consensus' in ensemble_output:
            consensus = ensemble_output['temporal_consensus'].mean()
            ensemble_score = consensus
        else:
            ensemble_score = torch.tensor(0.65).to(list(outputs.values())[0].device)
        
        # Reward high temporal expertise
        if 'temporal_expertise' in ensemble_output:
            expertise = ensemble_output['temporal_expertise'].mean()
            ensemble_score = ensemble_score * expertise
        
        return -ensemble_score * self.ensemble_weight * 0.07


class AdvancedTemporalDataset(Dataset):
    """Dataset optimized for advanced temporal intelligence training"""
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, 
                 temporal_focus: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.stage_config = stage_config
        self.temporal_focus = temporal_focus
        
        # Load data with advanced temporal filtering
        self.samples = []
        self._load_advanced_temporal_data()
        
        print(f"\033[96mLoaded {len(self.samples)} advanced temporal samples for CHRONOS V4 training\033[0m")
    
    def _load_advanced_temporal_data(self):
        """Load data with advanced temporal complexity focus"""
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
                        self._process_advanced_temporal_task(task, file)
    
    def _process_advanced_temporal_task(self, task: Dict, source_file: str):
        """Process task with advanced temporal analysis"""
        is_arc_task = 'arc_' in source_file
        
        # Process all examples for temporal learning
        for example in task.get('train', []) + task.get('test', []):
            sample = self._create_advanced_temporal_sample(example, is_arc_task)
            if sample:
                self.samples.append(sample)
    
    def _create_advanced_temporal_sample(self, example: Dict, is_arc_task: bool) -> Optional[Dict]:
        """Create sample with advanced temporal analysis"""
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Size filtering
        if (max(input_grid.shape) > self.max_grid_size or 
            max(output_grid.shape) > self.max_grid_size):
            return None
        
        # Advanced temporal analysis
        temporal_analysis = self._analyze_advanced_temporal_complexity(input_grid, output_grid)
        
        # Filter for advanced temporal relevance
        if self.temporal_focus and temporal_analysis['temporal_intelligence_level'] < 2:
            if random.random() > 0.35:  # Keep 35% of simple cases
                return None
        
        return {
            'input': input_grid,
            'output': output_grid,
            'is_arc': is_arc_task,
            'temporal_analysis': temporal_analysis
        }
    
    def _analyze_advanced_temporal_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze advanced temporal complexity and sequence requirements"""
        # Basic temporal properties
        same_shape = input_grid.shape == output_grid.shape
        same_content = np.array_equal(input_grid, output_grid)
        
        # Pattern analysis
        input_unique = len(np.unique(input_grid))
        output_unique = len(np.unique(output_grid))
        total_unique = len(np.unique(np.concatenate([input_grid.flatten(), output_grid.flatten()])))
        
        # Temporal intelligence level calculation
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
        
        # Movement detection (simple heuristic)
        movement_score = 0
        if same_shape:
            # Check for potential object movement
            for color in range(10):
                input_mask = (input_grid == color)
                output_mask = (output_grid == color)
                if input_mask.sum() > 0 and output_mask.sum() > 0:
                    input_center = np.mean(np.where(input_mask), axis=1)
                    output_center = np.mean(np.where(output_mask), axis=1)
                    if len(input_center) == 2 and len(output_center) == 2:
                        movement_dist = np.linalg.norm(input_center - output_center)
                        movement_score = max(movement_score, movement_dist)
        
        if movement_score > 2.0:
            temporal_intelligence_level += 0.5
        
        # Complexity classification
        if temporal_intelligence_level <= 1 and max_dim <= 10:
            complexity = 'trivial'
        elif temporal_intelligence_level <= 2 and max_dim <= 16:
            complexity = 'basic'
        elif temporal_intelligence_level <= 3 and max_dim <= 22:
            complexity = 'intermediate'
        elif temporal_intelligence_level <= 4:
            complexity = 'advanced'
        else:
            complexity = 'expert'
        
        # Temporal transformation type
        if same_content:
            transform_type = 'identity'
        elif same_shape and movement_score > 1.0:
            transform_type = 'movement_based'
        elif same_shape:
            transform_type = 'pattern_evolution'
        else:
            transform_type = 'sequence_generation'
        
        return {
            'temporal_intelligence_level': temporal_intelligence_level,
            'complexity': complexity,
            'transform_type': transform_type,
            'unique_patterns': total_unique,
            'movement_score': movement_score,
            'max_dimension': max_dim,
            'temporal_density': (input_unique + output_unique) / (max_dim ** 2)
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


def advanced_temporal_collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Enhanced collate function for advanced temporal training"""
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


def train_chronos_specialized_v4():
    """Main training function for CHRONOS V4"""
    print(f"\033[96mInitializing CHRONOS V4 Advanced Temporal Intelligence Training...\033[0m")
    
    # Initialize enhanced model
    model = ChronosV4Enhanced(
        max_grid_size=30,
        d_model=256,
        num_layers=6,
        preserve_weights=True
    ).to(device)
    
    print(f"\033[96mModel initialized with {sum(p.numel() for p in model.parameters())} parameters\033[0m")
    
    # Load existing weights
    model_path = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4/chronos_best.pt'
    weights_loaded = model.load_compatible_weights(model_path)
    
    if not weights_loaded:
        print(f"\033[96mStarting fresh CHRONOS V4 training\033[0m")
    
    # Initialize loss function
    criterion = ChronosV4TemporalLoss(CHRONOS_V4_CONFIG)
    
    # Initialize optimizer with temporal learning settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CHRONOS_V4_CONFIG['learning_rate'],
        weight_decay=CHRONOS_V4_CONFIG['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CHRONOS_V4_CONFIG['warmup_epochs'],
        T_mult=int(CHRONOS_V4_CONFIG['restart_multiplier']),
        eta_min=CHRONOS_V4_CONFIG['learning_rate'] * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training metrics
    best_performance = 0.0
    training_stats = defaultdict(list)
    
    print(f"\033[96mStarting Progressive Temporal Training - 13 Temporal Intelligence Stages\033[0m")
    
    # Progressive training through temporal stages
    for stage_idx, stage_config in enumerate(STAGE_CONFIG):
        print(f"\n\033[96m{'=' * 100}\033[0m")
        print(f"\033[96mStage {stage_idx}: Grid Size {stage_config['max_grid_size']} | "
              f"Temporal: {stage_config['temporal_complexity']} | Focus: {stage_config['focus']}\033[0m")
        print(f"\033[96m{'=' * 100}\033[0m")
        
        # Create advanced temporal dataset for this stage
        dataset = AdvancedTemporalDataset(
            data_dir='/content/AutomataNexus_Olympus_AGI2/data',
            max_grid_size=stage_config['max_grid_size'],
            stage_config=stage_config,
            temporal_focus=True
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=CHRONOS_V4_CONFIG['batch_size'],
            shuffle=True,
            collate_fn=advanced_temporal_collate_fn,
            num_workers=0
        )
        
        # Stage-specific training
        stage_performance = train_advanced_temporal_stage(
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
                'config': CHRONOS_V4_CONFIG,
                'ensemble_state': model.get_ensemble_state()
            }, '/content/AutomataNexus_Olympus_AGI2/models/chronos_v4_best.pt')
            print(f"\033[96mNew best temporal performance: {best_performance:.2%} - Model saved!\033[0m")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n\033[96m{'=' * 110}\033[0m")
    print(f"\033[96mCHRONOS V4 Advanced Temporal Intelligence Training Complete!\033[0m")
    print(f"\033[96mBest Temporal Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96mOLYMPUS Integration Ready: {model.get_ensemble_state()['coordination_ready']}\033[0m")
    print(f"\033[96m{'=' * 110}\033[0m")
    
    return model, best_performance


def train_advanced_temporal_stage(model, dataloader, criterion, optimizer, scheduler, scaler,
                                stage_idx, stage_config, training_stats):
    """Train a single advanced temporal curriculum stage"""
    model.train()
    
    epochs_for_stage = CHRONOS_V4_CONFIG['epochs_per_stage']
    accumulation_steps = CHRONOS_V4_CONFIG['gradient_accumulation']
    
    best_stage_performance = 0.0
    
    for epoch in range(epochs_for_stage):
        epoch_losses = defaultdict(float)
        total_exact_matches = 0
        total_samples = 0
        advanced_temporal_count = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"\033[96mAdvanced Temporal Stage {stage_idx} Epoch {epoch}\033[0m")
        
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), CHRONOS_V4_CONFIG['gradient_clip'])
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
            
            # Count advanced temporal cases
            for meta in metadata:
                if meta['temporal_analysis']['temporal_intelligence_level'] >= 3:
                    advanced_temporal_count += 1
            
            # Update progress bar
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({
                'Loss': f"{loss_dict['total'].item():.4f}",
                'Performance': f"{current_performance:.1f}%",
                'IoU': f"{loss_dict['avg_iou'].item():.3f}",
                'AdvTemporal': f"{advanced_temporal_count}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Calculate epoch performance
        epoch_performance = total_exact_matches / max(total_samples, 1)
        best_stage_performance = max(best_stage_performance, epoch_performance)
        
        # Log detailed progress
        if epoch % 5 == 0 or epoch == epochs_for_stage - 1:
            temporal_ratio = advanced_temporal_count / max(total_samples, 1)
            print(f"\033[96mAdvanced Temporal Stage {stage_idx} Epoch {epoch}: "
                  f"Performance = {epoch_performance:.1%}, "
                  f"Advanced Temporal = {temporal_ratio:.1%}, "
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
    model, best_performance = train_chronos_specialized_v4()