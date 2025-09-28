# AutomataNexus OLYMPUS AGI2 - V4 MEGA-SCALE with Curriculum Learning
# Combines massive scale with proven curriculum strategy
# Enhanced with: Early stopping, adaptive LR, stronger regularization, gradient clipping

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "matplotlib", "numpy", "pandas", "tqdm", "plotly>=6.1.1", "kaleido", "-q"])
print("✓ Packages installed")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
from typing import Dict, List, Tuple, Optional
import time
import gc
import random
from torch.amp import GradScaler, autocast
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd
from collections import deque, defaultdict

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    print(f'\n🚀 A100 80GB DETECTED! MEGA-SCALE + CURRICULUM MODE!')

# Setup paths (repository already exists in Colab)
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content')

from src.models.arc_models_enhanced import create_enhanced_models
from src.dsl import DSLTrainingIntegration, DSLProgramGenerator
from src.program_synthesis.synthesis_integration import LightweightProgramSynthesizer, ProgramSynthesisDataGenerator
from src.data.arc_data_synthesis import ARCDataSynthesizer, ARCDataAugmenter

# Import MEPT and LEAP components
try:
    from src.utils.mept_system import ExperienceReplayBuffer, PatternBank, MEPTLoss, MEPTAugmentedDataset, create_mept_system
    from src.utils.leap_system import AdaptivePatternGenerator, LEAPTrainer, WeakPatternDetector, create_leap_system
    MEPT_LEAP_AVAILABLE = True
except ImportError:
    MEPT_LEAP_AVAILABLE = False
    print("⚠️ MEPT/LEAP components not available")

# Import exact match boost components
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')
try:
    from stage0_exact_match_boost import ExactMatchBoostDataset, AggressiveLoss, inject_exact_match_training
    EXACT_BOOST_AVAILABLE = True
except ImportError:
    EXACT_BOOST_AVAILABLE = False
    print("⚠️ Exact match boost not available")

# MEGA-SCALE HYPERPARAMETERS FOR A100 80GB
BATCH_SIZE = 512  # 16x larger!
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size: 2048!
LEARNING_RATE = 0.01  # Scaled with batch size
NUM_EPOCHS = 300
MAX_GRID_SIZE = 30
NUM_COLORS = 10
NUM_WORKERS = 8  # Parallel data loading
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# Enhanced loss weights - FIXED!
RECONSTRUCTION_WEIGHT = 1.0
EDGE_WEIGHT = 0.3
COLOR_BALANCE_WEIGHT = 0.2
STRUCTURE_WEIGHT = 0.3
TRANSFORMATION_PENALTY = 2.0  # INCREASED: Much stronger penalty for copying!
EXACT_MATCH_BONUS = 5.0  # Reduced from 10.0 to prevent instability

# Curriculum settings
CURRICULUM_STAGES = 3
EPOCHS_PER_STAGE = 100

# Enable/Disable MEPT and LEAP
USE_MEPT = True
USE_LEAP = True  # NEW: Learning Enhancement through Adaptive Patterns

print(f"\n⚙️ V4 MEGA-SCALE + CURRICULUM Configuration:")
print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Workers: {NUM_WORKERS}")
print(f"  Curriculum stages: {CURRICULUM_STAGES}")
print(f"  Transformation penalty: {TRANSFORMATION_PENALTY} (2x stronger!)")
print(f"  Exact match bonus: {EXACT_MATCH_BONUS} (2x bigger!)")
print(f"  MEPT: {'Enabled' if USE_MEPT else 'Disabled'}")
print(f"  LEAP: {'Enabled' if USE_LEAP else 'Disabled'}")

# Data setup
DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
print(f"Using data directory: {DATA_DIR}")

# ================================================================================
# MEPT and LEAP Components are imported from utils
# ================================================================================

# The MEPT and LEAP classes (ExperienceReplayBuffer, PatternBank, MEPTLoss, 
# MEPTAugmentedDataset, AdaptivePatternGenerator, LEAPTrainer) are now 
# imported from src.utils.mept_system and src.utils.leap_system

'''
# Original class definitions commented out since we're importing them
class ExperienceReplayBuffer:
    """Maintains a buffer of successful exact match examples to prevent forgetting"""
    def __init__(self, capacity: int = 50000, prioritize_exact: bool = True):
        self.capacity = capacity
        self.prioritize_exact = prioritize_exact
        self.buffer = deque(maxlen=capacity)
        self.exact_matches = deque(maxlen=capacity // 2)  # Half for exact matches
        self.total_seen = 0
        
    def add(self, input_grid: torch.Tensor, output_grid: torch.Tensor, 
            predicted_grid: torch.Tensor, loss: float, is_exact: bool):
        """Add experience to buffer with priority for exact matches"""
        experience = {
            'input': input_grid.cpu(),
            'output': output_grid.cpu(),
            'predicted': predicted_grid.cpu(),
            'loss': loss,
            'is_exact': is_exact,
            'timestamp': self.total_seen
        }
        
        self.total_seen += 1
        
        if is_exact and self.prioritize_exact:
            self.exact_matches.append(experience)
        else:
            self.buffer.append(experience)
    
    def sample(self, batch_size: int, exact_ratio: float = 0.5) -> List[Dict]:
        """Sample from buffer with specified ratio of exact matches"""
        n_exact = int(batch_size * exact_ratio)
        n_regular = batch_size - n_exact
        
        exact_samples = []
        regular_samples = []
        
        # Sample exact matches
        if len(self.exact_matches) > 0:
            n_exact = min(n_exact, len(self.exact_matches))
            exact_indices = np.random.choice(len(self.exact_matches), n_exact, replace=True)
            exact_samples = [self.exact_matches[i] for i in exact_indices]
        
        # Sample regular experiences
        if len(self.buffer) > 0:
            n_regular = batch_size - len(exact_samples)
            regular_indices = np.random.choice(len(self.buffer), n_regular, replace=True)
            regular_samples = [self.buffer[i] for i in regular_indices]
        
        return exact_samples + regular_samples
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            'total_experiences': len(self.buffer) + len(self.exact_matches),
            'exact_matches': len(self.exact_matches),
            'regular_experiences': len(self.buffer),
            'total_seen': self.total_seen
        }


class PatternBank:
    """Stores successful transformation patterns for quick lookup"""
    def __init__(self, max_patterns: int = 10000):
        self.max_patterns = max_patterns
        self.patterns = {}  # hash -> (input, output, count)
        self.transformation_rules = {}  # transformation_type -> examples
        
    def add_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """Add a successful pattern to the bank"""
        # Create hash for quick lookup
        input_hash = hash(input_grid.tobytes())
        
        if input_hash not in self.patterns:
            self.patterns[input_hash] = {
                'input': input_grid,
                'output': output_grid,
                'count': 1
            }
        else:
            self.patterns[input_hash]['count'] += 1
        
        # Analyze transformation type
        trans_type = self._analyze_transformation(input_grid, output_grid)
        if trans_type not in self.transformation_rules:
            self.transformation_rules[trans_type] = []
        
        self.transformation_rules[trans_type].append((input_grid, output_grid))
        
        # Limit size
        if len(self.patterns) > self.max_patterns:
            # Remove least frequent patterns
            sorted_patterns = sorted(self.patterns.items(), 
                                   key=lambda x: x[1]['count'])
            to_remove = len(self.patterns) - self.max_patterns
            for hash_key, _ in sorted_patterns[:to_remove]:
                del self.patterns[hash_key]
    
    def _analyze_transformation(self, input_grid: np.ndarray, 
                               output_grid: np.ndarray) -> str:
        """Analyze the type of transformation"""
        if input_grid.shape != output_grid.shape:
            return 'resize'
        elif np.array_equal(input_grid, output_grid):
            return 'identity'
        elif np.array_equal(np.rot90(input_grid), output_grid):
            return 'rotate_90'
        elif np.array_equal(np.flip(input_grid, axis=0), output_grid):
            return 'flip_vertical'
        elif np.array_equal(np.flip(input_grid, axis=1), output_grid):
            return 'flip_horizontal'
        elif len(np.unique(output_grid)) < len(np.unique(input_grid)):
            return 'color_reduction'
        elif np.sum(output_grid > 0) < np.sum(input_grid > 0):
            return 'pattern_extraction'
        else:
            return 'complex'
    
    def lookup_pattern(self, input_grid: np.ndarray) -> Optional[np.ndarray]:
        """Look up a pattern in the bank"""
        input_hash = hash(input_grid.tobytes())
        if input_hash in self.patterns:
            return self.patterns[input_hash]['output']
        return None
    
    def get_similar_transformations(self, input_grid: np.ndarray, 
                                   output_grid: np.ndarray, k: int = 5) -> List:
        """Get similar transformations from the bank"""
        trans_type = self._analyze_transformation(input_grid, output_grid)
        
        if trans_type in self.transformation_rules:
            examples = self.transformation_rules[trans_type]
            # Return up to k random examples
            k = min(k, len(examples))
            return random.sample(examples, k)
        return []


class MEPTLoss(nn.Module):
    """Memory-Enhanced Progressive Training Loss with dynamic weighting"""
    def __init__(self, replay_buffer: ExperienceReplayBuffer, 
                 pattern_bank: PatternBank, use_mept: bool = True):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.pattern_bank = pattern_bank
        self.use_mept = use_mept
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # Dynamic weights that adapt during training
        self.weights = {
            'reconstruction': 1.0,
            'exact_match': 5.0,
            'consistency': 0.5,
            'diversity': 0.3,
            'memory_alignment': 0.4
        }
        
        # Track performance for dynamic adjustment
        self.performance_history = deque(maxlen=100)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = pred.shape
        
        # Get predictions
        pred_indices = pred.argmax(dim=1)
        target_indices = target.argmax(dim=1)
        input_indices = input_grid.argmax(dim=1)
        
        if self.use_mept:
            # MEPT Loss calculation
            # 1. Reconstruction loss with focal adjustment
            pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
            target_flat = target_indices.reshape(-1)
            ce_loss = self.ce_loss(pred_flat, target_flat)
            
            # Focal loss modification for hard examples
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** 2) * ce_loss
            reconstruction_loss = focal_loss.reshape(B, H, W).mean(dim=[1,2])
            
            # 2. Exact match bonus with progressive scaling
            exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
            exact_bonus = -exact_matches * self.weights['exact_match']
            
            # 3. Diversity loss - encourage diverse predictions
            pred_entropy = -(pred.softmax(dim=1) * pred.log_softmax(dim=1)).sum(dim=1).mean(dim=[1,2])
            diversity_loss = -pred_entropy  # Negative because we want to maximize entropy
            
            # 4. Memory alignment loss - align with successful past examples
            memory_loss = self._calculate_memory_loss(pred, input_indices, target_indices)
            
            # 5. Transformation penalty - prevent copying
            is_identity_task = (input_indices == target_indices).all(dim=[1,2]).float()
            same_as_input = (pred_indices == input_indices).float().mean(dim=[1,2])
            transformation_penalty = same_as_input * (1 - is_identity_task) * TRANSFORMATION_PENALTY
            
            # Combine losses with dynamic weighting
            total_loss = (
                self.weights['reconstruction'] * reconstruction_loss +
                exact_bonus +
                self.weights['diversity'] * diversity_loss +
                self.weights['memory_alignment'] * memory_loss +
                transformation_penalty
            )
            
            # Store experiences in replay buffer
            for i in range(B):
                self.replay_buffer.add(
                    input_grid[i], target[i], pred[i],
                    total_loss[i].item(), exact_matches[i].item() > 0.5
                )
            
            # Update pattern bank with exact matches
            for i in range(B):
                if exact_matches[i] > 0.5:
                    self.pattern_bank.add_pattern(
                        input_indices[i].cpu().numpy(),
                        target_indices[i].cpu().numpy()
                    )
            
            # Update performance history and adjust weights
            self.performance_history.append(exact_matches.mean().item())
            self._adjust_weights()
            
            return {
                'total': total_loss.mean(),
                'reconstruction': reconstruction_loss.mean(),
                'exact_bonus': -exact_bonus.mean(),
                'transformation': transformation_penalty.mean(),
                'diversity': diversity_loss.mean(),
                'memory': memory_loss.mean(),
                'exact_count': exact_matches.sum()
            }
        else:
            # Fall back to regular MegaScaleLoss behavior
            return self._regular_loss(pred, target, input_grid)
    
    def _regular_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                     input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Regular loss calculation (non-MEPT)"""
        B, C, H, W = pred.shape
        
        pred_indices = pred.argmax(dim=1)
        target_indices = target.argmax(dim=1)
        input_indices = input_grid.argmax(dim=1)
        
        # Standard reconstruction loss
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target_indices.reshape(-1)
        ce_loss = self.ce_loss(pred_flat, target_flat).reshape(B, H, W)
        
        # Exact match bonus
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_bonus = -exact_matches * EXACT_MATCH_BONUS
        
        # Edge-aware loss
        target_edges = self._detect_edges(target_indices)
        edge_weight = 1.0 + target_edges * (EDGE_WEIGHT * 10)
        weighted_loss = ce_loss * edge_weight
        reconstruction_loss = weighted_loss.mean(dim=[1,2])
        
        # Transformation penalty
        is_identity_task = (input_indices == target_indices).all(dim=[1,2]).float()
        same_as_input = (pred_indices == input_indices).float().mean(dim=[1,2])
        transformation_penalty = same_as_input * (1 - is_identity_task)
        
        total_loss = (
            RECONSTRUCTION_WEIGHT * reconstruction_loss +
            TRANSFORMATION_PENALTY * transformation_penalty +
            exact_bonus
        )
        
        return {
            'total': total_loss.mean(),
            'reconstruction': reconstruction_loss.mean(),
            'transformation': transformation_penalty.mean(),
            'exact_bonus': -exact_bonus.mean(),
            'exact_count': exact_matches.sum()
        }
    
    def _calculate_memory_loss(self, pred: torch.Tensor, 
                              input_indices: torch.Tensor,
                              target_indices: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on memory bank patterns"""
        B = pred.shape[0]
        memory_losses = []
        
        for i in range(B):
            # Look up pattern in bank
            input_np = input_indices[i].cpu().numpy()
            stored_output = self.pattern_bank.lookup_pattern(input_np)
            
            if stored_output is not None:
                # Convert to tensor
                stored_tensor = torch.tensor(stored_output, device=pred.device)
                stored_one_hot = F.one_hot(stored_tensor, num_classes=pred.shape[1])
                stored_one_hot = stored_one_hot.permute(2, 0, 1).float()
                
                # Calculate similarity loss
                memory_loss = F.kl_div(
                    pred[i].log_softmax(dim=0),
                    stored_one_hot,
                    reduction='batchmean'
                )
                memory_losses.append(memory_loss)
            else:
                memory_losses.append(torch.tensor(0.0, device=pred.device))
        
        return torch.stack(memory_losses)
    
    def _detect_edges(self, grid: torch.Tensor) -> torch.Tensor:
        """Detect edges in grid with NaN protection"""
        if grid.numel() == 0 or grid.shape[-1] < 2 or grid.shape[-2] < 2:
            return torch.zeros_like(grid)
        
        try:
            dx = torch.abs(grid[:, 1:, :] - grid[:, :-1, :])
            dy = torch.abs(grid[:, :, 1:] - grid[:, :, :-1])
            
            dx = F.pad(dx, (0, 0, 0, 1), value=0)
            dy = F.pad(dy, (0, 1, 0, 0), value=0)
            
            edges = ((dx + dy) > 0).float()
            
            if torch.isnan(edges).any():
                return torch.zeros_like(grid)
            
            return edges
        except Exception as e:
            return torch.zeros_like(grid)
    
    def _adjust_weights(self):
        """Dynamically adjust weights based on performance"""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = np.mean(list(self.performance_history)[-10:])
        
        # If exact match rate is low, increase exact match weight
        if recent_performance < 0.1:
            self.weights['exact_match'] = min(10.0, self.weights['exact_match'] * 1.1)
            self.weights['diversity'] = max(0.1, self.weights['diversity'] * 0.9)
        
        # If exact match rate is improving, balance weights
        elif recent_performance > 0.3:
            self.weights['exact_match'] = max(3.0, self.weights['exact_match'] * 0.95)
            self.weights['memory_alignment'] = min(1.0, self.weights['memory_alignment'] * 1.05)


class MEPTAugmentedDataset(Dataset):
    """Dataset that combines regular data with replay buffer samples"""
    def __init__(self, base_dataset, replay_buffer: ExperienceReplayBuffer,
                 replay_ratio: float = 0.3):
        self.base_dataset = base_dataset
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # With probability replay_ratio, return a replay sample
        if random.random() < self.replay_ratio and len(self.replay_buffer.buffer) > 0:
            experiences = self.replay_buffer.sample(1, exact_ratio=0.7)
            if experiences:
                exp = experiences[0]
                return {
                    'inputs': exp['input'],
                    'outputs': exp['output']
                }
        
        # Otherwise return regular sample
        return self.base_dataset[idx]


# ================================================================================
# LEAP (Learning Enhancement through Adaptive Patterns) Components for Stage 0
# ================================================================================

class AdaptivePatternGenerator:
    """Generates patterns that adapt based on model's current weaknesses"""
    
    def __init__(self, grid_size: int = 6):
        self.grid_size = grid_size
        self.performance_tracker = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        
        # Core pattern generators for Stage 0
        self.pattern_generators = {
            'identity': self._generate_identity,
            'solid_color': self._generate_solid_color,
            'horizontal_stripes': self._generate_horizontal_stripes,
            'vertical_stripes': self._generate_vertical_stripes,
            'checkerboard': self._generate_checkerboard,
            'center_dot': self._generate_center_dot,
            'border': self._generate_border,
            'corners': self._generate_corners,
            'color_swap': self._generate_color_swap,
            'extract_color': self._generate_extract_color,
            'binary_threshold': self._generate_binary_threshold,
            'counting': self._generate_counting,
            'diagonal': self._generate_diagonal,
            'cross': self._generate_cross,
            'invert': self._generate_invert
        }
    
    def generate_adaptive_batch(self, batch_size: int, weak_patterns: List[str] = None) -> List[Dict]:
        """Generate batch focusing on weak patterns"""
        if weak_patterns is None:
            weak_patterns = self._identify_weak_patterns()
        
        batch = []
        # 40% weak patterns, 30% identity (fundamental), 30% random
        weak_count = int(batch_size * 0.4)
        identity_count = int(batch_size * 0.3)
        random_count = batch_size - weak_count - identity_count
        
        # Generate weak patterns
        if weak_patterns:
            for _ in range(weak_count):
                pattern_type = random.choice(weak_patterns)
                pattern = self.pattern_generators[pattern_type]()
                pattern['pattern_type'] = pattern_type
                batch.append(pattern)
        else:
            # If no weak patterns, use random
            for _ in range(weak_count):
                pattern_type = random.choice(list(self.pattern_generators.keys()))
                pattern = self.pattern_generators[pattern_type]()
                pattern['pattern_type'] = pattern_type
                batch.append(pattern)
        
        # Always include identity
        for _ in range(identity_count):
            pattern = self._generate_identity()
            pattern['pattern_type'] = 'identity'
            batch.append(pattern)
        
        # Random patterns
        for _ in range(random_count):
            pattern_type = random.choice(list(self.pattern_generators.keys()))
            pattern = self.pattern_generators[pattern_type]()
            pattern['pattern_type'] = pattern_type
            batch.append(pattern)
        
        return batch
    
    def _identify_weak_patterns(self, threshold: float = 0.5) -> List[str]:
        """Identify patterns with low success rate"""
        weak_patterns = []
        for pattern_name, stats in self.performance_tracker.items():
            if stats['attempts'] > 10:  # Need sufficient attempts
                success_rate = stats['successes'] / stats['attempts']
                if success_rate < threshold:
                    weak_patterns.append(pattern_name)
        return weak_patterns
    
    def update_performance(self, pattern_type: str, success: bool):
        """Update performance tracking"""
        self.performance_tracker[pattern_type]['attempts'] += 1
        if success:
            self.performance_tracker[pattern_type]['successes'] += 1
    
    # Pattern generators
    def _generate_identity(self) -> Dict:
        """Identity - most fundamental"""
        grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        return {'input': grid, 'output': grid.copy()}
    
    def _generate_solid_color(self) -> Dict:
        """Fill with single color"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        color = np.random.randint(0, 4)
        output_grid = np.full_like(input_grid, color)
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_horizontal_stripes(self) -> Dict:
        """Horizontal stripes"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        colors = [1, 2]
        for i in range(self.grid_size):
            output_grid[i, :] = colors[i % 2]
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_vertical_stripes(self) -> Dict:
        """Vertical stripes"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        colors = [1, 3]
        for j in range(self.grid_size):
            output_grid[:, j] = colors[j % 2]
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_checkerboard(self) -> Dict:
        """Checkerboard pattern"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                output_grid[i, j] = 1 if (i + j) % 2 == 0 else 2
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_center_dot(self) -> Dict:
        """Single center pixel"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        center = self.grid_size // 2
        output_grid[center, center] = 2
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_border(self) -> Dict:
        """Border pattern"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        output_grid[0, :] = 1
        output_grid[-1, :] = 1
        output_grid[:, 0] = 1
        output_grid[:, -1] = 1
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_corners(self) -> Dict:
        """Corner dots"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        output_grid[0, 0] = 2
        output_grid[0, -1] = 2
        output_grid[-1, 0] = 2
        output_grid[-1, -1] = 2
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_color_swap(self) -> Dict:
        """Swap two colors"""
        input_grid = np.random.randint(0, 3, (self.grid_size, self.grid_size))
        output_grid = input_grid.copy()
        # Swap colors 1 and 2
        mask1 = input_grid == 1
        mask2 = input_grid == 2
        output_grid[mask1] = 2
        output_grid[mask2] = 1
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_extract_color(self) -> Dict:
        """Extract single color"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        # Ensure at least one non-zero color
        if np.all(input_grid == 0):
            input_grid[0, 0] = 1
        color_to_extract = np.random.choice(np.unique(input_grid[input_grid > 0]))
        output_grid = np.where(input_grid == color_to_extract, color_to_extract, 0)
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_binary_threshold(self) -> Dict:
        """Binary threshold"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.where(input_grid > 1, 1, 0)
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_counting(self) -> Dict:
        """Count non-zero and fill"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        count = np.count_nonzero(input_grid)
        color = min(3, count // 5)
        output_grid = np.full_like(input_grid, color)
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_diagonal(self) -> Dict:
        """Diagonal line"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        for i in range(min(self.grid_size, self.grid_size)):
            output_grid[i, i] = 3
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_cross(self) -> Dict:
        """Cross pattern"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        center = self.grid_size // 2
        output_grid[center, :] = 3
        output_grid[:, center] = 3
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_invert(self) -> Dict:
        """Invert 0/1"""
        input_grid = np.random.randint(0, 2, (self.grid_size, self.grid_size))
        output_grid = 1 - input_grid
        return {'input': input_grid, 'output': output_grid}


class LEAPTrainer:
    """Manages LEAP training integration for Stage 0"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.pattern_generator = AdaptivePatternGenerator()
        self.pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.leap_iterations = 0
        self.enabled = USE_LEAP
        
    def generate_leap_batch(self, batch_size: int = 64) -> Dict[str, torch.Tensor]:
        """Generate a LEAP training batch"""
        # Get weak patterns for focused training
        weak_patterns = self.pattern_generator._identify_weak_patterns()
        
        # Generate adaptive batch
        patterns = self.pattern_generator.generate_adaptive_batch(batch_size, weak_patterns)
        
        # Convert to tensors
        inputs = []
        outputs = []
        pattern_types = []
        
        for pattern in patterns:
            inputs.append(torch.tensor(pattern['input'], dtype=torch.long))
            outputs.append(torch.tensor(pattern['output'], dtype=torch.long))
            pattern_types.append(pattern['pattern_type'])
        
        return {
            'inputs': torch.stack(inputs),
            'outputs': torch.stack(outputs),
            'pattern_types': pattern_types
        }
    
    def update_pattern_stats(self, pattern_types: List[str], predictions: torch.Tensor, 
                           targets: torch.Tensor):
        """Update pattern performance statistics"""
        pred_indices = predictions.argmax(dim=1)
        target_indices = targets.argmax(dim=1)
        
        for i, pattern_type in enumerate(pattern_types):
            correct = (pred_indices[i] == target_indices[i]).all().item()
            self.pattern_stats[pattern_type]['total'] += 1
            if correct:
                self.pattern_stats[pattern_type]['correct'] += 1
            self.pattern_generator.update_performance(pattern_type, correct)
    
    def get_performance_report(self) -> str:
        """Get LEAP performance report"""
        if not self.enabled or not self.pattern_stats:
            return ""
            
        report_lines = ["\n🎯 LEAP Pattern Performance:"]
        
        # Sort by accuracy (lowest first)
        pattern_accs = []
        for pattern, stats in self.pattern_stats.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                pattern_accs.append((pattern, acc, stats['total']))
        
        pattern_accs.sort(key=lambda x: x[1])
        
        # Show worst 5 and best 5
        if len(pattern_accs) > 0:
            report_lines.append("Weakest patterns:")
            for pattern, acc, total in pattern_accs[:5]:
                report_lines.append(f"  {pattern}: {acc*100:.1f}% ({total} attempts)")
            
            if len(pattern_accs) > 5:
                report_lines.append("Strongest patterns:")
                for pattern, acc, total in pattern_accs[-5:]:
                    report_lines.append(f"  {pattern}: {acc*100:.1f}% ({total} attempts)")
        
        return "\n".join(report_lines)
'''  # End of commented out MEPT/LEAP classes


class CurriculumMegaScaleDataset(Dataset):
    """High-performance dataset with curriculum learning and ARC synthesis"""
    
    def __init__(self, data_dir: str, curriculum_stage: int = 0, augment_factor: int = 10, 
                 use_arc_synthesis: bool = True, synthesis_ratio: float = 0.3):
        self.samples = []
        self.curriculum_stage = curriculum_stage
        self.augment_factor = augment_factor
        self.use_arc_synthesis = use_arc_synthesis
        self.synthesis_ratio = synthesis_ratio
        
        # Initialize ARC synthesizer
        if self.use_arc_synthesis:
            try:
                self.arc_synthesizer = ARCDataSynthesizer(data_dir)
                self.arc_augmenter = ARCDataAugmenter()
                print(f"✅ ARC Data Synthesizer initialized for stage {curriculum_stage}")
            except Exception as e:
                print(f"⚠️ Could not initialize ARC synthesizer: {e}")
                self.use_arc_synthesis = False
        
        self._load_data(data_dir)
        
    def _load_data(self, data_dir: str):
        print(f"Loading curriculum stage {self.curriculum_stage} with {self.augment_factor}x augmentation...")
        
        with open(f'{data_dir}/arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        with open(f'{data_dir}/arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        for task_id, task_data in challenges.items():
            for i, example in enumerate(task_data['train']):
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                # Assess difficulty
                h, w = input_grid.shape
                oh, ow = output_grid.shape
                n_colors_in = len(np.unique(input_grid))
                n_colors_out = len(np.unique(output_grid))
                grid_size = max(h, w, oh, ow)
                size_diff = abs(h*w - oh*ow)
                
                # Create sample dictionary first
                sample = {
                    'input': input_grid,
                    'output': output_grid,
                    'task_id': task_id,
                    'example_idx': i
                }
                
                # Curriculum filtering
                if self.curriculum_stage == 0:  # Easy
                    # Stage 0: Only include very simple patterns
                    # Priority: same size, few colors, small grids
                    if grid_size > 12 or n_colors_in > 4 or n_colors_out > 4:  # Slightly relaxed
                        continue
                    # Give preference to same-size transformations
                    if size_diff == 0:
                        # Add extra copies for same-size patterns
                        for _ in range(5):  # More copies of perfect patterns
                            self.samples.append(sample)
                    # Also prioritize very small grids for exact learning
                    if grid_size <= 5:
                        for _ in range(3):
                            self.samples.append(sample)
                elif self.curriculum_stage == 1:  # Medium
                    if grid_size > 20 or n_colors_in > 5 or n_colors_out > 5:
                        continue
                # Stage 2 accepts all
                
                sample = {'input': input_grid, 'output': output_grid}
                
                # Add original
                self.samples.append(sample)
                
                # Stage 0: Add identity tasks to help model learn exact copying
                if self.curriculum_stage == 0:
                    # 60% chance for identity task to really emphasize exact copying
                    if np.random.random() < 0.6:
                        # Create identity task (output = input)
                        identity_sample = {
                            'input': input_grid.copy(),
                            'output': input_grid.copy()
                        }
                        self.samples.append(identity_sample)
                        # Add multiple copies to emphasize
                        for _ in range(4):  # More identity copies
                            self.samples.append(identity_sample)
                    
                    # Add pure color fill tasks (easy exact matches)
                    if grid_size <= 5 and np.random.random() < 0.3:
                        for color in range(1, min(4, n_colors_in + 1)):
                            filled = np.full_like(input_grid, color)
                            self.samples.append({
                                'input': input_grid.copy(),
                                'output': filled
                            })
                    
                    # Also add simple transformations
                    if n_colors_in == 2 and np.random.random() < 0.4:
                        # Swap colors
                        swapped = input_grid.copy()
                        unique_colors = np.unique(input_grid)
                        if len(unique_colors) == 2:
                            swapped[input_grid == unique_colors[0]] = unique_colors[1]
                            swapped[input_grid == unique_colors[1]] = unique_colors[0]
                            swap_sample = {
                                'input': input_grid.copy(),
                                'output': swapped
                            }
                            self.samples.append(swap_sample)
                            # Add extra copy
                            self.samples.append(swap_sample)
                    
                    # Add single color extraction tasks
                    if np.random.random() < 0.3 and n_colors_in <= 3:
                        # Extract each color as separate task
                        for color in np.unique(input_grid):
                            if color > 0:  # Don't extract background
                                extracted = np.where(input_grid == color, color, 0)
                                self.samples.append({
                                    'input': input_grid.copy(),
                                    'output': extracted
                                })
                    
                    # Add binary patterns (on/off)
                    if grid_size <= 5 and np.random.random() < 0.2:
                        # Convert to binary (non-zero becomes 1)
                        binary = np.where(input_grid > 0, 1, 0)
                        self.samples.append({
                            'input': input_grid.copy(),
                            'output': binary
                        })
                        # Inverted binary
                        inverted = np.where(input_grid == 0, 1, 0)
                        self.samples.append({
                            'input': input_grid.copy(),
                            'output': inverted
                        })
                
                # Add augmentations
                for _ in range(self.augment_factor - 1):
                    aug_type = np.random.choice(['rotate', 'flip', 'both'])
                    aug_input = input_grid.copy()
                    aug_output = output_grid.copy()
                    
                    if aug_type in ['rotate', 'both']:
                        k = np.random.randint(1, 4)
                        aug_input = np.rot90(aug_input, k)
                        aug_output = np.rot90(aug_output, k)
                    
                    if aug_type in ['flip', 'both']:
                        axis = np.random.randint(0, 2)
                        aug_input = np.flip(aug_input, axis=axis)
                        aug_output = np.flip(aug_output, axis=axis)
                    
                    self.samples.append({
                        'input': aug_input,
                        'output': aug_output
                    })
        
        # Add DSL-generated samples for all stages
        print(f"Adding DSL-generated deterministic samples for stage {self.curriculum_stage}...")
        dsl_samples = DSLTrainingIntegration.create_stage0_dsl_samples(self.curriculum_stage)
        for dsl_sample in dsl_samples:
            self.samples.append({
                'input': dsl_sample['input'],
                'output': dsl_sample['output']
            })
        print(f"Added {len(dsl_samples)} DSL samples")
        
        # Add program synthesis samples for all stages
        print("Adding program synthesis samples...")
        ps_generator = ProgramSynthesisDataGenerator()
        # More complex programs for higher stages
        num_ps_samples = 200 if self.curriculum_stage == 0 else 300
        ps_samples = ps_generator.generate_programmatic_examples(num_ps_samples)
        
        # Filter by difficulty for each stage
        stage_samples = []
        for ps_sample in ps_samples:
            if self.curriculum_stage == 0 and ps_sample['difficulty'] == 'easy':
                stage_samples.append(ps_sample)
            elif self.curriculum_stage == 1 and ps_sample['difficulty'] in ['easy', 'medium']:
                stage_samples.append(ps_sample)
            elif self.curriculum_stage == 2:  # Stage 2 gets all difficulties
                stage_samples.append(ps_sample)
        
        for ps_sample in stage_samples:
            self.samples.append({
                'input': ps_sample['input'],
                'output': ps_sample['output']
            })
        print(f"Added {len(stage_samples)} program synthesis samples")
        
        # Add ARC synthetic samples if available
        if self.use_arc_synthesis:
            print(f"Generating synthetic ARC samples (ratio: {self.synthesis_ratio})...")
            n_synthetic = int(len(self.samples) * self.synthesis_ratio)
            
            # Generate in batches for efficiency
            batch_size = 100
            n_batches = (n_synthetic + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                current_batch_size = min(batch_size, n_synthetic - batch_idx * batch_size)
                
                # Generate synthetic batch
                synthetic_data = self.arc_synthesizer.generate_synthetic_batch(
                    batch_size=current_batch_size,
                    stage=self.curriculum_stage,
                    exact_match_ratio=0.7 if self.curriculum_stage == 0 else 0.5
                )
                
                # Convert tensors back to numpy and add to samples
                inputs = synthetic_data['inputs'].cpu().numpy()
                outputs = synthetic_data['outputs'].cpu().numpy()
                
                for i in range(current_batch_size):
                    self.samples.append({
                        'input': inputs[i],
                        'output': outputs[i]
                    })
            
            print(f"Added {n_synthetic} synthetic ARC samples")
        
        print(f"Loaded {len(self.samples)} samples for stage {self.curriculum_stage}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Return raw numpy arrays as tensors (not one-hot)
        # Use .copy() to avoid negative stride issues
        input_grid = torch.tensor(sample['input'].copy(), dtype=torch.long)
        output_grid = torch.tensor(sample['output'].copy(), dtype=torch.long)
        
        # Clamp to valid range
        input_grid = torch.clamp(input_grid, 0, 9)
        output_grid = torch.clamp(output_grid, 0, 9)
        
        return {
            'inputs': input_grid,
            'outputs': output_grid
        }
    
    def _to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        h, w = grid.shape
        # Fast one-hot encoding
        one_hot = torch.zeros(NUM_COLORS, MAX_GRID_SIZE, MAX_GRID_SIZE)
        
        # Pad if needed
        if h > MAX_GRID_SIZE:
            grid = grid[:MAX_GRID_SIZE, :MAX_GRID_SIZE]
            h = MAX_GRID_SIZE
        if w > MAX_GRID_SIZE:
            grid = grid[:, :MAX_GRID_SIZE]
            w = MAX_GRID_SIZE
            
        # Vectorized one-hot
        for color in range(NUM_COLORS):
            mask = (grid == color)
            one_hot[color, :h, :w] = torch.from_numpy(mask.astype(np.float32))
        
        return one_hot


class TrainingReporter:
    """Generate comprehensive training reports with plotly visualizations"""
    
    def __init__(self, model_name: str, report_dir: str = '/content/AutomataNexus_Olympus_AGI2/src/models/reports'):
        self.model_name = model_name
        self.report_dir = os.path.join(report_dir, model_name)
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Training metrics storage
        self.metrics = {
            'epoch': [],
            'stage': [],
            'train_loss': [],
            'train_exact': [],
            'val_loss': [],
            'val_exact': [],
            'val_pixel_acc': [],
            'learning_rate': [],
            'transformation_penalty': [],
            'timestamp': []
        }
        
    def add_metrics(self, epoch: int, stage: int, train_loss: float, train_exact: float,
                   val_loss: float, val_exact: float, val_pixel_acc: float, lr: float, trans_penalty: float):
        """Add metrics for current epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['stage'].append(stage)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_exact'].append(train_exact)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_exact'].append(val_exact)
        self.metrics['val_pixel_acc'].append(val_pixel_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['transformation_penalty'].append(trans_penalty)
        self.metrics['timestamp'].append(datetime.now())
    
    def generate_report(self, final_stats: dict = None):
        """Generate comprehensive training report with visualizations"""
        # Create multiple visualizations
        self._create_line_plots()
        self._create_radar_chart()
        self._create_3d_surface_plot()
        self._create_loss_decomposition_plot()
        self._create_overview_html(final_stats)
        self._create_coa()
        
    def _create_line_plots(self):
        """Create line plots for training metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Exact Match Accuracy', 'Loss Curves', 'Pixel Accuracy', 'Learning Rate'],
            specs=[[{'secondary_y': False}, {'secondary_y': True}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Exact match accuracy
        fig.add_trace(
            go.Scatter(x=self.metrics['epoch'], y=self.metrics['train_exact'],
                      name='Train Exact', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.metrics['epoch'], y=self.metrics['val_exact'],
                      name='Val Exact', line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=self.metrics['epoch'], y=self.metrics['train_loss'],
                      name='Train Loss', line=dict(color='green', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=self.metrics['epoch'], y=self.metrics['val_loss'],
                      name='Val Loss', line=dict(color='orange', width=2)),
            row=1, col=2, secondary_y=False
        )
        
        # Add stage transitions
        for i, stage in enumerate(self.metrics['stage']):
            if i > 0 and stage != self.metrics['stage'][i-1]:
                fig.add_vline(x=self.metrics['epoch'][i], line_width=2, line_dash="dash",
                            line_color="gray", annotation_text=f"Stage {stage}")
        
        # Pixel accuracy
        fig.add_trace(
            go.Scatter(x=self.metrics['epoch'], y=self.metrics['val_pixel_acc'],
                      name='Val Pixel Acc', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # Learning rate
        fig.add_trace(
            go.Scatter(x=self.metrics['epoch'], y=self.metrics['learning_rate'],
                      name='Learning Rate', line=dict(color='brown', width=2)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=f"{self.model_name.upper()} Training Progress",
                         showlegend=True)
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Pixel Accuracy (%)", row=2, col=1)
        fig.update_yaxes(title_text="Learning Rate", row=2, col=2)
        
        fig.write_html(os.path.join(self.report_dir, 'training_curves.html'))
        try:
            fig.write_image(os.path.join(self.report_dir, 'training_curves.png'))
        except Exception as e:
            print(f"Warning: Could not save PNG image: {e}")
    
    def _create_radar_chart(self):
        """Create radar chart showing model performance across metrics"""
        # Get latest metrics
        latest_metrics = {
            'Exact Match': self.metrics['val_exact'][-1] if self.metrics['val_exact'] else 0,
            'Pixel Accuracy': self.metrics['val_pixel_acc'][-1] if self.metrics['val_pixel_acc'] else 0,
            'Loss (inverted)': max(0, 100 - self.metrics['val_loss'][-1] * 20) if self.metrics['val_loss'] else 0,
            'Stability': min(100, 100 - np.std(self.metrics['val_loss'][-10:]) * 50) if len(self.metrics['val_loss']) > 10 else 50,
            'Convergence': min(100, self.metrics['epoch'][-1] / 3) if self.metrics['epoch'] else 0
        }
        
        fig = go.Figure(data=go.Scatterpolar(
            r=list(latest_metrics.values()),
            theta=list(latest_metrics.keys()),
            fill='toself',
            name=self.model_name.upper()
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f"{self.model_name.upper()} Performance Radar"
        )
        
        fig.write_html(os.path.join(self.report_dir, 'performance_radar.html'))
        try:
            fig.write_image(os.path.join(self.report_dir, 'performance_radar.png'))
        except Exception as e:
            print(f"Warning: Could not save PNG image: {e}")
    
    def _create_3d_surface_plot(self):
        """Create 3D surface plot of loss landscape"""
        if len(self.metrics['epoch']) < 10:
            return
            
        # Create grid for visualization
        epochs = np.array(self.metrics['epoch'])
        stages = np.array(self.metrics['stage'])
        losses = np.array(self.metrics['val_loss'])
        
        fig = go.Figure(data=[go.Scatter3d(
            x=epochs,
            y=stages,
            z=losses,
            mode='markers+lines',
            marker=dict(
                size=5,
                color=losses,
                colorscale='Viridis',
                showscale=True
            ),
            line=dict(color='darkblue', width=4)
        )])
        
        fig.update_layout(
            title=f'{self.model_name.upper()} Loss Landscape',
            scene=dict(
                xaxis_title='Epoch',
                yaxis_title='Curriculum Stage',
                zaxis_title='Validation Loss',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        fig.write_html(os.path.join(self.report_dir, 'loss_landscape_3d.html'))
        try:
            fig.write_image(os.path.join(self.report_dir, 'loss_landscape_3d.png'))
        except Exception as e:
            print(f"Warning: Could not save PNG image: {e}")
    
    def _create_loss_decomposition_plot(self):
        """Create stacked area plot showing loss component breakdown"""
        # This would need loss component tracking in training loop
        # For now, create a placeholder
        epochs = self.metrics['epoch']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=self.metrics['val_loss'],
            mode='lines',
            name='Total Loss',
            line=dict(width=3, color='red')
        ))
        
        fig.update_layout(
            title=f'{self.model_name.upper()} Loss Components',
            xaxis_title='Epoch',
            yaxis_title='Loss Value',
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(self.report_dir, 'loss_decomposition.html'))
    
    def _create_overview_html(self, final_stats: dict = None):
        """Create overview HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.model_name.upper()} Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #333; color: white; padding: 20px; border-radius: 10px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #00d4aa; }}
                .metric-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
                .chart-container {{ margin: 20px 0; background: white; padding: 20px; border-radius: 10px; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 {self.model_name.upper()} Training Report</h1>
                <p>AutomataNexus OLYMPUS AGI2 - Model Training Summary</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Final Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{self.metrics['val_exact'][-1]:.2f}%</div>
                    <div class="metric-label">Validation Exact Match</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.metrics['val_pixel_acc'][-1]:.2f}%</div>
                    <div class="metric-label">Validation Pixel Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.metrics['val_loss'][-1]:.4f}</div>
                    <div class="metric-label">Final Validation Loss</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Training Progress</h2>
                <iframe src="training_curves.html" width="100%" height="850" frameborder="0"></iframe>
            </div>
            
            <div class="chart-container">
                <h2>Performance Radar</h2>
                <iframe src="performance_radar.html" width="100%" height="600" frameborder="0"></iframe>
            </div>
            
            <div class="chart-container">
                <h2>Loss Landscape</h2>
                <iframe src="loss_landscape_3d.html" width="100%" height="750" frameborder="0"></iframe>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Training Configuration</h3>
                    <ul>
                        <li>Batch Size: {BATCH_SIZE} (effective {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})</li>
                        <li>Learning Rate: {LEARNING_RATE}</li>
                        <li>Optimizer: SGD with Nesterov momentum</li>
                        <li>Curriculum Stages: {CURRICULUM_STAGES}</li>
                    </ul>
                </div>
                <div class="metric-card">
                    <h3>Model Architecture</h3>
                    <ul>
                        <li>Model Type: {self.model_name.upper()}</li>
                        <li>Parameters: ~{self._get_param_count()}</li>
                        <li>Mix Parameter: 0.05</li>
                        <li>Transformation Penalty: {TRANSFORMATION_PENALTY}</li>
                    </ul>
                </div>
                <div class="metric-card">
                    <h3>Best Performance</h3>
                    <ul>
                        <li>Best Exact Match: {max(self.metrics['val_exact'])}%</li>
                        <li>Best Pixel Accuracy: {max(self.metrics['val_pixel_acc'])}%</li>
                        <li>Achieved at Epoch: {self.metrics['epoch'][np.argmax(self.metrics['val_exact'])]}</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(self.report_dir, 'overview.html'), 'w') as f:
            f.write(html_content)
    
    def _get_param_count(self):
        """Get parameter count for model"""
        param_counts = {
            'minerva': '2.1M',
            'atlas': '1.2M',
            'iris': '0.9M',
            'chronos': '2.4M',
            'prometheus': '1.8M'
        }
        return param_counts.get(self.model_name, 'Unknown')
    
    def _create_coa(self):
        """Create Certificate of Authenticity"""
        model_descriptions = {
            'minerva': ('Strategic Pattern Analysis', 'Master Reasoning Engine'),
            'atlas': ('Spatial Transformation', 'Geometric Specialist'),
            'iris': ('Color Pattern Recognition', 'Visual Harmony Expert'),
            'chronos': ('Temporal Sequence Analysis', 'Evolution Predictor'),
            'prometheus': ('Creative Pattern Generation', 'Innovation Engine')
        }
        
        desc, subtitle = model_descriptions.get(self.model_name, ('Unknown', 'Unknown'))
        
        # Calculate model size
        param_count = self._get_param_count()
        model_size_mb = float(param_count.rstrip('M')) * 4  # Approximate MB for float32
        
        # Get epoch number for serial
        epoch_num = self.metrics['epoch'][-1] if self.metrics['epoch'] else 1
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Certificate of Authenticity - {self.model_name.upper()} | AutomataNexus LLC</title>
    <style>
        @page {{
            size: letter;
            margin: 0.5in;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}

        .certificate {{
            width: 8.5in;
            height: 12in;
            margin: 0 auto;
            background: white;
            position: relative;
            padding: 20px;
        }}

        .border {{
            border: 3px solid #00d4aa;
            border-radius: 15px;
            height: 100%;
            position: relative;
            padding: 20px;
        }}

        .watermark {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) rotate(-45deg);
            font-size: 120px;
            color: rgba(0, 212, 170, 0.06);
            font-weight: bold;
            z-index: 0;
        }}

        .content {{
            position: relative;
            z-index: 1;
            height: 100%;
            display: flex;
            flex-direction: column;
        }}

        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}

        .company {{
            font-size: 22px;
            font-weight: bold;
            color: #00d4aa;
            letter-spacing: 2px;
            margin-bottom: 8px;
        }}

        .title {{
            font-size: 32px;
            color: #2c3e50;
            margin-bottom: 8px;
        }}

        .subtitle {{
            font-size: 16px;
            color: #666;
            font-style: italic;
        }}

        .statement {{
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 15px;
            text-align: justify;
        }}

        .model-name {{
            font-size: 42px;
            font-weight: bold;
            color: #00d4aa;
            text-align: center;
            margin: 15px 0;
        }}

        .model-desc {{
            font-size: 18px;
            color: #666;
            text-align: center;
            font-style: italic;
            margin-bottom: 15px;
        }}

        .specs {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #00d4aa;
            margin: 15px 0;
        }}

        .specs h3 {{
            color: #00d4aa;
            margin-bottom: 8px;
            font-size: 16px;
        }}

        .specs-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
            font-size: 13px;
        }}

        ul {{
            margin: 15px 30px;
            font-size: 13px;
        }}

        li {{
            margin-bottom: 4px;
        }}

        .main-content {{
            flex: 1;
        }}

        .footer {{
            margin-top: 20px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}

        .signature {{
            text-align: left;
            flex: 1;
        }}

        .signature-name {{
            font-family: 'Brush Script MT', cursive;
            font-size: 22px;
            color: #2c3e50;
            margin-bottom: -8px;
        }}

        .signature-line {{
            border-top: 2px solid #333;
            width: 180px;
            margin: 0 0 4px 0;
        }}

        .signature-title {{
            font-size: 12px;
            line-height: 1.3;
        }}

        .seal {{
            width: 100px;
            height: 100px;
            border: 3px double #000;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: 9px;
            background: white;
            flex-shrink: 0;
        }}

        .seal-text {{
            font-weight: bold;
            color: #000;
            line-height: 0.9;
        }}

        .serial-footer {{
            text-align: center;
            font-size: 11px;
            color: #666;
            margin-top: 5px;
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="certificate">
        <div class="border">
            <div class="watermark">OLYMPUS AGI2</div>
            
            <div class="content">
                <div class="main-content">
                    <div class="header">
                        <div class="company">AUTOMATANEXUS LLC</div>
                        <div class="title">Certificate of Authenticity</div>
                        <div class="subtitle">OLYMPUS AGI2 Model Certification</div>
                    </div>

                    <p class="statement">
                        This is to certify that the following neural network model represents a genuine implementation 
                        of the OLYMPUS AGI2 ensemble system, developed and trained by AutomataNexus LLC using advanced deep 
                        learning techniques for Abstract Reasoning Corpus (ARC) task solving.
                    </p>

                    <div class="model-name">{self.model_name.upper()}</div>
                    <div class="model-desc">{desc}<br>{subtitle}</div>

                    <div class="specs">
                        <h3>Technical Specifications</h3>
                        <div class="specs-grid">
                            <div><strong>Version:</strong> v4.0</div>
                            <div><strong>Best Exact Match:</strong> {max(self.metrics['val_exact']) if self.metrics['val_exact'] else 0:.2f}%</div>
                            <div><strong>Parameters:</strong> {param_count}</div>
                            <div><strong>Model Size:</strong> ~{model_size_mb:.0f}MB</div>
                            <div><strong>Architecture:</strong> Enhanced{self.model_name.capitalize()}Net</div>
                            <div><strong>Framework:</strong> PyTorch {torch.__version__}</div>
                            <div><strong>Training Epochs:</strong> {self.metrics['epoch'][-1] if self.metrics['epoch'] else 0}</div>
                            <div><strong>Device:</strong> {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}</div>
                        </div>
                    </div>

                    <p class="statement">
                        This model has been rigorously trained using curriculum learning and validated on ARC tasks. 
                        The model demonstrates advanced reasoning capabilities including:
                    </p>
                    
                    <ul>
                        <li>{"Grid-aware attention with relational reasoning" if self.model_name == "minerva" else ""}</li>
                        <li>{"Spatial transformations with rotation/reflection prediction" if self.model_name == "atlas" else ""}</li>
                        <li>{"Color pattern recognition with attention mechanisms" if self.model_name == "iris" else ""}</li>
                        <li>{"Temporal sequence modeling with LSTM architecture" if self.model_name == "chronos" else ""}</li>
                        <li>{"VAE-based creative pattern generation" if self.model_name == "prometheus" else ""}</li>
                        <li>Mix parameter optimization (0.05) for transformation learning</li>
                        <li>Mega-scale training with batch size {BATCH_SIZE} (effective {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})</li>
                        <li>Curriculum learning across {CURRICULUM_STAGES} difficulty stages</li>
                        <li>Exact match bonus system with {EXACT_MATCH_BONUS}x reward</li>
                        <li>Final pixel accuracy: {self.metrics['val_pixel_acc'][-1] if self.metrics['val_pixel_acc'] else 0:.2f}%</li>
                    </ul>

                    <p class="statement">
                        {self.model_name.upper()} serves as a specialized component in the OLYMPUS AGI2 ensemble, 
                        contributing unique reasoning capabilities for solving abstract pattern recognition tasks.
                    </p>
                </div>

                <div class="footer">
                    <div class="signature">
                        <div class="signature-name">Andrew Jewell Sr.</div>
                        <div class="signature-line"></div>
                        <div class="signature-title">
                            Andrew Jewell Sr.<br>
                            Founder & AI Systems Engineer<br>
                            Date: {datetime.now().strftime('%B %d, %Y')}
                        </div>
                    </div>

                    <div class="seal">
                        <div class="seal-text">
                            AUTOMATANEXUS<br>
                            ★ OLYMPUS ★<br>
                            AGI2<br>
                            {datetime.now().year}
                        </div>
                    </div>
                </div>

                <div class="serial-footer">
                    Model Serial: ANX-{self.model_name.upper()}-{datetime.now().year}-{epoch_num:03d}
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
        
        with open(os.path.join(self.report_dir, 'COA.html'), 'w') as f:
            f.write(html_content)
    
    def _get_model_features(self):
        """Get model-specific features"""
        features = {
            'minerva': 'Grid attention, relational reasoning, pattern memory bank',
            'atlas': 'Spatial transformer network, rotation/reflection prediction',
            'iris': 'Color attention, mapping matrices, rule learning',
            'chronos': 'LSTM temporal reasoning, movement prediction, sequence modeling',
            'prometheus': 'VAE architecture, latent space manipulation, creative generation'
        }
        return features.get(self.model_name, 'Unknown')


def train_megascale_curriculum():
    """V4 Mega-scale training with curriculum learning and MEPT"""
    print("\n🚀 Starting V4 MEGA-SCALE + CURRICULUM Training")
    print("="*60)
    
    # Create models
    models = create_enhanced_models()
    
    # Initialize MEPT components if enabled
    if USE_MEPT:
        print("\n🧠 Initializing MEPT (Memory-Enhanced Progressive Training)")
        replay_buffer = ExperienceReplayBuffer(capacity=100000)
        pattern_bank = PatternBank(max_patterns=20000)
        loss_fn = MEPTLoss(replay_buffer, pattern_bank, use_mept=True)
        print(f"✅ MEPT initialized with:")
        print(f"   Replay buffer capacity: {replay_buffer.capacity}")
        print(f"   Pattern bank capacity: {pattern_bank.max_patterns}")
        print(f"   Dynamic loss weighting enabled")
    else:
        # Fallback to regular loss if MEPT disabled
        replay_buffer = ExperienceReplayBuffer(capacity=1)  # Minimal buffer
        pattern_bank = PatternBank(max_patterns=1)
        loss_fn = MEPTLoss(replay_buffer, pattern_bank, use_mept=False)
    
    # Initialize LEAP components if enabled
    if USE_LEAP:
        print("\n🎯 Initializing LEAP (Learning Enhancement through Adaptive Patterns)")
        leap_trainer = LEAPTrainer(device=device)
        print("✅ LEAP initialized for Stage 0 adaptive pattern training")
    
    # Initialize program synthesizer
    synthesizer = LightweightProgramSynthesizer()
    synthesis_stats = {
        'total_attempts': 0,
        'successful_syntheses': 0,
        'exact_improvements': 0
    }
    
    os.makedirs('/content/AutomataNexus_Olympus_AGI2/arc_models_v4', exist_ok=True)
    
    # Train all models
    for model_name, model in models.items():
            
        print(f"\n{'='*60}")
        print(f"🧠 Training {model_name.upper()} - MEGA-SCALE + CURRICULUM")
        print(f"{'='*60}")
        
        model = model.to(device)
        
        # Initialize reporter for this model
        reporter = TrainingReporter(model_name)
        
        # Stage-adaptive optimizer with better scaling for Stage 1
        # Use higher LR for Stage 0 to learn exact patterns faster
        # Lower LR for Stage 1 to prevent overfitting
        stage_lrs = [LEARNING_RATE, LEARNING_RATE * 0.5, LEARNING_RATE * 0.2]  # Much safer
        
        optimizer = optim.SGD(
            model.parameters(), 
            lr=stage_lrs[0],  # Start with Stage 0 LR
            momentum=0.9,
            weight_decay=0.0005,  # Increased weight decay
            nesterov=True
        )
        
        # Cosine annealing across all stages
        total_epochs = EPOCHS_PER_STAGE * CURRICULUM_STAGES
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        
        scaler = GradScaler('cuda')
        
        best_exact = 0
        best_val_loss = float('inf')
        global_epoch = 0
        patience_counter = 0
        max_patience = 20  # Early stopping patience
        
        # Check for existing checkpoint and resume if found
        best_checkpoint_path = f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/{model_name}_best.pt'
        latest_checkpoint_path = f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/{model_name}_checkpoint.pt'
        resume_stage = 0
        resume_epoch = 0
        
        # Try to load the best checkpoint first, then the latest checkpoint
        checkpoint_path = None
        if os.path.exists(best_checkpoint_path):
            checkpoint_path = best_checkpoint_path
            print(f"📂 Found best checkpoint for {model_name.upper()}, loading...")
        elif os.path.exists(latest_checkpoint_path):
            checkpoint_path = latest_checkpoint_path
            print(f"📂 Found latest checkpoint for {model_name.upper()}, loading...")
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model and optimizer states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Resume from checkpoint values
            global_epoch = checkpoint.get('epoch', checkpoint.get('global_epoch', 0))
            resume_stage = checkpoint.get('stage', checkpoint.get('curriculum_stage', 0))
            best_exact = checkpoint.get('val_exact', checkpoint.get('best_exact', 0))
            best_val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', float('inf')))
            
            # Calculate resume position in curriculum
            # Account for completed stages
            completed_epochs_in_previous_stages = resume_stage * EPOCHS_PER_STAGE
            resume_epoch = global_epoch - completed_epochs_in_previous_stages
            
            print(f"✅ Resumed from epoch {global_epoch} (Stage {resume_stage}, Epoch {resume_epoch})")
            print(f"   Best exact match: {best_exact:.2f}%")
            print(f"   Best val loss: {best_val_loss:.4f}")
            
            # Adjust scheduler state
            for _ in range(global_epoch):
                scheduler.step()
        else:
            print(f"🆕 No checkpoint found for {model_name.upper()}, starting fresh")
        
        # EXACT MATCH PRE-TRAINING for Stage 0 - Skip if resuming from a later stage
        if EXACT_BOOST_AVAILABLE and model_name in ['minerva', 'atlas', 'iris'] and resume_stage == 0 and global_epoch == 0:
            print(f"\n🎯 Running EXACT MATCH INJECTION for {model_name.upper()}")
            print("="*40)
            model = inject_exact_match_training(model, device=device, num_epochs=50)  # More epochs for higher %
            print("✅ Exact match injection complete!")
        
        # CURRICULUM LOOP - Resume from the correct stage
        for stage in range(resume_stage, CURRICULUM_STAGES):
            print(f"\n📚 Starting Curriculum Stage {stage}")
            print("="*40)
            
            # Adjust learning rate for each stage
            if stage > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = stage_lrs[stage]
                print(f"Learning rate adjusted to: {stage_lrs[stage]}")
            
            # Create dataset for this stage with ARC synthesis
            dataset = CurriculumMegaScaleDataset(
                DATA_DIR, 
                curriculum_stage=stage,
                use_arc_synthesis=True,
                synthesis_ratio=0.4 if stage == 0 else 0.3  # More synthesis for Stage 0
            )
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Initialize augmenter for this stage
            arc_augmenter = ARCDataAugmenter(device=device)
            
            # Custom collate function to handle different grid sizes
            def custom_collate_fn(batch):
                # Find max dimensions
                max_h_in = max(item['inputs'].shape[0] for item in batch)
                max_w_in = max(item['inputs'].shape[1] for item in batch)
                max_h_out = max(item['outputs'].shape[0] for item in batch)
                max_w_out = max(item['outputs'].shape[1] for item in batch)
                max_h = max(max_h_in, max_h_out, MAX_GRID_SIZE)
                max_w = max(max_w_in, max_w_out, MAX_GRID_SIZE)
                
                # Pad all grids to max size
                inputs = []
                outputs = []
                for item in batch:
                    # Pad inputs
                    h_in, w_in = item['inputs'].shape
                    pad_h = max_h - h_in
                    pad_w = max_w - w_in
                    padded_input = F.pad(item['inputs'], (0, pad_w, 0, pad_h), value=0)
                    
                    # Pad outputs
                    h_out, w_out = item['outputs'].shape
                    pad_h = max_h - h_out
                    pad_w = max_w - w_out
                    padded_output = F.pad(item['outputs'], (0, pad_w, 0, pad_h), value=0)
                    
                    inputs.append(padded_input)
                    outputs.append(padded_output)
                
                return {
                    'inputs': torch.stack(inputs),
                    'outputs': torch.stack(outputs)
                }
            
            # High-performance dataloaders with custom collate
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH_FACTOR,
                persistent_workers=True,
                collate_fn=custom_collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH_FACTOR,
                persistent_workers=True,
                collate_fn=custom_collate_fn
            )
            
            print(f"Stage {stage} - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
            print(f"Batches per epoch: {len(train_loader):,}")
            
            # Apply MEPT augmentation if enabled and buffer has samples
            if USE_MEPT and replay_buffer.get_stats()['total_experiences'] > 100:
                print(f"🔄 Applying MEPT augmentation with replay buffer...")
                print(f"   Buffer stats: {replay_buffer.get_stats()}")
                
                # Create augmented dataset
                augmented_dataset = MEPTAugmentedDataset(
                    train_dataset,
                    replay_buffer,
                    replay_ratio=0.3 if stage == 0 else 0.2  # More replay in early stages
                )
                
                # Recreate train loader with augmented dataset
                train_loader = DataLoader(
                    augmented_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                    prefetch_factor=PREFETCH_FACTOR,
                    persistent_workers=True,
                    collate_fn=custom_collate_fn
                )
            
            # Train for this stage - Resume from the correct epoch if needed
            start_epoch = resume_epoch if stage == resume_stage else 0
            for epoch in range(start_epoch, EPOCHS_PER_STAGE):
                # Don't increment global_epoch if we're resuming at the exact epoch
                if not (stage == resume_stage and epoch == start_epoch and global_epoch > 0):
                    global_epoch += 1
                
                # Training
                model.train()
                train_metrics = {'loss': 0, 'exact': 0, 'samples': 0}
                
                # Create exact match dataset for Stage 0 injection
                if stage == 0 and EXACT_BOOST_AVAILABLE:
                    exact_dataset = ExactMatchBoostDataset(1000, fixed_size=6)  # Use 6x6 for training
                    aggressive_loss = AggressiveLoss()
                
                pbar = tqdm(train_loader, desc=f"Stage {stage}, Epoch {epoch+1}/{EPOCHS_PER_STAGE}")
                optimizer.zero_grad()
                
                # LEAP integration for Stage 0
                leap_batch_counter = 0
                
                for batch_idx, batch in enumerate(pbar):
                    # EXACT MATCH INJECTION for Stage 0 - every 5th batch
                    if stage == 0 and EXACT_BOOST_AVAILABLE and batch_idx % 5 == 0:
                        # Replace with exact match batch
                        exact_batch_size = min(64, BATCH_SIZE)  # Smaller batches for exact match
                        exact_indices = random.sample(range(len(exact_dataset)), exact_batch_size)
                        
                        # Build exact match tensors
                        exact_inputs = []
                        exact_outputs = []
                        for idx in exact_indices:
                            sample = exact_dataset[idx]
                            exact_inputs.append(sample['input'])
                            exact_outputs.append(sample['output'])
                        
                        # Convert to tensors and one-hot
                        exact_inputs = torch.tensor(np.stack(exact_inputs), device=device)
                        exact_outputs = torch.tensor(np.stack(exact_outputs), device=device)
                        
                        # One-hot encode and pad to MAX_GRID_SIZE
                        B, H, W = exact_inputs.shape
                        # Pad to MAX_GRID_SIZE if needed
                        if H < MAX_GRID_SIZE or W < MAX_GRID_SIZE:
                            pad_h = MAX_GRID_SIZE - H
                            pad_w = MAX_GRID_SIZE - W
                            exact_inputs = F.pad(exact_inputs, (0, pad_w, 0, pad_h), value=0)
                            exact_outputs = F.pad(exact_outputs, (0, pad_w, 0, pad_h), value=0)
                        
                        input_grids = F.one_hot(exact_inputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        output_grids = F.one_hot(exact_outputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        
                        # Use aggressive loss for exact match batches
                        use_aggressive_loss = True
                    else:
                        # Regular batch - handle both key formats
                        if 'inputs' in batch:
                            inputs = batch['inputs'].to(device, non_blocking=True)
                            outputs = batch['outputs'].to(device, non_blocking=True)
                        else:
                            inputs = batch['input'].to(device, non_blocking=True)
                            outputs = batch['output'].to(device, non_blocking=True)
                        
                        # Validate ranges
                        if inputs.max() >= 10 or inputs.min() < 0:
                            print(f"WARNING: Invalid input values! Range: {inputs.min()}-{inputs.max()}")
                            inputs = torch.clamp(inputs, 0, 9)
                        if outputs.max() >= 10 or outputs.min() < 0:
                            print(f"WARNING: Invalid output values! Range: {outputs.min()}-{outputs.max()}")
                            outputs = torch.clamp(outputs, 0, 9)
                        
                        # Convert to one-hot
                        input_grids = F.one_hot(inputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        output_grids = F.one_hot(outputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        
                        # Apply ARC augmentation for Stage 0 (more augmentation for exact match training)
                        if stage == 0 and random.random() < 0.3:
                            # Note: augmenter expects raw indices, not one-hot
                            inputs_aug, outputs_aug = arc_augmenter.augment_batch(
                                inputs, outputs, augment_prob=0.5
                            )
                            input_grids = F.one_hot(inputs_aug, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                            output_grids = F.one_hot(outputs_aug, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        
                        use_aggressive_loss = False
                    
                    with autocast('cuda'):
                        # Handle CHRONOS differently - it expects a list of tensors
                        if model_name == 'chronos':
                            outputs = model([input_grids], target=output_grids)  # Pass as single-frame sequence with target
                            pred_output = outputs['predicted_output']
                        else:
                            outputs = model(input_grids, output_grids, mode='train')
                            pred_output = outputs['predicted_output']
                        
                        # Use appropriate loss function
                        if use_aggressive_loss and stage == 0 and EXACT_BOOST_AVAILABLE:
                            losses = aggressive_loss(pred_output, output_grids, input_grids)
                        else:
                            losses = loss_fn(pred_output, output_grids, input_grids)
                        
                        loss = losses['total'] / GRADIENT_ACCUMULATION_STEPS
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # More conservative clipping
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    # Update metrics
                    train_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                    train_metrics['exact'] += losses['exact_count'].item()
                    train_metrics['samples'] += input_grids.size(0)
                    
                    pbar.set_postfix({
                        'loss': f"{losses['total'].item():.3f}",
                        'exact': f"{losses['exact_count'].item():.0f}",
                        'trans': f"{losses['transformation'].item():.2f}"
                    })
                    
                    # LEAP training for Stage 0 - adaptive pattern injection
                    if USE_LEAP and stage == 0 and batch_idx % 3 == 0:  # Every 3rd batch
                        leap_batch = leap_trainer.generate_leap_batch(batch_size=64)
                        leap_inputs = leap_batch['inputs'].to(device)
                        leap_outputs = leap_batch['outputs'].to(device)
                        pattern_types = leap_batch['pattern_types']
                        
                        # Pad to MAX_GRID_SIZE if needed
                        if leap_inputs.shape[1] < MAX_GRID_SIZE or leap_inputs.shape[2] < MAX_GRID_SIZE:
                            pad_h = MAX_GRID_SIZE - leap_inputs.shape[1]
                            pad_w = MAX_GRID_SIZE - leap_inputs.shape[2]
                            leap_inputs = F.pad(leap_inputs, (0, pad_w, 0, pad_h), value=0)
                            leap_outputs = F.pad(leap_outputs, (0, pad_w, 0, pad_h), value=0)
                        
                        # Convert to one-hot
                        leap_input_oh = F.one_hot(leap_inputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        leap_output_oh = F.one_hot(leap_outputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                        
                        with autocast('cuda'):
                            # Forward pass
                            if model_name == 'chronos':
                                leap_pred = model([leap_input_oh], target=leap_output_oh)['predicted_output']
                            else:
                                leap_pred = model(leap_input_oh, leap_output_oh, mode='train')['predicted_output']
                            
                            # Compute loss with higher exact match bonus for LEAP patterns
                            leap_losses = loss_fn(leap_pred, leap_output_oh, leap_input_oh)
                            leap_loss = leap_losses['total'] / GRADIENT_ACCUMULATION_STEPS
                        
                        # Backward pass
                        scaler.scale(leap_loss).backward()
                        
                        # Update pattern statistics
                        leap_trainer.update_pattern_stats(pattern_types, leap_pred, leap_output_oh)
                        leap_batch_counter += 1
                        
                        # Apply gradients more frequently for LEAP
                        if leap_batch_counter % 2 == 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                
                # Step scheduler after optimizer step (moved from here)
                
                # Validation every 5 epochs
                if epoch % 5 == 0:
                    model.eval()
                    val_metrics = {'loss': 0, 'exact': 0, 'pixel_acc': 0, 'samples': 0}
                    synthesis_metrics = {'attempts': 0, 'successes': 0, 'exact_via_synthesis': 0}
                    
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc="Validation"):
                            inputs = batch['inputs'].to(device, non_blocking=True)
                            outputs = batch['outputs'].to(device, non_blocking=True)
                            
                            # Validate ranges
                            if inputs.max() >= 10 or inputs.min() < 0:
                                inputs = torch.clamp(inputs, 0, 9)
                            if outputs.max() >= 10 or outputs.min() < 0:
                                outputs = torch.clamp(outputs, 0, 9)
                            
                            # Convert to one-hot
                            input_grids = F.one_hot(inputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                            output_grids = F.one_hot(outputs, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
                            
                            with autocast('cuda'):
                                # Handle CHRONOS differently - it expects a list of tensors
                                if model_name == 'chronos':
                                    model_outputs = model([input_grids])  # Pass as single-frame sequence
                                else:
                                    model_outputs = model(input_grids)
                                pred_output = model_outputs['predicted_output']
                                losses = loss_fn(pred_output, output_grids, input_grids)
                            
                            # Metrics
                            pred_indices = pred_output.argmax(dim=1)
                            target_indices = output_grids.argmax(dim=1)
                            
                            exact = (pred_indices == target_indices).all(dim=[1,2]).sum().item()
                            pixel_acc = (pred_indices == target_indices).float().mean().item()
                            
                            val_metrics['loss'] += losses['total'].item() * input_grids.size(0)
                            val_metrics['exact'] += exact
                            val_metrics['pixel_acc'] += pixel_acc * input_grids.size(0)
                            val_metrics['samples'] += input_grids.size(0)
                            
                            # Try program synthesis on a subset of validation samples
                            if synthesis_metrics['attempts'] < 50:  # Limit for speed across all stages
                                for i in range(min(5, input_grids.size(0))):
                                    synthesis_metrics['attempts'] += 1
                                    input_np = inputs[i].cpu().numpy()
                                    output_np = outputs[i].cpu().numpy()
                                    
                                    program = synthesizer.quick_synthesize(input_np, output_np, max_depth=2)
                                    if program and program.verified:
                                        synthesis_metrics['successes'] += 1
                                        # Check if neural network failed but synthesis succeeded
                                        if not (pred_indices[i] == target_indices[i]).all():
                                            synthesis_metrics['exact_via_synthesis'] += 1
                    
                    # Calculate averages
                    train_loss = train_metrics['loss'] / train_metrics['samples']
                    train_exact_pct = train_metrics['exact'] / train_metrics['samples'] * 100
                    
                    val_loss = val_metrics['loss'] / val_metrics['samples']
                    val_exact_pct = val_metrics['exact'] / val_metrics['samples'] * 100
                    val_pixel_acc = val_metrics['pixel_acc'] / val_metrics['samples'] * 100
                    
                    print(f"\nGlobal Epoch {global_epoch} (Stage {stage}): "
                          f"Train Loss: {train_loss:.4f}, Train Exact: {train_exact_pct:.2f}%")
                    print(f"Val Loss: {val_loss:.4f}, Val Exact: {val_exact_pct:.2f}%, Pixel: {val_pixel_acc:.2f}%")
                    
                    # Report MEPT status if enabled
                    if USE_MEPT:
                        buffer_stats = replay_buffer.get_stats()
                        print(f"📊 MEPT Status:")
                        print(f"   Experience Buffer: {buffer_stats['total_experiences']:,} samples "
                              f"({buffer_stats['exact_matches']:,} exact matches)")
                        print(f"   Pattern Bank: {len(pattern_bank.patterns):,} unique patterns")
                        print(f"   Loss weights: {loss_fn.weights}")
                    
                    # Report LEAP status if enabled and in Stage 0
                    if USE_LEAP and stage == 0:
                        leap_report = leap_trainer.get_performance_report()
                        if leap_report:
                            print(leap_report)
                    
                    # Report synthesis results if any
                    if synthesis_metrics['attempts'] > 0:
                        synthesis_success_rate = synthesis_metrics['successes'] / synthesis_metrics['attempts'] * 100
                        print(f"🔧 Program Synthesis: {synthesis_success_rate:.1f}% success rate, "
                              f"{synthesis_metrics['exact_via_synthesis']} additional exact matches")
                        synthesis_stats['total_attempts'] += synthesis_metrics['attempts']
                        synthesis_stats['successful_syntheses'] += synthesis_metrics['successes']
                        synthesis_stats['exact_improvements'] += synthesis_metrics['exact_via_synthesis']
                    
                    # Add metrics to reporter
                    reporter.add_metrics(
                        epoch=global_epoch,
                        stage=stage,
                        train_loss=train_loss,
                        train_exact=train_exact_pct,
                        val_loss=val_loss,
                        val_exact=val_exact_pct,
                        val_pixel_acc=val_pixel_acc,
                        lr=optimizer.param_groups[0]['lr'],
                        trans_penalty=TRANSFORMATION_PENALTY
                    )
                    
                    # Early stopping and model saving
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= max_patience and stage > 0:
                            print(f"⚠️ Early stopping triggered! Val loss not improving for {max_patience} validations")
                            break
                
                # Step scheduler at end of epoch (after optimizer.step())
                scheduler.step()
                    
                    # Save best model based on exact match
                    if val_exact_pct > best_exact:
                        best_exact = val_exact_pct
                        torch.save({
                            'epoch': global_epoch,
                            'global_epoch': global_epoch,
                            'stage': stage,
                            'curriculum_stage': stage,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_exact': val_exact_pct,
                            'best_exact': val_exact_pct,
                            'val_pixel_acc': val_pixel_acc,
                            'val_loss': val_loss,
                            'best_val_loss': best_val_loss,
                            'model_name': model_name,
                            'transformation_penalty': TRANSFORMATION_PENALTY,
                            'exact_match_bonus': EXACT_MATCH_BONUS
                        }, f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/{model_name}_best.pt')
                        
                        print(f"✅ New best model! Exact: {val_exact_pct:.2f}%")
                        
                        # Log milestone achievements
                        if val_exact_pct >= 10.0 and val_exact_pct == best_exact:
                            print(f"🎉 Milestone: {val_exact_pct:.2f}% exact match!")
                    
                    # Save periodic checkpoint for resume capability
                    torch.save({
                        'epoch': global_epoch,
                        'global_epoch': global_epoch,
                        'stage': stage,
                        'curriculum_stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_exact': val_exact_pct,
                        'best_exact': best_exact,
                        'val_pixel_acc': val_pixel_acc,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'model_name': model_name,
                        'transformation_penalty': TRANSFORMATION_PENALTY,
                        'exact_match_bonus': EXACT_MATCH_BONUS
                    }, f'/content/AutomataNexus_Olympus_AGI2/arc_models_v4/{model_name}_checkpoint.pt')
                    
                    # Warning for exploding validation loss
                    if val_loss > 10.0:
                        print(f"⚠️ Warning: Validation loss is very high ({val_loss:.2f}), possible overfitting!")
                        # Reduce learning rate on explosion
                        if stage > 0:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            print(f"   Reduced learning rate to: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Check if early stopping was triggered
            if patience_counter >= max_patience and stage > 0:
                print(f"📛 Stage {stage} terminated early due to overfitting")
                break
            
            # Clear resume_epoch after first stage
            if stage == resume_stage:
                resume_epoch = 0
        
        # Generate comprehensive report
        print(f"\n📊 Generating training report for {model_name.upper()}...")
        reporter.generate_report({
            'best_exact': best_exact,
            'best_val_loss': best_val_loss,
            'total_epochs': global_epoch
        })
        print(f"✅ Report saved to: /content/AutomataNexus_Olympus_AGI2/src/models/reports/{model_name}/")
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n🎉 V4 MEGA-SCALE + CURRICULUM Training complete!")
    
    # Report overall synthesis statistics
    if synthesis_stats['total_attempts'] > 0:
        overall_success_rate = synthesis_stats['successful_syntheses'] / synthesis_stats['total_attempts'] * 100
        print(f"\n📊 Program Synthesis Summary:")
        print(f"  Total synthesis attempts: {synthesis_stats['total_attempts']}")
        print(f"  Successful syntheses: {synthesis_stats['successful_syntheses']} ({overall_success_rate:.1f}%)")
        print(f"  Additional exact matches via synthesis: {synthesis_stats['exact_improvements']}")
        print(f"  Synthesis cache size: {len(synthesizer.synthesis_cache)} programs")


if __name__ == "__main__":
    print("="*80)
    print("AUTOMATANEXUS OLYMPUS AGI2 - V4 MEGA-SCALE + CURRICULUM TRAINING")
    print("="*80)
    print("Training 5 specialized models with curriculum learning")
    print("Batch size: 512 (effective 2048 with gradient accumulation)")
    print("Models: MINERVA, ATLAS, IRIS, CHRONOS, PROMETHEUS")
    print("="*80)
    
    train_megascale_curriculum()