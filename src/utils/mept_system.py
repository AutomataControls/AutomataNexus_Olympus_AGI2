"""
MEPT (Memory-Enhanced Progressive Training) System
Prevents catastrophic forgetting through experience replay and pattern banking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


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
                 pattern_bank: PatternBank, use_mept: bool = True,
                 transformation_penalty: float = 2.0,
                 exact_match_bonus: float = 5.0):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.pattern_bank = pattern_bank
        self.use_mept = use_mept
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # Loss configuration
        self.transformation_penalty = transformation_penalty
        self.exact_match_bonus = exact_match_bonus
        
        # Dynamic weights that adapt during training
        self.weights = {
            'reconstruction': 1.0,
            'exact_match': exact_match_bonus,  # Use the passed parameter
            'consistency': 0.5,
            'diversity': 0.0,  # DISABLED - we want EXACT matches for ARC!
            'memory_alignment': 0.1  # Reduced - might interfere with learning
        }
        
        # Track performance for dynamic adjustment
        self.performance_history = deque(maxlen=100)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute MEPT loss with all components"""
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
            
            # Focal loss modification for hard examples - but more stable
            # Use alpha=0.25 and gamma=2.0 for stability
            alpha = 0.25
            gamma = 2.0
            pt = torch.exp(-ce_loss)
            focal_loss = alpha * ((1 - pt) ** gamma) * ce_loss
            reconstruction_loss = focal_loss.reshape(B, H, W).mean(dim=[1,2])
            
            # 2. Softer exact match with IoU-based scoring (IRIS improvement)
            exact_matches_strict = (pred_indices == target_indices).all(dim=[1,2]).float()
            
            # IoU-based soft exact match for better learning
            intersection = (pred_indices == target_indices).float().sum(dim=[1,2])
            union = (pred_indices.shape[1] * pred_indices.shape[2]) # Total pixels
            iou_scores = intersection / union
            
            # Combine strict and soft matches (weighted towards IoU for learning)
            combined_matches = 0.3 * exact_matches_strict + 0.7 * iou_scores
            exact_bonus = -combined_matches * self.weights['exact_match']
            
            # 3. Diversity loss - encourage diverse predictions
            pred_entropy = -(pred.softmax(dim=1) * pred.log_softmax(dim=1)).sum(dim=1).mean(dim=[1,2])
            diversity_loss = -pred_entropy  # Negative because we want to maximize entropy
            
            # 4. Memory alignment loss - align with successful past examples
            memory_loss = self._calculate_memory_loss(pred, input_indices, target_indices)
            
            # 5. Smart transformation penalty - task-aware (IRIS improvement)
            is_identity_task = (input_indices == target_indices).all(dim=[1,2]).float()
            same_as_input = (pred_indices == input_indices).float().mean(dim=[1,2])
            
            # Reduce penalty when making progress towards target
            progress_factor = 1.0 - iou_scores  # Less penalty when IoU is high
            transformation_penalty = same_as_input * (1 - is_identity_task) * self.transformation_penalty * progress_factor
            
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
            
            # Update pattern bank with high-quality matches (strict + soft)
            for i in range(B):
                if exact_matches_strict[i] > 0.5 or combined_matches[i] > 0.8:
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
                'exact_count': exact_matches_strict.sum(),
                'soft_exact_count': combined_matches.sum(),
                'avg_iou': iou_scores.mean()
            }
        else:
            # Fall back to regular loss behavior
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
        reconstruction_loss = ce_loss.mean(dim=[1,2])
        
        # Exact match bonus
        exact_matches = (pred_indices == target_indices).all(dim=[1,2]).float()
        exact_bonus = -exact_matches * self.exact_match_bonus
        
        # Transformation penalty
        is_identity_task = (input_indices == target_indices).all(dim=[1,2]).float()
        same_as_input = (pred_indices == input_indices).float().mean(dim=[1,2])
        transformation_penalty = same_as_input * (1 - is_identity_task) * self.transformation_penalty
        
        total_loss = reconstruction_loss + transformation_penalty + exact_bonus
        
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
    
    def _adjust_weights(self):
        """Dynamically adjust weights based on performance"""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = np.mean(list(self.performance_history)[-10:])
        
        # If exact match rate is low, increase exact match weight
        # But respect the original configured value as the base
        if recent_performance < 0.1:
            # Can go up to 1.5x the configured value
            max_exact_weight = self.exact_match_bonus * 1.5
            self.weights['exact_match'] = min(max_exact_weight, self.weights['exact_match'] * 1.05)
            # Keep diversity at 0 for ARC - we need exact matches!
            self.weights['diversity'] = 0.0
        
        # If exact match rate is improving, balance weights
        elif recent_performance > 0.3:
            # Don't go below 0.8x the configured value
            min_exact_weight = self.exact_match_bonus * 0.8
            self.weights['exact_match'] = max(min_exact_weight, self.weights['exact_match'] * 0.95)
            self.weights['memory_alignment'] = min(1.0, self.weights['memory_alignment'] * 1.05)


class MEPTAugmentedDataset(torch.utils.data.Dataset):
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
        if random.random() < self.replay_ratio and self.replay_buffer.get_stats()['total_experiences'] > 0:
            experiences = self.replay_buffer.sample(1, exact_ratio=0.7)
            if experiences:
                exp = experiences[0]
                return {
                    'inputs': exp['input'],
                    'outputs': exp['output']
                }
        
        # Otherwise return regular sample
        return self.base_dataset[idx]


def create_mept_system(capacity: int = 100000, pattern_bank_size: int = 20000,
                      transformation_penalty: float = 2.0, exact_match_bonus: float = 5.0):
    """Create a complete MEPT system"""
    replay_buffer = ExperienceReplayBuffer(capacity=capacity)
    pattern_bank = PatternBank(max_patterns=pattern_bank_size)
    loss_fn = MEPTLoss(
        replay_buffer, 
        pattern_bank, 
        use_mept=True,
        transformation_penalty=transformation_penalty,
        exact_match_bonus=exact_match_bonus
    )
    
    return {
        'replay_buffer': replay_buffer,
        'pattern_bank': pattern_bank,
        'loss_fn': loss_fn
    }