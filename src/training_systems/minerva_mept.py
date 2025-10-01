"""
MINERVA-specific MEPT (Memory-Enhanced Pattern Training) System
Focuses on storing and retrieving grid-based patterns and spatial relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
import heapq
import hashlib


class MinervaExperienceReplayBuffer:
    """Grid-pattern focused experience replay buffer for MINERVA"""
    
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.exact_matches = deque(maxlen=capacity // 2)
        
        # MINERVA-specific categorization
        self.grid_size_buckets = defaultdict(list)  # Organize by grid size
        self.pattern_type_buckets = defaultdict(list)  # Organize by pattern type
        self.symmetry_bucket = []  # Special bucket for symmetric patterns
        self.object_bucket = []  # Special bucket for object-based patterns
        
        # Priority scores for different pattern characteristics
        self.pattern_priorities = {
            'exact_match': 5.0,
            'symmetric': 3.0,
            'has_objects': 2.5,
            'complex_transformation': 2.0,
            'simple': 1.0
        }
    
    def add(self, input_grid: torch.Tensor, output_grid: torch.Tensor, 
            predicted_grid: torch.Tensor, loss: float, is_exact: bool = False,
            pattern_info: Optional[Dict] = None):
        """Add experience with MINERVA-specific metadata"""
        
        # Analyze grid characteristics
        grid_h, grid_w = input_grid.shape[-2:]
        has_symmetry = self._check_symmetry(input_grid)
        has_objects = self._detect_objects(input_grid)
        transformation_complexity = self._analyze_transformation(input_grid, output_grid)
        
        experience = {
            'input': input_grid.cpu(),
            'output': output_grid.cpu(),
            'predicted': predicted_grid.cpu(),
            'loss': loss,
            'is_exact': is_exact,
            'grid_size': (grid_h, grid_w),
            'has_symmetry': has_symmetry,
            'has_objects': has_objects,
            'transformation_complexity': transformation_complexity,
            'pattern_info': pattern_info or {},
            'priority': self._calculate_priority(is_exact, has_symmetry, has_objects, transformation_complexity)
        }
        
        self.buffer.append(experience)
        
        # Add to specialized buckets
        grid_key = f"{grid_h}x{grid_w}"
        self.grid_size_buckets[grid_key].append(len(self.buffer) - 1)
        
        if is_exact:
            self.exact_matches.append(experience)
        
        if has_symmetry:
            self.symmetry_bucket.append(len(self.buffer) - 1)
            
        if has_objects:
            self.object_bucket.append(len(self.buffer) - 1)
    
    def _check_symmetry(self, grid: torch.Tensor) -> bool:
        """Check if grid has symmetry properties"""
        if grid.dim() == 4:
            grid = grid.argmax(dim=1).squeeze(0)
        elif grid.dim() == 3:
            grid = grid.squeeze(0)
        
        grid_np = grid.cpu().numpy()
        
        # Check horizontal and vertical symmetry
        h_symmetric = np.array_equal(grid_np, np.fliplr(grid_np))
        v_symmetric = np.array_equal(grid_np, np.flipud(grid_np))
        
        return h_symmetric or v_symmetric
    
    def _detect_objects(self, grid: torch.Tensor) -> bool:
        """Detect if grid contains distinct objects"""
        if grid.dim() == 4:
            grid = grid.argmax(dim=1).squeeze(0)
        elif grid.dim() == 3:
            grid = grid.squeeze(0)
        
        unique_values = torch.unique(grid)
        
        # Simple heuristic: more than 2 colors likely indicates objects
        return len(unique_values) > 2
    
    def _analyze_transformation(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> str:
        """Analyze transformation complexity"""
        # Convert to indices if needed
        if input_grid.dim() == 4:
            input_grid = input_grid.argmax(dim=1)
            output_grid = output_grid.argmax(dim=1)
        
        # Check for common transformations
        if torch.equal(input_grid, output_grid):
            return 'identity'
        elif torch.equal(input_grid, torch.rot90(output_grid, k=1, dims=[-2, -1])):
            return 'rotation'
        elif torch.equal(input_grid, torch.flip(output_grid, dims=[-1])):
            return 'flip'
        else:
            # Check color changes
            input_colors = torch.unique(input_grid)
            output_colors = torch.unique(output_grid)
            
            if len(output_colors) > len(input_colors):
                return 'color_addition'
            elif not torch.equal(input_colors, output_colors):
                return 'color_transformation'
            else:
                return 'complex'
    
    def _calculate_priority(self, is_exact: bool, has_symmetry: bool, 
                          has_objects: bool, transformation: str) -> float:
        """Calculate priority score for experience"""
        priority = self.pattern_priorities['simple']
        
        if is_exact:
            priority = max(priority, self.pattern_priorities['exact_match'])
        if has_symmetry:
            priority = max(priority, self.pattern_priorities['symmetric'])
        if has_objects:
            priority = max(priority, self.pattern_priorities['has_objects'])
        if transformation in ['complex', 'color_transformation']:
            priority = max(priority, self.pattern_priorities['complex_transformation'])
        
        return priority
    
    def sample(self, batch_size: int, strategy: str = 'mixed') -> List[Dict]:
        """Sample experiences with MINERVA-specific strategies"""
        if len(self.buffer) == 0:
            return []
        
        if strategy == 'mixed':
            # Mix different sampling strategies
            samples = []
            
            # 40% exact matches
            exact_count = int(batch_size * 0.4)
            if self.exact_matches:
                exact_samples = np.random.choice(
                    list(self.exact_matches),
                    size=min(exact_count, len(self.exact_matches)),
                    replace=False
                )
                samples.extend(exact_samples)
            
            # 20% symmetric patterns
            sym_count = int(batch_size * 0.2)
            if self.symmetry_bucket:
                sym_indices = np.random.choice(
                    self.symmetry_bucket,
                    size=min(sym_count, len(self.symmetry_bucket)),
                    replace=False
                )
                samples.extend([self.buffer[i] for i in sym_indices])
            
            # 20% object-based patterns
            obj_count = int(batch_size * 0.2)
            if self.object_bucket:
                obj_indices = np.random.choice(
                    self.object_bucket,
                    size=min(obj_count, len(self.object_bucket)),
                    replace=False
                )
                samples.extend([self.buffer[i] for i in obj_indices])
            
            # Fill remaining with priority sampling
            remaining = batch_size - len(samples)
            if remaining > 0:
                # Priority-based sampling
                priorities = [exp['priority'] for exp in self.buffer]
                indices = np.random.choice(
                    len(self.buffer),
                    size=min(remaining, len(self.buffer)),
                    replace=False,
                    p=np.array(priorities) / sum(priorities)
                )
                samples.extend([self.buffer[i] for i in indices])
            
            return samples[:batch_size]
        
        elif strategy == 'grid_size':
            # Sample patterns of similar grid sizes
            grid_sizes = list(self.grid_size_buckets.keys())
            if not grid_sizes:
                return self.sample(batch_size, strategy='mixed')
            
            # Focus on specific grid size
            target_size = np.random.choice(grid_sizes)
            indices = self.grid_size_buckets[target_size]
            
            if len(indices) >= batch_size:
                selected = np.random.choice(indices, size=batch_size, replace=False)
                return [self.buffer[i] for i in selected]
            else:
                # Mix with other sizes
                samples = [self.buffer[i] for i in indices]
                remaining = batch_size - len(samples)
                other_samples = self.sample(remaining, strategy='mixed')
                samples.extend(other_samples)
                return samples
        
        else:
            # Default random sampling
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def get_stats(self) -> Dict[str, int]:
        """Get MINERVA-specific buffer statistics"""
        return {
            'total_experiences': len(self.buffer),
            'exact_matches': len(self.exact_matches),
            'symmetric_patterns': len(self.symmetry_bucket),
            'object_patterns': len(self.object_bucket),
            'unique_grid_sizes': len(self.grid_size_buckets),
            'grid_size_distribution': {
                size: len(indices) for size, indices in self.grid_size_buckets.items()
            }
        }


class MinervaPatternBank:
    """Grid pattern bank specifically for MINERVA's spatial reasoning"""
    
    def __init__(self, max_patterns: int = 10000):
        self.max_patterns = max_patterns
        self.patterns = {}
        
        # MINERVA-specific pattern categories
        self.spatial_patterns = {}  # Spatial transformations
        self.object_patterns = {}   # Object-based patterns
        self.symmetry_patterns = {} # Symmetry-based patterns
        self.grid_structure_patterns = {}  # Grid structure patterns
        
        # Pattern metadata
        self.pattern_usage = defaultdict(int)
        self.pattern_success = defaultdict(float)
    
    def add_pattern(self, input_pattern: torch.Tensor, output_pattern: torch.Tensor,
                   transformation_info: Dict, success_score: float = 1.0):
        """Store successful grid transformation pattern"""
        
        # Create pattern hash
        pattern_hash = self._compute_pattern_hash(input_pattern, output_pattern)
        
        if pattern_hash in self.patterns:
            # Update existing pattern
            self.pattern_usage[pattern_hash] += 1
            self.pattern_success[pattern_hash] = (
                self.pattern_success[pattern_hash] * 0.9 + success_score * 0.1
            )
        else:
            # Add new pattern
            pattern_data = {
                'input': input_pattern.cpu(),
                'output': output_pattern.cpu(),
                'transformation': transformation_info,
                'grid_size': input_pattern.shape[-2:],
                'usage_count': 1,
                'success_score': success_score
            }
            
            # Categorize pattern
            self._categorize_pattern(pattern_hash, pattern_data)
            
            # Add to main storage
            self.patterns[pattern_hash] = pattern_data
            
            # Evict if over capacity
            if len(self.patterns) > self.max_patterns:
                self._evict_least_useful()
    
    def _compute_pattern_hash(self, input_pattern: torch.Tensor, 
                            output_pattern: torch.Tensor) -> str:
        """Compute hash for pattern pair"""
        # Normalize patterns for hashing
        if input_pattern.dim() == 4:
            input_pattern = input_pattern.argmax(dim=1)
            output_pattern = output_pattern.argmax(dim=1)
        
        # Create hash
        combined = torch.cat([
            input_pattern.flatten(),
            output_pattern.flatten()
        ])
        
        return hashlib.md5(combined.cpu().numpy().tobytes()).hexdigest()
    
    def _categorize_pattern(self, pattern_hash: str, pattern_data: Dict):
        """Categorize pattern for MINERVA-specific retrieval"""
        input_pattern = pattern_data['input']
        output_pattern = pattern_data['output']
        
        # Check for spatial transformations
        if self._is_spatial_transformation(input_pattern, output_pattern):
            self.spatial_patterns[pattern_hash] = pattern_data
        
        # Check for object patterns
        if self._has_objects(input_pattern):
            self.object_patterns[pattern_hash] = pattern_data
        
        # Check for symmetry
        if self._has_symmetry(input_pattern) or self._has_symmetry(output_pattern):
            self.symmetry_patterns[pattern_hash] = pattern_data
        
        # Check for grid structure
        if self._has_grid_structure(input_pattern):
            self.grid_structure_patterns[pattern_hash] = pattern_data
    
    def _is_spatial_transformation(self, input_p: torch.Tensor, 
                                  output_p: torch.Tensor) -> bool:
        """Check if pattern represents spatial transformation"""
        # Simple checks for rotation, flip, etc.
        if input_p.shape != output_p.shape:
            return False
        
        # Check common transformations
        rotations = [
            torch.rot90(input_p, k=k, dims=[-2, -1]) 
            for k in range(1, 4)
        ]
        flips = [
            torch.flip(input_p, dims=[-1]),
            torch.flip(input_p, dims=[-2])
        ]
        
        for transformed in rotations + flips:
            if torch.equal(transformed, output_p):
                return True
        
        return False
    
    def _has_objects(self, pattern: torch.Tensor) -> bool:
        """Check if pattern contains distinct objects"""
        if pattern.dim() == 4:
            pattern = pattern.argmax(dim=1).squeeze(0)
        elif pattern.dim() == 3:
            pattern = pattern.squeeze(0)
        
        # Ensure we have a 2D tensor
        while pattern.dim() > 2:
            pattern = pattern.squeeze(0)
        if pattern.dim() < 2:
            return False
            
        unique_values = torch.unique(pattern)
        return len(unique_values) > 2
    
    def _has_symmetry(self, pattern: torch.Tensor) -> bool:
        """Check for symmetry in pattern"""
        if pattern.dim() == 4:
            pattern = pattern.argmax(dim=1).squeeze(0)
        elif pattern.dim() == 3:
            pattern = pattern.squeeze(0)
        
        # Ensure we have a 2D tensor
        while pattern.dim() > 2:
            pattern = pattern.squeeze(0)
        if pattern.dim() < 2:
            return False
            
        pattern_np = pattern.cpu().numpy()
        
        return (np.array_equal(pattern_np, np.fliplr(pattern_np)) or
                np.array_equal(pattern_np, np.flipud(pattern_np)))
    
    def _has_grid_structure(self, pattern: torch.Tensor) -> bool:
        """Check if pattern has regular grid structure"""
        if pattern.dim() == 4:
            pattern = pattern.argmax(dim=1).squeeze(0)
        elif pattern.dim() == 3:
            pattern = pattern.squeeze(0)
        
        # Ensure we have a 2D tensor
        while pattern.dim() > 2:
            pattern = pattern.squeeze(0)
        if pattern.dim() < 2:
            return False
            
        # Check for regular intervals or repeating structures
        h, w = pattern.shape
        
        # Check horizontal lines
        for i in range(1, h):
            if torch.equal(pattern[i], pattern[0]):
                return True
        
        # Check vertical lines
        for j in range(1, w):
            if torch.equal(pattern[:, j], pattern[:, 0]):
                return True
        
        return False
    
    def _evict_least_useful(self):
        """Remove least useful patterns"""
        # Score patterns by usage and success
        pattern_scores = {}
        for hash_val, data in self.patterns.items():
            usage = self.pattern_usage.get(hash_val, 1)
            success = self.pattern_success.get(hash_val, 0.5)
            score = usage * success
            pattern_scores[hash_val] = score
        
        # Remove bottom 10%
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1])
        to_remove = sorted_patterns[:len(sorted_patterns) // 10]
        
        for hash_val, _ in to_remove:
            del self.patterns[hash_val]
            # Remove from category indices
            for category in [self.spatial_patterns, self.object_patterns, 
                           self.symmetry_patterns, self.grid_structure_patterns]:
                if hash_val in category:
                    del category[hash_val]
    
    def retrieve_similar_patterns(self, input_grid: torch.Tensor, 
                                strategy: str = 'mixed', k: int = 5) -> List[Dict]:
        """Retrieve similar patterns for MINERVA's grid reasoning"""
        if not self.patterns:
            return []
        
        if strategy == 'spatial':
            candidates = self.spatial_patterns
        elif strategy == 'object':
            candidates = self.object_patterns
        elif strategy == 'symmetry':
            candidates = self.symmetry_patterns
        elif strategy == 'grid':
            candidates = self.grid_structure_patterns
        else:
            candidates = self.patterns
        
        if not candidates:
            return []
        
        # Compute similarities
        similarities = []
        for hash_val, pattern_data in candidates.items():
            sim_score = self._compute_grid_similarity(input_grid, pattern_data['input'])
            similarities.append((hash_val, sim_score))
        
        # Get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        return [self.patterns[hash_val] for hash_val, _ in top_k]
    
    def _compute_grid_similarity(self, grid1: torch.Tensor, grid2: torch.Tensor) -> float:
        """Compute similarity between two grids"""
        # Normalize dimensions
        if grid1.dim() == 4:
            grid1 = grid1.argmax(dim=1).squeeze(0)
        if grid2.dim() == 4:
            grid2 = grid2.argmax(dim=1).squeeze(0)
        
        # Size similarity
        size_sim = 1.0 - abs(grid1.numel() - grid2.numel()) / max(grid1.numel(), grid2.numel())
        
        # Color distribution similarity
        colors1 = torch.bincount(grid1.flatten(), minlength=10)[:10]
        colors2 = torch.bincount(grid2.flatten(), minlength=10)[:10]
        color_sim = F.cosine_similarity(colors1.float(), colors2.float(), dim=0)
        
        # Structure similarity (simple edge detection)
        edges1 = self._simple_edge_detection(grid1)
        edges2 = self._simple_edge_detection(grid2)
        edge_sim = 1.0 - torch.abs(edges1 - edges2).float().mean()
        
        return (size_sim + color_sim + edge_sim) / 3.0
    
    def _simple_edge_detection(self, grid: torch.Tensor) -> float:
        """Simple edge detection for structure comparison"""
        if grid.dim() > 2:
            grid = grid.squeeze()
        
        # Count transitions
        h_transitions = (grid[:, :-1] != grid[:, 1:]).sum()
        v_transitions = (grid[:-1, :] != grid[1:, :]).sum()
        
        total_transitions = (h_transitions + v_transitions).float()
        normalized = total_transitions / (grid.shape[0] * grid.shape[1])
        
        return normalized


class MinervaMEPTLoss(nn.Module):
    """MINERVA-specific MEPT loss focusing on grid reasoning"""
    
    def __init__(self, replay_buffer: MinervaExperienceReplayBuffer,
                pattern_bank: MinervaPatternBank,
                use_mept: bool = True,
                transformation_penalty: float = 0.5,
                exact_match_bonus: float = 5.0,
                spatial_consistency_weight: float = 0.3,
                object_preservation_weight: float = 0.2):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.pattern_bank = pattern_bank
        self.use_mept = use_mept
        self.transformation_penalty = transformation_penalty
        self.exact_match_bonus = exact_match_bonus
        self.spatial_consistency_weight = spatial_consistency_weight
        self.object_preservation_weight = object_preservation_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
               input_grid: torch.Tensor, model_outputs: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Compute MINERVA-specific MEPT loss"""
        
        # Base reconstruction loss
        ce_loss = F.cross_entropy(pred, target.argmax(dim=1))
        
        # Spatial consistency loss
        spatial_loss = self._spatial_consistency_loss(pred, target)
        
        # Object preservation loss
        object_loss = self._object_preservation_loss(pred, target, input_grid)
        
        # Grid structure loss
        structure_loss = self._grid_structure_loss(pred, target)
        
        # Transformation penalty
        transform_loss = self._transformation_penalty(pred, input_grid)
        
        # Exact match bonus
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        exact_matches = (pred_idx == target_idx).all(dim=[1,2]).float()
        exact_bonus = -self.exact_match_bonus * exact_matches.mean()
        
        # Combine losses
        total_loss = (
            ce_loss + 
            self.spatial_consistency_weight * spatial_loss +
            self.object_preservation_weight * object_loss +
            0.2 * structure_loss +
            self.transformation_penalty * transform_loss +
            exact_bonus
        )
        
        # Store successful patterns
        if self.use_mept and exact_matches.any():
            for i in range(pred.size(0)):
                if exact_matches[i]:
                    self.replay_buffer.add(
                        input_grid[i], target[i], pred[i],
                        total_loss.item(), is_exact=True
                    )
                    
                    # Add to pattern bank
                    transform_info = {
                        'type': 'learned',
                        'loss': total_loss.item()
                    }
                    self.pattern_bank.add_pattern(
                        input_grid[i], target[i],
                        transform_info, success_score=1.0
                    )
        
        return {
            'total': total_loss,
            'reconstruction': ce_loss,
            'spatial': spatial_loss,
            'object': object_loss,
            'structure': structure_loss,
            'transformation': transform_loss,
            'exact_bonus': exact_bonus,
            'exact_count': exact_matches.sum()
        }
    
    def _spatial_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ensure spatial relationships are preserved"""
        pred_idx = pred.argmax(dim=1).float()
        target_idx = target.argmax(dim=1).float()
        
        # Compute spatial gradients
        pred_grad_x = pred_idx[:, :, 1:] - pred_idx[:, :, :-1]
        pred_grad_y = pred_idx[:, 1:, :] - pred_idx[:, :-1, :]
        
        target_grad_x = target_idx[:, :, 1:] - target_idx[:, :, :-1]
        target_grad_y = target_idx[:, 1:, :] - target_idx[:, :-1, :]
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y
    
    def _object_preservation_loss(self, pred: torch.Tensor, target: torch.Tensor,
                                 input_grid: torch.Tensor) -> torch.Tensor:
        """Ensure objects are preserved correctly"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        input_idx = input_grid.argmax(dim=1) if input_grid.dim() == 4 else input_grid
        
        # Count unique values (objects)
        loss = 0
        for i in range(pred_idx.size(0)):
            pred_objects = torch.unique(pred_idx[i])
            target_objects = torch.unique(target_idx[i])
            input_objects = torch.unique(input_idx[i])
            
            # Penalize missing or extra objects
            missing = len(target_objects) - len(pred_objects)
            extra = len(pred_objects) - len(target_objects)
            
            loss += abs(missing) + abs(extra)
        
        return torch.tensor(loss / pred_idx.size(0), device=pred.device)
    
    def _grid_structure_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Preserve grid structural patterns"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Check row and column patterns
        loss = 0
        
        # Row consistency
        for i in range(pred_idx.size(0)):
            pred_rows = pred_idx[i]
            target_rows = target_idx[i]
            
            # Check if rows repeat
            for r in range(1, pred_rows.size(0)):
                pred_match = (pred_rows[r] == pred_rows[0]).float().mean()
                target_match = (target_rows[r] == target_rows[0]).float().mean()
                loss += abs(pred_match - target_match)
        
        return loss / pred_idx.size(0)
    
    def _transformation_penalty(self, pred: torch.Tensor, input_grid: torch.Tensor) -> torch.Tensor:
        """Penalize unnecessary transformations"""
        pred_idx = pred.argmax(dim=1)
        input_idx = input_grid.argmax(dim=1) if input_grid.dim() == 4 else input_grid
        
        # Compute difference
        diff = (pred_idx != input_idx).float().mean()
        
        return diff


def create_minerva_mept_system(capacity: int = 50000, pattern_bank_size: int = 10000,
                              transformation_penalty: float = 0.5,
                              exact_match_bonus: float = 5.0) -> Dict:
    """Create MINERVA-specific MEPT components"""
    replay_buffer = MinervaExperienceReplayBuffer(capacity)
    pattern_bank = MinervaPatternBank(pattern_bank_size)
    loss_fn = MinervaMEPTLoss(
        replay_buffer, pattern_bank,
        transformation_penalty=transformation_penalty,
        exact_match_bonus=exact_match_bonus
    )
    
    return {
        'replay_buffer': replay_buffer,
        'pattern_bank': pattern_bank,
        'loss_fn': loss_fn
    }