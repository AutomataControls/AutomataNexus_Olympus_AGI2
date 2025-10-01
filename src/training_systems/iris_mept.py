"""
IRIS-specific MEPT (Memory-Enhanced Pattern Training) System
Focuses on storing and retrieving color patterns and perceptual transformations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
import hashlib


class IrisExperienceReplayBuffer:
    """Color-pattern focused experience replay buffer for IRIS"""
    
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.exact_matches = deque(maxlen=capacity // 2)
        
        # IRIS-specific categorization
        self.color_count_buckets = defaultdict(list)  # Organize by number of colors
        self.color_transition_buckets = defaultdict(list)  # Organize by transition density
        self.color_pattern_bucket = []  # Special bucket for specific color patterns
        self.perceptual_bucket = []  # Special bucket for perceptual grouping
        
        # Priority scores for different color characteristics
        self.pattern_priorities = {
            'exact_match': 5.0,
            'complex_color_pattern': 3.5,
            'perceptual_grouping': 3.0,
            'color_transition': 2.5,
            'multi_color': 2.0,
            'simple': 1.0
        }
    
    def add(self, input_grid: torch.Tensor, output_grid: torch.Tensor, 
            predicted_grid: torch.Tensor, loss: float, is_exact: bool = False,
            pattern_info: Optional[Dict] = None):
        """Add experience with IRIS-specific metadata"""
        
        # Analyze color characteristics
        num_input_colors = len(torch.unique(input_grid))
        num_output_colors = len(torch.unique(output_grid))
        color_transitions = self._count_color_transitions(input_grid)
        has_perceptual_grouping = self._detect_perceptual_grouping(input_grid, output_grid)
        color_pattern_type = self._classify_color_pattern(input_grid, output_grid)
        
        experience = {
            'input': input_grid.cpu(),
            'output': output_grid.cpu(),
            'predicted': predicted_grid.cpu(),
            'loss': loss,
            'is_exact': is_exact,
            'num_input_colors': num_input_colors,
            'num_output_colors': num_output_colors,
            'color_transitions': color_transitions,
            'has_perceptual_grouping': has_perceptual_grouping,
            'color_pattern_type': color_pattern_type,
            'pattern_info': pattern_info or {},
            'priority': self._calculate_priority(
                is_exact, num_input_colors, num_output_colors, 
                color_transitions, has_perceptual_grouping
            )
        }
        
        self.buffer.append(experience)
        
        # Add to specialized buckets
        color_key = f"{num_input_colors}->{num_output_colors}"
        self.color_count_buckets[color_key].append(len(self.buffer) - 1)
        
        if is_exact:
            self.exact_matches.append(experience)
        
        if color_transitions > 20:  # High transition density
            self.color_transition_buckets['high'].append(len(self.buffer) - 1)
        elif color_transitions > 10:
            self.color_transition_buckets['medium'].append(len(self.buffer) - 1)
        
        if has_perceptual_grouping:
            self.perceptual_bucket.append(len(self.buffer) - 1)
            
        if color_pattern_type != 'simple':
            self.color_pattern_bucket.append(len(self.buffer) - 1)
    
    def _count_color_transitions(self, grid: torch.Tensor) -> int:
        """Count color transitions in grid"""
        if grid.dim() == 4:
            grid = grid.argmax(dim=1).squeeze(0)
        elif grid.dim() == 3:
            grid = grid.squeeze(0)
        
        # Count horizontal and vertical transitions
        h_transitions = torch.sum(grid[:, :-1] != grid[:, 1:]).item()
        v_transitions = torch.sum(grid[:-1, :] != grid[1:, :]).item()
        
        return h_transitions + v_transitions
    
    def _detect_perceptual_grouping(self, input_grid: torch.Tensor, 
                                   output_grid: torch.Tensor) -> bool:
        """Detect if transformation involves perceptual grouping"""
        # Convert to indices if needed
        if input_grid.dim() == 4:
            input_idx = input_grid.argmax(dim=1).squeeze(0)
            output_idx = output_grid.argmax(dim=1).squeeze(0)
        else:
            input_idx = input_grid.squeeze(0) if input_grid.dim() == 3 else input_grid
            output_idx = output_grid.squeeze(0) if output_grid.dim() == 3 else output_grid
        
        # Check if scattered colors become grouped
        input_colors = torch.unique(input_idx)
        output_colors = torch.unique(output_idx)
        
        # Simple heuristic: if output has more connected regions of same color
        return len(output_colors) < len(input_colors) and len(output_colors) > 1
    
    def _classify_color_pattern(self, input_grid: torch.Tensor, 
                              output_grid: torch.Tensor) -> str:
        """Classify the type of color pattern transformation"""
        if input_grid.dim() == 4:
            input_idx = input_grid.argmax(dim=1).squeeze(0)
            output_idx = output_grid.argmax(dim=1).squeeze(0)
        else:
            input_idx = input_grid.squeeze(0) if input_grid.dim() == 3 else input_grid
            output_idx = output_grid.squeeze(0) if output_grid.dim() == 3 else output_grid
        
        # Analyze transformation
        input_colors = set(torch.unique(input_idx).tolist())
        output_colors = set(torch.unique(output_idx).tolist())
        
        if output_colors - input_colors:  # New colors introduced
            return 'color_mixing'
        elif len(output_colors) < len(input_colors):
            return 'color_reduction'
        elif not torch.equal(input_idx, output_idx):
            # Check for specific patterns
            if self._is_gradient_pattern(output_idx):
                return 'gradient'
            elif self._is_alternating_pattern(output_idx):
                return 'alternating'
            else:
                return 'complex_transformation'
        else:
            return 'simple'
    
    def _is_gradient_pattern(self, grid: torch.Tensor) -> bool:
        """Check if grid has gradient pattern"""
        # Simple check for monotonic color changes
        for row in grid:
            if torch.all(row[:-1] <= row[1:]) or torch.all(row[:-1] >= row[1:]):
                return True
        return False
    
    def _is_alternating_pattern(self, grid: torch.Tensor) -> bool:
        """Check if grid has alternating pattern"""
        # Check for regular alternation
        unique_vals = torch.unique(grid)
        if len(unique_vals) == 2:
            return torch.sum(grid == unique_vals[0]) == torch.sum(grid == unique_vals[1])
        return False
    
    def _calculate_priority(self, is_exact: bool, num_input_colors: int,
                          num_output_colors: int, color_transitions: int,
                          has_perceptual_grouping: bool) -> float:
        """Calculate priority score for experience"""
        priority = self.pattern_priorities['simple']
        
        if is_exact:
            priority = max(priority, self.pattern_priorities['exact_match'])
        if num_input_colors > 3 or num_output_colors > 3:
            priority = max(priority, self.pattern_priorities['multi_color'])
        if color_transitions > 20:
            priority = max(priority, self.pattern_priorities['color_transition'])
        if has_perceptual_grouping:
            priority = max(priority, self.pattern_priorities['perceptual_grouping'])
        
        return priority
    
    def sample(self, batch_size: int, strategy: str = 'mixed') -> List[Dict]:
        """Sample experiences with IRIS-specific strategies"""
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
            
            # 20% perceptual grouping patterns
            percept_count = int(batch_size * 0.2)
            if self.perceptual_bucket:
                percept_indices = np.random.choice(
                    self.perceptual_bucket,
                    size=min(percept_count, len(self.perceptual_bucket)),
                    replace=False
                )
                samples.extend([self.buffer[i] for i in percept_indices])
            
            # 20% complex color patterns
            pattern_count = int(batch_size * 0.2)
            if self.color_pattern_bucket:
                pattern_indices = np.random.choice(
                    self.color_pattern_bucket,
                    size=min(pattern_count, len(self.color_pattern_bucket)),
                    replace=False
                )
                samples.extend([self.buffer[i] for i in pattern_indices])
            
            # Fill remaining with priority sampling
            remaining = batch_size - len(samples)
            if remaining > 0:
                priorities = [exp['priority'] for exp in self.buffer]
                indices = np.random.choice(
                    len(self.buffer),
                    size=min(remaining, len(self.buffer)),
                    replace=False,
                    p=np.array(priorities) / sum(priorities)
                )
                samples.extend([self.buffer[i] for i in indices])
            
            return samples[:batch_size]
        
        elif strategy == 'color_focused':
            # Sample patterns with similar color complexity
            color_keys = list(self.color_count_buckets.keys())
            if not color_keys:
                return self.sample(batch_size, strategy='mixed')
            
            # Focus on specific color transformation
            target_key = np.random.choice(color_keys)
            indices = self.color_count_buckets[target_key]
            
            if len(indices) >= batch_size:
                selected = np.random.choice(indices, size=batch_size, replace=False)
                return [self.buffer[i] for i in selected]
            else:
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
        """Get IRIS-specific buffer statistics"""
        color_distribution = defaultdict(int)
        for exp in self.buffer:
            key = f"{exp['num_input_colors']}->{exp['num_output_colors']}"
            color_distribution[key] += 1
        
        return {
            'total_experiences': len(self.buffer),
            'exact_matches': len(self.exact_matches),
            'perceptual_patterns': len(self.perceptual_bucket),
            'complex_color_patterns': len(self.color_pattern_bucket),
            'unique_color_transformations': len(self.color_count_buckets),
            'color_distribution': dict(color_distribution)
        }


class IrisPatternBank:
    """Color pattern bank specifically for IRIS's perception capabilities"""
    
    def __init__(self, max_patterns: int = 10000):
        self.max_patterns = max_patterns
        self.patterns = {}
        
        # IRIS-specific pattern categories
        self.gradient_patterns = {}     # Color gradient patterns
        self.alternating_patterns = {}  # Alternating color patterns
        self.mixing_patterns = {}       # Color mixing patterns
        self.perceptual_patterns = {}   # Perceptual grouping patterns
        
        # Pattern metadata
        self.pattern_usage = defaultdict(int)
        self.pattern_success = defaultdict(float)
        self.color_accuracy = defaultdict(lambda: defaultdict(float))
    
    def add_pattern(self, input_pattern: torch.Tensor, output_pattern: torch.Tensor,
                   transformation_info: Dict, success_score: float = 1.0):
        """Store successful color transformation pattern"""
        
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
                'input_colors': torch.unique(input_pattern).cpu().tolist(),
                'output_colors': torch.unique(output_pattern).cpu().tolist(),
                'usage_count': 1,
                'success_score': success_score
            }
            
            # Categorize pattern
            self._categorize_pattern(pattern_hash, pattern_data)
            
            # Track color-specific success
            for color in pattern_data['output_colors']:
                self.color_accuracy[pattern_hash][color] = success_score
            
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
        
        # Create hash including color information
        input_colors = torch.unique(input_pattern).cpu().numpy()
        output_colors = torch.unique(output_pattern).cpu().numpy()
        
        hash_data = np.concatenate([
            input_pattern.flatten().cpu().numpy(),
            output_pattern.flatten().cpu().numpy(),
            input_colors,
            output_colors
        ])
        
        return hashlib.md5(hash_data.tobytes()).hexdigest()
    
    def _categorize_pattern(self, pattern_hash: str, pattern_data: Dict):
        """Categorize pattern for IRIS-specific retrieval"""
        input_pattern = pattern_data['input']
        output_pattern = pattern_data['output']
        
        # Check for gradient patterns
        if self._is_gradient_transformation(input_pattern, output_pattern):
            self.gradient_patterns[pattern_hash] = pattern_data
        
        # Check for alternating patterns
        if self._is_alternating_transformation(input_pattern, output_pattern):
            self.alternating_patterns[pattern_hash] = pattern_data
        
        # Check for color mixing
        if self._is_color_mixing(pattern_data['input_colors'], 
                               pattern_data['output_colors']):
            self.mixing_patterns[pattern_hash] = pattern_data
        
        # Check for perceptual grouping
        if self._is_perceptual_grouping(input_pattern, output_pattern):
            self.perceptual_patterns[pattern_hash] = pattern_data
    
    def _is_gradient_transformation(self, input_p: torch.Tensor, 
                                  output_p: torch.Tensor) -> bool:
        """Check if pattern involves gradient transformation"""
        # Check if output has gradient structure
        if output_p.dim() > 2:
            output_p = output_p.squeeze()
        
        # Check rows for monotonic changes
        for row in output_p:
            sorted_row = torch.sort(row)[0]
            if torch.equal(row, sorted_row) or torch.equal(row, torch.flip(sorted_row, [0])):
                return True
        
        return False
    
    def _is_alternating_transformation(self, input_p: torch.Tensor, 
                                     output_p: torch.Tensor) -> bool:
        """Check if pattern involves alternating colors"""
        if output_p.dim() > 2:
            output_p = output_p.squeeze()
        
        unique_colors = torch.unique(output_p)
        if len(unique_colors) == 2:
            # Check for regular alternation
            for i in range(output_p.shape[0]):
                for j in range(output_p.shape[1] - 1):
                    if output_p[i, j] == output_p[i, j + 1]:
                        return False
            return True
        
        return False
    
    def _is_color_mixing(self, input_colors: List[int], 
                        output_colors: List[int]) -> bool:
        """Check if new colors are created through mixing"""
        input_set = set(input_colors)
        output_set = set(output_colors)
        
        # New colors that weren't in input
        new_colors = output_set - input_set
        
        # Heuristic: check if new colors are "between" input colors
        if new_colors and len(input_colors) >= 2:
            for new_color in new_colors:
                if min(input_colors) < new_color < max(input_colors):
                    return True
        
        return False
    
    def _is_perceptual_grouping(self, input_p: torch.Tensor, 
                               output_p: torch.Tensor) -> bool:
        """Check if pattern shows perceptual grouping"""
        # Simple check: output has larger contiguous regions
        input_transitions = self._count_transitions(input_p)
        output_transitions = self._count_transitions(output_p)
        
        return output_transitions < input_transitions * 0.7
    
    def _count_transitions(self, pattern: torch.Tensor) -> int:
        """Count color transitions in pattern"""
        if pattern.dim() > 2:
            pattern = pattern.squeeze()
        
        h_trans = torch.sum(pattern[:, :-1] != pattern[:, 1:]).item()
        v_trans = torch.sum(pattern[:-1, :] != pattern[1:, :]).item()
        
        return h_trans + v_trans
    
    def _evict_least_useful(self):
        """Remove least useful patterns focusing on color accuracy"""
        # Score patterns by usage, success, and color accuracy
        pattern_scores = {}
        for hash_val, data in self.patterns.items():
            usage = self.pattern_usage.get(hash_val, 1)
            success = self.pattern_success.get(hash_val, 0.5)
            
            # Average color accuracy
            color_acc = np.mean(list(self.color_accuracy[hash_val].values())) if hash_val in self.color_accuracy else 0.5
            
            score = usage * success * color_acc
            pattern_scores[hash_val] = score
        
        # Remove bottom 10%
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1])
        to_remove = sorted_patterns[:len(sorted_patterns) // 10]
        
        for hash_val, _ in to_remove:
            del self.patterns[hash_val]
            # Remove from category indices
            for category in [self.gradient_patterns, self.alternating_patterns, 
                           self.mixing_patterns, self.perceptual_patterns]:
                if hash_val in category:
                    del category[hash_val]
    
    def retrieve_similar_patterns(self, input_grid: torch.Tensor, 
                                strategy: str = 'mixed', k: int = 5) -> List[Dict]:
        """Retrieve similar color patterns for IRIS"""
        if not self.patterns:
            return []
        
        if strategy == 'gradient':
            candidates = self.gradient_patterns
        elif strategy == 'alternating':
            candidates = self.alternating_patterns
        elif strategy == 'mixing':
            candidates = self.mixing_patterns
        elif strategy == 'perceptual':
            candidates = self.perceptual_patterns
        else:
            candidates = self.patterns
        
        if not candidates:
            return []
        
        # Compute similarities based on color distribution
        similarities = []
        for hash_val, pattern_data in candidates.items():
            sim_score = self._compute_color_similarity(input_grid, pattern_data['input'])
            similarities.append((hash_val, sim_score))
        
        # Get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        return [self.patterns[hash_val] for hash_val, _ in top_k]
    
    def _compute_color_similarity(self, grid1: torch.Tensor, grid2: torch.Tensor) -> float:
        """Compute color-based similarity between two grids"""
        # Normalize dimensions
        if grid1.dim() == 4:
            grid1 = grid1.argmax(dim=1).squeeze(0)
        if grid2.dim() == 4:
            grid2 = grid2.argmax(dim=1).squeeze(0)
        
        # Color histogram similarity
        hist1 = torch.bincount(grid1.flatten(), minlength=10)[:10].float()
        hist2 = torch.bincount(grid2.flatten(), minlength=10)[:10].float()
        
        # Normalize histograms
        hist1 = hist1 / (hist1.sum() + 1e-8)
        hist2 = hist2 / (hist2.sum() + 1e-8)
        
        # Compute similarity (1 - histogram distance)
        hist_sim = 1.0 - torch.abs(hist1 - hist2).mean()
        
        # Color palette similarity
        colors1 = set(torch.unique(grid1).tolist())
        colors2 = set(torch.unique(grid2).tolist())
        
        palette_sim = len(colors1 & colors2) / max(len(colors1 | colors2), 1)
        
        return (hist_sim + palette_sim) / 2.0


class IrisMEPTLoss(nn.Module):
    """IRIS-specific MEPT loss focusing on color perception"""
    
    def __init__(self, replay_buffer: IrisExperienceReplayBuffer,
                pattern_bank: IrisPatternBank,
                use_mept: bool = True,
                transformation_penalty: float = 0.3,
                exact_match_bonus: float = 5.0,
                color_accuracy_weight: float = 0.4,
                perceptual_grouping_weight: float = 0.3):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.pattern_bank = pattern_bank
        self.use_mept = use_mept
        self.transformation_penalty = transformation_penalty
        self.exact_match_bonus = exact_match_bonus
        self.color_accuracy_weight = color_accuracy_weight
        self.perceptual_grouping_weight = perceptual_grouping_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
               input_grid: torch.Tensor, model_outputs: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Compute IRIS-specific MEPT loss"""
        
        # Base reconstruction loss
        ce_loss = F.cross_entropy(pred, target.argmax(dim=1))
        
        # Color accuracy loss
        color_loss = self._color_accuracy_loss(pred, target)
        
        # Color distribution loss
        distribution_loss = self._color_distribution_loss(pred, target)
        
        # Perceptual grouping loss
        perceptual_loss = self._perceptual_grouping_loss(pred, target)
        
        # Color transition loss
        transition_loss = self._color_transition_loss(pred, target)
        
        # Transformation penalty (less aggressive for IRIS)
        transform_loss = self._transformation_penalty(pred, input_grid)
        
        # Exact match bonus
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        exact_matches = (pred_idx == target_idx).all(dim=[1,2]).float()
        exact_bonus = -self.exact_match_bonus * exact_matches.mean()
        
        # Combine losses
        total_loss = (
            ce_loss + 
            self.color_accuracy_weight * color_loss +
            0.3 * distribution_loss +
            self.perceptual_grouping_weight * perceptual_loss +
            0.2 * transition_loss +
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
                        'color_accuracy': 1.0 - color_loss.item()
                    }
                    self.pattern_bank.add_pattern(
                        input_grid[i], target[i],
                        transform_info, success_score=1.0
                    )
        
        return {
            'total': total_loss,
            'reconstruction': ce_loss,
            'color_accuracy': color_loss,
            'distribution': distribution_loss,
            'perceptual': perceptual_loss,
            'transition': transition_loss,
            'transformation': transform_loss,
            'exact_bonus': exact_bonus,
            'exact_count': exact_matches.sum()
        }
    
    def _color_accuracy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ensure accurate color perception"""
        # Per-color accuracy loss
        pred_probs = F.softmax(pred, dim=1)
        target_onehot = target
        
        # Compute per-color IoU
        color_ious = []
        for c in range(pred.size(1)):
            pred_c = pred_probs[:, c]
            target_c = target_onehot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=[1, 2])
            union = pred_c.sum(dim=[1, 2]) + target_c.sum(dim=[1, 2]) - intersection
            
            iou = intersection / (union + 1e-8)
            color_ious.append(1.0 - iou.mean())
        
        return torch.stack(color_ious).mean()
    
    def _color_distribution_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Preserve color distribution"""
        pred_probs = F.softmax(pred, dim=1)
        
        # Global color distributions
        pred_dist = pred_probs.mean(dim=[2, 3])  # [B, C]
        target_dist = target.mean(dim=[2, 3])    # [B, C]
        
        # KL divergence between distributions
        kl_div = F.kl_div(
            torch.log(pred_dist + 1e-8),
            target_dist,
            reduction='batchmean'
        )
        
        return kl_div
    
    def _perceptual_grouping_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Encourage perceptual grouping of colors"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Compute connected components similarity
        loss = 0
        for i in range(pred_idx.size(0)):
            # Count color regions
            pred_regions = self._count_color_regions(pred_idx[i])
            target_regions = self._count_color_regions(target_idx[i])
            
            # Penalize difference in number of regions
            loss += abs(pred_regions - target_regions) / max(target_regions, 1)
        
        return loss / pred_idx.size(0)
    
    def _count_color_regions(self, grid: torch.Tensor) -> int:
        """Count distinct color regions (simplified)"""
        unique_colors = torch.unique(grid)
        total_regions = 0
        
        for color in unique_colors:
            mask = (grid == color)
            # Simple approximation: count transitions
            h_trans = torch.sum(mask[:, :-1] != mask[:, 1:])
            v_trans = torch.sum(mask[:-1, :] != mask[1:, :])
            # Rough estimate of regions
            total_regions += max(1, (h_trans + v_trans) // 4)
        
        return total_regions
    
    def _color_transition_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Preserve color transition patterns"""
        pred_idx = pred.argmax(dim=1)
        target_idx = target.argmax(dim=1)
        
        # Compute transition matrices
        pred_trans = self._compute_transition_matrix(pred_idx)
        target_trans = self._compute_transition_matrix(target_idx)
        
        return F.mse_loss(pred_trans, target_trans)
    
    def _compute_transition_matrix(self, grids: torch.Tensor) -> torch.Tensor:
        """Compute color transition statistics"""
        batch_size = grids.size(0)
        trans_matrix = torch.zeros(batch_size, 10, 10, device=grids.device)
        
        for b in range(batch_size):
            grid = grids[b]
            # Horizontal transitions
            for i in range(grid.size(0)):
                for j in range(grid.size(1) - 1):
                    c1, c2 = grid[i, j].item(), grid[i, j + 1].item()
                    if c1 < 10 and c2 < 10:
                        trans_matrix[b, c1, c2] += 1
            
            # Normalize
            trans_matrix[b] = trans_matrix[b] / (trans_matrix[b].sum() + 1e-8)
        
        return trans_matrix
    
    def _transformation_penalty(self, pred: torch.Tensor, input_grid: torch.Tensor) -> torch.Tensor:
        """Lighter penalty for IRIS as color changes are expected"""
        pred_idx = pred.argmax(dim=1)
        input_idx = input_grid.argmax(dim=1) if input_grid.dim() == 4 else input_grid
        
        # Only penalize structural changes, not color changes
        pred_mask = pred_idx > 0
        input_mask = input_idx > 0
        
        structure_diff = (pred_mask != input_mask).float().mean()
        
        return structure_diff * 0.5  # Reduced penalty


def create_iris_mept_system(capacity: int = 50000, pattern_bank_size: int = 10000,
                           transformation_penalty: float = 0.3,
                           exact_match_bonus: float = 5.0) -> Dict:
    """Create IRIS-specific MEPT components"""
    replay_buffer = IrisExperienceReplayBuffer(capacity)
    pattern_bank = IrisPatternBank(pattern_bank_size)
    loss_fn = IrisMEPTLoss(
        replay_buffer, pattern_bank,
        transformation_penalty=transformation_penalty,
        exact_match_bonus=exact_match_bonus
    )
    
    return {
        'replay_buffer': replay_buffer,
        'pattern_bank': pattern_bank,
        'loss_fn': loss_fn
    }