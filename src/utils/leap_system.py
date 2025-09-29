"""
LEAP (Learning Enhancement through Adaptive Patterns) System
Adaptive pattern generation for Stage 0 training focusing on weak patterns
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random


class AdaptivePatternGenerator:
    """Generates patterns that adapt based on model's current weaknesses"""
    
    def __init__(self, grid_size: int = 8):  # Increased from 6 to 8
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
            'invert': self._generate_invert,
            # Add simpler patterns
            'fill_background': self._generate_fill_background,
            'keep_largest': self._generate_keep_largest,
            'remove_color': self._generate_remove_color,
            'flip_horizontal': self._generate_flip_horizontal,
            'flip_vertical': self._generate_flip_vertical
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
    
    def get_pattern_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for all patterns"""
        stats = {}
        for pattern_name, perf in self.performance_tracker.items():
            if perf['attempts'] > 0:
                stats[pattern_name] = {
                    'attempts': perf['attempts'],
                    'successes': perf['successes'],
                    'accuracy': perf['successes'] / perf['attempts']
                }
        return stats
    
    # Pattern generators
    def _generate_identity(self) -> Dict:
        """Identity - most fundamental"""
        grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        return {'input': grid, 'output': grid.copy()}
    
    def _generate_solid_color(self) -> Dict:
        """Fill non-zero pixels with single color"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        # Only fill where input is non-zero
        color = np.random.randint(1, 4)
        output_grid = np.where(input_grid > 0, color, 0)
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_horizontal_stripes(self) -> Dict:
        """Color rows based on input presence"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        colors = [1, 2]
        for i in range(self.grid_size):
            # Only color where input has values
            row_mask = input_grid[i, :] > 0
            if np.any(row_mask):
                output_grid[i, row_mask] = colors[i % 2]
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_vertical_stripes(self) -> Dict:
        """Color columns based on input presence"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        colors = [1, 3]
        for j in range(self.grid_size):
            # Only color where input has values
            col_mask = input_grid[:, j] > 0
            if np.any(col_mask):
                output_grid[col_mask, j] = colors[j % 2]
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_checkerboard(self) -> Dict:
        """Checkerboard pattern where input exists"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if input_grid[i, j] > 0:
                    output_grid[i, j] = 1 if (i + j) % 2 == 0 else 2
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_center_dot(self) -> Dict:
        """Highlight center if input exists there"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = input_grid.copy()
        center = self.grid_size // 2
        if input_grid[center, center] > 0:
            output_grid[center, center] = 3
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_border(self) -> Dict:
        """Highlight border pixels from input"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        # Copy border pixels from input
        output_grid[0, :] = input_grid[0, :]
        output_grid[-1, :] = input_grid[-1, :]
        output_grid[:, 0] = input_grid[:, 0]
        output_grid[:, -1] = input_grid[:, -1]
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_corners(self) -> Dict:
        """Extract corner pixels from input"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        # Copy corner pixels
        output_grid[0, 0] = input_grid[0, 0]
        output_grid[0, -1] = input_grid[0, -1]
        output_grid[-1, 0] = input_grid[-1, 0]
        output_grid[-1, -1] = input_grid[-1, -1]
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
        """Fill non-zero pixels with count-based color"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        count = np.count_nonzero(input_grid)
        color = min(3, max(1, count // 5))  # Ensure color is at least 1
        # Only fill where input is non-zero
        output_grid = np.where(input_grid > 0, color, 0)
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_diagonal(self) -> Dict:
        """Extract diagonal from input"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        for i in range(self.grid_size):
            output_grid[i, i] = input_grid[i, i]
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_cross(self) -> Dict:
        """Extract cross pattern from input"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        center = self.grid_size // 2
        # Copy center row and column
        output_grid[center, :] = input_grid[center, :]
        output_grid[:, center] = input_grid[:, center]
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_invert(self) -> Dict:
        """Invert 0/1"""
        input_grid = np.random.randint(0, 2, (self.grid_size, self.grid_size))
        output_grid = 1 - input_grid
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_fill_background(self) -> Dict:
        """Fill background (0) with a color"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = input_grid.copy()
        fill_color = np.random.randint(1, 4)
        output_grid[output_grid == 0] = fill_color
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_keep_largest(self) -> Dict:
        """Keep only the most frequent non-zero color"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.zeros_like(input_grid)
        unique, counts = np.unique(input_grid[input_grid > 0], return_counts=True)
        if len(unique) > 0:
            largest_color = unique[np.argmax(counts)]
            output_grid[input_grid == largest_color] = largest_color
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_remove_color(self) -> Dict:
        """Remove a specific color"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = input_grid.copy()
        # Remove color 1
        output_grid[output_grid == 1] = 0
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_flip_horizontal(self) -> Dict:
        """Flip horizontally"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.fliplr(input_grid)
        return {'input': input_grid, 'output': output_grid}
    
    def _generate_flip_vertical(self) -> Dict:
        """Flip vertically"""
        input_grid = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        output_grid = np.flipud(input_grid)
        return {'input': input_grid, 'output': output_grid}


class LEAPTrainer:
    """Manages LEAP training integration for Stage 0"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.pattern_generator = AdaptivePatternGenerator()
        self.pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.leap_iterations = 0
        self.enabled = True
        
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
            # Use .copy() to ensure contiguous memory layout
            inputs.append(torch.tensor(pattern['input'].copy(), dtype=torch.long))
            outputs.append(torch.tensor(pattern['output'].copy(), dtype=torch.long))
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
    
    def should_inject_leap_batch(self, stage: int, batch_idx: int, 
                               injection_frequency: int = 10) -> bool:
        """Determine if LEAP batch should be injected"""
        # Only inject in Stage 0
        if stage != 0 or not self.enabled:
            return False
        
        # Inject every N batches
        return batch_idx % injection_frequency == 0
    
    def get_injection_ratio(self, stage: int, epoch: int) -> float:
        """Get adaptive injection ratio based on stage and epoch"""
        if stage != 0:
            return 0.0
        
        # Start with higher injection rate, decrease over time
        base_ratio = 0.3
        decay_factor = 0.95 ** epoch  # Decay by 5% each epoch
        
        return max(0.1, base_ratio * decay_factor)  # Minimum 10% injection


class WeakPatternDetector:
    """Detects and tracks weak pattern types during training"""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.pattern_history = defaultdict(list)
        self.pattern_types = {
            'identity': self._is_identity,
            'solid_color': self._is_solid_color,
            'stripes': self._is_stripes,
            'checkerboard': self._is_checkerboard,
            'border': self._is_border,
            'color_swap': self._is_color_swap,
            'extraction': self._is_extraction,
            'counting': self._is_counting
        }
    
    def detect_pattern_type(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Detect the type of transformation pattern"""
        for pattern_name, detector_fn in self.pattern_types.items():
            if detector_fn(input_grid, output_grid):
                return pattern_name
        return 'complex'
    
    def update_history(self, pattern_type: str, success: bool):
        """Update pattern success history"""
        self.pattern_history[pattern_type].append(1.0 if success else 0.0)
        
        # Keep only recent history (last 100 attempts)
        if len(self.pattern_history[pattern_type]) > 100:
            self.pattern_history[pattern_type] = self.pattern_history[pattern_type][-100:]
    
    def get_weak_patterns(self) -> List[str]:
        """Get list of weak pattern types"""
        weak_patterns = []
        
        for pattern_type, history in self.pattern_history.items():
            if len(history) >= 10:  # Need sufficient data
                success_rate = np.mean(history)
                if success_rate < self.threshold:
                    weak_patterns.append(pattern_type)
        
        return weak_patterns
    
    # Pattern detection functions
    def _is_identity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output equals input"""
        return np.array_equal(input_grid, output_grid)
    
    def _is_solid_color(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is single color"""
        return len(np.unique(output_grid)) == 1
    
    def _is_stripes(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output has stripe pattern"""
        # Check horizontal stripes
        for i in range(1, output_grid.shape[0]):
            if not np.array_equal(output_grid[i], output_grid[0]):
                break
        else:
            return True
        
        # Check vertical stripes
        for j in range(1, output_grid.shape[1]):
            if not np.array_equal(output_grid[:, j], output_grid[:, 0]):
                break
        else:
            return True
        
        return False
    
    def _is_checkerboard(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is checkerboard pattern"""
        if len(np.unique(output_grid)) != 2:
            return False
        
        for i in range(output_grid.shape[0]):
            for j in range(output_grid.shape[1]):
                expected = output_grid[0, 0] if (i + j) % 2 == 0 else output_grid[0, 1]
                if output_grid[i, j] != expected:
                    return False
        return True
    
    def _is_border(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output is border pattern"""
        # Check if edges are non-zero and interior is zero
        if np.any(output_grid[1:-1, 1:-1] != 0):
            return False
        
        # Check if all edges have same non-zero value
        edge_vals = np.concatenate([
            output_grid[0, :], output_grid[-1, :],
            output_grid[1:-1, 0], output_grid[1:-1, -1]
        ])
        
        return len(np.unique(edge_vals)) == 1 and edge_vals[0] != 0
    
    def _is_color_swap(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if colors are swapped"""
        if input_grid.shape != output_grid.shape:
            return False
        
        input_colors = np.unique(input_grid)
        output_colors = np.unique(output_grid)
        
        if not np.array_equal(sorted(input_colors), sorted(output_colors)):
            return False
        
        # Check if it's a permutation
        color_map = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = input_grid[i, j]
                out_color = output_grid[i, j]
                
                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        return False
                else:
                    color_map[in_color] = out_color
        
        return len(color_map) > 1 and any(k != v for k, v in color_map.items())
    
    def _is_extraction(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if single color is extracted"""
        output_colors = np.unique(output_grid)
        if len(output_colors) > 2:  # Should be 0 and one other color
            return False
        
        if 0 not in output_colors:
            return False
        
        # Check if non-zero pixels in output are subset of input
        non_zero_mask = output_grid > 0
        if np.any(non_zero_mask):
            extracted_color = output_grid[non_zero_mask][0]
            return np.all(output_grid[non_zero_mask] == extracted_color) and \
                   np.all(output_grid[non_zero_mask] == input_grid[non_zero_mask])
        
        return False
    
    def _is_counting(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if output represents counting pattern"""
        return len(np.unique(output_grid)) == 1 and output_grid[0, 0] > 0


def create_leap_system(device: str = 'cuda', grid_size: int = 6):
    """Create a complete LEAP system"""
    trainer = LEAPTrainer(device=device)
    detector = WeakPatternDetector()
    
    return {
        'trainer': trainer,
        'detector': detector,
        'pattern_generator': trainer.pattern_generator
    }