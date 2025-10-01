"""
IRIS-specific LEAP (Learning Enhancement through Adaptive Patterns) System
Focuses on color patterns, transformations, and perceptual grouping
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random


class IrisPatternGenerator:
    """Generates color-based patterns specifically for IRIS's capabilities"""
    
    def __init__(self):
        self.pattern_types = [
            'color_gradient', 'color_alternation', 'color_blocks', 'color_rings',
            'color_stripes', 'color_mosaic', 'color_fade', 'color_wave',
            'color_clustering', 'color_symmetry', 'color_propagation', 'rainbow',
            'color_mixing', 'perceptual_grouping'
        ]
        self.complexity_levels = {
            'basic': ['color_blocks', 'color_stripes', 'color_alternation', 'color_rings'],
            'simple': ['color_gradient', 'color_fade', 'color_mosaic', 'color_symmetry'],
            'medium': ['color_wave', 'color_clustering', 'rainbow', 'color_mixing'],
            'complex': ['color_propagation', 'perceptual_grouping']
        }
    
    def generate_pattern(self, pattern_type: str, grid_size: int = 10, complexity: str = 'basic') -> Tuple[np.ndarray, np.ndarray]:
        """Generate input-output pairs for specific color patterns"""
        
        if pattern_type == 'color_gradient':
            return self._generate_color_gradient(grid_size)
        elif pattern_type == 'color_alternation':
            return self._generate_color_alternation(grid_size)
        elif pattern_type == 'color_blocks':
            return self._generate_color_blocks(grid_size)
        elif pattern_type == 'color_rings':
            return self._generate_color_rings(grid_size)
        elif pattern_type == 'color_stripes':
            return self._generate_color_stripes(grid_size)
        elif pattern_type == 'color_mosaic':
            return self._generate_color_mosaic(grid_size)
        elif pattern_type == 'color_fade':
            return self._generate_color_fade(grid_size)
        elif pattern_type == 'color_wave':
            return self._generate_color_wave(grid_size)
        elif pattern_type == 'color_clustering':
            return self._generate_color_clustering(grid_size)
        elif pattern_type == 'color_symmetry':
            return self._generate_color_symmetry(grid_size)
        elif pattern_type == 'color_propagation':
            return self._generate_color_propagation(grid_size)
        elif pattern_type == 'rainbow':
            return self._generate_rainbow(grid_size)
        elif pattern_type == 'color_mixing':
            return self._generate_color_mixing(grid_size)
        elif pattern_type == 'perceptual_grouping':
            return self._generate_perceptual_grouping(grid_size)
        else:
            # Default to simple pattern
            return self._generate_color_blocks(grid_size)
    
    def _generate_color_gradient(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate color gradient patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create input with partial gradient
        for i in range(size):
            color = i * 8 // size  # Map to color range 0-8
            input_grid[i, :size//2] = color
        
        # Output: complete the gradient
        for i in range(size):
            color = i * 8 // size
            output_grid[i, :] = color
        
        return input_grid, output_grid
    
    def _generate_color_alternation(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate alternating color patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create alternating pattern
        colors = [1, 2, 3, 4]
        for i in range(size):
            for j in range(size):
                idx = (i + j) % len(colors)
                input_grid[i, j] = colors[idx]
        
        # Transform: shift colors
        output_grid = (input_grid + 1) % 9
        
        return input_grid, output_grid
    
    def _generate_color_blocks(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate color block patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create color blocks
        block_size = max(2, size // 4)
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                color = ((i // block_size) + (j // block_size)) % 8 + 1
                input_grid[i:i+block_size, j:j+block_size] = color
        
        # Transform: invert colors
        max_color = np.max(input_grid)
        output_grid = max_color + 1 - input_grid
        
        return input_grid, output_grid
    
    def _generate_color_rings(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate concentric color rings"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        center = size // 2
        # Create partial rings
        for i in range(size):
            for j in range(size//2):  # Only left half
                dist = int(np.sqrt((i - center)**2 + (j - center)**2))
                input_grid[i, j] = min(dist % 8 + 1, 8)
        
        # Complete the rings
        for i in range(size):
            for j in range(size):
                dist = int(np.sqrt((i - center)**2 + (j - center)**2))
                output_grid[i, j] = min(dist % 8 + 1, 8)
        
        return input_grid, output_grid
    
    def _generate_color_stripes(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stripe patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Horizontal stripes
        stripe_width = max(1, size // 6)
        colors = [1, 2, 3, 4, 5]
        for i in range(size):
            color_idx = (i // stripe_width) % len(colors)
            input_grid[i, :] = colors[color_idx]
        
        # Transform to vertical stripes
        output_grid = input_grid.T
        
        return input_grid, output_grid
    
    def _generate_color_mosaic(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mosaic patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create random mosaic
        tile_size = max(2, size // 5)
        for i in range(0, size, tile_size):
            for j in range(0, size, tile_size):
                color = random.randint(1, 8)
                input_grid[i:i+tile_size, j:j+tile_size] = color
        
        # Transform: blur boundaries
        output_grid = input_grid.copy()
        for i in range(1, size-1):
            for j in range(1, size-1):
                if (i % tile_size == 0 or j % tile_size == 0):
                    # Average neighboring colors
                    neighbors = [
                        input_grid[i-1, j], input_grid[i+1, j],
                        input_grid[i, j-1], input_grid[i, j+1]
                    ]
                    output_grid[i, j] = int(np.mean(neighbors))
        
        return input_grid, output_grid
    
    def _generate_color_fade(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate fading color patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create base color
        base_color = 8
        input_grid[:, :size//3] = base_color
        
        # Create fade effect
        for j in range(size):
            fade_factor = 1.0 - (j / size)
            color = int(base_color * fade_factor)
            output_grid[:, j] = max(1, color)
        
        return input_grid, output_grid
    
    def _generate_color_wave(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate wave-like color patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create color wave
        for i in range(size):
            for j in range(size):
                wave = np.sin(i * np.pi / 4) + np.sin(j * np.pi / 3)
                color = int((wave + 2) * 2) % 8 + 1
                input_grid[i, j] = color
        
        # Transform: phase shift
        for i in range(size):
            for j in range(size):
                wave = np.sin((i + 2) * np.pi / 4) + np.sin((j + 2) * np.pi / 3)
                color = int((wave + 2) * 2) % 8 + 1
                output_grid[i, j] = color
        
        return input_grid, output_grid
    
    def _generate_color_clustering(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate clustered color patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create random points
        num_points = min(10, size)
        points = []
        for _ in range(num_points):
            x = random.randint(0, size-1)
            y = random.randint(0, size-1)
            color = random.randint(1, 6)
            points.append((x, y, color))
            input_grid[y, x] = color
        
        # Voronoi-like clustering
        for i in range(size):
            for j in range(size):
                if output_grid[i, j] == 0:
                    # Find nearest point
                    min_dist = float('inf')
                    nearest_color = 1
                    for x, y, color in points:
                        dist = (i - y)**2 + (j - x)**2
                        if dist < min_dist:
                            min_dist = dist
                            nearest_color = color
                    output_grid[i, j] = nearest_color
        
        return input_grid, output_grid
    
    def _generate_color_symmetry(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate color symmetry patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create half pattern with colors
        half = size // 2
        for i in range(size):
            for j in range(half):
                color = ((i + j) % 7) + 1
                input_grid[i, j] = color
        
        # Complete with color symmetry
        output_grid = input_grid.copy()
        for i in range(size):
            for j in range(half, size):
                output_grid[i, j] = output_grid[i, size - 1 - j]
        
        return input_grid, output_grid
    
    def _generate_color_propagation(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate color propagation patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Seed colors
        seeds = [(size//4, size//4, 2), (3*size//4, size//4, 3), 
                 (size//2, 3*size//4, 4)]
        
        for x, y, color in seeds:
            if x < size and y < size:
                input_grid[y, x] = color
        
        # Propagate colors
        output_grid = input_grid.copy()
        for _ in range(size//2):
            temp_grid = output_grid.copy()
            for i in range(1, size-1):
                for j in range(1, size-1):
                    if output_grid[i, j] == 0:
                        # Check neighbors
                        neighbors = [
                            temp_grid[i-1, j], temp_grid[i+1, j],
                            temp_grid[i, j-1], temp_grid[i, j+1]
                        ]
                        non_zero = [n for n in neighbors if n > 0]
                        if non_zero:
                            output_grid[i, j] = random.choice(non_zero)
        
        return input_grid, output_grid
    
    def _generate_rainbow(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rainbow patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Partial rainbow
        colors = [1, 2, 3, 4, 5, 6, 7]
        band_width = max(1, size // len(colors))
        
        for i in range(size//2):
            color_idx = min(i // band_width, len(colors) - 1)
            input_grid[i, :] = colors[color_idx]
        
        # Complete rainbow
        for i in range(size):
            color_idx = min(i // band_width, len(colors) - 1)
            output_grid[i, :] = colors[color_idx]
        
        return input_grid, output_grid
    
    def _generate_color_mixing(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate color mixing patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create two color regions
        input_grid[:size//2, :] = 2  # Blue
        input_grid[size//2:, :] = 4  # Yellow
        
        # Mix colors in the middle
        output_grid = input_grid.copy()
        mix_zone = max(1, size // 8)
        mid = size // 2
        
        for i in range(mid - mix_zone, mid + mix_zone):
            if 0 <= i < size:
                output_grid[i, :] = 3  # Green (blue + yellow)
        
        return input_grid, output_grid
    
    def _generate_perceptual_grouping(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate perceptual grouping patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create scattered colored dots
        num_dots = size * size // 10
        colors = [1, 2, 3]
        
        for _ in range(num_dots):
            x = random.randint(0, size-1)
            y = random.randint(0, size-1)
            color = random.choice(colors)
            input_grid[y, x] = color
        
        # Group by color proximity
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Find connected components for each color
        for color in colors:
            mask = (input_grid == color)
            # Simple flood fill for each color region
            for i in range(size):
                for j in range(size):
                    if mask[i, j]:
                        # Expand region
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < size and 0 <= nj < size:
                                    if input_grid[ni, nj] == 0:
                                        output_grid[ni, nj] = color
        
        # Include original dots
        output_grid[input_grid > 0] = input_grid[input_grid > 0]
        
        return input_grid, output_grid


class IrisLEAPTrainer:
    """IRIS-specific LEAP trainer focusing on color perception"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.pattern_generator = IrisPatternGenerator()
        self.pattern_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        self.difficulty_schedule = {
            0: 'basic',
            10: 'simple', 
            20: 'medium',
            30: 'complex'
        }
        self.weak_patterns: Set[str] = set()
        self.color_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    def generate_leap_batch(self, batch_size: int = 64, stage: int = 0, grid_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Generate a batch of IRIS-specific LEAP patterns"""
        
        # Determine complexity based on stage
        complexity = 'basic'
        for threshold, level in self.difficulty_schedule.items():
            if stage >= threshold:
                complexity = level
        
        # Get appropriate pattern types
        available_patterns = self.pattern_generator.complexity_levels[complexity]
        
        # Prioritize weak patterns
        if self.weak_patterns:
            pattern_weights = {
                p: 3.0 if p in self.weak_patterns else 1.0 
                for p in available_patterns
            }
        else:
            pattern_weights = {p: 1.0 for p in available_patterns}
        
        # Generate batch
        inputs = []
        outputs = []
        pattern_types = []
        
        for _ in range(batch_size):
            # Select pattern type
            pattern_type = random.choices(
                list(pattern_weights.keys()),
                weights=list(pattern_weights.values())
            )[0]
            
            # Generate pattern
            if grid_size is None:
                grid_size = random.randint(6, min(30, 6 + stage))
            
            input_grid, output_grid = self.pattern_generator.generate_pattern(
                pattern_type, grid_size, complexity
            )
            
            inputs.append(torch.tensor(input_grid, dtype=torch.long))
            outputs.append(torch.tensor(output_grid, dtype=torch.long))
            pattern_types.append(pattern_type)
        
        # Stack into batch
        max_h = max(inp.shape[0] for inp in inputs)
        max_w = max(inp.shape[1] for inp in inputs)
        
        batch_inputs = torch.zeros(batch_size, max_h, max_w, dtype=torch.long)
        batch_outputs = torch.zeros(batch_size, max_h, max_w, dtype=torch.long)
        
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            h, w = inp.shape
            batch_inputs[i, :h, :w] = inp
            batch_outputs[i, :h, :w] = out
        
        return {
            'inputs': batch_inputs,
            'outputs': batch_outputs,
            'pattern_types': pattern_types,
            'complexity': complexity,
            'grid_sizes': [(inp.shape[0], inp.shape[1]) for inp in inputs]
        }
    
    def update_pattern_stats(self, pattern_types: List[str], predictions: torch.Tensor, targets: torch.Tensor):
        """Update statistics for pattern performance"""
        pred_indices = predictions.argmax(dim=1)
        target_indices = targets.argmax(dim=1)
        
        for i, pattern_type in enumerate(pattern_types):
            self.pattern_stats[pattern_type]['attempts'] += 1
            
            if (pred_indices[i] == target_indices[i]).all():
                self.pattern_stats[pattern_type]['successes'] += 1
            
            # Track color-specific accuracy
            pred_colors = torch.unique(pred_indices[i])
            target_colors = torch.unique(target_indices[i])
            
            for color in target_colors:
                self.color_accuracy[color.item()]['total'] += 1
                if color in pred_colors:
                    mask_pred = (pred_indices[i] == color)
                    mask_target = (target_indices[i] == color)
                    accuracy = (mask_pred & mask_target).sum().float() / mask_target.sum().float()
                    if accuracy > 0.8:  # 80% threshold
                        self.color_accuracy[color.item()]['correct'] += 1
        
        # Update weak patterns
        self.weak_patterns.clear()
        for pattern, stats in self.pattern_stats.items():
            if stats['attempts'] >= 10:
                success_rate = stats['successes'] / stats['attempts']
                if success_rate < 0.7:
                    self.weak_patterns.add(pattern)
    
    def analyze_color_perception_gaps(self) -> Dict[str, float]:
        """Analyze which colors IRIS struggles with"""
        gaps = {}
        
        for color, stats in self.color_accuracy.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                gaps[f'color_{color}'] = 1.0 - accuracy
        
        return gaps
    
    def get_performance_report(self) -> str:
        """Get IRIS-specific performance report"""
        if not self.pattern_stats:
            return "No IRIS LEAP patterns trained yet"
        
        report_lines = ["IRIS LEAP Color Perception Performance:"]
        
        # Pattern performance
        sorted_patterns = sorted(
            self.pattern_stats.items(),
            key=lambda x: x[1]['successes'] / max(1, x[1]['attempts'])
        )
        
        for pattern, stats in sorted_patterns:
            if stats['attempts'] > 0:
                success_rate = stats['successes'] / stats['attempts'] * 100
                status = "⚠️" if pattern in self.weak_patterns else "✅"
                report_lines.append(
                    f"  {status} {pattern}: {success_rate:.1f}% "
                    f"({stats['successes']}/{stats['attempts']})"
                )
        
        # Color-specific performance
        report_lines.append("\nColor Recognition Accuracy:")
        for color in range(10):
            if color in self.color_accuracy and self.color_accuracy[color]['total'] > 0:
                acc = self.color_accuracy[color]['correct'] / self.color_accuracy[color]['total'] * 100
                report_lines.append(f"  Color {color}: {acc:.1f}%")
        
        # Summary
        total_attempts = sum(s['attempts'] for s in self.pattern_stats.values())
        total_successes = sum(s['successes'] for s in self.pattern_stats.values())
        
        if total_attempts > 0:
            overall_rate = total_successes / total_attempts * 100
            report_lines.append(f"\nOverall Color Pattern Recognition: {overall_rate:.1f}%")
            
            if self.weak_patterns:
                report_lines.append(f"Weak patterns: {', '.join(self.weak_patterns)}")
        
        return "\n".join(report_lines)


def create_iris_leap_system(device='cuda') -> Dict:
    """Create IRIS-specific LEAP components"""
    return {
        'trainer': IrisLEAPTrainer(device),
        'pattern_generator': IrisPatternGenerator(),
    }