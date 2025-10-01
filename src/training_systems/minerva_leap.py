"""
MINERVA-specific LEAP (Learning Enhancement through Adaptive Patterns) System
Focuses on grid reasoning patterns and spatial relationships
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random


class MinervaPatternGenerator:
    """Generates grid-based patterns specifically for MINERVA's capabilities"""
    
    def __init__(self):
        self.pattern_types = [
            'grid_lines', 'checkerboard', 'symmetry', 'frames', 'corners',
            'diagonals', 'quadrants', 'nested_squares', 'spirals', 'fractals',
            'object_boundaries', 'connected_regions', 'repeating_motifs'
        ]
        self.complexity_levels = {
            'basic': ['grid_lines', 'checkerboard', 'frames', 'corners'],
            'simple': ['symmetry', 'diagonals', 'quadrants', 'repeating_motifs'],
            'medium': ['nested_squares', 'spirals', 'object_boundaries'],
            'complex': ['fractals', 'connected_regions']
        }
    
    def generate_pattern(self, pattern_type: str, grid_size: int = 10, complexity: str = 'basic') -> Tuple[np.ndarray, np.ndarray]:
        """Generate input-output pairs for specific grid patterns"""
        
        if pattern_type == 'grid_lines':
            return self._generate_grid_lines(grid_size)
        elif pattern_type == 'checkerboard':
            return self._generate_checkerboard(grid_size)
        elif pattern_type == 'symmetry':
            return self._generate_symmetry(grid_size)
        elif pattern_type == 'frames':
            return self._generate_frames(grid_size)
        elif pattern_type == 'corners':
            return self._generate_corners(grid_size)
        elif pattern_type == 'diagonals':
            return self._generate_diagonals(grid_size)
        elif pattern_type == 'quadrants':
            return self._generate_quadrants(grid_size)
        elif pattern_type == 'nested_squares':
            return self._generate_nested_squares(grid_size)
        elif pattern_type == 'spirals':
            return self._generate_spirals(grid_size)
        elif pattern_type == 'fractals':
            return self._generate_fractals(grid_size)
        elif pattern_type == 'object_boundaries':
            return self._generate_object_boundaries(grid_size)
        elif pattern_type == 'connected_regions':
            return self._generate_connected_regions(grid_size)
        elif pattern_type == 'repeating_motifs':
            return self._generate_repeating_motifs(grid_size)
        else:
            # Default to simple pattern
            return self._generate_grid_lines(grid_size)
    
    def _generate_grid_lines(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate horizontal and vertical line patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create input with sparse lines
        for i in range(0, size, 3):
            input_grid[i, :] = 1
            input_grid[:, i] = 2
        
        # Output: fill regions between lines
        output_grid = input_grid.copy()
        for i in range(size):
            for j in range(size):
                if input_grid[i, j] == 0:
                    output_grid[i, j] = 3
        
        return input_grid, output_grid
    
    def _generate_checkerboard(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate checkerboard patterns with transformations"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create checkerboard
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    input_grid[i, j] = 1
                else:
                    input_grid[i, j] = 2
        
        # Transform: rotate colors
        output_grid = np.where(input_grid == 1, 2, 1)
        
        return input_grid, output_grid
    
    def _generate_symmetry(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate patterns that test symmetry detection"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create half pattern
        half = size // 2
        for i in range(size):
            for j in range(half):
                input_grid[i, j] = random.randint(0, 3)
        
        # Complete with symmetry
        output_grid = input_grid.copy()
        for i in range(size):
            for j in range(half, size):
                output_grid[i, j] = output_grid[i, size - 1 - j]
        
        return input_grid, output_grid
    
    def _generate_frames(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate frame/border patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create multiple frames
        for frame in range(min(3, size // 2)):
            color = frame + 1
            # Top and bottom
            input_grid[frame, frame:size-frame] = color
            input_grid[size-1-frame, frame:size-frame] = color
            # Left and right
            input_grid[frame:size-frame, frame] = color
            input_grid[frame:size-frame, size-1-frame] = color
        
        # Transform: fill interior
        output_grid = input_grid.copy()
        center = size // 2
        if size > 6:
            output_grid[center-1:center+2, center-1:center+2] = 4
        
        return input_grid, output_grid
    
    def _generate_corners(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate corner-based patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Place objects in corners
        corner_size = max(2, size // 4)
        input_grid[:corner_size, :corner_size] = 1  # Top-left
        input_grid[:corner_size, -corner_size:] = 2  # Top-right
        input_grid[-corner_size:, :corner_size] = 3  # Bottom-left
        input_grid[-corner_size:, -corner_size:] = 4  # Bottom-right
        
        # Transform: connect corners
        output_grid = input_grid.copy()
        mid = size // 2
        output_grid[mid-1:mid+1, :] = 5
        output_grid[:, mid-1:mid+1] = 5
        
        return input_grid, output_grid
    
    def _generate_diagonals(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate diagonal patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Main diagonal
        for i in range(size):
            input_grid[i, i] = 1
            if size - 1 - i >= 0:
                input_grid[i, size - 1 - i] = 2
        
        # Transform: thicken diagonals
        for i in range(size):
            for j in range(size):
                if abs(i - j) <= 1:
                    output_grid[i, j] = 1
                if abs(i - (size - 1 - j)) <= 1:
                    output_grid[i, j] = 2
        
        return input_grid, output_grid
    
    def _generate_quadrants(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate quadrant-based patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        mid_x, mid_y = size // 2, size // 2
        
        # Create different patterns in each quadrant
        input_grid[:mid_x, :mid_y] = 1  # Top-left
        input_grid[:mid_x, mid_y:] = 2  # Top-right
        input_grid[mid_x:, :mid_y] = 3  # Bottom-left
        input_grid[mid_x:, mid_y:] = 4  # Bottom-right
        
        # Transform: rotate quadrants
        output_grid = np.zeros((size, size), dtype=np.int32)
        output_grid[:mid_x, :mid_y] = input_grid[mid_x:, :mid_y]  # BL -> TL
        output_grid[:mid_x, mid_y:] = input_grid[:mid_x, :mid_y]  # TL -> TR
        output_grid[mid_x:, mid_y:] = input_grid[:mid_x, mid_y:]  # TR -> BR
        output_grid[mid_x:, :mid_y] = input_grid[mid_x:, mid_y:]  # BR -> BL
        
        return input_grid, output_grid
    
    def _generate_nested_squares(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate nested square patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create nested squares
        for layer in range(size // 2):
            color = (layer % 4) + 1
            input_grid[layer:size-layer, layer:size-layer] = color
        
        # Transform: extract specific layer
        target_layer = 1
        for i in range(size):
            for j in range(size):
                dist_to_edge = min(i, j, size-1-i, size-1-j)
                if dist_to_edge == target_layer:
                    output_grid[i, j] = input_grid[i, j]
        
        return input_grid, output_grid
    
    def _generate_spirals(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate spiral patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create outward spiral
        x, y = size // 2, size // 2
        dx, dy = 0, -1
        color = 1
        
        for i in range(size * size):
            if 0 <= x < size and 0 <= y < size:
                input_grid[y, x] = color
                color = (color % 4) + 1
            
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy
        
        # Transform: reverse spiral colors
        output_grid = 5 - input_grid
        
        return input_grid, output_grid
    
    def _generate_fractals(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simple fractal-like patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        output_grid = np.zeros((size, size), dtype=np.int32)
        
        def sierpinski_triangle(x, y, s, depth, color):
            if depth == 0 or s < 2:
                input_grid[y:y+s, x:x+s] = color
            else:
                half = s // 2
                sierpinski_triangle(x, y, half, depth-1, color)
                sierpinski_triangle(x + half, y, half, depth-1, (color % 4) + 1)
                sierpinski_triangle(x + half//2, y + half, half, depth-1, (color % 4) + 1)
        
        sierpinski_triangle(0, 0, size, 3, 1)
        
        # Transform: fill empty spaces
        output_grid = input_grid.copy()
        output_grid[output_grid == 0] = 5
        
        return input_grid, output_grid
    
    def _generate_object_boundaries(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate patterns focused on object boundaries"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create random objects
        num_objects = min(3, size // 3)
        for obj in range(num_objects):
            x = random.randint(1, size - 4)
            y = random.randint(1, size - 4)
            w = random.randint(2, min(4, size - x - 1))
            h = random.randint(2, min(4, size - y - 1))
            input_grid[y:y+h, x:x+w] = obj + 1
        
        # Extract boundaries
        output_grid = np.zeros((size, size), dtype=np.int32)
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                if input_grid[i, j] != 0:
                    # Check if on boundary
                    neighbors = [
                        input_grid[i-1, j], input_grid[i+1, j],
                        input_grid[i, j-1], input_grid[i, j+1]
                    ]
                    if 0 in neighbors or any(n != input_grid[i, j] for n in neighbors if n != 0):
                        output_grid[i, j] = input_grid[i, j]
        
        return input_grid, output_grid
    
    def _generate_connected_regions(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate patterns with connected regions"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create regions
        seeds = [(size//4, size//4), (3*size//4, size//4), (size//2, 3*size//4)]
        for idx, (x, y) in enumerate(seeds):
            color = idx + 1
            # Grow region from seed
            stack = [(x, y)]
            grown = 0
            max_grow = size * size // 4
            
            while stack and grown < max_grow:
                cx, cy = stack.pop()
                if 0 <= cx < size and 0 <= cy < size and input_grid[cy, cx] == 0:
                    input_grid[cy, cx] = color
                    grown += 1
                    
                    # Add neighbors with probability
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if random.random() > 0.3:
                            stack.append((cx + dx, cy + dy))
        
        # Transform: merge touching regions
        output_grid = input_grid.copy()
        # Simple merge logic - would be more complex in practice
        
        return input_grid, output_grid
    
    def _generate_repeating_motifs(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate repeating motif patterns"""
        input_grid = np.zeros((size, size), dtype=np.int32)
        
        # Create a small motif
        motif = np.array([[1, 2], [2, 1]])
        motif_h, motif_w = motif.shape
        
        # Tile the motif
        for i in range(0, size - motif_h + 1, motif_h):
            for j in range(0, size - motif_w + 1, motif_w):
                input_grid[i:i+motif_h, j:j+motif_w] = motif
        
        # Transform: shift motif
        output_grid = np.roll(input_grid, shift=(1, 1), axis=(0, 1))
        
        return input_grid, output_grid


class MinervaLEAPTrainer:
    """MINERVA-specific LEAP trainer focusing on grid reasoning"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.pattern_generator = MinervaPatternGenerator()
        self.pattern_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        self.difficulty_schedule = {
            0: 'basic',
            10: 'simple', 
            20: 'medium',
            30: 'complex'
        }
        self.weak_patterns: Set[str] = set()
        self.pattern_history = []
    
    def generate_leap_batch(self, batch_size: int = 64, stage: int = 0, grid_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Generate a batch of MINERVA-specific LEAP patterns"""
        
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
        
        # Update weak patterns
        self.weak_patterns.clear()
        for pattern, stats in self.pattern_stats.items():
            if stats['attempts'] >= 10:
                success_rate = stats['successes'] / stats['attempts']
                if success_rate < 0.7:
                    self.weak_patterns.add(pattern)
    
    def analyze_grid_reasoning_gaps(self) -> Dict[str, float]:
        """Analyze which grid reasoning capabilities need improvement"""
        gaps = {}
        
        for pattern, stats in self.pattern_stats.items():
            if stats['attempts'] > 0:
                success_rate = stats['successes'] / stats['attempts']
                gaps[pattern] = 1.0 - success_rate
        
        return gaps
    
    def get_performance_report(self) -> str:
        """Get MINERVA-specific performance report"""
        if not self.pattern_stats:
            return "No MINERVA LEAP patterns trained yet"
        
        report_lines = ["MINERVA LEAP Grid Reasoning Performance:"]
        
        # Sort by success rate
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
        
        # Add summary
        total_attempts = sum(s['attempts'] for s in self.pattern_stats.values())
        total_successes = sum(s['successes'] for s in self.pattern_stats.values())
        
        if total_attempts > 0:
            overall_rate = total_successes / total_attempts * 100
            report_lines.append(f"\nOverall Grid Reasoning: {overall_rate:.1f}%")
            
            if self.weak_patterns:
                report_lines.append(f"Weak patterns: {', '.join(self.weak_patterns)}")
        
        return "\n".join(report_lines)


def create_minerva_leap_system(device='cuda') -> Dict:
    """Create MINERVA-specific LEAP components"""
    return {
        'trainer': MinervaLEAPTrainer(device),
        'pattern_generator': MinervaPatternGenerator(),
    }