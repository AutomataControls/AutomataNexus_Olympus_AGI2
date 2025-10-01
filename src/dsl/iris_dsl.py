"""
IRIS-specific DSL integration for color patterns and transformations
COMPLETELY INDEPENDENT - No imports from base DSL
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum


class IrisOperation(Enum):
    """IRIS-specific operations for color perception and transformation"""
    # Basic color transformations
    SWAP_COLORS = "swap_colors"
    INVERT_COLORS = "invert_colors"
    CYCLE_COLORS = "cycle_colors"
    MAP_COLORS = "map_colors"
    
    # Color filtering
    FILTER_BY_COLOR = "filter_by_color"
    EXTRACT_COLOR_MASK = "extract_color_mask"
    REMOVE_COLOR = "remove_color"
    ISOLATE_COLOR = "isolate_color"
    
    # Gradient operations
    CREATE_GRADIENT = "create_gradient"
    APPLY_COLOR_FADE = "apply_color_fade"
    COLOR_DISTANCE_MAP = "color_distance_map"
    RADIAL_GRADIENT = "radial_gradient"
    
    # Multi-color patterns
    ALTERNATE_COLORS = "alternate_colors"
    RAINBOW_PATTERN = "rainbow_pattern"
    COLOR_BY_REGION = "color_by_region"
    CHECKERBOARD_COLORS = "checkerboard_colors"
    
    # Color mixing
    BLEND_COLORS = "blend_colors"
    COLOR_ARITHMETIC = "color_arithmetic"
    CONDITIONAL_COLOR = "conditional_color"
    MIX_ADJACENT = "mix_adjacent"
    
    # Perceptual operations
    COLOR_CLUSTERING = "color_clustering"
    PALETTE_REDUCTION = "palette_reduction"
    PERCEPTUAL_GROUPING = "perceptual_grouping"
    COLOR_QUANTIZATION = "color_quantization"
    
    # Advanced color ops
    COLOR_WAVE = "color_wave"
    COLOR_SPIRAL = "color_spiral"
    COLOR_MOSAIC = "color_mosaic"
    COLOR_PROPAGATION = "color_propagation"


class IrisDSLProgram:
    """IRIS-specific DSL program representation"""
    
    def __init__(self, operations: List[Tuple[IrisOperation, Dict[str, Any]]]):
        self.operations = operations
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute the IRIS DSL program on input grid"""
        result = grid.copy()
        for op, params in self.operations:
            result = IrisDSLExecutor.execute_operation(result, op, params)
        return result
    
    def to_string(self) -> str:
        """Convert program to readable string"""
        prog_str = []
        for op, params in self.operations:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            prog_str.append(f"{op.value}({param_str})")
        return " -> ".join(prog_str)


class IrisDSLExecutor:
    """Executes IRIS-specific DSL operations"""
    
    @staticmethod
    def execute_operation(grid: np.ndarray, op: IrisOperation, params: Dict[str, Any]) -> np.ndarray:
        """Execute a single IRIS DSL operation"""
        
        # Basic color transformations
        if op == IrisOperation.SWAP_COLORS:
            return IrisDSLExecutor._swap_colors(grid, params)
        elif op == IrisOperation.INVERT_COLORS:
            return IrisDSLExecutor._invert_colors(grid, params)
        elif op == IrisOperation.CYCLE_COLORS:
            return IrisDSLExecutor._cycle_colors(grid, params)
        elif op == IrisOperation.MAP_COLORS:
            return IrisDSLExecutor._map_colors(grid, params)
        
        # Color filtering
        elif op == IrisOperation.FILTER_BY_COLOR:
            return IrisDSLExecutor._filter_by_color(grid, params)
        elif op == IrisOperation.EXTRACT_COLOR_MASK:
            return IrisDSLExecutor._extract_color_mask(grid, params)
        elif op == IrisOperation.REMOVE_COLOR:
            return IrisDSLExecutor._remove_color(grid, params)
        elif op == IrisOperation.ISOLATE_COLOR:
            return IrisDSLExecutor._isolate_color(grid, params)
        
        # Gradient operations
        elif op == IrisOperation.CREATE_GRADIENT:
            return IrisDSLExecutor._create_gradient(grid, params)
        elif op == IrisOperation.APPLY_COLOR_FADE:
            return IrisDSLExecutor._apply_color_fade(grid, params)
        elif op == IrisOperation.COLOR_DISTANCE_MAP:
            return IrisDSLExecutor._color_distance_map(grid, params)
        elif op == IrisOperation.RADIAL_GRADIENT:
            return IrisDSLExecutor._radial_gradient(grid, params)
        
        # Multi-color patterns
        elif op == IrisOperation.ALTERNATE_COLORS:
            return IrisDSLExecutor._alternate_colors(grid, params)
        elif op == IrisOperation.RAINBOW_PATTERN:
            return IrisDSLExecutor._rainbow_pattern(grid, params)
        elif op == IrisOperation.COLOR_BY_REGION:
            return IrisDSLExecutor._color_by_region(grid, params)
        elif op == IrisOperation.CHECKERBOARD_COLORS:
            return IrisDSLExecutor._checkerboard_colors(grid, params)
        
        # Color mixing
        elif op == IrisOperation.BLEND_COLORS:
            return IrisDSLExecutor._blend_colors(grid, params)
        elif op == IrisOperation.COLOR_ARITHMETIC:
            return IrisDSLExecutor._color_arithmetic(grid, params)
        elif op == IrisOperation.CONDITIONAL_COLOR:
            return IrisDSLExecutor._conditional_color(grid, params)
        elif op == IrisOperation.MIX_ADJACENT:
            return IrisDSLExecutor._mix_adjacent(grid, params)
        
        # Perceptual operations
        elif op == IrisOperation.COLOR_CLUSTERING:
            return IrisDSLExecutor._color_clustering(grid, params)
        elif op == IrisOperation.PALETTE_REDUCTION:
            return IrisDSLExecutor._palette_reduction(grid, params)
        elif op == IrisOperation.PERCEPTUAL_GROUPING:
            return IrisDSLExecutor._perceptual_grouping(grid, params)
        elif op == IrisOperation.COLOR_QUANTIZATION:
            return IrisDSLExecutor._color_quantization(grid, params)
        
        # Advanced color operations
        elif op == IrisOperation.COLOR_WAVE:
            return IrisDSLExecutor._color_wave(grid, params)
        elif op == IrisOperation.COLOR_SPIRAL:
            return IrisDSLExecutor._color_spiral(grid, params)
        elif op == IrisOperation.COLOR_MOSAIC:
            return IrisDSLExecutor._color_mosaic(grid, params)
        elif op == IrisOperation.COLOR_PROPAGATION:
            return IrisDSLExecutor._color_propagation(grid, params)
        
        else:
            return grid
    
    @staticmethod
    def _swap_colors(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Swap two colors"""
        color1 = params.get('color1', 0)
        color2 = params.get('color2', 1)
        result = grid.copy()
        
        mask1 = grid == color1
        mask2 = grid == color2
        result[mask1] = color2
        result[mask2] = color1
        
        return result
    
    @staticmethod
    def _invert_colors(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Invert color values"""
        max_color = params.get('max_color', 9)
        return max_color - grid
    
    @staticmethod
    def _cycle_colors(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Cycle colors by offset"""
        offset = params.get('offset', 1)
        max_color = params.get('max_color', 9)
        return (grid + offset) % (max_color + 1)
    
    @staticmethod
    def _map_colors(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Map colors according to dictionary"""
        mapping = params.get('mapping', {})
        result = grid.copy()
        
        for old_color, new_color in mapping.items():
            result[grid == old_color] = new_color
        
        return result
    
    @staticmethod
    def _filter_by_color(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Keep only specified colors"""
        colors_to_keep = params.get('colors', [1, 2, 3])
        background = params.get('background', 0)
        result = np.full_like(grid, background)
        
        for color in colors_to_keep:
            result[grid == color] = color
        
        return result
    
    @staticmethod
    def _extract_color_mask(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract binary mask for color"""
        target_color = params.get('color', 1)
        result = np.zeros_like(grid)
        result[grid == target_color] = 1
        return result
    
    @staticmethod
    def _remove_color(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Remove specific color"""
        color_to_remove = params.get('color', 0)
        replacement = params.get('replacement', 0)
        result = grid.copy()
        result[grid == color_to_remove] = replacement
        return result
    
    @staticmethod
    def _isolate_color(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Isolate single color, set others to background"""
        target_color = params.get('color', 1)
        background = params.get('background', 0)
        result = np.full_like(grid, background)
        result[grid == target_color] = target_color
        return result
    
    @staticmethod
    def _create_gradient(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create color gradient"""
        direction = params.get('direction', 'horizontal')
        start_color = params.get('start', 0)
        end_color = params.get('end', 9)
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        if direction == 'horizontal':
            for j in range(w):
                color = start_color + (end_color - start_color) * j // max(1, w - 1)
                result[:, j] = color
        elif direction == 'vertical':
            for i in range(h):
                color = start_color + (end_color - start_color) * i // max(1, h - 1)
                result[i, :] = color
        elif direction == 'diagonal':
            for i in range(h):
                for j in range(w):
                    progress = (i + j) / max(1, h + w - 2)
                    color = int(start_color + (end_color - start_color) * progress)
                    result[i, j] = color
        
        return result
    
    @staticmethod
    def _apply_color_fade(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply fading effect to colors"""
        fade_direction = params.get('direction', 'right')
        fade_strength = params.get('strength', 0.5)
        
        h, w = grid.shape
        result = grid.copy().astype(float)
        
        if fade_direction == 'right':
            for j in range(w):
                factor = 1.0 - (j / w) * fade_strength
                result[:, j] *= factor
        elif fade_direction == 'left':
            for j in range(w):
                factor = 1.0 - ((w - 1 - j) / w) * fade_strength
                result[:, j] *= factor
        elif fade_direction == 'down':
            for i in range(h):
                factor = 1.0 - (i / h) * fade_strength
                result[i, :] *= factor
        elif fade_direction == 'up':
            for i in range(h):
                factor = 1.0 - ((h - 1 - i) / h) * fade_strength
                result[i, :] *= factor
        
        return np.round(result).astype(np.int32)
    
    @staticmethod
    def _color_distance_map(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create map based on distance from color"""
        reference_color = params.get('reference', 5)
        result = np.abs(grid - reference_color)
        return np.clip(result, 0, 9)
    
    @staticmethod
    def _radial_gradient(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create radial color gradient"""
        center_color = params.get('center', 0)
        edge_color = params.get('edge', 9)
        
        h, w = grid.shape
        center_h, center_w = h // 2, w // 2
        max_dist = np.sqrt(center_h**2 + center_w**2)
        
        result = np.zeros_like(grid)
        
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                progress = dist / max_dist
                color = int(center_color + (edge_color - center_color) * progress)
                result[i, j] = np.clip(color, 0, 9)
        
        return result
    
    @staticmethod
    def _alternate_colors(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create alternating color pattern"""
        colors = params.get('colors', [1, 2, 3])
        pattern = params.get('pattern', 'row')
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        if pattern == 'row':
            for i in range(h):
                result[i, :] = colors[i % len(colors)]
        elif pattern == 'column':
            for j in range(w):
                result[:, j] = colors[j % len(colors)]
        elif pattern == 'diagonal':
            for i in range(h):
                for j in range(w):
                    result[i, j] = colors[(i + j) % len(colors)]
        
        return result
    
    @staticmethod
    def _rainbow_pattern(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create rainbow pattern"""
        direction = params.get('direction', 'horizontal')
        colors = [1, 2, 3, 4, 5, 6, 7]  # Rainbow colors
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        if direction == 'horizontal':
            band_width = max(1, w // len(colors))
            for j in range(w):
                color_idx = min(j // band_width, len(colors) - 1)
                result[:, j] = colors[color_idx]
        else:
            band_width = max(1, h // len(colors))
            for i in range(h):
                color_idx = min(i // band_width, len(colors) - 1)
                result[i, :] = colors[color_idx]
        
        return result
    
    @staticmethod
    def _color_by_region(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Color by connected regions"""
        background = params.get('background', 0)
        
        try:
            from scipy import ndimage
            # Label connected components
            labeled, num_features = ndimage.label(grid != background)
            
            result = np.zeros_like(grid)
            for i in range(1, min(num_features + 1, 10)):
                result[labeled == i] = i
            
            return result
        except:
            # Fallback: simple region coloring
            return grid
    
    @staticmethod
    def _checkerboard_colors(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create checkerboard with specified colors"""
        color1 = params.get('color1', 0)
        color2 = params.get('color2', 1)
        size = params.get('size', 1)
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        for i in range(h):
            for j in range(w):
                if ((i // size) + (j // size)) % 2 == 0:
                    result[i, j] = color1
                else:
                    result[i, j] = color2
        
        return result
    
    @staticmethod
    def _blend_colors(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Blend adjacent colors"""
        mode = params.get('mode', 'average')
        
        h, w = grid.shape
        result = grid.copy().astype(float)
        
        if mode == 'average':
            # Average with neighbors
            for i in range(1, h-1):
                for j in range(1, w-1):
                    neighbors = [
                        grid[i-1, j], grid[i+1, j],
                        grid[i, j-1], grid[i, j+1]
                    ]
                    result[i, j] = np.mean(neighbors)
        
        return np.round(result).astype(np.int32)
    
    @staticmethod
    def _color_arithmetic(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Perform arithmetic operations on colors"""
        operation = params.get('operation', 'add')
        value = params.get('value', 1)
        
        if operation == 'add':
            result = grid + value
        elif operation == 'subtract':
            result = grid - value
        elif operation == 'multiply':
            result = grid * value
        elif operation == 'divide':
            result = grid // max(1, value)
        else:
            result = grid
        
        return np.clip(result, 0, 9)
    
    @staticmethod
    def _conditional_color(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply color based on condition"""
        condition = params.get('condition', 'threshold')
        threshold = params.get('threshold', 5)
        true_color = params.get('true_color', 9)
        false_color = params.get('false_color', 0)
        
        if condition == 'threshold':
            result = np.where(grid >= threshold, true_color, false_color)
        elif condition == 'even':
            result = np.where(grid % 2 == 0, true_color, false_color)
        elif condition == 'odd':
            result = np.where(grid % 2 == 1, true_color, false_color)
        else:
            result = grid
        
        return result
    
    @staticmethod
    def _mix_adjacent(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Mix colors with adjacent cells"""
        direction = params.get('direction', 'horizontal')
        
        h, w = grid.shape
        result = grid.copy()
        
        if direction == 'horizontal' and w > 1:
            for i in range(h):
                for j in range(1, w):
                    result[i, j] = (grid[i, j] + grid[i, j-1]) // 2
        elif direction == 'vertical' and h > 1:
            for i in range(1, h):
                for j in range(w):
                    result[i, j] = (grid[i, j] + grid[i-1, j]) // 2
        
        return result
    
    @staticmethod
    def _color_clustering(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Cluster similar colors together"""
        n_clusters = params.get('n_clusters', 3)
        
        # Simple clustering by value ranges
        unique_colors = np.unique(grid)
        if len(unique_colors) <= n_clusters:
            return grid
        
        # Create clusters
        result = np.zeros_like(grid)
        cluster_size = 10 // n_clusters
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cluster_id = min(grid[i, j] // cluster_size, n_clusters - 1)
                result[i, j] = cluster_id * cluster_size
        
        return result
    
    @staticmethod
    def _palette_reduction(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Reduce color palette"""
        n_colors = params.get('n_colors', 4)
        
        # Get unique colors sorted by frequency
        unique, counts = np.unique(grid, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        keep_colors = unique[sorted_idx][:n_colors]
        
        # Map other colors to nearest kept color
        result = grid.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] not in keep_colors:
                    # Find nearest kept color
                    distances = np.abs(keep_colors - grid[i, j])
                    nearest_idx = np.argmin(distances)
                    result[i, j] = keep_colors[nearest_idx]
        
        return result
    
    @staticmethod
    def _perceptual_grouping(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Group colors by perceptual similarity"""
        threshold = params.get('threshold', 2)
        
        h, w = grid.shape
        result = grid.copy()
        
        # Simple perceptual grouping by color distance
        processed = np.zeros((h, w), dtype=bool)
        group_color = 0
        
        for i in range(h):
            for j in range(w):
                if not processed[i, j]:
                    # Start new group
                    base_color = grid[i, j]
                    
                    # Find all similar colors
                    for ii in range(h):
                        for jj in range(w):
                            if not processed[ii, jj]:
                                if abs(grid[ii, jj] - base_color) <= threshold:
                                    result[ii, jj] = group_color
                                    processed[ii, jj] = True
                    
                    group_color = (group_color + 1) % 10
        
        return result
    
    @staticmethod
    def _color_quantization(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Quantize colors to levels"""
        levels = params.get('levels', 4)
        
        # Quantize to specified levels
        step = 10 // levels
        result = (grid // step) * step
        
        return np.clip(result, 0, 9)
    
    @staticmethod
    def _color_wave(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create wave-like color pattern"""
        frequency = params.get('frequency', 1)
        amplitude = params.get('amplitude', 3)
        phase = params.get('phase', 0)
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        for i in range(h):
            for j in range(w):
                wave = np.sin((i + phase) * frequency * np.pi / h) + \
                       np.sin((j + phase) * frequency * np.pi / w)
                color = int((wave + 2) * amplitude) % 10
                result[i, j] = color
        
        return result
    
    @staticmethod
    def _color_spiral(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create spiral color pattern"""
        colors = params.get('colors', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        # Generate spiral coordinates
        top, bottom = 0, h - 1
        left, right = 0, w - 1
        color_idx = 0
        
        while top <= bottom and left <= right:
            # Top row
            for j in range(left, right + 1):
                result[top, j] = colors[color_idx % len(colors)]
                color_idx += 1
            top += 1
            
            # Right column
            for i in range(top, bottom + 1):
                result[i, right] = colors[color_idx % len(colors)]
                color_idx += 1
            right -= 1
            
            # Bottom row
            if top <= bottom:
                for j in range(right, left - 1, -1):
                    result[bottom, j] = colors[color_idx % len(colors)]
                    color_idx += 1
                bottom -= 1
            
            # Left column
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    result[i, left] = colors[color_idx % len(colors)]
                    color_idx += 1
                left += 1
        
        return result
    
    @staticmethod
    def _color_mosaic(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create mosaic color pattern"""
        tile_size = params.get('tile_size', 2)
        colors = params.get('colors', [1, 2, 3, 4])
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                # Random color for each tile
                color_idx = ((i // tile_size) * (w // tile_size) + (j // tile_size)) % len(colors)
                color = colors[color_idx]
                
                # Fill tile
                for di in range(min(tile_size, h - i)):
                    for dj in range(min(tile_size, w - j)):
                        result[i + di, j + dj] = color
        
        return result
    
    @staticmethod
    def _color_propagation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Propagate colors from seed points"""
        iterations = params.get('iterations', 5)
        
        h, w = grid.shape
        result = grid.copy()
        
        # Find seed points (non-zero colors)
        seeds = []
        for i in range(h):
            for j in range(w):
                if grid[i, j] > 0:
                    seeds.append((i, j, grid[i, j]))
        
        # Propagate colors
        for _ in range(iterations):
            new_result = result.copy()
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if result[i, j] == 0:
                        # Check neighbors
                        neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w and result[ni, nj] > 0:
                                    neighbors.append(result[ni, nj])
                        
                        if neighbors:
                            # Take most common neighbor color
                            from collections import Counter
                            color_counts = Counter(neighbors)
                            new_result[i, j] = color_counts.most_common(1)[0][0]
            
            result = new_result
        
        return result


class IRISDSLGenerator:
    """Generates DSL programs specifically for IRIS's color perception capabilities"""
    
    @staticmethod
    def generate_color_pattern_programs() -> List[IrisDSLProgram]:
        """Generate programs focused on color-based patterns and transformations"""
        programs = []
        
        # Basic color transformations
        programs.append(IrisDSLProgram([(IrisOperation.SWAP_COLORS, {'color1': 1, 'color2': 2})]))
        programs.append(IrisDSLProgram([(IrisOperation.INVERT_COLORS, {'max_color': 9})]))
        programs.append(IrisDSLProgram([(IrisOperation.CYCLE_COLORS, {'offset': 1, 'max_color': 9})]))
        
        # Color filtering
        programs.append(IrisDSLProgram([(IrisOperation.FILTER_BY_COLOR, {'colors': [1, 2, 3], 'background': 0})]))
        programs.append(IrisDSLProgram([(IrisOperation.EXTRACT_COLOR_MASK, {'color': 1})]))
        programs.append(IrisDSLProgram([(IrisOperation.ISOLATE_COLOR, {'color': 2, 'background': 0})]))
        
        # Gradient patterns
        programs.append(IrisDSLProgram([(IrisOperation.CREATE_GRADIENT, {'direction': 'horizontal', 'start': 0, 'end': 9})]))
        programs.append(IrisDSLProgram([(IrisOperation.CREATE_GRADIENT, {'direction': 'vertical', 'start': 1, 'end': 8})]))
        programs.append(IrisDSLProgram([(IrisOperation.RADIAL_GRADIENT, {'center': 0, 'edge': 9})]))
        
        # Multi-color patterns
        programs.append(IrisDSLProgram([(IrisOperation.ALTERNATE_COLORS, {'colors': [1, 2, 3], 'pattern': 'row'})]))
        programs.append(IrisDSLProgram([(IrisOperation.RAINBOW_PATTERN, {'direction': 'horizontal'})]))
        programs.append(IrisDSLProgram([(IrisOperation.CHECKERBOARD_COLORS, {'color1': 0, 'color2': 1, 'size': 1})]))
        
        # Color mixing
        programs.append(IrisDSLProgram([(IrisOperation.BLEND_COLORS, {'mode': 'average'})]))
        programs.append(IrisDSLProgram([(IrisOperation.MIX_ADJACENT, {'direction': 'horizontal'})]))
        
        # Advanced patterns
        programs.append(IrisDSLProgram([(IrisOperation.COLOR_WAVE, {'frequency': 1, 'amplitude': 3, 'phase': 0})]))
        programs.append(IrisDSLProgram([(IrisOperation.COLOR_MOSAIC, {'tile_size': 2, 'colors': [1, 2, 3, 4]})]))
        
        # Combined operations
        programs.append(IrisDSLProgram([
            (IrisOperation.CREATE_GRADIENT, {'direction': 'horizontal', 'start': 0, 'end': 5}),
            (IrisOperation.CYCLE_COLORS, {'offset': 2, 'max_color': 9})
        ]))
        
        programs.append(IrisDSLProgram([
            (IrisOperation.FILTER_BY_COLOR, {'colors': [1, 2, 3], 'background': 0}),
            (IrisOperation.INVERT_COLORS, {'max_color': 9})
        ]))
        
        return programs
    
    @staticmethod
    def generate_perceptual_programs() -> List[IrisDSLProgram]:
        """Generate programs for perceptual color analysis"""
        programs = []
        
        # Color grouping and clustering
        programs.append(IrisDSLProgram([(IrisOperation.COLOR_CLUSTERING, {'n_clusters': 3})]))
        programs.append(IrisDSLProgram([(IrisOperation.PALETTE_REDUCTION, {'n_colors': 4})]))
        programs.append(IrisDSLProgram([(IrisOperation.PERCEPTUAL_GROUPING, {'threshold': 2})]))
        programs.append(IrisDSLProgram([(IrisOperation.COLOR_QUANTIZATION, {'levels': 4})]))
        
        # Conditional operations
        programs.append(IrisDSLProgram([(IrisOperation.CONDITIONAL_COLOR, {'condition': 'threshold', 'threshold': 5, 'true_color': 9, 'false_color': 0})]))
        programs.append(IrisDSLProgram([(IrisOperation.COLOR_ARITHMETIC, {'operation': 'add', 'value': 2})]))
        
        # Spatial color operations
        programs.append(IrisDSLProgram([(IrisOperation.COLOR_BY_REGION, {'background': 0})]))
        programs.append(IrisDSLProgram([(IrisOperation.COLOR_PROPAGATION, {'iterations': 5})]))
        programs.append(IrisDSLProgram([(IrisOperation.COLOR_SPIRAL, {'colors': list(range(10))})]))
        
        # Complex transformations
        programs.append(IrisDSLProgram([
            (IrisOperation.PALETTE_REDUCTION, {'n_colors': 3}),
            (IrisOperation.COLOR_WAVE, {'frequency': 2, 'amplitude': 2, 'phase': 0})
        ]))
        
        programs.append(IrisDSLProgram([
            (IrisOperation.PERCEPTUAL_GROUPING, {'threshold': 2}),
            (IrisOperation.ALTERNATE_COLORS, {'colors': [1, 4, 7], 'pattern': 'diagonal'})
        ]))
        
        return programs


class IRISDSLTraining:
    """DSL training integration specifically for IRIS"""
    
    @staticmethod
    def create_iris_dsl_samples(curriculum_stage: int = 0) -> List[Dict[str, np.ndarray]]:
        """Create DSL samples tailored for IRIS's color perception at each curriculum stage"""
        samples = []
        
        # Stage-specific color patterns optimized for IRIS
        if curriculum_stage == 0:
            # Stage 0: Simple 3x3 to 6x6 grids with basic color patterns
            base_grids = IRISDSLTraining._create_stage0_color_grids()
            programs = IRISDSLGenerator.generate_color_pattern_programs()[:15]
            
        elif curriculum_stage == 1:
            # Stage 1: 4x4 to 8x8 grids with intermediate color patterns
            base_grids = IRISDSLTraining._create_stage1_color_grids()
            programs = IRISDSLGenerator.generate_color_pattern_programs()[:25]
            
        elif curriculum_stage in [2, 3]:
            # Stages 2-3: 6x6 to 15x15 grids with complex color patterns
            base_grids = IRISDSLTraining._create_stage2_3_color_grids(curriculum_stage)
            programs = IRISDSLGenerator.generate_color_pattern_programs()
            
        elif curriculum_stage in [4, 5]:
            # Stages 4-5: 10x10 to 20x20 grids with perceptual color reasoning
            base_grids = IRISDSLTraining._create_stage4_5_color_grids(curriculum_stage)
            programs = IRISDSLGenerator.generate_perceptual_programs()[:20]
            
        else:  # Stages 6-7
            # Stages 6-7: 15x15 to 30x30 grids with advanced color patterns
            base_grids = IRISDSLTraining._create_advanced_color_grids(curriculum_stage)
            programs = IRISDSLGenerator.generate_perceptual_programs()
        
        # Generate samples
        for grid in base_grids:
            for program in programs:
                try:
                    output = program.execute(grid)
                    if output.shape == grid.shape and not np.array_equal(output, grid):
                        samples.append({
                            'input': grid.copy(),
                            'output': output,
                            'program': program,
                            'stage': curriculum_stage,
                            'type': 'iris_dsl'
                        })
                except Exception:
                    continue
        
        return samples
    
    @staticmethod
    def _create_stage0_color_grids() -> List[np.ndarray]:
        """Create simple color grids for Stage 0 (3x3 to 6x6)"""
        grids = []
        
        # Basic color patterns that IRIS excels at
        patterns = [
            # Single color fills
            np.full((3, 3), 1),
            np.full((4, 4), 2),
            np.full((5, 5), 3),
            
            # Two-color patterns
            np.array([[0, 1], [1, 0]]),
            np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
            np.array([[2, 2, 2], [0, 0, 0], [1, 1, 1]]),
            
            # Color gradients (discrete)
            np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            
            # Color corners
            np.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]]),
            np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]),
            
            # Color stripes
            np.array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]]),
            np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
            
            # Color rings
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
            np.array([[2, 2, 2, 2], [2, 1, 1, 2], [2, 1, 1, 2], [2, 2, 2, 2]]),
            
            # Multi-color mosaics
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]]),
            
            # Color clusters
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 2]]),
            np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]),
            
            # Diagonal color patterns
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            np.array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]),
        ]
        
        # Extend patterns with variations
        for pattern in patterns:
            grids.append(pattern)
            # Add rotated versions for color pattern learning
            if pattern.shape[0] == pattern.shape[1]:
                grids.append(np.rot90(pattern))
            # Add color-shifted versions
            shifted = (pattern + 1) % 9
            grids.append(shifted)
        
        return grids
    
    @staticmethod
    def _create_stage1_color_grids() -> List[np.ndarray]:
        """Create intermediate color grids for Stage 1 (4x4 to 8x8)"""
        grids = []
        
        # More complex color patterns
        patterns = [
            # Color waves
            np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1], 
                      [3, 4, 0, 1, 2], [4, 0, 1, 2, 3]]),
            
            # Concentric color rings
            np.array([[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 3, 2, 1], 
                      [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]]),
            
            # Color checkerboard variations
            np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]]),
            
            # Rainbow patterns
            np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0], [2, 3, 4, 5, 0, 1],
                      [3, 4, 5, 0, 1, 2], [4, 5, 0, 1, 2, 3], [5, 0, 1, 2, 3, 4]]),
            
            # Color gradients with boundaries
            np.array([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5],
                      [3, 3, 4, 4, 5, 5], [6, 6, 7, 7, 8, 8], [6, 6, 7, 7, 8, 8]]),
        ]
        
        for pattern in patterns:
            grids.append(pattern)
            # Add mirrored versions
            grids.append(np.fliplr(pattern))
            # Add inverted color versions
            max_color = np.max(pattern)
            if max_color > 0:
                grids.append(max_color - pattern)
        
        return grids
    
    @staticmethod
    def _create_stage2_3_color_grids(stage: int) -> List[np.ndarray]:
        """Create complex color grids for Stages 2-3 (6x6 to 15x15)"""
        grids = []
        max_size = 10 if stage == 2 else 15
        
        # Complex color patterns
        for size in range(6, max_size + 1):
            # Radial color gradient
            grid = np.zeros((size, size), dtype=np.int32)
            center = size // 2
            for i in range(size):
                for j in range(size):
                    dist = int(np.sqrt((i - center)**2 + (j - center)**2))
                    grid[i, j] = min(dist, 8)
            grids.append(grid)
            
            # Color spiral
            grid = np.zeros((size, size), dtype=np.int32)
            x, y = size // 2, size // 2
            dx, dy = 0, -1
            color = 0
            for _ in range(size * size):
                if 0 <= x < size and 0 <= y < size:
                    grid[y, x] = color % 9
                    color += 1
                if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                    dx, dy = -dy, dx
                x, y = x + dx, y + dy
            grids.append(grid)
            
            # Color mosaic
            grid = np.zeros((size, size), dtype=np.int32)
            tile_size = 3
            for i in range(0, size, tile_size):
                for j in range(0, size, tile_size):
                    color = ((i // tile_size) + (j // tile_size)) % 9
                    grid[i:min(i+tile_size, size), j:min(j+tile_size, size)] = color
            grids.append(grid)
        
        return grids
    
    @staticmethod
    def _create_stage4_5_color_grids(stage: int) -> List[np.ndarray]:
        """Create perceptual color grids for Stages 4-5 (10x10 to 20x20)"""
        grids = []
        max_size = 15 if stage == 4 else 20
        
        # Perceptual color patterns
        for size in range(10, min(max_size + 1, 21), 2):
            # Color frequency pattern
            grid = np.zeros((size, size), dtype=np.int32)
            for i in range(size):
                for j in range(size):
                    # Create interference pattern
                    freq1 = np.sin(i * np.pi / 4) + np.sin(j * np.pi / 4)
                    freq2 = np.sin(i * np.pi / 6) + np.sin(j * np.pi / 3)
                    color = int((freq1 + freq2 + 2) * 2) % 9
                    grid[i, j] = color
            grids.append(grid)
            
            # Color clustering pattern
            grid = np.zeros((size, size), dtype=np.int32)
            num_clusters = 5
            centers = [(np.random.randint(0, size), np.random.randint(0, size)) 
                      for _ in range(num_clusters)]
            
            for i in range(size):
                for j in range(size):
                    # Find nearest cluster center
                    min_dist = float('inf')
                    cluster_id = 0
                    for k, (cx, cy) in enumerate(centers):
                        dist = (i - cx)**2 + (j - cy)**2
                        if dist < min_dist:
                            min_dist = dist
                            cluster_id = k
                    grid[i, j] = cluster_id % 9
            grids.append(grid)
        
        return grids
    
    @staticmethod
    def _create_advanced_color_grids(stage: int) -> List[np.ndarray]:
        """Create advanced color grids for Stages 6-7 (15x15 to 30x30)"""
        grids = []
        max_size = 24 if stage == 6 else 30
        
        # Advanced color patterns with multiple rules
        for size in range(15, min(max_size + 1, 31), 3):
            # Complex color field
            grid = np.zeros((size, size), dtype=np.int32)
            
            # Create base color field with multiple influences
            for i in range(size):
                for j in range(size):
                    # Multiple color influences
                    influence1 = (i + j) % 5
                    influence2 = abs(i - j) % 4
                    influence3 = (i * j) % 3
                    
                    # Combine influences
                    color = (influence1 + influence2 + influence3) % 9
                    grid[i, j] = color
            
            grids.append(grid)
            
            # Perceptual grouping pattern
            grid = np.zeros((size, size), dtype=np.int32)
            
            # Create regions with perceptual grouping
            region_size = 5
            for i in range(0, size, region_size):
                for j in range(0, size, region_size):
                    base_color = ((i // region_size) * 3 + (j // region_size)) % 9
                    
                    # Add variation within region
                    for di in range(min(region_size, size - i)):
                        for dj in range(min(region_size, size - j)):
                            variation = (di + dj) % 2
                            grid[i + di, j + dj] = (base_color + variation) % 9
            
            grids.append(grid)
        
        return grids
    
    @staticmethod
    def augment_batch_with_iris_dsl(
        batch: Dict[str, torch.Tensor],
        curriculum_stage: int,
        dsl_ratio: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """Augment training batch with IRIS-specific DSL examples"""
        
        B = batch['input'].shape[0]
        num_dsl = int(B * dsl_ratio)
        
        if num_dsl > 0:
            # Generate IRIS-specific DSL examples
            dsl_samples = IRISDSLTraining.create_iris_dsl_samples(curriculum_stage)
            
            if dsl_samples:
                # Replace some batch samples with DSL examples
                indices = torch.randperm(B)[:num_dsl]
                for idx, i in enumerate(indices):
                    if idx < len(dsl_samples):
                        sample_idx = idx % len(dsl_samples)
                        batch['input'][i] = torch.tensor(dsl_samples[sample_idx]['input'])
                        batch['output'][i] = torch.tensor(dsl_samples[sample_idx]['output'])
        
        return batch
    
    @staticmethod
    def analyze_color_complexity(grid: np.ndarray) -> Dict[str, Any]:
        """Analyze color complexity to guide DSL program selection"""
        h, w = grid.shape
        unique_colors = np.unique(grid)
        num_colors = len(unique_colors)
        
        # Color distribution
        color_counts = np.bincount(grid.flatten(), minlength=10)[:10]
        color_entropy = -np.sum(
            (color_counts[color_counts > 0] / grid.size) * 
            np.log2(color_counts[color_counts > 0] / grid.size)
        )
        
        # Color transitions
        h_transitions = np.sum(grid[:, :-1] != grid[:, 1:])
        v_transitions = np.sum(grid[:-1, :] != grid[1:, :])
        total_transitions = h_transitions + v_transitions
        
        # Color clustering
        try:
            from scipy import ndimage
            labeled, num_regions = ndimage.label(grid)
        except:
            num_regions = num_colors
        
        return {
            'size': (h, w),
            'num_colors': num_colors,
            'unique_colors': unique_colors.tolist(),
            'color_entropy': color_entropy,
            'transition_density': total_transitions / (h * w),
            'num_color_regions': num_regions,
            'dominant_color': int(np.argmax(color_counts)),
            'complexity_score': color_entropy * (num_colors / 10.0)
        }