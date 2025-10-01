"""
IRIS-specific DSL integration for color patterns and transformations
Generates deterministic training samples optimized for IRIS's color perception
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from .arc_dsl import DSLProgram, DSLProgramGenerator, DSLExecutor, Operation


class IRISDSLGenerator:
    """Generates DSL programs specifically for IRIS's color perception capabilities"""
    
    @staticmethod
    def generate_color_pattern_programs() -> List[DSLProgram]:
        """Generate programs focused on color-based patterns and transformations"""
        programs = []
        
        # Color-specific operations for IRIS
        color_operations = [
            # Basic color transformations
            [Operation.SWAP_COLORS],
            [Operation.INVERT_COLORS],
            [Operation.CYCLE_COLORS],
            [Operation.MAP_COLORS],
            
            # Color filtering operations
            [Operation.FILTER_BY_COLOR, 1],
            [Operation.FILTER_BY_COLOR, 2],
            [Operation.EXTRACT_COLOR_MASK, 3],
            
            # Color gradient operations
            [Operation.CREATE_GRADIENT],
            [Operation.APPLY_COLOR_FADE],
            [Operation.COLOR_DISTANCE_MAP],
            
            # Multi-color patterns
            [Operation.ALTERNATE_COLORS],
            [Operation.RAINBOW_PATTERN],
            [Operation.COLOR_BY_REGION],
            
            # Color mixing operations
            [Operation.BLEND_COLORS],
            [Operation.COLOR_ARITHMETIC],
            [Operation.CONDITIONAL_COLOR],
            
            # Complex color transformations
            [Operation.COLOR_PROPAGATION],
            [Operation.COLOR_CLUSTERING],
            [Operation.PALETTE_REDUCTION],
        ]
        
        for ops in color_operations:
            program = DSLProgram(ops)
            programs.append(program)
            
        return programs
    
    @staticmethod
    def generate_perceptual_programs() -> List[DSLProgram]:
        """Generate programs for perceptual color analysis"""
        programs = []
        
        # Perceptual operations for advanced color reasoning
        perceptual_ops = [
            # Color relationship detection
            [Operation.FIND_COLOR_PAIRS],
            [Operation.DETECT_COLOR_HARMONY],
            [Operation.IDENTIFY_DOMINANT_COLOR],
            
            # Color-based segmentation
            [Operation.SEGMENT_BY_COLOR],
            [Operation.GROUP_SIMILAR_COLORS],
            [Operation.COLOR_CONNECTIVITY],
            
            # Conditional color operations
            [Operation.IF_COLOR_PRESENT, Operation.APPLY_TRANSFORM],
            [Operation.COLOR_BASED_RULE, Operation.EXECUTE_ACTION],
            
            # Complex color manipulations
            [Operation.COLOR_WAVE_PATTERN],
            [Operation.RADIAL_COLOR_GRADIENT],
            [Operation.COLOR_SYMMETRY_CHECK],
        ]
        
        for ops in perceptual_ops:
            program = DSLProgram(ops)
            programs.append(program)
            
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
        from scipy import ndimage
        labeled, num_regions = ndimage.label(grid)
        
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