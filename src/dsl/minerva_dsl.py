"""
MINERVA-specific DSL integration for grid reasoning and strategic analysis
Generates deterministic training samples optimized for MINERVA's architecture
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
try:
    from scipy import ndimage
except ImportError:
    ndimage = None
from .arc_dsl import DSLProgram, DSLProgramGenerator, DSLExecutor, Operation


class MINERVADSLGenerator:
    """Generates DSL programs specifically for MINERVA's grid reasoning capabilities"""
    
    @staticmethod
    def generate_grid_reasoning_programs() -> List[DSLProgram]:
        """Generate programs focused on grid-based patterns and transformations"""
        programs = []
        
        # Grid-specific operations for MINERVA
        grid_operations = [
            # Basic grid transformations
            [Operation.ROTATE_90],
            [Operation.ROTATE_180],
            [Operation.ROTATE_270],
            [Operation.FLIP_HORIZONTAL],
            [Operation.FLIP_VERTICAL],
            [Operation.TRANSPOSE],
            
            # Strategic pattern operations
            [Operation.FLOOD_FILL, 2],  # Fill with color 2
            [Operation.FLOOD_FILL, 3],  # Fill with color 3
            [Operation.EXTRACT_OBJECTS],
            [Operation.SORT_BY_SIZE],
            
            # Combined transformations for complex reasoning
            [Operation.ROTATE_90, Operation.FLIP_HORIZONTAL],
            [Operation.TRANSPOSE, Operation.ROTATE_90],
            [Operation.FLOOD_FILL, 1, Operation.ROTATE_180],
            
            # Pattern completion operations
            [Operation.COMPLETE_PATTERN],
            [Operation.MIRROR_PATTERN],
            [Operation.REPEAT_PATTERN],
            
            # Grid analysis operations
            [Operation.FIND_SYMMETRY],
            [Operation.DETECT_GRID_LINES],
            [Operation.EXTRACT_SUBGRIDS],
        ]
        
        for ops in grid_operations:
            program = DSLProgram(ops)
            programs.append(program)
            
        return programs
    
    @staticmethod
    def generate_strategic_programs() -> List[DSLProgram]:
        """Generate programs for strategic pattern analysis"""
        programs = []
        
        # Strategic operations for advanced grid reasoning
        strategic_ops = [
            # Multi-step transformations
            [Operation.EXTRACT_OBJECTS, Operation.SORT_BY_SIZE, Operation.APPLY_RULE],
            [Operation.FIND_SYMMETRY, Operation.COMPLETE_PATTERN],
            [Operation.DETECT_GRID_LINES, Operation.FLOOD_FILL, 4],
            
            # Conditional operations
            [Operation.IF_SYMMETRIC, Operation.MIRROR_PATTERN],
            [Operation.IF_HAS_PATTERN, Operation.REPEAT_PATTERN],
            
            # Complex grid manipulations
            [Operation.SPLIT_GRID, Operation.TRANSFORM_EACH, Operation.MERGE_GRIDS],
            [Operation.EXTRACT_BOUNDARY, Operation.FILL_INTERIOR],
            [Operation.FIND_CONNECTED_COMPONENTS, Operation.COLOR_BY_SIZE],
        ]
        
        for ops in strategic_ops:
            program = DSLProgram(ops)
            programs.append(program)
            
        return programs


class MINERVADSLTraining:
    """DSL training integration specifically for MINERVA"""
    
    @staticmethod
    def create_minerva_dsl_samples(curriculum_stage: int = 0) -> List[Dict[str, np.ndarray]]:
        """Create DSL samples tailored for MINERVA's grid reasoning at each curriculum stage"""
        samples = []
        
        # Stage-specific grid patterns optimized for MINERVA
        if curriculum_stage == 0:
            # Stage 0: Simple 3x3 to 6x6 grids with basic patterns
            base_grids = MINERVADSLTraining._create_stage0_grids()
            programs = MINERVADSLGenerator.generate_grid_reasoning_programs()[:20]
            
        elif curriculum_stage == 1:
            # Stage 1: 4x4 to 7x7 grids with intermediate patterns
            base_grids = MINERVADSLTraining._create_stage1_grids()
            programs = MINERVADSLGenerator.generate_grid_reasoning_programs()[:30]
            
        elif curriculum_stage in [2, 3]:
            # Stages 2-3: 5x5 to 12x12 grids with complex patterns
            base_grids = MINERVADSLTraining._create_stage2_3_grids(curriculum_stage)
            programs = MINERVADSLGenerator.generate_grid_reasoning_programs()
            
        elif curriculum_stage in [4, 5]:
            # Stages 4-5: 8x8 to 19x19 grids with strategic reasoning
            base_grids = MINERVADSLTraining._create_stage4_5_grids(curriculum_stage)
            programs = MINERVADSLGenerator.generate_strategic_programs()[:20]
            
        else:  # Stages 6-7
            # Stages 6-7: 15x15 to 30x30 grids with advanced patterns
            base_grids = MINERVADSLTraining._create_advanced_grids(curriculum_stage)
            programs = MINERVADSLGenerator.generate_strategic_programs()
        
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
                            'type': 'minerva_dsl'
                        })
                except Exception:
                    continue
        
        return samples
    
    @staticmethod
    def _create_stage0_grids() -> List[np.ndarray]:
        """Create simple grids for Stage 0 (3x3 to 6x6)"""
        grids = []
        
        # Basic patterns that MINERVA excels at
        patterns = [
            # Checkerboard patterns
            np.array([[0, 1], [1, 0]]),
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]),
            
            # Line patterns
            np.array([[1, 1, 1], [0, 0, 0], [2, 2, 2]]),
            np.array([[1, 0, 2], [1, 0, 2], [1, 0, 2]]),
            
            # Corner patterns
            np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
            np.array([[2, 2, 0], [2, 0, 0], [0, 0, 0]]),
            
            # Center patterns
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
            
            # Diagonal patterns
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
            
            # Frame patterns
            np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
            np.array([[2, 2, 2, 2], [2, 0, 0, 2], [2, 0, 0, 2], [2, 2, 2, 2]]),
            
            # L-shapes
            np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),
            np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]]),
            
            # T-shapes
            np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            
            # Plus patterns
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]),
            
            # Multi-color patterns
            np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
            np.array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]),
        ]
        
        # Extend patterns to different sizes
        for pattern in patterns:
            grids.append(pattern)
            # Add variations with different colors
            if pattern.max() < 3:
                grids.append(pattern + 1)
                grids.append((pattern + 2) % 4)
        
        return grids
    
    @staticmethod
    def _create_stage1_grids() -> List[np.ndarray]:
        """Create intermediate grids for Stage 1 (4x4 to 7x7)"""
        grids = []
        
        # More complex patterns
        patterns = [
            # Symmetric patterns
            np.array([[1, 2, 2, 1], [2, 1, 1, 2], [2, 1, 1, 2], [1, 2, 2, 1]]),
            np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]),
            
            # Nested patterns
            np.array([[3, 3, 3, 3, 3], [3, 2, 2, 2, 3], [3, 2, 1, 2, 3], [3, 2, 2, 2, 3], [3, 3, 3, 3, 3]]),
            
            # Grid subdivisions
            np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]),
            
            # Repeating motifs
            np.array([[1, 0, 1, 0, 1, 0], [0, 2, 0, 2, 0, 2], [1, 0, 1, 0, 1, 0], 
                      [0, 2, 0, 2, 0, 2], [1, 0, 1, 0, 1, 0], [0, 2, 0, 2, 0, 2]]),
        ]
        
        for pattern in patterns:
            grids.append(pattern)
            # Add rotated versions
            grids.append(np.rot90(pattern))
            # Add flipped versions
            grids.append(np.fliplr(pattern))
        
        return grids
    
    @staticmethod
    def _create_stage2_3_grids(stage: int) -> List[np.ndarray]:
        """Create complex grids for Stages 2-3 (5x5 to 12x12)"""
        grids = []
        max_size = 9 if stage == 2 else 12
        
        # Complex geometric patterns
        for size in range(5, max_size + 1):
            # Concentric squares
            grid = np.zeros((size, size), dtype=np.int32)
            for i in range(size // 2 + 1):
                grid[i:size-i, i:size-i] = i % 4
            grids.append(grid)
            
            # Radial pattern
            grid = np.zeros((size, size), dtype=np.int32)
            center = size // 2
            for i in range(size):
                for j in range(size):
                    dist = abs(i - center) + abs(j - center)
                    grid[i, j] = dist % 5
            grids.append(grid)
            
            # Quadrant pattern
            grid = np.zeros((size, size), dtype=np.int32)
            mid = size // 2
            grid[:mid, :mid] = 1
            grid[:mid, mid:] = 2
            grid[mid:, :mid] = 3
            grid[mid:, mid:] = 4
            grids.append(grid)
        
        return grids
    
    @staticmethod
    def _create_stage4_5_grids(stage: int) -> List[np.ndarray]:
        """Create strategic reasoning grids for Stages 4-5 (8x8 to 19x19)"""
        grids = []
        max_size = 15 if stage == 4 else 19
        
        # Strategic patterns requiring reasoning
        for size in range(8, min(max_size + 1, 20), 2):
            # Maze-like pattern
            grid = np.zeros((size, size), dtype=np.int32)
            for i in range(0, size, 3):
                grid[i, :] = 1
                grid[:, i] = 1
            for i in range(1, size-1, 3):
                for j in range(1, size-1, 3):
                    grid[i, j] = 2
            grids.append(grid)
            
            # Fractal-like pattern
            grid = np.zeros((size, size), dtype=np.int32)
            def fill_fractal(x, y, s, color):
                if s < 2:
                    return
                grid[x:x+s//2, y:y+s//2] = color
                grid[x+s//2:x+s, y+s//2:y+s] = color
                fill_fractal(x, y+s//2, s//2, (color + 1) % 5)
                fill_fractal(x+s//2, y, s//2, (color + 2) % 5)
            
            fill_fractal(0, 0, size, 1)
            grids.append(grid)
        
        return grids
    
    @staticmethod
    def _create_advanced_grids(stage: int) -> List[np.ndarray]:
        """Create advanced grids for Stages 6-7 (15x15 to 30x30)"""
        grids = []
        max_size = 24 if stage == 6 else 30
        
        # Advanced patterns with multiple rules
        for size in range(15, min(max_size + 1, 31), 3):
            # Complex tiling pattern
            grid = np.zeros((size, size), dtype=np.int32)
            tiles = [
                np.array([[1, 2], [2, 1]]),
                np.array([[3, 4], [4, 3]]),
                np.array([[5, 0], [0, 5]])
            ]
            
            for i in range(0, size-1, 2):
                for j in range(0, size-1, 2):
                    tile_idx = ((i//2) + (j//2)) % len(tiles)
                    grid[i:i+2, j:j+2] = tiles[tile_idx]
            
            grids.append(grid)
            
            # Rule-based generation
            grid = np.zeros((size, size), dtype=np.int32)
            for i in range(size):
                for j in range(size):
                    # Complex rule based on position
                    if (i + j) % 5 == 0:
                        grid[i, j] = 1
                    elif (i * j) % 7 == 0:
                        grid[i, j] = 2
                    elif i % 3 == j % 3:
                        grid[i, j] = 3
                    else:
                        grid[i, j] = (i + j) % 4
            
            grids.append(grid)
        
        return grids
    
    @staticmethod
    def augment_batch_with_minerva_dsl(
        batch: Dict[str, torch.Tensor],
        curriculum_stage: int,
        dsl_ratio: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """Augment training batch with MINERVA-specific DSL examples"""
        
        B = batch['input'].shape[0]
        num_dsl = int(B * dsl_ratio)
        
        if num_dsl > 0:
            # Generate MINERVA-specific DSL examples
            dsl_samples = MINERVADSLTraining.create_minerva_dsl_samples(curriculum_stage)
            
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
    def analyze_grid_complexity(grid: np.ndarray) -> Dict[str, Any]:
        """Analyze grid complexity to guide DSL program selection"""
        h, w = grid.shape
        unique_colors = len(np.unique(grid))
        
        # Check for patterns
        has_symmetry_h = np.array_equal(grid, np.fliplr(grid))
        has_symmetry_v = np.array_equal(grid, np.flipud(grid))
        has_rotation = np.array_equal(grid, np.rot90(grid, k=4))
        
        # Count connected components
        if ndimage:
            components = ndimage.label(grid > 0)[1]
        else:
            # Simple approximation if scipy not available
            components = len(np.unique(grid)) - 1
        
        return {
            'size': (h, w),
            'unique_colors': unique_colors,
            'has_horizontal_symmetry': has_symmetry_h,
            'has_vertical_symmetry': has_symmetry_v,
            'has_rotational_symmetry': has_rotation,
            'connected_components': components,
            'complexity_score': unique_colors * components / (h * w)
        }