"""
MINERVA-specific DSL integration for grid reasoning and strategic analysis
COMPLETELY INDEPENDENT - No imports from base DSL
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum


class MinervaOperation(Enum):
    """MINERVA-specific operations for grid reasoning"""
    # Basic grid transformations
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    TRANSPOSE = "transpose"
    
    # Grid-specific operations
    EXTRACT_GRID_LINES = "extract_grid_lines"
    FILL_GRID_CELLS = "fill_grid_cells"
    CREATE_CHECKERBOARD = "create_checkerboard"
    EXTRACT_CORNERS = "extract_corners"
    EXTRACT_EDGES = "extract_edges"
    EXTRACT_CENTER = "extract_center"
    
    # Pattern operations
    COMPLETE_SYMMETRY = "complete_symmetry"
    EXTRACT_PATTERN = "extract_pattern"
    REPEAT_PATTERN = "repeat_pattern"
    MIRROR_PATTERN = "mirror_pattern"
    
    # Strategic operations
    FIND_LARGEST_REGION = "find_largest_region"
    FIND_SMALLEST_REGION = "find_smallest_region"
    COUNT_REGIONS = "count_regions"
    EXTRACT_BY_SIZE = "extract_by_size"
    
    # Logical grid operations
    GRID_AND = "grid_and"
    GRID_OR = "grid_or"
    GRID_XOR = "grid_xor"
    GRID_NOT = "grid_not"
    
    # Subdivision operations
    SPLIT_QUADRANTS = "split_quadrants"
    EXTRACT_QUADRANT = "extract_quadrant"
    MERGE_QUADRANTS = "merge_quadrants"
    
    # Frame operations
    EXTRACT_FRAME = "extract_frame"
    FILL_FRAME = "fill_frame"
    REMOVE_FRAME = "remove_frame"


class MinervaDSLProgram:
    """MINERVA-specific DSL program representation"""
    
    def __init__(self, operations: List[Tuple[MinervaOperation, Dict[str, Any]]]):
        self.operations = operations
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute the MINERVA DSL program on input grid"""
        result = grid.copy()
        for op, params in self.operations:
            result = MinervaDSLExecutor.execute_operation(result, op, params)
        return result
    
    def to_string(self) -> str:
        """Convert program to readable string"""
        prog_str = []
        for op, params in self.operations:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            prog_str.append(f"{op.value}({param_str})")
        return " -> ".join(prog_str)


class MinervaDSLExecutor:
    """Executes MINERVA-specific DSL operations"""
    
    @staticmethod
    def execute_operation(grid: np.ndarray, op: MinervaOperation, params: Dict[str, Any]) -> np.ndarray:
        """Execute a single MINERVA DSL operation"""
        
        # Basic transformations
        if op == MinervaOperation.ROTATE_90:
            return np.rot90(grid, k=1)
        elif op == MinervaOperation.ROTATE_180:
            return np.rot90(grid, k=2)
        elif op == MinervaOperation.ROTATE_270:
            return np.rot90(grid, k=3)
        elif op == MinervaOperation.FLIP_HORIZONTAL:
            return np.fliplr(grid)
        elif op == MinervaOperation.FLIP_VERTICAL:
            return np.flipud(grid)
        elif op == MinervaOperation.TRANSPOSE:
            return grid.T
        
        # Grid-specific operations
        elif op == MinervaOperation.EXTRACT_GRID_LINES:
            return MinervaDSLExecutor._extract_grid_lines(grid, params)
        elif op == MinervaOperation.FILL_GRID_CELLS:
            return MinervaDSLExecutor._fill_grid_cells(grid, params)
        elif op == MinervaOperation.CREATE_CHECKERBOARD:
            return MinervaDSLExecutor._create_checkerboard(grid, params)
        elif op == MinervaOperation.EXTRACT_CORNERS:
            return MinervaDSLExecutor._extract_corners(grid, params)
        elif op == MinervaOperation.EXTRACT_EDGES:
            return MinervaDSLExecutor._extract_edges(grid, params)
        elif op == MinervaOperation.EXTRACT_CENTER:
            return MinervaDSLExecutor._extract_center(grid, params)
        
        # Pattern operations
        elif op == MinervaOperation.COMPLETE_SYMMETRY:
            return MinervaDSLExecutor._complete_symmetry(grid, params)
        elif op == MinervaOperation.EXTRACT_PATTERN:
            return MinervaDSLExecutor._extract_pattern(grid, params)
        elif op == MinervaOperation.REPEAT_PATTERN:
            return MinervaDSLExecutor._repeat_pattern(grid, params)
        elif op == MinervaOperation.MIRROR_PATTERN:
            return MinervaDSLExecutor._mirror_pattern(grid, params)
        
        # Strategic operations
        elif op == MinervaOperation.FIND_LARGEST_REGION:
            return MinervaDSLExecutor._find_largest_region(grid, params)
        elif op == MinervaOperation.FIND_SMALLEST_REGION:
            return MinervaDSLExecutor._find_smallest_region(grid, params)
        elif op == MinervaOperation.COUNT_REGIONS:
            return MinervaDSLExecutor._count_regions(grid, params)
        elif op == MinervaOperation.EXTRACT_BY_SIZE:
            return MinervaDSLExecutor._extract_by_size(grid, params)
        
        # Logical operations
        elif op == MinervaOperation.GRID_AND:
            other = params.get('other', grid)
            return np.minimum(grid, other)
        elif op == MinervaOperation.GRID_OR:
            other = params.get('other', grid)
            return np.maximum(grid, other)
        elif op == MinervaOperation.GRID_XOR:
            other = params.get('other', grid)
            return np.where(grid != other, np.maximum(grid, other), 0)
        elif op == MinervaOperation.GRID_NOT:
            max_val = params.get('max_val', 9)
            return max_val - grid
        
        # Subdivision operations
        elif op == MinervaOperation.SPLIT_QUADRANTS:
            return MinervaDSLExecutor._split_quadrants(grid, params)
        elif op == MinervaOperation.EXTRACT_QUADRANT:
            return MinervaDSLExecutor._extract_quadrant(grid, params)
        elif op == MinervaOperation.MERGE_QUADRANTS:
            return MinervaDSLExecutor._merge_quadrants(grid, params)
        
        # Frame operations
        elif op == MinervaOperation.EXTRACT_FRAME:
            return MinervaDSLExecutor._extract_frame(grid, params)
        elif op == MinervaOperation.FILL_FRAME:
            return MinervaDSLExecutor._fill_frame(grid, params)
        elif op == MinervaOperation.REMOVE_FRAME:
            return MinervaDSLExecutor._remove_frame(grid, params)
        
        else:
            return grid
    
    @staticmethod
    def _extract_grid_lines(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract horizontal and vertical lines"""
        spacing = params.get('spacing', 3)
        color = params.get('color', 1)
        result = np.zeros_like(grid)
        
        # Extract lines at regular intervals
        for i in range(0, grid.shape[0], spacing):
            result[i, :] = color
        for j in range(0, grid.shape[1], spacing):
            result[:, j] = color
        
        return result
    
    @staticmethod
    def _fill_grid_cells(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Fill alternating grid cells"""
        cell_size = params.get('cell_size', 2)
        colors = params.get('colors', [1, 2])
        result = grid.copy()
        
        for i in range(0, grid.shape[0], cell_size):
            for j in range(0, grid.shape[1], cell_size):
                color_idx = ((i // cell_size) + (j // cell_size)) % len(colors)
                result[i:i+cell_size, j:j+cell_size] = colors[color_idx]
        
        return result
    
    @staticmethod
    def _create_checkerboard(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create checkerboard pattern"""
        color1 = params.get('color1', 0)
        color2 = params.get('color2', 1)
        result = np.zeros_like(grid)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if (i + j) % 2 == 0:
                    result[i, j] = color1
                else:
                    result[i, j] = color2
        
        return result
    
    @staticmethod
    def _extract_corners(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract corner regions"""
        size = params.get('size', 2)
        result = np.zeros_like(grid)
        
        # Top-left
        result[:size, :size] = grid[:size, :size]
        # Top-right
        result[:size, -size:] = grid[:size, -size:]
        # Bottom-left
        result[-size:, :size] = grid[-size:, :size]
        # Bottom-right
        result[-size:, -size:] = grid[-size:, -size:]
        
        return result
    
    @staticmethod
    def _extract_edges(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract edge pixels"""
        width = params.get('width', 1)
        result = np.zeros_like(grid)
        
        # Top and bottom edges
        result[:width, :] = grid[:width, :]
        result[-width:, :] = grid[-width:, :]
        
        # Left and right edges
        result[:, :width] = grid[:, :width]
        result[:, -width:] = grid[:, -width:]
        
        return result
    
    @staticmethod
    def _extract_center(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract center region"""
        ratio = params.get('ratio', 0.5)
        h, w = grid.shape
        center_h = int(h * ratio)
        center_w = int(w * ratio)
        
        start_h = (h - center_h) // 2
        start_w = (w - center_w) // 2
        
        result = np.zeros_like(grid)
        result[start_h:start_h+center_h, start_w:start_w+center_w] = \
            grid[start_h:start_h+center_h, start_w:start_w+center_w]
        
        return result
    
    @staticmethod
    def _complete_symmetry(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Complete symmetrical pattern"""
        axis = params.get('axis', 'horizontal')
        
        if axis == 'horizontal':
            # Use top half to complete bottom
            half = grid[:grid.shape[0]//2]
            return np.vstack([half, np.flipud(half)])
        else:
            # Use left half to complete right
            half = grid[:, :grid.shape[1]//2]
            return np.hstack([half, np.fliplr(half)])
    
    @staticmethod
    def _extract_pattern(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract repeating pattern"""
        pattern_size = params.get('pattern_size', 3)
        result = np.zeros_like(grid)
        
        # Extract first occurrence of pattern
        result[:pattern_size, :pattern_size] = grid[:pattern_size, :pattern_size]
        
        return result
    
    @staticmethod
    def _repeat_pattern(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Repeat a small pattern across the grid"""
        pattern_h = params.get('pattern_h', 2)
        pattern_w = params.get('pattern_w', 2)
        
        # Extract pattern from top-left
        pattern = grid[:pattern_h, :pattern_w]
        
        # Repeat pattern
        result = np.tile(pattern, (grid.shape[0] // pattern_h + 1, 
                                  grid.shape[1] // pattern_w + 1))
        
        return result[:grid.shape[0], :grid.shape[1]]
    
    @staticmethod
    def _mirror_pattern(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Mirror pattern in specified direction"""
        direction = params.get('direction', 'horizontal')
        
        if direction == 'horizontal':
            return np.hstack([grid, np.fliplr(grid)])
        else:
            return np.vstack([grid, np.flipud(grid)])
    
    @staticmethod
    def _find_largest_region(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Find and extract largest connected region"""
        background = params.get('background', 0)
        
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(grid != background)
            
            if num_features == 0:
                return np.full_like(grid, background)
            
            # Find largest region
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_idx = np.argmax(sizes) + 1
            
            result = np.full_like(grid, background)
            mask = (labeled == largest_idx)
            result[mask] = grid[mask]
            
            return result
        except:
            # Fallback: return original grid
            return grid
    
    @staticmethod
    def _find_smallest_region(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Find and extract smallest connected region"""
        background = params.get('background', 0)
        
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(grid != background)
            
            if num_features == 0:
                return np.full_like(grid, background)
            
            # Find smallest region
            sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            smallest_idx = np.argmin(sizes) + 1
            
            result = np.full_like(grid, background)
            mask = (labeled == smallest_idx)
            result[mask] = grid[mask]
            
            return result
        except:
            # Fallback: return original grid
            return grid
    
    @staticmethod
    def _count_regions(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Count and label connected regions"""
        background = params.get('background', 0)
        
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(grid != background)
            
            # Create result showing region count
            result = np.full_like(grid, min(num_features, 9))
            
            return result
        except:
            # Fallback: return count of 1
            return np.ones_like(grid)
    
    @staticmethod
    def _extract_by_size(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract regions by size threshold"""
        min_size = params.get('min_size', 3)
        background = params.get('background', 0)
        
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(grid != background)
            
            result = np.full_like(grid, background)
            
            for i in range(1, num_features + 1):
                region_size = np.sum(labeled == i)
                if region_size >= min_size:
                    mask = (labeled == i)
                    result[mask] = grid[mask]
            
            return result
        except:
            # Fallback: return original grid
            return grid
    
    @staticmethod
    def _split_quadrants(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Split grid into quadrants with different colors"""
        h, w = grid.shape
        mid_h, mid_w = h // 2, w // 2
        
        result = grid.copy()
        
        # Assign different values to each quadrant
        result[:mid_h, :mid_w] = 1  # Top-left
        result[:mid_h, mid_w:] = 2  # Top-right
        result[mid_h:, :mid_w] = 3  # Bottom-left
        result[mid_h:, mid_w:] = 4  # Bottom-right
        
        return result
    
    @staticmethod
    def _extract_quadrant(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract specific quadrant"""
        quadrant = params.get('quadrant', 1)  # 1=TL, 2=TR, 3=BL, 4=BR
        h, w = grid.shape
        mid_h, mid_w = h // 2, w // 2
        
        result = np.zeros_like(grid)
        
        if quadrant == 1:
            result[:mid_h, :mid_w] = grid[:mid_h, :mid_w]
        elif quadrant == 2:
            result[:mid_h, mid_w:] = grid[:mid_h, mid_w:]
        elif quadrant == 3:
            result[mid_h:, :mid_w] = grid[mid_h:, :mid_w]
        elif quadrant == 4:
            result[mid_h:, mid_w:] = grid[mid_h:, mid_w:]
        
        return result
    
    @staticmethod
    def _merge_quadrants(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Merge quadrants with specific rule"""
        rule = params.get('rule', 'max')
        h, w = grid.shape
        mid_h, mid_w = h // 2, w // 2
        
        quadrants = [
            grid[:mid_h, :mid_w],
            grid[:mid_h, mid_w:],
            grid[mid_h:, :mid_w],
            grid[mid_h:, mid_w:]
        ]
        
        if rule == 'max':
            merged = np.maximum.reduce(quadrants)
        elif rule == 'min':
            merged = np.minimum.reduce(quadrants)
        elif rule == 'sum':
            merged = np.sum(quadrants, axis=0)
        else:
            merged = quadrants[0]
        
        # Tile the merged result
        result = np.tile(merged, (2, 2))[:h, :w]
        
        return result
    
    @staticmethod
    def _extract_frame(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract frame of the grid"""
        width = params.get('width', 1)
        result = np.zeros_like(grid)
        
        # Extract frame
        result[:width, :] = grid[:width, :]
        result[-width:, :] = grid[-width:, :]
        result[:, :width] = grid[:, :width]
        result[:, -width:] = grid[:, -width:]
        
        return result
    
    @staticmethod
    def _fill_frame(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Fill frame with specific color"""
        width = params.get('width', 1)
        color = params.get('color', 1)
        result = grid.copy()
        
        # Fill frame
        result[:width, :] = color
        result[-width:, :] = color
        result[:, :width] = color
        result[:, -width:] = color
        
        return result
    
    @staticmethod
    def _remove_frame(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Remove frame by setting to background"""
        width = params.get('width', 1)
        background = params.get('background', 0)
        result = grid.copy()
        
        # Remove frame
        result[:width, :] = background
        result[-width:, :] = background
        result[:, :width] = background
        result[:, -width:] = background
        
        return result


class MINERVADSLGenerator:
    """Generates DSL programs specifically for MINERVA's grid reasoning capabilities"""
    
    @staticmethod
    def generate_grid_reasoning_programs() -> List[MinervaDSLProgram]:
        """Generate programs focused on grid-based patterns and transformations"""
        programs = []
        
        # Basic transformations
        programs.append(MinervaDSLProgram([(MinervaOperation.ROTATE_90, {})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.ROTATE_180, {})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.ROTATE_270, {})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.FLIP_HORIZONTAL, {})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.FLIP_VERTICAL, {})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.TRANSPOSE, {})]))
        
        # Grid patterns
        programs.append(MinervaDSLProgram([(MinervaOperation.CREATE_CHECKERBOARD, {'color1': 0, 'color2': 1})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.EXTRACT_GRID_LINES, {'spacing': 3, 'color': 1})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.FILL_GRID_CELLS, {'cell_size': 2, 'colors': [1, 2]})]))
        
        # Corner and edge operations
        programs.append(MinervaDSLProgram([(MinervaOperation.EXTRACT_CORNERS, {'size': 2})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.EXTRACT_EDGES, {'width': 1})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.EXTRACT_CENTER, {'ratio': 0.5})]))
        
        # Frame operations
        programs.append(MinervaDSLProgram([(MinervaOperation.EXTRACT_FRAME, {'width': 1})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.FILL_FRAME, {'width': 1, 'color': 2})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.REMOVE_FRAME, {'width': 1, 'background': 0})]))
        
        # Symmetry operations
        programs.append(MinervaDSLProgram([(MinervaOperation.COMPLETE_SYMMETRY, {'axis': 'horizontal'})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.COMPLETE_SYMMETRY, {'axis': 'vertical'})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.MIRROR_PATTERN, {'direction': 'horizontal'})]))
        
        # Combined operations
        programs.append(MinervaDSLProgram([
            (MinervaOperation.ROTATE_90, {}),
            (MinervaOperation.FLIP_HORIZONTAL, {})
        ]))
        
        programs.append(MinervaDSLProgram([
            (MinervaOperation.EXTRACT_CORNERS, {'size': 2}),
            (MinervaOperation.ROTATE_180, {})
        ]))
        
        return programs
    
    @staticmethod
    def generate_strategic_programs() -> List[MinervaDSLProgram]:
        """Generate programs for strategic pattern analysis"""
        programs = []
        
        # Region-based operations
        programs.append(MinervaDSLProgram([(MinervaOperation.FIND_LARGEST_REGION, {'background': 0})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.FIND_SMALLEST_REGION, {'background': 0})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.COUNT_REGIONS, {'background': 0})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.EXTRACT_BY_SIZE, {'min_size': 3, 'background': 0})]))
        
        # Quadrant operations
        programs.append(MinervaDSLProgram([(MinervaOperation.SPLIT_QUADRANTS, {})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.EXTRACT_QUADRANT, {'quadrant': 1})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.MERGE_QUADRANTS, {'rule': 'max'})]))
        
        # Pattern completion
        programs.append(MinervaDSLProgram([(MinervaOperation.EXTRACT_PATTERN, {'pattern_size': 3})]))
        programs.append(MinervaDSLProgram([(MinervaOperation.REPEAT_PATTERN, {'pattern_h': 2, 'pattern_w': 2})]))
        
        # Logical operations
        programs.append(MinervaDSLProgram([(MinervaOperation.GRID_NOT, {'max_val': 9})]))
        
        # Complex multi-step operations
        programs.append(MinervaDSLProgram([
            (MinervaOperation.FIND_LARGEST_REGION, {'background': 0}),
            (MinervaOperation.ROTATE_90, {})
        ]))
        
        programs.append(MinervaDSLProgram([
            (MinervaOperation.EXTRACT_EDGES, {'width': 1}),
            (MinervaOperation.FILL_FRAME, {'width': 1, 'color': 3})
        ]))
        
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
        
        # Handle both 'input'/'output' and 'inputs'/'outputs' keys
        input_key = 'inputs' if 'inputs' in batch else 'input'
        output_key = 'outputs' if 'outputs' in batch else 'output'
        
        B = batch[input_key].shape[0]
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
                        batch[input_key][i] = torch.tensor(dsl_samples[sample_idx]['input'])
                        batch[output_key][i] = torch.tensor(dsl_samples[sample_idx]['output'])
        
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
        try:
            from scipy import ndimage
            components = ndimage.label(grid > 0)[1]
        except:
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