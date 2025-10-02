"""
ATLAS-specific Program Synthesis System
Specialized for spatial transformation program synthesis and execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
import random
import itertools
import json
from dataclasses import dataclass, asdict
import time


@dataclass
class AtlasSynthesisProgram:
    """Represents a synthesized spatial transformation program"""
    program_id: str
    operations: List[Tuple[str, Dict[str, Any]]]
    input_signature: Dict[str, Any]
    output_signature: Dict[str, Any]
    execution_time: float
    success_rate: float
    confidence_score: float
    spatial_complexity: int
    transformation_type: str
    generalization_score: float
    
    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """Execute the synthesis program"""
        result = input_grid.copy()
        
        for operation, params in self.operations:
            result = AtlasSynthesisExecutor.execute_operation(result, operation, params)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AtlasSynthesisProgram':
        """Create from dictionary"""
        return cls(**data)
    
    def get_complexity_score(self) -> float:
        """Get program complexity score"""
        base_complexity = len(self.operations)
        param_complexity = sum(len(params) for _, params in self.operations)
        return base_complexity + 0.1 * param_complexity


class AtlasSynthesisExecutor:
    """Executes ATLAS synthesis programs with advanced spatial operations"""
    
    @staticmethod
    def execute_operation(grid: np.ndarray, operation: str, params: Dict[str, Any]) -> np.ndarray:
        """Execute a single synthesis operation"""
        
        # Basic transformations
        if operation == "identity":
            return grid
        elif operation == "rotate_cw_90":
            return np.rot90(grid, k=3)  # Clockwise
        elif operation == "rotate_cw_180":
            return np.rot90(grid, k=2)
        elif operation == "rotate_cw_270":
            return np.rot90(grid, k=1)  # Counter-clockwise 90 = Clockwise 270
        elif operation == "rotate_ccw_90":
            return np.rot90(grid, k=1)
        elif operation == "flip_horizontal":
            return np.flip(grid, axis=1)
        elif operation == "flip_vertical":
            return np.flip(grid, axis=0)
        elif operation == "transpose":
            return np.transpose(grid)
        elif operation == "antitranspose":
            return np.flip(np.flip(np.transpose(grid), axis=0), axis=1)
        
        # Scaling operations
        elif operation == "scale_up_2x":
            return AtlasSynthesisExecutor._scale_operation(grid, 2.0)
        elif operation == "scale_down_2x":
            return AtlasSynthesisExecutor._scale_operation(grid, 0.5)
        elif operation == "scale_up_3x":
            return AtlasSynthesisExecutor._scale_operation(grid, 3.0)
        elif operation == "scale_custom":
            factor = params.get('factor', 1.0)
            return AtlasSynthesisExecutor._scale_operation(grid, factor)
        
        # Translation operations
        elif operation == "translate":
            dx = params.get('dx', 0)
            dy = params.get('dy', 0)
            return AtlasSynthesisExecutor._translate_operation(grid, dx, dy)
        elif operation == "translate_cyclic":
            dx = params.get('dx', 0)
            dy = params.get('dy', 0)
            return AtlasSynthesisExecutor._translate_cyclic_operation(grid, dx, dy)
        
        # Mirroring operations
        elif operation == "mirror_horizontal":
            return AtlasSynthesisExecutor._mirror_horizontal_operation(grid)
        elif operation == "mirror_vertical":
            return AtlasSynthesisExecutor._mirror_vertical_operation(grid)
        elif operation == "mirror_both":
            return AtlasSynthesisExecutor._mirror_both_operation(grid)
        elif operation == "mirror_diagonal":
            return AtlasSynthesisExecutor._mirror_diagonal_operation(grid)
        
        # Tiling operations
        elif operation == "tile_2x2":
            return AtlasSynthesisExecutor._tile_operation(grid, 2, 2)
        elif operation == "tile_3x3":
            return AtlasSynthesisExecutor._tile_operation(grid, 3, 3)
        elif operation == "tile_custom":
            tiles_x = params.get('tiles_x', 2)
            tiles_y = params.get('tiles_y', 2)
            return AtlasSynthesisExecutor._tile_operation(grid, tiles_x, tiles_y)
        
        # Pattern operations
        elif operation == "checkerboard":
            return AtlasSynthesisExecutor._checkerboard_operation(grid, params)
        elif operation == "radial_pattern":
            return AtlasSynthesisExecutor._radial_pattern_operation(grid, params)
        elif operation == "stripe_pattern":
            return AtlasSynthesisExecutor._stripe_pattern_operation(grid, params)
        
        # Morphological operations
        elif operation == "dilate":
            return AtlasSynthesisExecutor._dilate_operation(grid, params)
        elif operation == "erode":
            return AtlasSynthesisExecutor._erode_operation(grid, params)
        elif operation == "open":
            eroded = AtlasSynthesisExecutor._erode_operation(grid, params)
            return AtlasSynthesisExecutor._dilate_operation(eroded, params)
        elif operation == "close":
            dilated = AtlasSynthesisExecutor._dilate_operation(grid, params)
            return AtlasSynthesisExecutor._erode_operation(dilated, params)
        
        # Advanced transformations
        elif operation == "perspective_transform":
            return AtlasSynthesisExecutor._perspective_transform_operation(grid, params)
        elif operation == "barrel_distortion":
            return AtlasSynthesisExecutor._barrel_distortion_operation(grid, params)
        elif operation == "wave_transform":
            return AtlasSynthesisExecutor._wave_transform_operation(grid, params)
        
        else:
            return grid
    
    # Implementation of operation methods
    @staticmethod
    def _scale_operation(grid: np.ndarray, factor: float) -> np.ndarray:
        """Scale operation with proper handling"""
        if factor == 1.0:
            return grid
        
        h, w = grid.shape
        new_h, new_w = max(1, int(h * factor)), max(1, int(w * factor))
        
        # Nearest neighbor scaling
        scaled = np.zeros((new_h, new_w), dtype=grid.dtype)
        for i in range(new_h):
            for j in range(new_w):
                orig_i = min(int(i / factor), h - 1)
                orig_j = min(int(j / factor), w - 1)
                scaled[i, j] = grid[orig_i, orig_j]
        
        return scaled
    
    @staticmethod
    def _translate_operation(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate with zero padding"""
        if dx == 0 and dy == 0:
            return grid
        
        result = np.zeros_like(grid)
        h, w = grid.shape
        
        for i in range(h):
            for j in range(w):
                new_i = i + dy
                new_j = j + dx
                if 0 <= new_i < h and 0 <= new_j < w:
                    result[new_i, new_j] = grid[i, j]
        
        return result
    
    @staticmethod
    def _translate_cyclic_operation(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate with cyclic boundary conditions"""
        if dx == 0 and dy == 0:
            return grid
        
        return np.roll(np.roll(grid, dx, axis=1), dy, axis=0)
    
    @staticmethod
    def _mirror_horizontal_operation(grid: np.ndarray) -> np.ndarray:
        """Create horizontal mirror pattern"""
        h, w = grid.shape
        w_half = w // 2
        
        if w_half == 0:
            return grid
        
        left_half = grid[:, :w_half]
        result = np.zeros_like(grid)
        result[:, :w_half] = left_half
        
        # Mirror to right side
        mirror_w = min(w_half, w - w_half)
        result[:, w_half:w_half + mirror_w] = np.flip(left_half[:, :mirror_w], axis=1)
        
        return result
    
    @staticmethod
    def _mirror_vertical_operation(grid: np.ndarray) -> np.ndarray:
        """Create vertical mirror pattern"""
        h, w = grid.shape
        h_half = h // 2
        
        if h_half == 0:
            return grid
        
        top_half = grid[:h_half, :]
        result = np.zeros_like(grid)
        result[:h_half, :] = top_half
        
        # Mirror to bottom
        mirror_h = min(h_half, h - h_half)
        result[h_half:h_half + mirror_h, :] = np.flip(top_half[:mirror_h, :], axis=0)
        
        return result
    
    @staticmethod
    def _mirror_both_operation(grid: np.ndarray) -> np.ndarray:
        """Create quadrant mirror pattern"""
        h, w = grid.shape
        h_half, w_half = h // 2, w // 2
        
        if h_half == 0 or w_half == 0:
            return grid
        
        quarter = grid[:h_half, :w_half]
        result = np.zeros_like(grid)
        
        # Top-left (original)
        result[:h_half, :w_half] = quarter
        
        # Top-right (horizontal mirror)
        mirror_w = min(w_half, w - w_half)
        result[:h_half, w_half:w_half + mirror_w] = np.flip(quarter[:, :mirror_w], axis=1)
        
        # Bottom-left (vertical mirror)
        mirror_h = min(h_half, h - h_half)
        result[h_half:h_half + mirror_h, :w_half] = np.flip(quarter[:mirror_h, :], axis=0)
        
        # Bottom-right (both mirrors)
        result[h_half:h_half + mirror_h, w_half:w_half + mirror_w] = \
            np.flip(np.flip(quarter[:mirror_h, :mirror_w], axis=0), axis=1)
        
        return result
    
    @staticmethod
    def _mirror_diagonal_operation(grid: np.ndarray) -> np.ndarray:
        """Create diagonal mirror pattern"""
        h, w = grid.shape
        size = min(h, w)
        
        if size <= 1:
            return grid
        
        # Extract upper triangle
        result = grid.copy()
        for i in range(size):
            for j in range(i + 1, size):
                result[j, i] = result[i, j]  # Mirror across main diagonal
        
        return result
    
    @staticmethod
    def _tile_operation(grid: np.ndarray, tiles_x: int, tiles_y: int) -> np.ndarray:
        """Create tiled pattern"""
        h, w = grid.shape
        tile_h = max(1, h // tiles_y)
        tile_w = max(1, w // tiles_x)
        
        base_tile = grid[:tile_h, :tile_w]
        result = np.zeros_like(grid)
        
        for i in range(tiles_y):
            for j in range(tiles_x):
                start_i = i * tile_h
                start_j = j * tile_w
                end_i = min(start_i + tile_h, h)
                end_j = min(start_j + tile_w, w)
                
                tile_h_actual = end_i - start_i
                tile_w_actual = end_j - start_j
                
                result[start_i:end_i, start_j:end_j] = base_tile[:tile_h_actual, :tile_w_actual]
        
        return result
    
    @staticmethod
    def _checkerboard_operation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create checkerboard pattern"""
        h, w = grid.shape
        check_size = params.get('check_size', 2)
        
        result = grid.copy()
        
        for i in range(h):
            for j in range(w):
                check_i = i // check_size
                check_j = j // check_size
                if (check_i + check_j) % 2 == 1:
                    result[i, j] = 0  # Or some other transformation
        
        return result
    
    @staticmethod
    def _radial_pattern_operation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create radial pattern"""
        h, w = grid.shape
        center_i, center_j = h // 2, w // 2
        max_radius = params.get('max_radius', min(h, w) // 2)
        
        result = grid.copy()
        
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if distance <= max_radius:
                    # Apply some radial transformation
                    angle = np.arctan2(i - center_i, j - center_j)
                    intensity = 1.0 - (distance / max_radius)
                    result[i, j] = grid[i, j] * intensity
        
        return result
    
    @staticmethod
    def _stripe_pattern_operation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create stripe pattern"""
        h, w = grid.shape
        stripe_width = params.get('stripe_width', 3)
        direction = params.get('direction', 'horizontal')  # 'horizontal' or 'vertical'
        
        result = grid.copy()
        
        if direction == 'horizontal':
            for i in range(h):
                if (i // stripe_width) % 2 == 1:
                    result[i, :] = 0  # Or some other transformation
        else:  # vertical
            for j in range(w):
                if (j // stripe_width) % 2 == 1:
                    result[:, j] = 0
        
        return result
    
    @staticmethod
    def _dilate_operation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Morphological dilation"""
        kernel_size = params.get('kernel_size', 3)
        
        h, w = grid.shape
        result = grid.copy()
        
        offset = kernel_size // 2
        
        for i in range(h):
            for j in range(w):
                max_val = 0
                for di in range(-offset, offset + 1):
                    for dj in range(-offset, offset + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            max_val = max(max_val, grid[ni, nj])
                result[i, j] = max_val
        
        return result
    
    @staticmethod
    def _erode_operation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Morphological erosion"""
        kernel_size = params.get('kernel_size', 3)
        
        h, w = grid.shape
        result = grid.copy()
        
        offset = kernel_size // 2
        
        for i in range(h):
            for j in range(w):
                min_val = float('inf')
                for di in range(-offset, offset + 1):
                    for dj in range(-offset, offset + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            min_val = min(min_val, grid[ni, nj])
                        else:
                            min_val = 0  # Assume boundary is 0
                result[i, j] = min_val if min_val != float('inf') else 0
        
        return result
    
    @staticmethod
    def _perspective_transform_operation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Simple perspective transformation"""
        # Simplified implementation - just return original for now
        return grid
    
    @staticmethod
    def _barrel_distortion_operation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Barrel distortion transformation"""
        # Simplified implementation - just return original for now
        return grid
    
    @staticmethod
    def _wave_transform_operation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Wave transformation"""
        h, w = grid.shape
        amplitude = params.get('amplitude', 2)
        frequency = params.get('frequency', 1)
        direction = params.get('direction', 'horizontal')
        
        result = np.zeros_like(grid)
        
        for i in range(h):
            for j in range(w):
                if direction == 'horizontal':
                    offset = int(amplitude * np.sin(2 * np.pi * frequency * j / w))
                    new_i = (i + offset) % h
                    result[new_i, j] = grid[i, j]
                else:  # vertical
                    offset = int(amplitude * np.sin(2 * np.pi * frequency * i / h))
                    new_j = (j + offset) % w
                    result[i, new_j] = grid[i, j]
        
        return result


class ATLASProgramSynthesizer:
    """Advanced ATLAS program synthesizer with learning capabilities"""
    
    def __init__(self, max_program_length: int = 5):
        self.max_program_length = max_program_length
        self.program_library = {}
        self.synthesis_cache = {}
        self.operation_weights = defaultdict(lambda: 1.0)
        self.success_patterns = deque(maxlen=2000)
        self.failure_patterns = deque(maxlen=500)
        
        # Define operation categories
        self.basic_operations = [
            "identity", "rotate_cw_90", "rotate_cw_180", "rotate_cw_270",
            "flip_horizontal", "flip_vertical", "transpose", "antitranspose"
        ]
        
        self.scaling_operations = [
            "scale_up_2x", "scale_down_2x", "scale_up_3x"
        ]
        
        self.translation_operations = [
            ("translate", [
                {"dx": 1, "dy": 0}, {"dx": -1, "dy": 0},
                {"dx": 0, "dy": 1}, {"dx": 0, "dy": -1},
                {"dx": 1, "dy": 1}, {"dx": -1, "dy": -1}
            ]),
            ("translate_cyclic", [
                {"dx": 1, "dy": 0}, {"dx": -1, "dy": 0},
                {"dx": 0, "dy": 1}, {"dx": 0, "dy": -1}
            ])
        ]
        
        self.mirroring_operations = [
            "mirror_horizontal", "mirror_vertical", "mirror_both", "mirror_diagonal"
        ]
        
        self.tiling_operations = [
            "tile_2x2", "tile_3x3"
        ]
        
        self.pattern_operations = [
            ("checkerboard", [{"check_size": 2}, {"check_size": 3}]),
            ("stripe_pattern", [
                {"stripe_width": 2, "direction": "horizontal"},
                {"stripe_width": 3, "direction": "vertical"}
            ])
        ]
        
        self.morphological_operations = [
            ("dilate", [{"kernel_size": 3}]),
            ("erode", [{"kernel_size": 3}]),
            ("open", [{"kernel_size": 3}]),
            ("close", [{"kernel_size": 3}])
        ]
    
    def synthesize_program(self, input_grid: np.ndarray, 
                          output_grid: np.ndarray,
                          max_attempts: int = 200,
                          beam_width: int = 5) -> Optional[AtlasSynthesisProgram]:
        """Synthesize program using beam search"""
        
        # Check cache first
        cache_key = self._create_cache_key(input_grid, output_grid)
        if cache_key in self.synthesis_cache:
            return self.synthesis_cache[cache_key]
        
        start_time = time.time()
        
        # Initialize beam with empty programs
        beam = [AtlasSynthesisProgram(
            program_id=f"prog_{random.randint(1000, 9999)}",
            operations=[],
            input_signature=self._extract_signature(input_grid),
            output_signature=self._extract_signature(output_grid),
            execution_time=0.0,
            success_rate=0.0,
            confidence_score=0.0,
            spatial_complexity=0,
            transformation_type="unknown",
            generalization_score=0.0
        )]
        
        best_program = None
        best_score = float('inf')
        
        # Beam search
        for length in range(1, self.max_program_length + 1):
            new_beam = []
            
            for program in beam:
                # Generate extensions
                extensions = self._generate_program_extensions(program, input_grid, output_grid)
                
                for extended_program in extensions[:beam_width]:
                    try:
                        result = extended_program.execute(input_grid)
                        score = self._evaluate_program(result, output_grid)
                        
                        extended_program.confidence_score = 1.0 / (1.0 + score)
                        extended_program.success_rate = 1.0 if score < 0.01 else 0.0
                        extended_program.spatial_complexity = self._calculate_spatial_complexity(extended_program)
                        extended_program.transformation_type = self._classify_transformation(extended_program)
                        
                        if score < best_score:
                            best_score = score
                            best_program = extended_program
                            
                            # Early termination for exact match
                            if score < 0.01:
                                break
                        
                        if extended_program.confidence_score > 0.1:  # Only keep promising programs
                            new_beam.append(extended_program)
                            
                    except Exception:
                        continue
                
                if best_program and best_program.success_rate > 0.9:
                    break
            
            # Keep best programs for next iteration
            beam = sorted(new_beam, key=lambda p: p.confidence_score, reverse=True)[:beam_width]
            
            if not beam:
                break
        
        # Finalize best program
        if best_program:
            best_program.execution_time = time.time() - start_time
            best_program.generalization_score = self._estimate_generalization(best_program)
            
            # Cache the result
            self.synthesis_cache[cache_key] = best_program
            
            # Update pattern libraries
            if best_program.success_rate > 0.5:
                self.success_patterns.append(best_program)
                self._update_operation_weights(best_program, success=True)
            else:
                self.failure_patterns.append(best_program)
                self._update_operation_weights(best_program, success=False)
        
        return best_program
    
    def _generate_program_extensions(self, program: AtlasSynthesisProgram,
                                   input_grid: np.ndarray,
                                   output_grid: np.ndarray) -> List[AtlasSynthesisProgram]:
        """Generate extensions of current program"""
        extensions = []
        
        # Get weighted operation candidates
        operation_candidates = self._get_weighted_operations()
        
        for operation, params in operation_candidates[:20]:  # Limit candidates
            new_operations = program.operations + [(operation, params)]
            
            extended_program = AtlasSynthesisProgram(
                program_id=f"{program.program_id}_ext_{len(extensions)}",
                operations=new_operations,
                input_signature=program.input_signature,
                output_signature=program.output_signature,
                execution_time=0.0,
                success_rate=0.0,
                confidence_score=0.0,
                spatial_complexity=0,
                transformation_type="unknown",
                generalization_score=0.0
            )
            
            extensions.append(extended_program)
        
        return extensions
    
    def _get_weighted_operations(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get operations weighted by past success"""
        candidates = []
        
        # Basic operations
        for op in self.basic_operations:
            weight = self.operation_weights[op]
            candidates.append((op, {}, weight))
        
        # Scaling operations
        for op in self.scaling_operations:
            weight = self.operation_weights[op]
            candidates.append((op, {}, weight))
        
        # Translation operations
        for op, param_list in self.translation_operations:
            for params in param_list:
                weight = self.operation_weights[op]
                candidates.append((op, params, weight))
        
        # Mirroring operations
        for op in self.mirroring_operations:
            weight = self.operation_weights[op]
            candidates.append((op, {}, weight))
        
        # Tiling operations
        for op in self.tiling_operations:
            weight = self.operation_weights[op]
            candidates.append((op, {}, weight))
        
        # Pattern operations
        for op, param_list in self.pattern_operations:
            for params in param_list:
                weight = self.operation_weights[op]
                candidates.append((op, params, weight))
        
        # Morphological operations
        for op, param_list in self.morphological_operations:
            for params in param_list:
                weight = self.operation_weights[op]
                candidates.append((op, params, weight))
        
        # Sort by weight and return operation, params pairs
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(op, params) for op, params, weight in candidates]
    
    def _extract_signature(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract spatial signature from grid"""
        return {
            'shape': grid.shape,
            'mean': float(np.mean(grid)),
            'std': float(np.std(grid)),
            'max': float(np.max(grid)),
            'min': float(np.min(grid)),
            'unique_values': len(np.unique(grid)),
            'symmetry_h': self._calculate_symmetry(grid, axis=0),
            'symmetry_v': self._calculate_symmetry(grid, axis=1),
            'edge_density': self._calculate_edge_density(grid)
        }
    
    def _calculate_symmetry(self, grid: np.ndarray, axis: int) -> float:
        """Calculate symmetry along axis"""
        flipped = np.flip(grid, axis=axis)
        return float(1.0 - np.mean(np.abs(grid - flipped)))
    
    def _calculate_edge_density(self, grid: np.ndarray) -> float:
        """Calculate edge density in grid"""
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return 0.0
        
        grad_x = np.abs(grid[1:, :] - grid[:-1, :])
        grad_y = np.abs(grid[:, 1:] - grid[:, :-1])
        
        edge_magnitude = np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
        return float(np.mean(edge_magnitude))
    
    def _create_cache_key(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Create cache key for grid pair"""
        input_hash = hash(input_grid.tobytes())
        output_hash = hash(output_grid.tobytes())
        return f"{input_hash}_{output_hash}"
    
    def _evaluate_program(self, result: np.ndarray, target: np.ndarray) -> float:
        """Evaluate program output quality"""
        if result.shape != target.shape:
            return float('inf')
        
        # Primary metric: MSE
        mse = np.mean((result - target) ** 2)
        
        # Secondary metrics
        structural_penalty = self._compute_structural_penalty(result, target)
        complexity_penalty = 0.01 * len(result.shape)  # Prefer simpler results
        
        return mse + 0.1 * structural_penalty + complexity_penalty
    
    def _compute_structural_penalty(self, result: np.ndarray, target: np.ndarray) -> float:
        """Compute structural difference penalty"""
        # Edge preservation
        result_edges = self._calculate_edge_density(result)
        target_edges = self._calculate_edge_density(target)
        edge_penalty = abs(result_edges - target_edges)
        
        # Symmetry preservation
        result_sym_h = self._calculate_symmetry(result, axis=0)
        target_sym_h = self._calculate_symmetry(target, axis=0)
        sym_penalty_h = abs(result_sym_h - target_sym_h)
        
        result_sym_v = self._calculate_symmetry(result, axis=1)
        target_sym_v = self._calculate_symmetry(target, axis=1)
        sym_penalty_v = abs(result_sym_v - target_sym_v)
        
        return edge_penalty + sym_penalty_h + sym_penalty_v
    
    def _calculate_spatial_complexity(self, program: AtlasSynthesisProgram) -> int:
        """Calculate spatial complexity of program"""
        complexity = 0
        
        for operation, params in program.operations:
            # Base complexity for operation
            complexity += 1
            
            # Additional complexity for parameters
            complexity += len(params)
            
            # Special complexity for certain operations
            if 'tile' in operation or 'pattern' in operation:
                complexity += 2
            elif 'transform' in operation or 'distortion' in operation:
                complexity += 3
        
        return complexity
    
    def _classify_transformation(self, program: AtlasSynthesisProgram) -> str:
        """Classify the type of transformation"""
        operations = [op for op, _ in program.operations]
        
        if any('rotate' in op for op in operations):
            return "rotation"
        elif any('flip' in op or 'mirror' in op for op in operations):
            return "reflection"
        elif any('scale' in op for op in operations):
            return "scaling"
        elif any('translate' in op for op in operations):
            return "translation"
        elif any('tile' in op for op in operations):
            return "tiling"
        elif any('pattern' in op for op in operations):
            return "pattern_generation"
        elif any(op in ['dilate', 'erode', 'open', 'close'] for op in operations):
            return "morphological"
        else:
            return "composite"
    
    def _estimate_generalization(self, program: AtlasSynthesisProgram) -> float:
        """Estimate how well program might generalize"""
        # Simpler programs tend to generalize better
        complexity_penalty = program.spatial_complexity * 0.1
        
        # Programs with basic operations generalize better
        basic_ops = sum(1 for op, _ in program.operations if op in self.basic_operations)
        basic_bonus = basic_ops * 0.2
        
        # High confidence suggests good generalization
        confidence_bonus = program.confidence_score * 0.5
        
        return max(0.0, min(1.0, confidence_bonus + basic_bonus - complexity_penalty))
    
    def _update_operation_weights(self, program: AtlasSynthesisProgram, success: bool):
        """Update operation weights based on success/failure"""
        weight_delta = 0.1 if success else -0.05
        
        for operation, _ in program.operations:
            self.operation_weights[operation] += weight_delta
            # Keep weights positive
            self.operation_weights[operation] = max(0.1, self.operation_weights[operation])
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get synthesis statistics"""
        return {
            'cache_size': len(self.synthesis_cache),
            'success_patterns': len(self.success_patterns),
            'failure_patterns': len(self.failure_patterns),
            'operation_weights': dict(self.operation_weights),
            'avg_success_complexity': self._compute_avg_complexity(self.success_patterns),
            'avg_failure_complexity': self._compute_avg_complexity(self.failure_patterns)
        }
    
    def _compute_avg_complexity(self, programs: deque) -> float:
        """Compute average complexity of programs"""
        if not programs:
            return 0.0
        
        total_complexity = sum(p.spatial_complexity for p in programs)
        return total_complexity / len(programs)


def create_atlas_synthesis_system(device: str = 'cuda') -> Dict[str, Any]:
    """Create complete ATLAS synthesis system"""
    
    synthesizer = ATLASProgramSynthesizer()
    
    def synthesize_spatial_program(input_grid: np.ndarray,
                                  output_grid: np.ndarray) -> Optional[AtlasSynthesisProgram]:
        """Synthesize spatial transformation program"""
        return synthesizer.synthesize_program(input_grid, output_grid)
    
    def synthesize_from_examples(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[AtlasSynthesisProgram]:
        """Synthesize program from multiple examples"""
        if not examples:
            return None
        
        # Use first example as primary
        input_grid, output_grid = examples[0]
        program = synthesizer.synthesize_program(input_grid, output_grid)
        
        if program is None:
            return None
        
        # Test on other examples
        success_count = 0
        for test_input, test_output in examples:
            try:
                result = program.execute(test_input)
                score = synthesizer._evaluate_program(result, test_output)
                if score < 0.1:  # Reasonable threshold
                    success_count += 1
            except Exception:
                continue
        
        # Update generalization score
        program.generalization_score = success_count / len(examples)
        
        return program if program.generalization_score > 0.5 else None
    
    def execute_program_on_grid(program: AtlasSynthesisProgram,
                               input_grid: np.ndarray) -> np.ndarray:
        """Execute synthesis program on grid"""
        return program.execute(input_grid)
    
    def get_program_library() -> List[AtlasSynthesisProgram]:
        """Get current program library"""
        return list(synthesizer.success_patterns)
    
    return {
        'synthesizer': synthesizer,
        'synthesize_program': synthesize_spatial_program,
        'synthesize_from_examples': synthesize_from_examples,
        'execute_program': execute_program_on_grid,
        'get_library': get_program_library,
        'get_statistics': synthesizer.get_synthesis_statistics,
        'name': 'ATLAS_Synthesis_v1'
    }