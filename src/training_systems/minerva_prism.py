"""
MINERVA-specific PRISM (Program Reasoning through Inductive Synthesis) System
Focuses on synthesizing grid transformation programs and spatial reasoning rules
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
from enum import Enum


class GridOperation(Enum):
    """MINERVA-specific grid operations"""
    # Basic transformations
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_H = "flip_horizontal"
    FLIP_V = "flip_vertical"
    TRANSPOSE = "transpose"
    
    # Color operations
    SWAP_COLORS = "swap_colors"
    FILL_COLOR = "fill_color"
    REPLACE_COLOR = "replace_color"
    
    # Grid structure operations
    EXTRACT_SUBGRID = "extract_subgrid"
    TILE_PATTERN = "tile_pattern"
    MIRROR_HALF = "mirror_half"
    COMPLETE_SYMMETRY = "complete_symmetry"
    
    # Object operations
    EXTRACT_OBJECTS = "extract_objects"
    MERGE_OBJECTS = "merge_objects"
    MOVE_OBJECT = "move_object"
    DUPLICATE_OBJECT = "duplicate_object"
    
    # Boundary operations
    EXTRACT_BOUNDARY = "extract_boundary"
    FILL_INTERIOR = "fill_interior"
    THICKEN_BOUNDARY = "thicken_boundary"
    
    # Pattern operations
    REPEAT_PATTERN = "repeat_pattern"
    DETECT_PATTERN = "detect_pattern"
    APPLY_RULE = "apply_rule"


@dataclass
class GridProgram:
    """A program for grid transformation"""
    operations: List[Tuple[GridOperation, Dict[str, Any]]]
    confidence: float = 1.0
    complexity: int = 0
    
    def __post_init__(self):
        self.complexity = len(self.operations)
    
    def to_string(self) -> str:
        """Convert program to readable string"""
        parts = []
        for op, params in self.operations:
            if params:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                parts.append(f"{op.value}({param_str})")
            else:
                parts.append(op.value)
        return " -> ".join(parts)
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute program on grid"""
        result = grid.copy()
        for op, params in self.operations:
            result = apply_grid_operation(result, op, params)
        return result


def apply_grid_operation(grid: np.ndarray, operation: GridOperation, 
                        params: Dict[str, Any]) -> np.ndarray:
    """Apply a single grid operation"""
    
    if operation == GridOperation.ROTATE_90:
        return np.rot90(grid, k=1)
    elif operation == GridOperation.ROTATE_180:
        return np.rot90(grid, k=2)
    elif operation == GridOperation.ROTATE_270:
        return np.rot90(grid, k=3)
    elif operation == GridOperation.FLIP_H:
        return np.fliplr(grid)
    elif operation == GridOperation.FLIP_V:
        return np.flipud(grid)
    elif operation == GridOperation.TRANSPOSE:
        return grid.T
    
    elif operation == GridOperation.SWAP_COLORS:
        color1 = params.get('color1', 0)
        color2 = params.get('color2', 1)
        result = grid.copy()
        mask1 = grid == color1
        mask2 = grid == color2
        result[mask1] = color2
        result[mask2] = color1
        return result
    
    elif operation == GridOperation.FILL_COLOR:
        color = params.get('color', 0)
        target = params.get('target', None)
        if target is None:
            return np.full_like(grid, color)
        else:
            result = grid.copy()
            result[grid == target] = color
            return result
    
    elif operation == GridOperation.MIRROR_HALF:
        direction = params.get('direction', 'horizontal')
        if direction == 'horizontal':
            half = grid.shape[1] // 2
            result = grid.copy()
            result[:, half:] = np.fliplr(grid[:, :half])[:, :grid.shape[1]-half]
            return result
        else:
            half = grid.shape[0] // 2
            result = grid.copy()
            result[half:, :] = np.flipud(grid[:half, :])[:grid.shape[0]-half, :]
            return result
    
    elif operation == GridOperation.EXTRACT_BOUNDARY:
        thickness = params.get('thickness', 1)
        result = np.zeros_like(grid)
        result[:thickness, :] = grid[:thickness, :]
        result[-thickness:, :] = grid[-thickness:, :]
        result[:, :thickness] = grid[:, :thickness]
        result[:, -thickness:] = grid[:, -thickness:]
        return result
    
    # Add more operations as needed
    else:
        return grid


class MinervaProgramSynthesizer:
    """MINERVA-specific program synthesizer for grid transformations"""
    
    def __init__(self, max_program_length: int = 5, beam_size: int = 10):
        self.max_program_length = max_program_length
        self.beam_size = beam_size
        
        # MINERVA-specific operation sets
        self.basic_ops = [
            GridOperation.ROTATE_90, GridOperation.ROTATE_180, 
            GridOperation.ROTATE_270, GridOperation.FLIP_H, 
            GridOperation.FLIP_V, GridOperation.TRANSPOSE
        ]
        
        self.color_ops = [
            GridOperation.SWAP_COLORS, GridOperation.FILL_COLOR,
            GridOperation.REPLACE_COLOR
        ]
        
        self.structure_ops = [
            GridOperation.MIRROR_HALF, GridOperation.EXTRACT_BOUNDARY,
            GridOperation.COMPLETE_SYMMETRY
        ]
        
        # Learned program templates
        self.program_templates = self._initialize_templates()
        
        # Performance tracking
        self.synthesis_stats = defaultdict(int)
        self.successful_programs = []
    
    def _initialize_templates(self) -> List[List[GridOperation]]:
        """Initialize common program templates"""
        return [
            # Rotation templates
            [GridOperation.ROTATE_90],
            [GridOperation.ROTATE_180],
            [GridOperation.FLIP_H],
            [GridOperation.FLIP_V],
            
            # Symmetry templates
            [GridOperation.MIRROR_HALF],
            [GridOperation.EXTRACT_BOUNDARY, GridOperation.FILL_COLOR],
            
            # Combined transformations
            [GridOperation.ROTATE_90, GridOperation.FLIP_H],
            [GridOperation.TRANSPOSE, GridOperation.FLIP_V],
        ]
    
    def synthesize(self, input_grid: np.ndarray, output_grid: np.ndarray,
                  timeout: float = 2.0) -> Optional[GridProgram]:
        """Synthesize a program that transforms input to output"""
        
        # Quick check for simple transformations
        quick_program = self._check_simple_transformations(input_grid, output_grid)
        if quick_program:
            self.synthesis_stats['quick_success'] += 1
            return quick_program
        
        # Analyze grids
        analysis = self._analyze_transformation(input_grid, output_grid)
        
        # Try template-based synthesis
        template_program = self._template_synthesis(input_grid, output_grid, analysis)
        if template_program:
            self.synthesis_stats['template_success'] += 1
            return template_program
        
        # Beam search for program
        beam_program = self._beam_search_synthesis(input_grid, output_grid, analysis)
        if beam_program:
            self.synthesis_stats['beam_success'] += 1
            self.successful_programs.append(beam_program)
            return beam_program
        
        self.synthesis_stats['failed'] += 1
        return None
    
    def _check_simple_transformations(self, input_grid: np.ndarray, 
                                    output_grid: np.ndarray) -> Optional[GridProgram]:
        """Check for simple single-operation transformations"""
        
        # Check basic transformations
        for op in self.basic_ops:
            transformed = apply_grid_operation(input_grid, op, {})
            if np.array_equal(transformed, output_grid):
                return GridProgram([(op, {})], confidence=1.0)
        
        # Check color swaps
        input_colors = np.unique(input_grid)
        output_colors = np.unique(output_grid)
        
        if len(input_colors) == 2 and len(output_colors) == 2:
            if set(input_colors) == set(output_colors):
                # Try color swap
                op = GridOperation.SWAP_COLORS
                params = {'color1': int(input_colors[0]), 'color2': int(input_colors[1])}
                transformed = apply_grid_operation(input_grid, op, params)
                if np.array_equal(transformed, output_grid):
                    return GridProgram([(op, params)], confidence=1.0)
        
        return None
    
    def _analyze_transformation(self, input_grid: np.ndarray, 
                              output_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze the transformation characteristics"""
        
        analysis = {
            'size_changed': input_grid.shape != output_grid.shape,
            'colors_changed': not np.array_equal(np.unique(input_grid), np.unique(output_grid)),
            'structure_preserved': self._check_structure_preservation(input_grid, output_grid),
            'has_symmetry': self._check_symmetry(output_grid),
            'transformation_type': self._classify_transformation(input_grid, output_grid)
        }
        
        return analysis
    
    def _check_structure_preservation(self, input_grid: np.ndarray, 
                                    output_grid: np.ndarray) -> bool:
        """Check if basic structure is preserved"""
        if input_grid.shape != output_grid.shape:
            return False
        
        # Check if non-zero positions are similar
        input_mask = input_grid != 0
        output_mask = output_grid != 0
        
        overlap = np.sum(input_mask & output_mask)
        total = np.sum(input_mask | output_mask)
        
        return overlap / max(1, total) > 0.5
    
    def _check_symmetry(self, grid: np.ndarray) -> Dict[str, bool]:
        """Check various symmetries"""
        return {
            'horizontal': np.array_equal(grid, np.fliplr(grid)),
            'vertical': np.array_equal(grid, np.flipud(grid)),
            'diagonal': np.array_equal(grid, grid.T),
            'rotational': np.array_equal(grid, np.rot90(grid, k=2))
        }
    
    def _classify_transformation(self, input_grid: np.ndarray, 
                               output_grid: np.ndarray) -> str:
        """Classify the type of transformation"""
        
        if input_grid.shape != output_grid.shape:
            return 'resize'
        
        # Check for geometric transformations
        for k in range(1, 4):
            if np.array_equal(np.rot90(input_grid, k=k), output_grid):
                return f'rotation_{k*90}'
        
        if np.array_equal(np.fliplr(input_grid), output_grid):
            return 'flip_horizontal'
        if np.array_equal(np.flipud(input_grid), output_grid):
            return 'flip_vertical'
        
        # Check for color transformations
        if not np.array_equal(np.unique(input_grid), np.unique(output_grid)):
            return 'color_change'
        
        # Check for pattern transformations
        if self._is_pattern_completion(input_grid, output_grid):
            return 'pattern_completion'
        
        return 'complex'
    
    def _is_pattern_completion(self, input_grid: np.ndarray, 
                              output_grid: np.ndarray) -> bool:
        """Check if output completes a pattern from input"""
        # Simple check: output contains input
        if input_grid.shape != output_grid.shape:
            return False
        
        mask = input_grid != 0
        return np.all(output_grid[mask] == input_grid[mask])
    
    def _template_synthesis(self, input_grid: np.ndarray, output_grid: np.ndarray,
                          analysis: Dict[str, Any]) -> Optional[GridProgram]:
        """Try synthesis using templates"""
        
        for template in self.program_templates:
            # Try template as-is
            program = GridProgram([(op, {}) for op in template])
            result = program.execute(input_grid)
            
            if np.array_equal(result, output_grid):
                return program
            
            # Try with color operations
            if analysis['colors_changed']:
                for color_op in self.color_ops:
                    # Add color operation to template
                    extended_ops = [(op, {}) for op in template]
                    
                    # Infer color parameters
                    if color_op == GridOperation.SWAP_COLORS:
                        colors = np.unique(input_grid)
                        if len(colors) >= 2:
                            params = {'color1': int(colors[0]), 'color2': int(colors[1])}
                            extended_ops.append((color_op, params))
                            
                            extended_program = GridProgram(extended_ops)
                            result = extended_program.execute(input_grid)
                            
                            if np.array_equal(result, output_grid):
                                return extended_program
        
        return None
    
    def _beam_search_synthesis(self, input_grid: np.ndarray, output_grid: np.ndarray,
                             analysis: Dict[str, Any]) -> Optional[GridProgram]:
        """Beam search for program synthesis"""
        
        # Initialize beam with empty program
        beam = [([], input_grid, 0.0)]  # (operations, current_grid, cost)
        
        for depth in range(self.max_program_length):
            new_beam = []
            
            for ops, current_grid, cost in beam:
                # Check if we've reached the target
                if np.array_equal(current_grid, output_grid):
                    return GridProgram(ops, confidence=1.0 - cost)
                
                # Generate candidates
                candidates = self._generate_candidates(current_grid, output_grid, analysis)
                
                for op, params in candidates:
                    try:
                        new_grid = apply_grid_operation(current_grid, op, params)
                        new_ops = ops + [(op, params)]
                        
                        # Compute cost
                        new_cost = cost + self._compute_step_cost(
                            current_grid, new_grid, output_grid
                        )
                        
                        new_beam.append((new_ops, new_grid, new_cost))
                    except:
                        continue
            
            # Prune beam
            new_beam.sort(key=lambda x: x[2])
            beam = new_beam[:self.beam_size]
            
            if not beam:
                break
        
        # Return best program if any reaches target
        for ops, grid, cost in beam:
            if np.array_equal(grid, output_grid):
                return GridProgram(ops, confidence=1.0 - cost)
        
        return None
    
    def _generate_candidates(self, current_grid: np.ndarray, target_grid: np.ndarray,
                           analysis: Dict[str, Any]) -> List[Tuple[GridOperation, Dict]]:
        """Generate candidate operations"""
        candidates = []
        
        # Always try basic operations
        for op in self.basic_ops:
            candidates.append((op, {}))
        
        # Add color operations if needed
        if analysis['colors_changed'] or not analysis['structure_preserved']:
            current_colors = np.unique(current_grid)
            target_colors = np.unique(target_grid)
            
            # Color swaps
            if len(current_colors) >= 2:
                for i, c1 in enumerate(current_colors[:3]):
                    for c2 in current_colors[i+1:4]:
                        candidates.append((
                            GridOperation.SWAP_COLORS,
                            {'color1': int(c1), 'color2': int(c2)}
                        ))
            
            # Fill operations
            for color in target_colors[:3]:
                candidates.append((
                    GridOperation.FILL_COLOR,
                    {'color': int(color)}
                ))
        
        # Add structure operations
        if analysis['has_symmetry']['horizontal'] or analysis['has_symmetry']['vertical']:
            candidates.append((GridOperation.MIRROR_HALF, {'direction': 'horizontal'}))
            candidates.append((GridOperation.MIRROR_HALF, {'direction': 'vertical'}))
        
        candidates.append((GridOperation.EXTRACT_BOUNDARY, {'thickness': 1}))
        
        return candidates
    
    def _compute_step_cost(self, prev_grid: np.ndarray, current_grid: np.ndarray,
                         target_grid: np.ndarray) -> float:
        """Compute cost of a transformation step"""
        
        # Distance to target
        target_distance = np.mean(current_grid != target_grid)
        
        # Complexity penalty
        complexity = 0.1
        
        # Progress reward
        prev_distance = np.mean(prev_grid != target_grid)
        progress = max(0, prev_distance - target_distance)
        
        return target_distance + complexity - progress
    
    def get_synthesis_stats(self) -> Dict[str, int]:
        """Get synthesis statistics"""
        return dict(self.synthesis_stats)


class MinervaProgramLibrary:
    """Library of learned programs for MINERVA"""
    
    def __init__(self, max_programs: int = 1000):
        self.max_programs = max_programs
        self.programs = {}
        self.program_index = defaultdict(list)  # Index by transformation type
        self.usage_stats = defaultdict(int)
    
    def add_program(self, program: GridProgram, input_example: np.ndarray,
                   output_example: np.ndarray):
        """Add a successful program to the library"""
        
        # Create program signature
        signature = self._compute_signature(input_example, output_example)
        
        if signature not in self.programs:
            self.programs[signature] = {
                'program': program,
                'examples': [(input_example, output_example)],
                'usage_count': 0,
                'success_rate': 1.0
            }
            
            # Index by transformation type
            transform_type = self._classify_program(program)
            self.program_index[transform_type].append(signature)
        else:
            # Add example to existing program
            self.programs[signature]['examples'].append((input_example, output_example))
    
    def _compute_signature(self, input_grid: np.ndarray, 
                         output_grid: np.ndarray) -> str:
        """Compute signature for transformation"""
        # Simple signature based on shapes and color counts
        sig_parts = [
            f"shape_{input_grid.shape}_{output_grid.shape}",
            f"colors_{len(np.unique(input_grid))}_{len(np.unique(output_grid))}"
        ]
        return "_".join(sig_parts)
    
    def _classify_program(self, program: GridProgram) -> str:
        """Classify program by its operations"""
        if not program.operations:
            return "empty"
        
        first_op = program.operations[0][0]
        
        if first_op in [GridOperation.ROTATE_90, GridOperation.ROTATE_180, 
                       GridOperation.ROTATE_270, GridOperation.FLIP_H, 
                       GridOperation.FLIP_V, GridOperation.TRANSPOSE]:
            return "geometric"
        elif first_op in [GridOperation.SWAP_COLORS, GridOperation.FILL_COLOR]:
            return "color"
        elif first_op in [GridOperation.MIRROR_HALF, GridOperation.EXTRACT_BOUNDARY]:
            return "structure"
        else:
            return "complex"
    
    def find_similar_programs(self, input_grid: np.ndarray, 
                            output_grid: np.ndarray, k: int = 5) -> List[GridProgram]:
        """Find similar programs that might work"""
        
        signature = self._compute_signature(input_grid, output_grid)
        
        # Direct match
        if signature in self.programs:
            return [self.programs[signature]['program']]
        
        # Find similar programs
        candidates = []
        
        for sig, prog_data in self.programs.items():
            # Simple similarity based on signature overlap
            if self._signatures_similar(signature, sig):
                candidates.append((prog_data['program'], prog_data['success_rate']))
        
        # Sort by success rate
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [prog for prog, _ in candidates[:k]]
    
    def _signatures_similar(self, sig1: str, sig2: str) -> bool:
        """Check if signatures are similar"""
        parts1 = sig1.split("_")
        parts2 = sig2.split("_")
        
        # Check shape similarity
        if "shape" in parts1[0] and "shape" in parts2[0]:
            # Both have same dimensionality
            return parts1[0].count(",") == parts2[0].count(",")
        
        return False


def create_minerva_prism_system() -> Dict:
    """Create MINERVA-specific PRISM components"""
    return {
        'synthesizer': MinervaProgramSynthesizer(),
        'library': MinervaProgramLibrary(),
        'operations': GridOperation
    }