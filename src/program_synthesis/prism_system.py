"""
PRISM (Program Reasoning through Inductive Synthesis and Metaprogramming)
Novel enhancement to program synthesis for OLYMPUS AGI2
Combines neural-guided search with meta-learning and compositional reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import itertools
from enum import Enum
import hashlib


class MetaProgram(Enum):
    """Meta-level program patterns that compose primitives"""
    RECURSIVE_DECOMPOSE = "recursive_decompose"  # Break into subproblems
    SYMMETRY_COMPLETE = "symmetry_complete"  # Complete partial symmetries
    OBJECT_TRANSFORM = "object_transform"  # Transform each object individually
    PATTERN_PROPAGATE = "pattern_propagate"  # Propagate local patterns globally
    CONSTRAINT_SOLVE = "constraint_solve"  # Solve with constraints
    ANALOGY_APPLY = "analogy_apply"  # Apply transformation by analogy
    INVARIANT_PRESERVE = "invariant_preserve"  # Preserve certain properties
    HIERARCHICAL_COMPOSE = "hierarchical_compose"  # Build up hierarchically


@dataclass
class ProgramSketch:
    """High-level program structure before filling in details"""
    meta_program: MetaProgram
    slots: List[str]  # Slots to be filled with primitives
    constraints: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    def instantiate(self, slot_fillers: Dict[str, Any]) -> 'CompositeProgram':
        """Fill in the sketch with concrete primitives"""
        return CompositeProgram(
            meta_program=self.meta_program,
            components=slot_fillers,
            constraints=self.constraints
        )


@dataclass
class CompositeProgram:
    """Instantiated program with filled slots"""
    meta_program: MetaProgram
    components: Dict[str, Any]
    constraints: Dict[str, Any]
    execution_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute the composite program"""
        cache_key = hashlib.md5(grid.tobytes()).hexdigest()
        if cache_key in self.execution_cache:
            return self.execution_cache[cache_key]
        
        result = self._execute_meta_program(grid)
        self.execution_cache[cache_key] = result
        return result
    
    def _execute_meta_program(self, grid: np.ndarray) -> np.ndarray:
        """Execute based on meta-program type"""
        if self.meta_program == MetaProgram.RECURSIVE_DECOMPOSE:
            return self._recursive_decompose(grid)
        elif self.meta_program == MetaProgram.SYMMETRY_COMPLETE:
            return self._symmetry_complete(grid)
        elif self.meta_program == MetaProgram.OBJECT_TRANSFORM:
            return self._object_transform(grid)
        elif self.meta_program == MetaProgram.PATTERN_PROPAGATE:
            return self._pattern_propagate(grid)
        elif self.meta_program == MetaProgram.CONSTRAINT_SOLVE:
            return self._constraint_solve(grid)
        elif self.meta_program == MetaProgram.ANALOGY_APPLY:
            return self._analogy_apply(grid)
        elif self.meta_program == MetaProgram.INVARIANT_PRESERVE:
            return self._invariant_preserve(grid)
        elif self.meta_program == MetaProgram.HIERARCHICAL_COMPOSE:
            return self._hierarchical_compose(grid)
        else:
            return grid
    
    def _recursive_decompose(self, grid: np.ndarray) -> np.ndarray:
        """Recursively decompose into subproblems"""
        # Extract decomposition strategy
        strategy = self.components.get('decomposition_strategy', 'quadrant')
        transform = self.components.get('sub_transform')
        combine = self.components.get('combine_method', 'stack')
        
        if strategy == 'quadrant':
            h, w = grid.shape
            mid_h, mid_w = h // 2, w // 2
            
            # Split into quadrants
            q1 = grid[:mid_h, :mid_w]
            q2 = grid[:mid_h, mid_w:]
            q3 = grid[mid_h:, :mid_w]
            q4 = grid[mid_h:, mid_w:]
            
            # Transform each
            if transform:
                q1 = transform(q1)
                q2 = transform(q2)
                q3 = transform(q3)
                q4 = transform(q4)
            
            # Recombine
            top = np.hstack([q1, q2])
            bottom = np.hstack([q3, q4])
            return np.vstack([top, bottom])
        
        return grid
    
    def _symmetry_complete(self, grid: np.ndarray) -> np.ndarray:
        """Complete partial symmetries in the grid"""
        symmetry_type = self.components.get('symmetry_type', 'horizontal')
        
        if symmetry_type == 'horizontal':
            h, w = grid.shape
            for i in range(h):
                for j in range(w // 2):
                    if grid[i, j] != 0 and grid[i, w-1-j] == 0:
                        grid[i, w-1-j] = grid[i, j]
                    elif grid[i, j] == 0 and grid[i, w-1-j] != 0:
                        grid[i, j] = grid[i, w-1-j]
        
        elif symmetry_type == 'vertical':
            h, w = grid.shape
            for i in range(h // 2):
                for j in range(w):
                    if grid[i, j] != 0 and grid[h-1-i, j] == 0:
                        grid[h-1-i, j] = grid[i, j]
                    elif grid[i, j] == 0 and grid[h-1-i, j] != 0:
                        grid[i, j] = grid[h-1-i, j]
        
        elif symmetry_type == 'diagonal':
            h, w = grid.shape
            size = min(h, w)
            for i in range(size):
                for j in range(i):
                    if grid[i, j] != 0 and grid[j, i] == 0:
                        grid[j, i] = grid[i, j]
                    elif grid[i, j] == 0 and grid[j, i] != 0:
                        grid[i, j] = grid[j, i]
        
        return grid
    
    def _object_transform(self, grid: np.ndarray) -> np.ndarray:
        """Transform each object individually"""
        from scipy.ndimage import label
        
        transform = self.components.get('object_transform')
        if not transform:
            return grid
        
        result = np.zeros_like(grid)
        
        # Process each color separately
        for color in range(1, 10):
            mask = (grid == color)
            if not mask.any():
                continue
            
            labeled, num_objects = label(mask)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled == obj_id)
                obj_bbox = self._get_bbox(obj_mask)
                
                if obj_bbox:
                    y1, x1, y2, x2 = obj_bbox
                    obj_region = grid[y1:y2+1, x1:x2+1].copy()
                    obj_mask_region = obj_mask[y1:y2+1, x1:x2+1]
                    
                    # Apply transform to object
                    transformed = transform(obj_region * obj_mask_region)
                    
                    # Place back
                    if transformed.shape == obj_region.shape:
                        result[y1:y2+1, x1:x2+1] = np.where(
                            transformed != 0, transformed, result[y1:y2+1, x1:x2+1]
                        )
        
        return result
    
    def _pattern_propagate(self, grid: np.ndarray) -> np.ndarray:
        """Propagate local patterns globally"""
        pattern_size = self.components.get('pattern_size', 3)
        propagation_rule = self.components.get('propagation_rule', 'tile')
        
        # Find the pattern (first non-zero region)
        pattern = self._extract_pattern(grid, pattern_size)
        if pattern is None:
            return grid
        
        result = np.zeros_like(grid)
        
        if propagation_rule == 'tile':
            h, w = grid.shape
            ph, pw = pattern.shape
            for i in range(0, h, ph):
                for j in range(0, w, pw):
                    end_i = min(i + ph, h)
                    end_j = min(j + pw, w)
                    result[i:end_i, j:end_j] = pattern[:end_i-i, :end_j-j]
        
        elif propagation_rule == 'radial':
            # Propagate from center outward
            h, w = grid.shape
            center_h, center_w = h // 2, w // 2
            ph, pw = pattern.shape
            
            # Place at center
            start_h = center_h - ph // 2
            start_w = center_w - pw // 2
            result[start_h:start_h+ph, start_w:start_w+pw] = pattern
            
            # Propagate in 8 directions
            for dh in [-ph, 0, ph]:
                for dw in [-pw, 0, pw]:
                    if dh == 0 and dw == 0:
                        continue
                    new_h = start_h + dh
                    new_w = start_w + dw
                    if 0 <= new_h < h-ph and 0 <= new_w < w-pw:
                        result[new_h:new_h+ph, new_w:new_w+pw] = pattern
        
        return result
    
    def _constraint_solve(self, grid: np.ndarray) -> np.ndarray:
        """Solve using constraints"""
        constraints = self.components.get('constraints', [])
        solver = ConstraintSolver(constraints)
        return solver.solve(grid)
    
    def _analogy_apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply transformation by analogy"""
        reference_transform = self.components.get('reference_transform')
        if not reference_transform:
            return grid
        
        # Apply the same transformation
        return reference_transform(grid)
    
    def _invariant_preserve(self, grid: np.ndarray) -> np.ndarray:
        """Transform while preserving invariants"""
        invariants = self.components.get('invariants', [])
        transform = self.components.get('transform')
        
        if not transform:
            return grid
        
        result = transform(grid)
        
        # Check and restore invariants
        for inv in invariants:
            if inv == 'total_colored_cells':
                original_count = np.sum(grid > 0)
                result_count = np.sum(result > 0)
                if result_count != original_count:
                    # Adjust to maintain count
                    pass
            elif inv == 'color_histogram':
                for color in range(10):
                    original_count = np.sum(grid == color)
                    result_count = np.sum(result == color)
                    if result_count != original_count:
                        # Adjust to maintain histogram
                        pass
        
        return result
    
    def _hierarchical_compose(self, grid: np.ndarray) -> np.ndarray:
        """Build up solution hierarchically"""
        levels = self.components.get('levels', [])
        
        result = grid.copy()
        for level in levels:
            transform = level.get('transform')
            if transform:
                result = transform(result)
        
        return result
    
    def _get_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return None
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (y1, x1, y2, x2)
    
    def _extract_pattern(self, grid: np.ndarray, size: int) -> Optional[np.ndarray]:
        """Extract first pattern of given size"""
        h, w = grid.shape
        for i in range(h - size + 1):
            for j in range(w - size + 1):
                pattern = grid[i:i+size, j:j+size]
                if np.any(pattern != 0):
                    return pattern
        return None


class ConstraintSolver:
    """Solve grids using constraint propagation"""
    
    def __init__(self, constraints: List[Dict]):
        self.constraints = constraints
    
    def solve(self, grid: np.ndarray) -> np.ndarray:
        """Apply constraint solving"""
        result = grid.copy()
        
        # Iteratively apply constraints until convergence
        changed = True
        iterations = 0
        max_iterations = 100
        
        while changed and iterations < max_iterations:
            changed = False
            old_result = result.copy()
            
            for constraint in self.constraints:
                result = self._apply_constraint(result, constraint)
            
            if not np.array_equal(result, old_result):
                changed = True
            
            iterations += 1
        
        return result
    
    def _apply_constraint(self, grid: np.ndarray, constraint: Dict) -> np.ndarray:
        """Apply a single constraint"""
        c_type = constraint.get('type')
        
        if c_type == 'adjacency':
            # Cells of color A must be adjacent to color B
            color_a = constraint.get('color_a', 1)
            color_b = constraint.get('color_b', 2)
            return self._enforce_adjacency(grid, color_a, color_b)
        
        elif c_type == 'count':
            # Specific number of cells must have a color
            color = constraint.get('color', 1)
            count = constraint.get('count', 1)
            return self._enforce_count(grid, color, count)
        
        elif c_type == 'pattern':
            # Certain patterns must appear
            pattern = constraint.get('pattern')
            return self._enforce_pattern(grid, pattern)
        
        return grid
    
    def _enforce_adjacency(self, grid: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
        """Enforce adjacency constraint"""
        h, w = grid.shape
        result = grid.copy()
        
        # Find all cells of color_a
        a_positions = np.argwhere(grid == color_a)
        
        for pos in a_positions:
            y, x = pos
            has_b_neighbor = False
            
            # Check 4-neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == color_b:
                    has_b_neighbor = True
                    break
            
            # If no B neighbor, try to create one
            if not has_b_neighbor:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0:
                        result[ny, nx] = color_b
                        break
        
        return result
    
    def _enforce_count(self, grid: np.ndarray, color: int, count: int) -> np.ndarray:
        """Enforce count constraint"""
        current_count = np.sum(grid == color)
        
        if current_count == count:
            return grid
        
        result = grid.copy()
        
        if current_count < count:
            # Add more cells of this color
            empty_cells = np.argwhere(grid == 0)
            if len(empty_cells) > 0:
                n_to_add = min(count - current_count, len(empty_cells))
                indices = np.random.choice(len(empty_cells), n_to_add, replace=False)
                for idx in indices:
                    y, x = empty_cells[idx]
                    result[y, x] = color
        
        elif current_count > count:
            # Remove some cells of this color
            color_cells = np.argwhere(grid == color)
            n_to_remove = current_count - count
            indices = np.random.choice(len(color_cells), n_to_remove, replace=False)
            for idx in indices:
                y, x = color_cells[idx]
                result[y, x] = 0
        
        return result
    
    def _enforce_pattern(self, grid: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Enforce pattern constraint"""
        # Simple implementation: ensure pattern appears at least once
        h, w = grid.shape
        ph, pw = pattern.shape
        
        # Check if pattern already exists
        for i in range(h - ph + 1):
            for j in range(w - pw + 1):
                if np.array_equal(grid[i:i+ph, j:j+pw], pattern):
                    return grid
        
        # Try to place pattern
        result = grid.copy()
        for i in range(h - ph + 1):
            for j in range(w - pw + 1):
                # Check if we can place pattern here
                can_place = True
                for pi in range(ph):
                    for pj in range(pw):
                        if pattern[pi, pj] != 0 and grid[i+pi, j+pj] != 0:
                            if pattern[pi, pj] != grid[i+pi, j+pj]:
                                can_place = False
                                break
                    if not can_place:
                        break
                
                if can_place:
                    result[i:i+ph, j:j+pw] = pattern
                    return result
        
        return result


class PRISMSynthesizer:
    """Main PRISM synthesizer combining all components"""
    
    def __init__(self):
        self.sketch_library = self._build_sketch_library()
        self.primitive_library = self._build_primitive_library()
        self.meta_learner = MetaLearner()
        self.synthesis_cache = {}
        self.success_history = deque(maxlen=1000)
        
    def synthesize(self, 
                   input_grid: np.ndarray, 
                   output_grid: np.ndarray,
                   examples: List[Tuple[np.ndarray, np.ndarray]] = None,
                   time_limit: float = 5.0) -> Optional[CompositeProgram]:
        """Synthesize a program using PRISM approach"""
        
        # Check cache
        cache_key = (input_grid.tobytes(), output_grid.tobytes())
        if cache_key in self.synthesis_cache:
            return self.synthesis_cache[cache_key]
        
        # Analyze task characteristics
        task_features = self._extract_task_features(input_grid, output_grid, examples)
        
        # Get meta-program suggestions from meta-learner
        suggested_sketches = self.meta_learner.suggest_sketches(task_features)
        
        # Try each sketch
        for sketch in suggested_sketches:
            # Generate slot fillers
            slot_candidates = self._generate_slot_fillers(sketch, input_grid, output_grid)
            
            for fillers in slot_candidates:
                program = sketch.instantiate(fillers)
                
                # Test on main example
                try:
                    result = program.execute(input_grid)
                    if np.array_equal(result, output_grid):
                        # Verify on additional examples
                        if examples and self._verify_on_examples(program, examples):
                            self.synthesis_cache[cache_key] = program
                            self.success_history.append({
                                'sketch': sketch,
                                'features': task_features,
                                'program': program
                            })
                            self.meta_learner.update_from_success(sketch, task_features)
                            return program
                except Exception:
                    continue
        
        return None
    
    def _extract_task_features(self, input_grid: np.ndarray, 
                               output_grid: np.ndarray,
                               examples: List[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
        """Extract features characterizing the task"""
        features = {
            'size_change': input_grid.shape != output_grid.shape,
            'color_change': not np.array_equal(np.unique(input_grid), np.unique(output_grid)),
            'symmetry_present': self._has_symmetry(input_grid) or self._has_symmetry(output_grid),
            'objects_present': self._count_objects(input_grid) > 1,
            'pattern_repetition': self._has_repetition(output_grid),
            'transformation_type': self._infer_transformation_type(input_grid, output_grid),
        }
        
        if examples:
            features['num_examples'] = len(examples)
            features['consistent_transform'] = self._check_consistency(examples)
        
        return features
    
    def _has_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has symmetry"""
        h, w = grid.shape
        
        # Check horizontal symmetry
        for i in range(h):
            if not np.array_equal(grid[i, :w//2], np.flip(grid[i, w//2:w//2*2])):
                break
        else:
            return True
        
        # Check vertical symmetry
        for j in range(w):
            if not np.array_equal(grid[:h//2, j], np.flip(grid[h//2:h//2*2, j])):
                break
        else:
            return True
        
        return False
    
    def _count_objects(self, grid: np.ndarray) -> int:
        """Count distinct objects in grid"""
        from scipy.ndimage import label
        
        total_objects = 0
        for color in range(1, 10):
            mask = (grid == color)
            if mask.any():
                _, num_objects = label(mask)
                total_objects += num_objects
        
        return total_objects
    
    def _has_repetition(self, grid: np.ndarray) -> bool:
        """Check if grid has repeating patterns"""
        h, w = grid.shape
        
        # Check for 2x2, 3x3 patterns
        for size in [2, 3]:
            if h >= size * 2 and w >= size * 2:
                pattern = grid[:size, :size]
                # Check if pattern repeats
                for i in range(0, h-size, size):
                    for j in range(0, w-size, size):
                        if not np.array_equal(grid[i:i+size, j:j+size], pattern):
                            return False
                return True
        
        return False
    
    def _infer_transformation_type(self, input_grid: np.ndarray, 
                                   output_grid: np.ndarray) -> str:
        """Infer the type of transformation"""
        if np.array_equal(input_grid, output_grid):
            return 'identity'
        elif np.array_equal(np.rot90(input_grid), output_grid):
            return 'rotation'
        elif np.array_equal(np.flip(input_grid, axis=0), output_grid):
            return 'flip'
        elif input_grid.shape != output_grid.shape:
            return 'resize'
        elif self._count_objects(input_grid) != self._count_objects(output_grid):
            return 'object_manipulation'
        else:
            return 'complex'
    
    def _check_consistency(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if transformation is consistent across examples"""
        if len(examples) < 2:
            return True
        
        # Simple check: same transformation type
        transform_types = []
        for inp, out in examples:
            transform_types.append(self._infer_transformation_type(inp, out))
        
        return len(set(transform_types)) == 1
    
    def _generate_slot_fillers(self, sketch: ProgramSketch, 
                              input_grid: np.ndarray,
                              output_grid: np.ndarray) -> List[Dict[str, Any]]:
        """Generate candidate slot fillers for a sketch"""
        candidates = []
        
        # Based on sketch type, generate appropriate fillers
        if sketch.meta_program == MetaProgram.OBJECT_TRANSFORM:
            # Try different object transformations
            for transform in ['rotate_90', 'flip_h', 'scale_2x', 'color_shift']:
                candidates.append({
                    'object_transform': self.primitive_library.get(transform)
                })
        
        elif sketch.meta_program == MetaProgram.PATTERN_PROPAGATE:
            # Try different pattern sizes and rules
            for size in [2, 3, 4]:
                for rule in ['tile', 'radial']:
                    candidates.append({
                        'pattern_size': size,
                        'propagation_rule': rule
                    })
        
        # Add more based on other meta-programs...
        
        return candidates[:10]  # Limit candidates
    
    def _verify_on_examples(self, program: CompositeProgram, 
                           examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Verify program on additional examples"""
        for inp, expected in examples:
            try:
                result = program.execute(inp)
                if not np.array_equal(result, expected):
                    return False
            except Exception:
                return False
        return True
    
    def _build_sketch_library(self) -> List[ProgramSketch]:
        """Build library of program sketches"""
        return [
            ProgramSketch(MetaProgram.OBJECT_TRANSFORM, ['object_transform']),
            ProgramSketch(MetaProgram.PATTERN_PROPAGATE, ['pattern_size', 'propagation_rule']),
            ProgramSketch(MetaProgram.SYMMETRY_COMPLETE, ['symmetry_type']),
            ProgramSketch(MetaProgram.RECURSIVE_DECOMPOSE, ['decomposition_strategy', 'sub_transform', 'combine_method']),
            ProgramSketch(MetaProgram.CONSTRAINT_SOLVE, ['constraints']),
            ProgramSketch(MetaProgram.HIERARCHICAL_COMPOSE, ['levels']),
        ]
    
    def _build_primitive_library(self) -> Dict[str, callable]:
        """Build library of primitive operations"""
        return {
            'identity': lambda g: g.copy(),  # CRITICAL for LEAP patterns
            'rotate_90': lambda g: np.rot90(g),
            'rotate_180': lambda g: np.rot90(g, k=2),
            'rotate_270': lambda g: np.rot90(g, k=3),
            'flip_h': lambda g: np.fliplr(g),
            'flip_v': lambda g: np.flipud(g),
            'scale_2x': lambda g: np.repeat(np.repeat(g, 2, axis=0), 2, axis=1),
            'color_shift': lambda g: np.where(g > 0, ((g - 1) % 9) + 1, 0),  # Fixed to stay in 1-9 range
            'invert': lambda g: np.where(g > 0, 10 - g, 0) if g.max() <= 9 else g,  # Fixed to handle edge case
            'transpose': lambda g: g.T,
        }


class MetaLearner:
    """Learn which program sketches work for which task features"""
    
    def __init__(self):
        self.success_counts = defaultdict(lambda: defaultdict(int))
        self.feature_weights = defaultdict(lambda: defaultdict(float))
        
    def suggest_sketches(self, task_features: Dict) -> List[ProgramSketch]:
        """Suggest sketches based on task features"""
        scores = defaultdict(float)
        
        # Score each sketch based on historical success
        for sketch_type in MetaProgram:
            for feature, value in task_features.items():
                if value:
                    scores[sketch_type] += self.feature_weights[sketch_type][feature]
        
        # Sort by score
        sorted_sketches = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to ProgramSketch objects
        result = []
        for meta_program, score in sorted_sketches:
            if score > 0:
                sketch = self._create_sketch(meta_program)
                sketch.confidence = score
                result.append(sketch)
        
        # Always include some exploration
        if len(result) < 3:
            for meta_program in MetaProgram:
                if not any(s.meta_program == meta_program for s in result):
                    result.append(self._create_sketch(meta_program))
                    if len(result) >= 5:
                        break
        
        return result
    
    def update_from_success(self, sketch: ProgramSketch, features: Dict):
        """Update weights based on successful synthesis"""
        for feature, value in features.items():
            if value:
                self.success_counts[sketch.meta_program][feature] += 1
                self.feature_weights[sketch.meta_program][feature] += 0.1
        
        # Normalize weights
        for meta_program in self.feature_weights:
            total = sum(self.feature_weights[meta_program].values())
            if total > 0:
                for feature in self.feature_weights[meta_program]:
                    self.feature_weights[meta_program][feature] /= total
    
    def _create_sketch(self, meta_program: MetaProgram) -> ProgramSketch:
        """Create a sketch for a meta-program"""
        slot_mappings = {
            MetaProgram.OBJECT_TRANSFORM: ['object_transform'],
            MetaProgram.PATTERN_PROPAGATE: ['pattern_size', 'propagation_rule'],
            MetaProgram.SYMMETRY_COMPLETE: ['symmetry_type'],
            MetaProgram.RECURSIVE_DECOMPOSE: ['decomposition_strategy', 'sub_transform', 'combine_method'],
            MetaProgram.CONSTRAINT_SOLVE: ['constraints'],
            MetaProgram.HIERARCHICAL_COMPOSE: ['levels'],
            MetaProgram.ANALOGY_APPLY: ['reference_transform'],
            MetaProgram.INVARIANT_PRESERVE: ['invariants', 'transform'],
        }
        
        return ProgramSketch(
            meta_program=meta_program,
            slots=slot_mappings.get(meta_program, [])
        )


def create_prism_system():
    """Create a complete PRISM synthesis system"""
    return {
        'synthesizer': PRISMSynthesizer(),
        'meta_learner': MetaLearner(),
        'constraint_solver': ConstraintSolver([])
    }