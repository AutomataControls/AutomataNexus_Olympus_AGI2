"""
Integration layer between OLYMPUS models and Program Synthesis
Lightweight implementation to work alongside active training
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class SynthesizedProgram:
    """A synthesized program with metadata"""
    operations: List[str]
    parameters: List[Dict]
    confidence: float
    verified: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'operations': self.operations,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'verified': self.verified
        }


class LightweightProgramSynthesizer:
    """Lightweight synthesizer that can run during training"""
    
    def __init__(self):
        self.primitive_library = self._build_primitive_library()
        self.synthesis_cache = {}  # Cache successful programs
        
    def _build_primitive_library(self) -> Dict:
        """Build library of fast primitive operations"""
        return {
            # Simple geometric transforms (very fast)
            'rotate_90': lambda g: np.rot90(g, k=1),
            'rotate_180': lambda g: np.rot90(g, k=2),
            'rotate_270': lambda g: np.rot90(g, k=3),
            'flip_h': lambda g: np.fliplr(g),
            'flip_v': lambda g: np.flipud(g),
            'transpose': lambda g: g.T,
            
            # Color operations (fast)
            'invert_colors': lambda g: self._invert_colors(g),
            'extract_non_zero': lambda g: np.where(g > 0, g, 0),
            'binarize': lambda g: np.where(g > 0, 1, 0),
            
            # Pattern operations (medium complexity)
            'fill_background': lambda g: self._fill_background(g),
            'extract_largest': lambda g: self._extract_largest_component(g),
            'symmetrize_h': lambda g: self._make_symmetric_h(g),
            'symmetrize_v': lambda g: self._make_symmetric_v(g),
        }
    
    def quick_synthesize(self, 
                        input_grid: np.ndarray, 
                        output_grid: np.ndarray,
                        max_depth: int = 3) -> Optional[SynthesizedProgram]:
        """Quick synthesis for common patterns"""
        
        # Check cache first
        cache_key = (input_grid.tobytes(), output_grid.tobytes())
        if cache_key in self.synthesis_cache:
            return self.synthesis_cache[cache_key]
        
        # Try single operations first
        for op_name, op_func in self.primitive_library.items():
            try:
                result = op_func(input_grid)
                if np.array_equal(result, output_grid):
                    program = SynthesizedProgram(
                        operations=[op_name],
                        parameters=[{}],
                        confidence=1.0,
                        verified=True
                    )
                    self.synthesis_cache[cache_key] = program
                    return program
            except:
                continue
        
        # Try two-operation sequences
        if max_depth >= 2:
            for op1_name, op1_func in self.primitive_library.items():
                try:
                    intermediate = op1_func(input_grid)
                    for op2_name, op2_func in self.primitive_library.items():
                        try:
                            result = op2_func(intermediate)
                            if np.array_equal(result, output_grid):
                                program = SynthesizedProgram(
                                    operations=[op1_name, op2_name],
                                    parameters=[{}, {}],
                                    confidence=0.9,
                                    verified=True
                                )
                                self.synthesis_cache[cache_key] = program
                                return program
                        except:
                            continue
                except:
                    continue
        
        return None
    
    def _invert_colors(self, grid: np.ndarray) -> np.ndarray:
        """Invert all non-zero colors"""
        result = grid.copy()
        unique_colors = np.unique(grid)
        color_map = {c: 9-c if c > 0 else 0 for c in unique_colors}
        for old, new in color_map.items():
            result[grid == old] = new
        return result
    
    def _fill_background(self, grid: np.ndarray) -> np.ndarray:
        """Fill background (0) with most common non-zero color"""
        if not (grid == 0).any():
            return grid
        non_zero = grid[grid > 0]
        if len(non_zero) == 0:
            return grid
        most_common = np.bincount(non_zero).argmax()
        result = grid.copy()
        result[grid == 0] = most_common
        return result
    
    def _extract_largest_component(self, grid: np.ndarray) -> np.ndarray:
        """Extract largest connected component"""
        # Simplified version for efficiency
        result = np.zeros_like(grid)
        colors, counts = np.unique(grid[grid > 0], return_counts=True)
        if len(colors) > 0:
            largest_color = colors[np.argmax(counts)]
            result[grid == largest_color] = largest_color
        return result
    
    def _make_symmetric_h(self, grid: np.ndarray) -> np.ndarray:
        """Make grid horizontally symmetric"""
        h, w = grid.shape
        result = grid.copy()
        for i in range(h):
            for j in range(w // 2):
                result[i, w-1-j] = result[i, j]
        return result
    
    def _make_symmetric_v(self, grid: np.ndarray) -> np.ndarray:
        """Make grid vertically symmetric"""
        h, w = grid.shape
        result = grid.copy()
        for i in range(h // 2):
            for j in range(w):
                result[h-1-i, j] = result[i, j]
        return result


class HybridARCSolver:
    """Combines neural predictions with program synthesis"""
    
    def __init__(self, models: Dict[str, torch.nn.Module]):
        self.models = models
        self.synthesizer = LightweightProgramSynthesizer()
        self.synthesis_stats = {
            'attempts': 0,
            'successes': 0,
            'cache_hits': 0
        }
    
    def solve_with_synthesis(self, 
                           task_input: np.ndarray,
                           training_examples: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Solve using both neural models and program synthesis"""
        
        # First, try to synthesize from training examples
        if training_examples:
            self.synthesis_stats['attempts'] += 1
            
            # Try to find a program that works for all examples
            candidate_program = None
            for train_in, train_out in training_examples[:3]:  # Limit for speed
                program = self.synthesizer.quick_synthesize(train_in, train_out)
                if program:
                    # Verify on other examples
                    if self._verify_program(program, training_examples):
                        candidate_program = program
                        self.synthesis_stats['successes'] += 1
                        break
            
            if candidate_program:
                # Apply the program to test input
                result = task_input.copy()
                for op in candidate_program.operations:
                    result = self.synthesizer.primitive_library[op](result)
                return result
        
        # Fall back to neural prediction
        return self._neural_prediction(task_input)
    
    def _verify_program(self, 
                       program: SynthesizedProgram, 
                       examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Verify program on all examples"""
        for in_grid, out_grid in examples:
            result = in_grid.copy()
            try:
                for op in program.operations:
                    result = self.synthesizer.primitive_library[op](result)
                if not np.array_equal(result, out_grid):
                    return False
            except:
                return False
        return True
    
    def _neural_prediction(self, input_grid: np.ndarray) -> np.ndarray:
        """Standard neural model prediction"""
        # Placeholder - would use actual OLYMPUS models
        return input_grid
    
    def get_synthesis_report(self) -> Dict:
        """Get synthesis statistics"""
        success_rate = (self.synthesis_stats['successes'] / 
                       max(1, self.synthesis_stats['attempts'])) * 100
        return {
            'total_attempts': self.synthesis_stats['attempts'],
            'successful_syntheses': self.synthesis_stats['successes'],
            'success_rate': f"{success_rate:.1f}%",
            'cache_size': len(self.synthesizer.synthesis_cache),
            'available_primitives': len(self.synthesizer.primitive_library)
        }


class ProgramSynthesisDataGenerator:
    """Generate training data with known programs"""
    
    def __init__(self):
        self.synthesizer = LightweightProgramSynthesizer()
        
    def generate_programmatic_examples(self, num_examples: int = 100) -> List[Dict]:
        """Generate examples with known program solutions"""
        examples = []
        
        # Generate base patterns
        base_patterns = [
            # Checkerboard
            np.array([[i%2 if j%2==0 else (i+1)%2 for j in range(5)] 
                     for i in range(5)]) * 3,
            # Diagonal
            np.array([[1 if i==j else 0 for j in range(5)] 
                     for i in range(5)]),
            # Cross
            np.array([[1 if i==2 or j==2 else 0 for j in range(5)] 
                     for i in range(5)]),
            # L-shape
            np.array([[1 if (i==4 or j==0) and not (i==0) else 0 
                     for j in range(5)] for i in range(5)]),
            # Square
            np.array([[2 if 1<=i<=3 and 1<=j<=3 else 0 
                     for j in range(5)] for i in range(5)]),
        ]
        
        # Apply transformations
        for pattern in base_patterns:
            for op_name, op_func in self.synthesizer.primitive_library.items():
                try:
                    output = op_func(pattern)
                    if output.shape == pattern.shape:
                        examples.append({
                            'input': pattern,
                            'output': output,
                            'program': [op_name],
                            'difficulty': 'easy'
                        })
                except:
                    continue
        
        # Two-step transformations
        for pattern in base_patterns[:3]:  # Limit for speed
            ops = list(self.synthesizer.primitive_library.items())
            for i, (op1_name, op1_func) in enumerate(ops):
                for op2_name, op2_func in ops[i:i+3]:  # Local combinations
                    try:
                        intermediate = op1_func(pattern)
                        output = op2_func(intermediate)
                        if output.shape == pattern.shape:
                            examples.append({
                                'input': pattern,
                                'output': output,
                                'program': [op1_name, op2_name],
                                'difficulty': 'medium'
                            })
                    except:
                        continue
        
        return examples[:num_examples]