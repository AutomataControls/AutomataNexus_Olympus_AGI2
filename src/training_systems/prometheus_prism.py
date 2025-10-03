"""
PROMETHEUS-specific PRISM (Program Synthesis and Inductive Reasoning Module) System
Specialized for meta-learning and ensemble coordination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import defaultdict, deque

# Try to import base PRISM system
try:
    from .prism_system import PRISMSynthesizer, ProgramLibrary
    BASE_PRISM_AVAILABLE = True
except ImportError:
    BASE_PRISM_AVAILABLE = False


class PrometheusMetaProgramSynthesizer:
    """PROMETHEUS-specific program synthesizer for meta-learning"""
    
    def __init__(self, max_program_length: int = 20, vocab_size: int = 50):
        self.max_program_length = max_program_length
        self.vocab_size = vocab_size
        
        # Meta-program vocabulary (operations that work across different domains)
        self.meta_operations = {
            0: 'NOP',           # No operation
            1: 'COPY',          # Copy input to output
            2: 'ROTATE_90',     # Rotate 90 degrees
            3: 'ROTATE_180',    # Rotate 180 degrees  
            4: 'ROTATE_270',    # Rotate 270 degrees
            5: 'FLIP_H',        # Horizontal flip
            6: 'FLIP_V',        # Vertical flip
            7: 'TRANSPOSE',     # Matrix transpose
            8: 'SHIFT_UP',      # Shift up
            9: 'SHIFT_DOWN',    # Shift down
            10: 'SHIFT_LEFT',   # Shift left
            11: 'SHIFT_RIGHT',  # Shift right
            12: 'COLOR_MAP',    # Apply color mapping
            13: 'SCALE_2X',     # Scale by 2x
            14: 'SCALE_HALF',   # Scale by 0.5x
            15: 'EXTRACT_OBJ',  # Extract objects
            16: 'MERGE_OBJ',    # Merge objects
            17: 'FILL_HOLES',   # Fill holes in objects
            18: 'OUTLINE',      # Create outline
            19: 'PATTERN_REP',  # Repeat pattern
            20: 'SYMMETRY',     # Apply symmetry
            21: 'CONNECT',      # Connect similar objects
            22: 'SEPARATE',     # Separate touching objects
            23: 'COUNT_OBJ',    # Count objects
            24: 'SIZE_FILTER',  # Filter by size
            25: 'COLOR_FILTER', # Filter by color
            26: 'IF_THEN',      # Conditional operation
            27: 'FOR_EACH',     # Iterate over objects
            28: 'WHILE_LOOP',   # While loop
            29: 'RECURSIVE',    # Recursive operation
            # Meta-learning specific operations
            30: 'ADAPT',        # Adaptation operation
            31: 'GENERALIZE',   # Generalization operation
            32: 'SPECIALIZE',   # Specialization operation
            33: 'COMPOSE',      # Compose transformations
            34: 'DECOMPOSE',    # Decompose complex operations
            35: 'ANALOGIZE',    # Apply analogy
            36: 'ABSTRACT',     # Create abstraction
            37: 'INSTANTIATE',  # Instantiate template
            38: 'INTERPOLATE',  # Interpolate between examples
            39: 'EXTRAPOLATE',  # Extrapolate pattern
            # Ensemble coordination operations
            40: 'ENSEMBLE_AVG', # Average ensemble predictions
            41: 'ENSEMBLE_VOTE',# Voting among ensemble
            42: 'ENSEMBLE_WEIGHT', # Weighted ensemble
            43: 'MODEL_SELECT', # Select best model
            44: 'CONFLICT_RES', # Resolve conflicts
            45: 'CONSENSUS',    # Find consensus
            46: 'DIVERSE_GEN',  # Generate diverse solutions
            47: 'QUALITY_CTRL', # Quality control
            48: 'UNCERTAINTY',  # Handle uncertainty
            49: 'META_LEARN'    # Meta-learning operation
        }
        
        self.reverse_vocab = {v: k for k, v in self.meta_operations.items()}
        
        # Program synthesis network
        self.synthesis_network = nn.Sequential(
            nn.Linear(256, 512),  # Input feature dimension
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_program_length * self.vocab_size)
        )
        
        # Program execution engine
        self.execution_engine = PrometheusMetaProgramExecutor()
        
    def synthesize_meta_program(self, input_examples: List[torch.Tensor], 
                               output_examples: List[torch.Tensor],
                               context: Dict = None) -> Dict:
        """Synthesize meta-program from input-output examples"""
        
        # Extract features from examples
        features = self._extract_features(input_examples, output_examples, context)
        
        # Generate program logits
        program_logits = self.synthesis_network(features)
        program_logits = program_logits.view(-1, self.max_program_length, self.vocab_size)
        
        # Sample programs (beam search or greedy)
        programs = self._sample_programs(program_logits, num_samples=5)
        
        # Evaluate programs on examples
        best_program, best_score = self._evaluate_programs(programs, input_examples, output_examples)
        
        return {
            'program': best_program,
            'score': best_score,
            'program_tokens': [self.meta_operations[op] for op in best_program],
            'meta_learning_context': context
        }
    
    def _extract_features(self, inputs: List[torch.Tensor], outputs: List[torch.Tensor],
                         context: Dict = None) -> torch.Tensor:
        """Extract features from input-output examples"""
        
        features = []
        
        for inp, out in zip(inputs, outputs):
            # Basic statistical features
            input_stats = torch.tensor([
                inp.float().mean(), inp.float().std(),
                inp.max().float(), inp.min().float(),
                (inp == 0).float().mean(),  # Background ratio
                inp.unique().numel()        # Number of colors
            ])
            
            output_stats = torch.tensor([
                out.float().mean(), out.float().std(),
                out.max().float(), out.min().float(),
                (out == 0).float().mean(),
                out.unique().numel()
            ])
            
            # Transformation features
            transform_features = torch.tensor([
                float(torch.equal(inp, out)),           # Identity
                float(torch.equal(inp, torch.rot90(out, k=1))), # Rotation
                float(torch.equal(inp, torch.flip(out, dims=[0]))), # Flip
                float((inp.shape[0] != out.shape[0]) or (inp.shape[1] != out.shape[1])), # Size change
            ])
            
            example_features = torch.cat([input_stats, output_stats, transform_features])
            features.append(example_features)
        
        # Aggregate features across examples
        if features:
            aggregated = torch.stack(features).mean(dim=0)
            
            # Add meta-learning context features if available
            if context:
                meta_features = torch.tensor([
                    context.get('difficulty', 0.5),
                    context.get('num_examples', len(inputs)) / 10.0,
                    float(context.get('requires_adaptation', False)),
                    float(context.get('requires_ensemble', False))
                ])
                aggregated = torch.cat([aggregated, meta_features])
            
            # Pad to expected feature dimension
            if aggregated.size(0) < 256:
                padding = torch.zeros(256 - aggregated.size(0))
                aggregated = torch.cat([aggregated, padding])
            
            return aggregated[:256]  # Truncate if too long
        else:
            return torch.zeros(256)
    
    def _sample_programs(self, program_logits: torch.Tensor, num_samples: int = 5) -> List[List[int]]:
        """Sample programs from logits"""
        programs = []
        
        for _ in range(num_samples):
            program = []
            for step in range(self.max_program_length):
                # Sample operation
                probs = F.softmax(program_logits[0, step], dim=0)
                op = torch.multinomial(probs, 1).item()
                program.append(op)
                
                # Early stopping for NOP
                if op == 0 and len(program) > 2:  # Allow some operations before stopping
                    break
            
            programs.append(program)
        
        return programs
    
    def _evaluate_programs(self, programs: List[List[int]], 
                          inputs: List[torch.Tensor], 
                          outputs: List[torch.Tensor]) -> Tuple[List[int], float]:
        """Evaluate programs and return best one"""
        
        best_program = programs[0] if programs else [0]
        best_score = 0.0
        
        for program in programs:
            total_score = 0.0
            valid_executions = 0
            
            for inp, expected_out in zip(inputs, outputs):
                try:
                    actual_out = self.execution_engine.execute(program, inp)
                    if actual_out is not None:
                        score = self._compute_similarity(actual_out, expected_out)
                        total_score += score
                        valid_executions += 1
                except Exception:
                    continue  # Skip failed executions
            
            if valid_executions > 0:
                avg_score = total_score / valid_executions
                if avg_score > best_score:
                    best_score = avg_score
                    best_program = program
        
        return best_program, best_score
    
    def _compute_similarity(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Compute similarity between predicted and target grids"""
        
        if predicted.shape != target.shape:
            return 0.0
        
        exact_match = torch.equal(predicted, target)
        if exact_match:
            return 1.0
        
        # Pixel-wise accuracy
        pixel_accuracy = (predicted == target).float().mean().item()
        return pixel_accuracy


class PrometheusMetaProgramExecutor:
    """Execute meta-programs for PROMETHEUS"""
    
    def __init__(self):
        self.max_iterations = 100  # Prevent infinite loops
        
    def execute(self, program: List[int], input_grid: torch.Tensor) -> Optional[torch.Tensor]:
        """Execute program on input grid"""
        
        try:
            current_grid = input_grid.clone()
            iteration = 0
            
            for op_id in program:
                if iteration >= self.max_iterations:
                    break
                
                current_grid = self._execute_operation(op_id, current_grid)
                if current_grid is None:
                    return None
                
                iteration += 1
            
            return current_grid
            
        except Exception:
            return None
    
    def _execute_operation(self, op_id: int, grid: torch.Tensor) -> Optional[torch.Tensor]:
        """Execute single operation"""
        
        if op_id == 0:  # NOP
            return grid
        elif op_id == 1:  # COPY
            return grid.clone()
        elif op_id == 2:  # ROTATE_90
            return torch.rot90(grid, k=1)
        elif op_id == 3:  # ROTATE_180
            return torch.rot90(grid, k=2)
        elif op_id == 4:  # ROTATE_270
            return torch.rot90(grid, k=3)
        elif op_id == 5:  # FLIP_H
            return torch.flip(grid, dims=[1])
        elif op_id == 6:  # FLIP_V
            return torch.flip(grid, dims=[0])
        elif op_id == 7:  # TRANSPOSE
            return grid.t()
        elif op_id == 8:  # SHIFT_UP
            return torch.roll(grid, -1, dims=0)
        elif op_id == 9:  # SHIFT_DOWN
            return torch.roll(grid, 1, dims=0)
        elif op_id == 10:  # SHIFT_LEFT
            return torch.roll(grid, -1, dims=1)
        elif op_id == 11:  # SHIFT_RIGHT
            return torch.roll(grid, 1, dims=1)
        elif op_id == 12:  # COLOR_MAP
            return self._apply_color_mapping(grid)
        elif op_id == 30:  # ADAPT (meta-learning specific)
            return self._adaptive_transform(grid)
        elif op_id == 36:  # ABSTRACT
            return self._create_abstraction(grid)
        elif op_id == 40:  # ENSEMBLE_AVG (placeholder)
            return grid  # Would coordinate with ensemble in real implementation
        else:
            # Default to identity for unimplemented operations
            return grid
    
    def _apply_color_mapping(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply simple color mapping"""
        # Cycle colors: 0->0, 1->2, 2->3, 3->1, 4->5, 5->4, etc.
        mapped = grid.clone()
        unique_colors = grid.unique()
        
        for color in unique_colors:
            if color == 0:
                continue  # Keep background
            new_color = (color % 5) + 1 if color > 0 else 0
            mapped[grid == color] = new_color
        
        return mapped
    
    def _adaptive_transform(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply adaptive transformation based on grid properties"""
        
        # Analyze grid properties
        height, width = grid.shape
        num_colors = grid.unique().numel()
        
        # Adaptive strategy based on properties
        if num_colors <= 2:
            # Binary grid - apply pattern repetition
            return self._repeat_pattern(grid)
        elif height != width:
            # Rectangular grid - try to make square
            min_size = min(height, width)
            return grid[:min_size, :min_size]
        else:
            # Square grid with many colors - apply rotation
            return torch.rot90(grid)
    
    def _repeat_pattern(self, grid: torch.Tensor) -> torch.Tensor:
        """Repeat pattern in grid"""
        h, w = grid.shape
        if h >= 4 and w >= 4:
            # Extract top-left quarter and repeat
            quarter = grid[:h//2, :w//2]
            repeated = quarter.repeat(2, 2)[:h, :w]
            return repeated
        return grid
    
    def _create_abstraction(self, grid: torch.Tensor) -> torch.Tensor:
        """Create abstraction of grid"""
        # Simple abstraction: reduce to binary based on most common color
        most_common = torch.bincount(grid.flatten()).argmax()
        abstracted = (grid == most_common).long()
        return abstracted


class PrometheusMetaProgramLibrary:
    """PROMETHEUS-specific program library for meta-learning"""
    
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.programs = deque(maxlen=capacity)
        self.meta_programs = deque(maxlen=capacity // 4)
        self.ensemble_programs = deque(maxlen=capacity // 4)
        self.program_scores = {}
        
    def add_program(self, program: List[int], score: float, context: Dict = None):
        """Add program to library"""
        
        program_key = tuple(program)
        self.program_scores[program_key] = score
        
        program_entry = {
            'program': program,
            'score': score,
            'context': context or {},
            'usage_count': 0,
            'success_rate': score
        }
        
        self.programs.append(program_entry)
        
        # Add to specialized collections
        if context and context.get('requires_meta_learning'):
            self.meta_programs.append(program_entry)
        if context and context.get('requires_ensemble'):
            self.ensemble_programs.append(program_entry)
    
    def get_best_programs(self, k: int = 10, context: Dict = None) -> List[Dict]:
        """Get best programs, optionally filtered by context"""
        
        candidates = list(self.programs)
        
        # Filter by context if provided
        if context:
            if context.get('meta_learning_focus'):
                candidates = list(self.meta_programs)
            elif context.get('ensemble_focus'):
                candidates = list(self.ensemble_programs)
        
        # Sort by score and return top k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:k]
    
    def update_program_stats(self, program: List[int], success: bool):
        """Update program statistics"""
        program_key = tuple(program)
        
        for prog_entry in self.programs:
            if tuple(prog_entry['program']) == program_key:
                prog_entry['usage_count'] += 1
                # Update success rate with exponential moving average
                alpha = 0.1
                old_rate = prog_entry['success_rate']
                prog_entry['success_rate'] = alpha * float(success) + (1 - alpha) * old_rate
                break


def create_prometheus_prism_system() -> Dict:
    """Create PROMETHEUS-specific PRISM system"""
    
    synthesizer = PrometheusMetaProgramSynthesizer()
    program_library = PrometheusMetaProgramLibrary()
    
    return {
        'synthesizer': synthesizer,
        'program_bank': program_library,  # Use 'program_bank' for PROMETHEUS
        'meta_synthesis_enabled': True,
        'ensemble_coordination_enabled': True,
        'max_program_length': 20,
        'vocab_size': 50
    }