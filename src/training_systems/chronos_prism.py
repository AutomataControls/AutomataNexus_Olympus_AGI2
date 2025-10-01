"""
CHRONOS PRISM Training System
Program Synthesis for Intelligent Sequence Modeling - Temporal Program Synthesis
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Set
import random
import numpy as np
from dataclasses import dataclass
from enum import Enum
import itertools


class TemporalPrimitive(Enum):
    """Basic temporal operations"""
    SHIFT_RIGHT = "shift_right"
    SHIFT_LEFT = "shift_left"
    SHIFT_UP = "shift_up"
    SHIFT_DOWN = "shift_down"
    ROTATE_CW = "rotate_clockwise"
    ROTATE_CCW = "rotate_counterclockwise"
    GROW = "grow"
    SHRINK = "shrink"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    COPY = "copy"
    MAINTAIN = "maintain"
    OSCILLATE = "oscillate"
    ACCUMULATE = "accumulate"
    DELAY = "delay"


@dataclass
class TemporalProgram:
    """Represents a temporal transformation program"""
    instructions: List[Tuple[TemporalPrimitive, Dict[str, Any]]]
    description: str
    complexity: int
    pattern_type: str
    expected_length: int
    
    
@dataclass
class ProgramExecutionResult:
    """Result of program execution"""
    sequence: List[torch.Tensor]
    success: bool
    error_message: Optional[str] = None
    execution_trace: List[Dict] = None


class ChronosPRISM:
    """PRISM system specialized for CHRONOS temporal program synthesis"""
    
    def __init__(self, grid_size: int = 12, max_sequence_length: int = 10):
        self.grid_size = grid_size
        self.max_sequence_length = max_sequence_length
        self.primitive_implementations = self._init_primitive_implementations()
        self.program_templates = self._init_program_templates()
        self.synthesis_strategies = self._init_synthesis_strategies()
        
    def _init_primitive_implementations(self) -> Dict[TemporalPrimitive, callable]:
        """Initialize implementations for temporal primitives"""
        return {
            TemporalPrimitive.SHIFT_RIGHT: self._shift_right,
            TemporalPrimitive.SHIFT_LEFT: self._shift_left,
            TemporalPrimitive.SHIFT_UP: self._shift_up,
            TemporalPrimitive.SHIFT_DOWN: self._shift_down,
            TemporalPrimitive.ROTATE_CW: self._rotate_clockwise,
            TemporalPrimitive.ROTATE_CCW: self._rotate_counterclockwise,
            TemporalPrimitive.GROW: self._grow,
            TemporalPrimitive.SHRINK: self._shrink,
            TemporalPrimitive.FADE_IN: self._fade_in,
            TemporalPrimitive.FADE_OUT: self._fade_out,
            TemporalPrimitive.COPY: self._copy,
            TemporalPrimitive.MAINTAIN: self._maintain,
            TemporalPrimitive.OSCILLATE: self._oscillate,
            TemporalPrimitive.ACCUMULATE: self._accumulate,
            TemporalPrimitive.DELAY: self._delay
        }
    
    def _init_program_templates(self) -> Dict[str, List[Tuple[TemporalPrimitive, Dict]]]:
        """Initialize temporal program templates"""
        return {
            'linear_motion': [
                (TemporalPrimitive.SHIFT_RIGHT, {'amount': 1}),
                (TemporalPrimitive.MAINTAIN, {})
            ],
            'bouncing': [
                (TemporalPrimitive.SHIFT_RIGHT, {'amount': 1}),
                (TemporalPrimitive.SHIFT_DOWN, {'amount': 1}),
                (TemporalPrimitive.SHIFT_LEFT, {'amount': 1}),
                (TemporalPrimitive.SHIFT_UP, {'amount': 1})
            ],
            'rotation': [
                (TemporalPrimitive.ROTATE_CW, {'angle': 90}),
                (TemporalPrimitive.MAINTAIN, {})
            ],
            'pulsing': [
                (TemporalPrimitive.GROW, {'factor': 1.5}),
                (TemporalPrimitive.MAINTAIN, {}),
                (TemporalPrimitive.SHRINK, {'factor': 0.7}),
                (TemporalPrimitive.MAINTAIN, {})
            ],
            'fading': [
                (TemporalPrimitive.FADE_OUT, {'rate': 0.2}),
                (TemporalPrimitive.DELAY, {'steps': 2}),
                (TemporalPrimitive.FADE_IN, {'rate': 0.3})
            ],
            'accumulating': [
                (TemporalPrimitive.SHIFT_RIGHT, {'amount': 1}),
                (TemporalPrimitive.ACCUMULATE, {'decay': 0.8})
            ]
        }
    
    def _init_synthesis_strategies(self) -> Dict[str, callable]:
        """Initialize program synthesis strategies"""
        return {
            'template_based': self._template_based_synthesis,
            'search_based': self._search_based_synthesis,
            'compositional': self._compositional_synthesis,
            'example_guided': self._example_guided_synthesis,
            'constraint_based': self._constraint_based_synthesis
        }
    
    def synthesize_program(self, target_sequence: Optional[List[torch.Tensor]] = None,
                          constraints: Optional[Dict] = None,
                          strategy: str = 'template_based') -> TemporalProgram:
        """Synthesize a temporal program"""
        if strategy not in self.synthesis_strategies:
            strategy = 'template_based'
            
        synthesis_func = self.synthesis_strategies[strategy]
        return synthesis_func(target_sequence, constraints)
    
    def _template_based_synthesis(self, target: Optional[List[torch.Tensor]],
                                 constraints: Optional[Dict]) -> TemporalProgram:
        """Synthesize using predefined templates"""
        # Select template based on constraints or randomly
        if constraints and 'pattern_type' in constraints:
            pattern_type = constraints['pattern_type']
            if pattern_type in self.program_templates:
                template = self.program_templates[pattern_type]
            else:
                template = random.choice(list(self.program_templates.values()))
        else:
            template = random.choice(list(self.program_templates.values()))
            
        # Modify template parameters
        instructions = []
        for primitive, params in template:
            # Randomize parameters
            new_params = params.copy()
            if 'amount' in new_params:
                new_params['amount'] = random.randint(1, 3)
            if 'angle' in new_params:
                new_params['angle'] = random.choice([90, 180, 270])
            if 'factor' in new_params:
                new_params['factor'] = random.uniform(0.5, 2.0)
            if 'rate' in new_params:
                new_params['rate'] = random.uniform(0.1, 0.5)
                
            instructions.append((primitive, new_params))
            
        # Repeat to achieve desired length
        target_length = constraints.get('length', self.max_sequence_length) if constraints else self.max_sequence_length
        
        program = TemporalProgram(
            instructions=instructions,
            description="Template-based temporal program",
            complexity=len(set(p for p, _ in instructions)),
            pattern_type='template',
            expected_length=target_length
        )
        
        return program
    
    def _search_based_synthesis(self, target: Optional[List[torch.Tensor]],
                               constraints: Optional[Dict]) -> TemporalProgram:
        """Synthesize using search through program space"""
        max_attempts = 100
        best_program = None
        best_score = float('-inf')
        
        for _ in range(max_attempts):
            # Generate random program
            length = random.randint(2, 8)
            instructions = []
            
            for _ in range(length):
                primitive = random.choice(list(TemporalPrimitive))
                params = self._generate_random_params(primitive)
                instructions.append((primitive, params))
                
            program = TemporalProgram(
                instructions=instructions,
                description="Search-based temporal program",
                complexity=len(set(p for p, _ in instructions)),
                pattern_type='search',
                expected_length=self.max_sequence_length
            )
            
            # Evaluate program
            if target:
                score = self._evaluate_program(program, target)
                if score > best_score:
                    best_score = score
                    best_program = program
            else:
                # If no target, use diversity metric
                result = self.execute_program(program, self._create_random_initial())
                if result.success:
                    score = self._calculate_sequence_diversity(result.sequence)
                    if score > best_score:
                        best_score = score
                        best_program = program
                        
        return best_program if best_program else self._template_based_synthesis(target, constraints)
    
    def _compositional_synthesis(self, target: Optional[List[torch.Tensor]],
                                constraints: Optional[Dict]) -> TemporalProgram:
        """Synthesize by composing sub-programs"""
        # Create sub-programs
        sub_programs = []
        
        # Movement sub-program
        movement_primitives = [
            TemporalPrimitive.SHIFT_RIGHT,
            TemporalPrimitive.SHIFT_LEFT,
            TemporalPrimitive.SHIFT_UP,
            TemporalPrimitive.SHIFT_DOWN
        ]
        movement_prog = [(random.choice(movement_primitives), {'amount': 1}) 
                        for _ in range(random.randint(2, 4))]
        sub_programs.append(movement_prog)
        
        # Transformation sub-program
        transform_primitives = [
            TemporalPrimitive.GROW,
            TemporalPrimitive.SHRINK,
            TemporalPrimitive.ROTATE_CW,
            TemporalPrimitive.ROTATE_CCW
        ]
        transform_prog = [(random.choice(transform_primitives), 
                          self._generate_random_params(random.choice(transform_primitives)))
                         for _ in range(random.randint(1, 3))]
        sub_programs.append(transform_prog)
        
        # Effect sub-program
        effect_primitives = [
            TemporalPrimitive.FADE_IN,
            TemporalPrimitive.FADE_OUT,
            TemporalPrimitive.OSCILLATE,
            TemporalPrimitive.ACCUMULATE
        ]
        effect_prog = [(random.choice(effect_primitives),
                       self._generate_random_params(random.choice(effect_primitives)))
                      for _ in range(random.randint(1, 2))]
        sub_programs.append(effect_prog)
        
        # Compose sub-programs
        composition_order = random.sample(range(len(sub_programs)), len(sub_programs))
        instructions = []
        
        for idx in composition_order:
            instructions.extend(sub_programs[idx])
            
        # Add delays between compositions
        final_instructions = []
        for i, inst in enumerate(instructions):
            final_instructions.append(inst)
            if i < len(instructions) - 1 and random.random() < 0.3:
                final_instructions.append((TemporalPrimitive.DELAY, {'steps': 1}))
                
        program = TemporalProgram(
            instructions=final_instructions,
            description="Compositional temporal program",
            complexity=len(set(p for p, _ in final_instructions)),
            pattern_type='compositional',
            expected_length=self.max_sequence_length
        )
        
        return program
    
    def _example_guided_synthesis(self, target: Optional[List[torch.Tensor]],
                                 constraints: Optional[Dict]) -> TemporalProgram:
        """Synthesize by analyzing target sequence"""
        if not target or len(target) < 2:
            return self._template_based_synthesis(target, constraints)
            
        # Analyze target sequence
        instructions = []
        
        for i in range(1, len(target)):
            prev_frame = target[i-1]
            curr_frame = target[i]
            
            # Infer operation
            operation = self._infer_operation(prev_frame, curr_frame)
            if operation:
                instructions.append(operation)
            else:
                instructions.append((TemporalPrimitive.MAINTAIN, {}))
                
        # Simplify program
        simplified = self._simplify_program(instructions)
        
        program = TemporalProgram(
            instructions=simplified,
            description="Example-guided temporal program",
            complexity=len(set(p for p, _ in simplified)),
            pattern_type='example_guided',
            expected_length=len(target)
        )
        
        return program
    
    def _constraint_based_synthesis(self, target: Optional[List[torch.Tensor]],
                                   constraints: Optional[Dict]) -> TemporalProgram:
        """Synthesize based on constraints"""
        if not constraints:
            return self._template_based_synthesis(target, constraints)
            
        instructions = []
        
        # Parse constraints
        must_include = constraints.get('must_include', [])
        forbidden = constraints.get('forbidden', [])
        min_length = constraints.get('min_length', 2)
        max_length = constraints.get('max_length', 8)
        pattern_properties = constraints.get('properties', {})
        
        # Build valid primitive set
        valid_primitives = [p for p in TemporalPrimitive if p not in forbidden]
        
        # Ensure must_include primitives
        for primitive in must_include:
            if isinstance(primitive, str):
                primitive = TemporalPrimitive(primitive)
            params = self._generate_random_params(primitive)
            instructions.append((primitive, params))
            
        # Add additional primitives
        remaining_length = random.randint(
            max(0, min_length - len(instructions)),
            max_length - len(instructions)
        )
        
        for _ in range(remaining_length):
            primitive = random.choice(valid_primitives)
            params = self._generate_constrained_params(primitive, pattern_properties)
            instructions.append((primitive, params))
            
        # Shuffle if not ordered
        if not constraints.get('preserve_order', False):
            random.shuffle(instructions)
            
        program = TemporalProgram(
            instructions=instructions,
            description="Constraint-based temporal program",
            complexity=len(set(p for p, _ in instructions)),
            pattern_type='constraint_based',
            expected_length=self.max_sequence_length
        )
        
        return program
    
    def execute_program(self, program: TemporalProgram, 
                       initial_state: torch.Tensor) -> ProgramExecutionResult:
        """Execute a temporal program"""
        sequence = [initial_state.clone()]
        execution_trace = []
        current_state = initial_state.clone()
        
        try:
            # Execute instructions cyclically until reaching expected length
            instruction_idx = 0
            
            while len(sequence) < program.expected_length:
                if instruction_idx >= len(program.instructions):
                    instruction_idx = 0  # Cycle back
                    
                primitive, params = program.instructions[instruction_idx]
                
                # Execute primitive
                if primitive in self.primitive_implementations:
                    implementation = self.primitive_implementations[primitive]
                    new_state = implementation(current_state, **params)
                    
                    # Handle special primitives
                    if primitive == TemporalPrimitive.ACCUMULATE:
                        # Accumulate over history
                        new_state = self._accumulate_sequence(sequence, params.get('decay', 0.9))
                    elif primitive == TemporalPrimitive.DELAY:
                        # Repeat current state
                        for _ in range(params.get('steps', 1)):
                            sequence.append(current_state.clone())
                            if len(sequence) >= program.expected_length:
                                break
                        instruction_idx += 1
                        continue
                    elif primitive == TemporalPrimitive.OSCILLATE:
                        # Oscillate between states
                        if len(sequence) > 1:
                            new_state = sequence[-2].clone() if len(sequence) % 2 == 0 else current_state
                            
                    sequence.append(new_state)
                    current_state = new_state
                    
                    # Log execution
                    execution_trace.append({
                        'step': len(sequence) - 1,
                        'primitive': primitive.value,
                        'params': params
                    })
                else:
                    raise ValueError(f"Unknown primitive: {primitive}")
                    
                instruction_idx += 1
                
            return ProgramExecutionResult(
                sequence=sequence[:program.expected_length],
                success=True,
                execution_trace=execution_trace
            )
            
        except Exception as e:
            return ProgramExecutionResult(
                sequence=sequence,
                success=False,
                error_message=str(e),
                execution_trace=execution_trace
            )
    
    def _create_random_initial(self) -> torch.Tensor:
        """Create random initial state"""
        state = torch.zeros(10, self.grid_size, self.grid_size)
        
        # Add random object
        obj_type = random.choice(['square', 'line', 'cross', 'random'])
        color = random.randint(1, 9)
        
        if obj_type == 'square':
            size = random.randint(2, 4)
            x = random.randint(0, self.grid_size - size)
            y = random.randint(0, self.grid_size - size)
            state[color, y:y+size, x:x+size] = 1
            
        elif obj_type == 'line':
            if random.random() < 0.5:
                # Horizontal
                y = random.randint(2, self.grid_size - 3)
                state[color, y, 2:-2] = 1
            else:
                # Vertical
                x = random.randint(2, self.grid_size - 3)
                state[color, 2:-2, x] = 1
                
        elif obj_type == 'cross':
            center = self.grid_size // 2
            state[color, center, :] = 1
            state[color, :, center] = 1
            
        else:
            # Random pattern
            num_pixels = random.randint(5, 15)
            for _ in range(num_pixels):
                y = random.randint(0, self.grid_size - 1)
                x = random.randint(0, self.grid_size - 1)
                state[color, y, x] = 1
                
        return state
    
    # Primitive implementations
    def _shift_right(self, state: torch.Tensor, amount: int = 1) -> torch.Tensor:
        """Shift pattern right"""
        return torch.roll(state, shifts=amount, dims=-1)
    
    def _shift_left(self, state: torch.Tensor, amount: int = 1) -> torch.Tensor:
        """Shift pattern left"""
        return torch.roll(state, shifts=-amount, dims=-1)
    
    def _shift_up(self, state: torch.Tensor, amount: int = 1) -> torch.Tensor:
        """Shift pattern up"""
        return torch.roll(state, shifts=-amount, dims=-2)
    
    def _shift_down(self, state: torch.Tensor, amount: int = 1) -> torch.Tensor:
        """Shift pattern down"""
        return torch.roll(state, shifts=amount, dims=-2)
    
    def _rotate_clockwise(self, state: torch.Tensor, angle: int = 90) -> torch.Tensor:
        """Rotate pattern clockwise"""
        k = angle // 90
        return torch.rot90(state, k=k, dims=[-2, -1])
    
    def _rotate_counterclockwise(self, state: torch.Tensor, angle: int = 90) -> torch.Tensor:
        """Rotate pattern counterclockwise"""
        k = angle // 90
        return torch.rot90(state, k=-k, dims=[-2, -1])
    
    def _grow(self, state: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
        """Grow pattern"""
        # Simple dilation
        kernel = torch.ones(1, 1, 3, 3) / 9
        grown = F.conv2d(state.unsqueeze(0).float(), kernel, padding=1).squeeze(0)
        return (grown > 0.2).float() * state.max()
    
    def _shrink(self, state: torch.Tensor, factor: float = 0.7) -> torch.Tensor:
        """Shrink pattern"""
        # Simple erosion
        kernel = torch.ones(1, 1, 3, 3) / 9
        shrunk = F.conv2d(state.unsqueeze(0).float(), kernel, padding=1).squeeze(0)
        return (shrunk > 0.8).float() * state.max()
    
    def _fade_in(self, state: torch.Tensor, rate: float = 0.3) -> torch.Tensor:
        """Fade in effect"""
        return state * (1 + rate)
    
    def _fade_out(self, state: torch.Tensor, rate: float = 0.3) -> torch.Tensor:
        """Fade out effect"""
        return state * (1 - rate)
    
    def _copy(self, state: torch.Tensor) -> torch.Tensor:
        """Copy state"""
        return state.clone()
    
    def _maintain(self, state: torch.Tensor) -> torch.Tensor:
        """Maintain current state"""
        return state.clone()
    
    def _oscillate(self, state: torch.Tensor) -> torch.Tensor:
        """Oscillate (handled in execute_program)"""
        return state.clone()
    
    def _accumulate(self, state: torch.Tensor, decay: float = 0.9) -> torch.Tensor:
        """Accumulate (handled in execute_program)"""
        return state.clone()
    
    def _delay(self, state: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Delay (handled in execute_program)"""
        return state.clone()
    
    def _accumulate_sequence(self, sequence: List[torch.Tensor], decay: float) -> torch.Tensor:
        """Accumulate over sequence history"""
        if not sequence:
            return torch.zeros_like(sequence[0])
            
        result = torch.zeros_like(sequence[0])
        for i, frame in enumerate(reversed(sequence[-5:])):  # Last 5 frames
            weight = decay ** i
            result = torch.maximum(result, frame * weight)
            
        return result
    
    def _generate_random_params(self, primitive: TemporalPrimitive) -> Dict[str, Any]:
        """Generate random parameters for primitive"""
        params = {}
        
        if primitive in [TemporalPrimitive.SHIFT_RIGHT, TemporalPrimitive.SHIFT_LEFT,
                        TemporalPrimitive.SHIFT_UP, TemporalPrimitive.SHIFT_DOWN]:
            params['amount'] = random.randint(1, 3)
            
        elif primitive in [TemporalPrimitive.ROTATE_CW, TemporalPrimitive.ROTATE_CCW]:
            params['angle'] = random.choice([90, 180, 270])
            
        elif primitive in [TemporalPrimitive.GROW, TemporalPrimitive.SHRINK]:
            params['factor'] = random.uniform(0.5, 2.0)
            
        elif primitive in [TemporalPrimitive.FADE_IN, TemporalPrimitive.FADE_OUT]:
            params['rate'] = random.uniform(0.1, 0.5)
            
        elif primitive == TemporalPrimitive.ACCUMULATE:
            params['decay'] = random.uniform(0.7, 0.95)
            
        elif primitive == TemporalPrimitive.DELAY:
            params['steps'] = random.randint(1, 3)
            
        return params
    
    def _generate_constrained_params(self, primitive: TemporalPrimitive, 
                                   constraints: Dict) -> Dict[str, Any]:
        """Generate parameters respecting constraints"""
        params = self._generate_random_params(primitive)
        
        # Apply constraints
        if 'max_movement' in constraints:
            if 'amount' in params:
                params['amount'] = min(params['amount'], constraints['max_movement'])
                
        if 'rotation_step' in constraints:
            if 'angle' in params:
                params['angle'] = constraints['rotation_step']
                
        if 'growth_bounds' in constraints:
            if 'factor' in params:
                min_g, max_g = constraints['growth_bounds']
                params['factor'] = max(min_g, min(max_g, params['factor']))
                
        return params
    
    def _infer_operation(self, prev: torch.Tensor, curr: torch.Tensor) -> Optional[Tuple[TemporalPrimitive, Dict]]:
        """Infer operation from frame transition"""
        # Check for shifts
        for shift_amount in [1, 2, 3]:
            if torch.allclose(torch.roll(prev, shift_amount, dims=-1), curr):
                return (TemporalPrimitive.SHIFT_RIGHT, {'amount': shift_amount})
            if torch.allclose(torch.roll(prev, -shift_amount, dims=-1), curr):
                return (TemporalPrimitive.SHIFT_LEFT, {'amount': shift_amount})
            if torch.allclose(torch.roll(prev, shift_amount, dims=-2), curr):
                return (TemporalPrimitive.SHIFT_DOWN, {'amount': shift_amount})
            if torch.allclose(torch.roll(prev, -shift_amount, dims=-2), curr):
                return (TemporalPrimitive.SHIFT_UP, {'amount': shift_amount})
                
        # Check for rotations
        for k in [1, 2, 3]:
            if torch.allclose(torch.rot90(prev, k=k, dims=[-2, -1]), curr):
                return (TemporalPrimitive.ROTATE_CW, {'angle': k * 90})
            if torch.allclose(torch.rot90(prev, k=-k, dims=[-2, -1]), curr):
                return (TemporalPrimitive.ROTATE_CCW, {'angle': k * 90})
                
        # Check for scaling
        prev_size = (prev > 0).sum().item()
        curr_size = (curr > 0).sum().item()
        
        if curr_size > prev_size * 1.2:
            return (TemporalPrimitive.GROW, {'factor': curr_size / max(prev_size, 1)})
        elif curr_size < prev_size * 0.8:
            return (TemporalPrimitive.SHRINK, {'factor': curr_size / max(prev_size, 1)})
            
        # Check for fading
        if torch.allclose(prev * 0.5, curr, atol=0.1):
            return (TemporalPrimitive.FADE_OUT, {'rate': 0.5})
        elif torch.allclose(prev * 1.5, curr, atol=0.1):
            return (TemporalPrimitive.FADE_IN, {'rate': 0.5})
            
        return None
    
    def _simplify_program(self, instructions: List[Tuple[TemporalPrimitive, Dict]]) -> List[Tuple[TemporalPrimitive, Dict]]:
        """Simplify program by removing redundancies"""
        if not instructions:
            return instructions
            
        simplified = []
        prev_primitive = None
        
        for primitive, params in instructions:
            # Merge consecutive same operations
            if primitive == prev_primitive and primitive in [
                TemporalPrimitive.SHIFT_RIGHT, TemporalPrimitive.SHIFT_LEFT,
                TemporalPrimitive.SHIFT_UP, TemporalPrimitive.SHIFT_DOWN
            ]:
                # Accumulate amounts
                simplified[-1] = (primitive, {'amount': simplified[-1][1]['amount'] + params['amount']})
            else:
                simplified.append((primitive, params))
                prev_primitive = primitive
                
        return simplified
    
    def _evaluate_program(self, program: TemporalProgram, target: List[torch.Tensor]) -> float:
        """Evaluate how well program matches target"""
        # Execute program
        initial = target[0] if target else self._create_random_initial()
        result = self.execute_program(program, initial)
        
        if not result.success:
            return 0.0
            
        # Compare sequences
        score = 0.0
        min_len = min(len(result.sequence), len(target))
        
        for i in range(min_len):
            similarity = 1.0 - torch.abs(result.sequence[i] - target[i]).mean().item()
            score += similarity
            
        # Normalize
        score /= len(target)
        
        # Penalize length mismatch
        length_penalty = abs(len(result.sequence) - len(target)) / len(target)
        score *= (1 - length_penalty * 0.5)
        
        return max(0, score)
    
    def _calculate_sequence_diversity(self, sequence: List[torch.Tensor]) -> float:
        """Calculate diversity of sequence"""
        if len(sequence) < 2:
            return 0.0
            
        total_diff = 0.0
        for i in range(1, len(sequence)):
            diff = torch.abs(sequence[i] - sequence[i-1]).sum().item()
            total_diff += diff
            
        # Normalize
        max_possible_diff = 10 * self.grid_size * self.grid_size * (len(sequence) - 1)
        diversity = total_diff / max_possible_diff
        
        return min(1.0, diversity)
    
    def analyze_program(self, program: TemporalProgram) -> Dict[str, Any]:
        """Analyze program properties"""
        analysis = {
            'length': len(program.instructions),
            'unique_primitives': len(set(p for p, _ in program.instructions)),
            'complexity': program.complexity,
            'pattern_type': program.pattern_type,
            'primitive_counts': {},
            'has_loops': False,
            'has_accumulation': False,
            'dominant_operation': None
        }
        
        # Count primitives
        primitive_counts = {}
        for primitive, _ in program.instructions:
            primitive_counts[primitive.value] = primitive_counts.get(primitive.value, 0) + 1
            
        analysis['primitive_counts'] = primitive_counts
        
        # Check for loops (repeated subsequences)
        if len(program.instructions) > 3:
            for sub_len in range(2, len(program.instructions) // 2 + 1):
                for start in range(len(program.instructions) - sub_len * 2 + 1):
                    sub1 = program.instructions[start:start+sub_len]
                    sub2 = program.instructions[start+sub_len:start+sub_len*2]
                    if sub1 == sub2:
                        analysis['has_loops'] = True
                        break
                        
        # Check for accumulation
        analysis['has_accumulation'] = any(p == TemporalPrimitive.ACCUMULATE 
                                         for p, _ in program.instructions)
        
        # Find dominant operation
        if primitive_counts:
            analysis['dominant_operation'] = max(primitive_counts.items(), 
                                                key=lambda x: x[1])[0]
            
        return analysis
    
    def mutate_program(self, program: TemporalProgram, mutation_rate: float = 0.2) -> TemporalProgram:
        """Mutate a temporal program"""
        new_instructions = []
        
        for primitive, params in program.instructions:
            if random.random() < mutation_rate:
                # Choose mutation type
                mutation_type = random.choice(['replace', 'modify_params', 'delete', 'insert'])
                
                if mutation_type == 'replace':
                    # Replace with different primitive
                    new_primitive = random.choice(list(TemporalPrimitive))
                    new_params = self._generate_random_params(new_primitive)
                    new_instructions.append((new_primitive, new_params))
                    
                elif mutation_type == 'modify_params':
                    # Modify parameters
                    new_params = self._generate_random_params(primitive)
                    new_instructions.append((primitive, new_params))
                    
                elif mutation_type == 'delete':
                    # Skip this instruction
                    pass
                    
                else:  # insert
                    # Insert new instruction
                    new_instructions.append((primitive, params))
                    insert_primitive = random.choice(list(TemporalPrimitive))
                    insert_params = self._generate_random_params(insert_primitive)
                    new_instructions.append((insert_primitive, insert_params))
            else:
                new_instructions.append((primitive, params))
                
        # Ensure minimum length
        if len(new_instructions) < 2:
            for _ in range(2 - len(new_instructions)):
                primitive = random.choice(list(TemporalPrimitive))
                params = self._generate_random_params(primitive)
                new_instructions.append((primitive, params))
                
        return TemporalProgram(
            instructions=new_instructions,
            description=f"Mutated: {program.description}",
            complexity=len(set(p for p, _ in new_instructions)),
            pattern_type=program.pattern_type,
            expected_length=program.expected_length
        )
    
    def crossover_programs(self, program1: TemporalProgram, 
                          program2: TemporalProgram) -> TemporalProgram:
        """Create offspring program via crossover"""
        # Choose crossover point
        point1 = random.randint(1, len(program1.instructions) - 1)
        point2 = random.randint(1, len(program2.instructions) - 1)
        
        # Create offspring
        new_instructions = (program1.instructions[:point1] + 
                          program2.instructions[point2:])
        
        return TemporalProgram(
            instructions=new_instructions,
            description="Crossover offspring",
            complexity=len(set(p for p, _ in new_instructions)),
            pattern_type='crossover',
            expected_length=(program1.expected_length + program2.expected_length) // 2
        )


def create_chronos_prism_system(hidden_dim: int = 256):
    """Factory function to create CHRONOS PRISM system"""
    synthesizer = ChronosPRISM()
    # hidden_dim parameter can be used for future enhancements
    
    return {
        'synthesizer': synthesizer,
        'evaluator': synthesizer,  # Same object handles evaluation
        'library': None,  # CHRONOS uses integrated program library
        'description': 'CHRONOS Temporal Program Synthesis System'
    }