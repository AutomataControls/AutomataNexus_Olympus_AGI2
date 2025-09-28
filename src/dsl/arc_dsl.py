"""
ARC Domain-Specific Language (DSL) for Program Synthesis
Provides deterministic transformations for exact output generation
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Callable
from enum import Enum
import torch


class Operation(Enum):
    """DSL primitive operations"""
    # Basic transforms
    IDENTITY = "identity"
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_H = "flip_horizontal"
    FLIP_V = "flip_vertical"
    TRANSPOSE = "transpose"
    
    # Color operations
    COLOR_MAP = "color_map"
    FILL_COLOR = "fill_color"
    REPLACE_COLOR = "replace_color"
    EXTRACT_COLOR = "extract_color"
    
    # Object operations
    EXTRACT_OBJECTS = "extract_objects"
    COUNT_OBJECTS = "count_objects"
    LARGEST_OBJECT = "largest_object"
    SMALLEST_OBJECT = "smallest_object"
    
    # Grid operations
    CROP = "crop"
    PAD = "pad"
    RESIZE = "resize"
    TILE = "tile"
    
    # Logical operations
    AND = "and"
    OR = "or"
    XOR = "xor"
    NOT = "not"
    
    # Pattern operations
    FIND_PATTERN = "find_pattern"
    APPLY_PATTERN = "apply_pattern"
    SYMMETRIZE = "symmetrize"


class DSLProgram:
    """Represents a sequence of DSL operations"""
    
    def __init__(self, operations: List[Tuple[Operation, Dict[str, Any]]]):
        self.operations = operations
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute the program on input grid"""
        result = grid.copy()
        for op, params in self.operations:
            result = DSLExecutor.execute_operation(result, op, params)
        return result
    
    def to_string(self) -> str:
        """Convert program to readable string"""
        prog_str = []
        for op, params in self.operations:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            prog_str.append(f"{op.value}({param_str})")
        return " -> ".join(prog_str)


class DSLExecutor:
    """Executes DSL operations on grids"""
    
    @staticmethod
    def execute_operation(grid: np.ndarray, op: Operation, params: Dict[str, Any]) -> np.ndarray:
        """Execute a single DSL operation"""
        
        # Basic transforms
        if op == Operation.IDENTITY:
            return grid.copy()
        elif op == Operation.ROTATE_90:
            return np.rot90(grid, k=1)
        elif op == Operation.ROTATE_180:
            return np.rot90(grid, k=2)
        elif op == Operation.ROTATE_270:
            return np.rot90(grid, k=3)
        elif op == Operation.FLIP_H:
            return np.fliplr(grid)
        elif op == Operation.FLIP_V:
            return np.flipud(grid)
        elif op == Operation.TRANSPOSE:
            return grid.T
        
        # Color operations
        elif op == Operation.COLOR_MAP:
            mapping = params.get('mapping', {})
            result = grid.copy()
            for old_color, new_color in mapping.items():
                result[grid == old_color] = new_color
            return result
        
        elif op == Operation.FILL_COLOR:
            color = params.get('color', 0)
            return np.full_like(grid, color)
        
        elif op == Operation.REPLACE_COLOR:
            old_color = params.get('old_color', 0)
            new_color = params.get('new_color', 1)
            result = grid.copy()
            result[grid == old_color] = new_color
            return result
        
        elif op == Operation.EXTRACT_COLOR:
            color = params.get('color', 1)
            background = params.get('background', 0)
            result = np.full_like(grid, background)
            result[grid == color] = color
            return result
        
        # Object operations
        elif op == Operation.EXTRACT_OBJECTS:
            return DSLExecutor._extract_objects(grid, params)
        
        elif op == Operation.LARGEST_OBJECT:
            return DSLExecutor._get_largest_object(grid, params)
        
        elif op == Operation.SMALLEST_OBJECT:
            return DSLExecutor._get_smallest_object(grid, params)
        
        # Grid operations
        elif op == Operation.CROP:
            x1 = params.get('x1', 0)
            y1 = params.get('y1', 0)
            x2 = params.get('x2', grid.shape[0])
            y2 = params.get('y2', grid.shape[1])
            return grid[x1:x2, y1:y2]
        
        elif op == Operation.PAD:
            pad_val = params.get('value', 0)
            pad_width = params.get('width', 1)
            return np.pad(grid, pad_width, constant_values=pad_val)
        
        elif op == Operation.TILE:
            reps = params.get('reps', (2, 2))
            return np.tile(grid, reps)
        
        # Logical operations
        elif op == Operation.AND:
            other = params.get('other')
            if other is not None:
                return np.minimum(grid, other)
            return grid
        
        elif op == Operation.OR:
            other = params.get('other')
            if other is not None:
                return np.maximum(grid, other)
            return grid
        
        elif op == Operation.XOR:
            other = params.get('other')
            if other is not None:
                return np.where(grid != other, np.maximum(grid, other), 0)
            return grid
        
        elif op == Operation.NOT:
            max_val = params.get('max_val', 9)
            return max_val - grid
        
        # Pattern operations
        elif op == Operation.SYMMETRIZE:
            axis = params.get('axis', 'horizontal')
            if axis == 'horizontal':
                top = grid[:grid.shape[0]//2]
                return np.vstack([top, np.flipud(top)])
            else:
                left = grid[:, :grid.shape[1]//2]
                return np.hstack([left, np.fliplr(left)])
        
        else:
            return grid
    
    @staticmethod
    def _extract_objects(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract connected components"""
        background = params.get('background', 0)
        min_size = params.get('min_size', 1)
        
        from scipy import ndimage
        
        # Label connected components
        labeled, num_features = ndimage.label(grid != background)
        
        # Filter by size
        result = np.zeros_like(grid)
        for i in range(1, num_features + 1):
            component = (labeled == i)
            if np.sum(component) >= min_size:
                result[component] = grid[component]
        
        return result
    
    @staticmethod
    def _get_largest_object(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract largest connected component"""
        background = params.get('background', 0)
        
        from scipy import ndimage
        
        labeled, num_features = ndimage.label(grid != background)
        
        if num_features == 0:
            return np.full_like(grid, background)
        
        # Find largest
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        largest_idx = np.argmax(sizes) + 1
        
        result = np.full_like(grid, background)
        mask = (labeled == largest_idx)
        result[mask] = grid[mask]
        
        return result
    
    @staticmethod
    def _get_smallest_object(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Extract smallest connected component"""
        background = params.get('background', 0)
        
        from scipy import ndimage
        
        labeled, num_features = ndimage.label(grid != background)
        
        if num_features == 0:
            return np.full_like(grid, background)
        
        # Find smallest
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        smallest_idx = np.argmin(sizes) + 1
        
        result = np.full_like(grid, background)
        mask = (labeled == smallest_idx)
        result[mask] = grid[mask]
        
        return result


class DSLProgramGenerator:
    """Generates DSL programs for Stage 0 training"""
    
    @staticmethod
    def generate_simple_programs() -> List[DSLProgram]:
        """Generate simple deterministic programs for Stage 0"""
        programs = []
        
        # Identity
        programs.append(DSLProgram([(Operation.IDENTITY, {})]))
        
        # Rotations
        programs.append(DSLProgram([(Operation.ROTATE_90, {})]))
        programs.append(DSLProgram([(Operation.ROTATE_180, {})]))
        programs.append(DSLProgram([(Operation.ROTATE_270, {})]))
        
        # Flips
        programs.append(DSLProgram([(Operation.FLIP_H, {})]))
        programs.append(DSLProgram([(Operation.FLIP_V, {})]))
        programs.append(DSLProgram([(Operation.TRANSPOSE, {})]))
        
        # Color operations
        for old in range(3):
            for new in range(3):
                if old != new:
                    programs.append(DSLProgram([
                        (Operation.REPLACE_COLOR, {'old_color': old, 'new_color': new})
                    ]))
        
        # Fill operations
        for color in range(3):
            programs.append(DSLProgram([(Operation.FILL_COLOR, {'color': color})]))
        
        # Extract color
        for color in range(1, 3):
            programs.append(DSLProgram([
                (Operation.EXTRACT_COLOR, {'color': color, 'background': 0})
            ]))
        
        # Composite operations
        programs.append(DSLProgram([
            (Operation.ROTATE_90, {}),
            (Operation.FLIP_H, {})
        ]))
        
        programs.append(DSLProgram([
            (Operation.EXTRACT_COLOR, {'color': 1, 'background': 0}),
            (Operation.ROTATE_180, {})
        ]))
        
        # Symmetrize
        programs.append(DSLProgram([(Operation.SYMMETRIZE, {'axis': 'horizontal'})]))
        programs.append(DSLProgram([(Operation.SYMMETRIZE, {'axis': 'vertical'})]))
        
        return programs
    
    @staticmethod
    def generate_random_program(max_ops: int = 3) -> DSLProgram:
        """Generate a random DSL program"""
        import random
        
        num_ops = random.randint(1, max_ops)
        operations = []
        
        simple_ops = [
            Operation.IDENTITY, Operation.ROTATE_90, Operation.ROTATE_180,
            Operation.ROTATE_270, Operation.FLIP_H, Operation.FLIP_V,
            Operation.TRANSPOSE
        ]
        
        for _ in range(num_ops):
            op = random.choice(simple_ops)
            operations.append((op, {}))
        
        return DSLProgram(operations)


class DSLProgramPredictor(torch.nn.Module):
    """Neural network that predicts DSL programs instead of pixels"""
    
    def __init__(self, input_channels: int = 10, hidden_dim: int = 256, max_program_length: int = 5):
        super().__init__()
        self.max_program_length = max_program_length
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        
        # Program prediction head
        self.program_head = torch.nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Operation classifier
        num_operations = len(Operation)
        self.op_classifier = torch.nn.Linear(hidden_dim, num_operations)
        
        # Parameter predictors
        self.color_param = torch.nn.Linear(hidden_dim, 10)  # For color parameters
        self.spatial_param = torch.nn.Linear(hidden_dim, 4)  # For spatial parameters
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict DSL program for input grid"""
        B = x.shape[0]
        
        # Encode input
        features = self.encoder(x).squeeze(-1).squeeze(-1)  # B, 128
        
        # Initialize LSTM input
        lstm_input = features.unsqueeze(1).repeat(1, self.max_program_length, 1)  # B, L, 128
        
        # Predict program sequence
        lstm_out, _ = self.program_head(lstm_input)  # B, L, hidden_dim
        
        # Predict operations
        op_logits = self.op_classifier(lstm_out)  # B, L, num_operations
        
        # Predict parameters
        color_params = self.color_param(lstm_out)  # B, L, 10
        spatial_params = self.spatial_param(lstm_out)  # B, L, 4
        
        return {
            'operations': op_logits,
            'color_params': color_params,
            'spatial_params': spatial_params
        }
    
    def predict_program(self, x: torch.Tensor) -> List[DSLProgram]:
        """Convert network output to executable DSL programs"""
        with torch.no_grad():
            output = self.forward(x)
            op_logits = output['operations']
            
            B, L, _ = op_logits.shape
            programs = []
            
            for b in range(B):
                operations = []
                for l in range(L):
                    op_idx = op_logits[b, l].argmax().item()
                    op = list(Operation)[op_idx]
                    
                    # Simple parameter assignment based on operation type
                    params = {}
                    if 'color' in op.value.lower():
                        color_idx = output['color_params'][b, l].argmax().item()
                        params['color'] = color_idx
                    
                    operations.append((op, params))
                
                programs.append(DSLProgram(operations))
            
            return programs