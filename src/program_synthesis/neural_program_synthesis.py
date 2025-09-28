"""
Neural-Guided Program Synthesis for ARC Tasks
Enhanced implementation based on François Chollet's insights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
import itertools
from collections import defaultdict

class ProgramPrimitive(Enum):
    """Extended set of program primitives for ARC tasks"""
    # Geometric transforms
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_HORIZONTAL = "flip_h"
    FLIP_VERTICAL = "flip_v"
    TRANSPOSE = "transpose"
    
    # Object operations
    EXTRACT_OBJECTS = "extract_objects"
    LARGEST_OBJECT = "largest_object"
    SMALLEST_OBJECT = "smallest_object"
    COUNT_OBJECTS = "count_objects"
    SORT_BY_SIZE = "sort_by_size"
    SORT_BY_COLOR = "sort_by_color"
    
    # Color operations
    REPLACE_COLOR = "replace_color"
    EXTRACT_COLOR = "extract_color"
    COLOR_MAP = "color_map"
    FILL = "fill"
    FLOOD_FILL = "flood_fill"
    
    # Spatial operations
    GRAVITY = "gravity"
    ALIGN_LEFT = "align_left"
    ALIGN_RIGHT = "align_right"
    ALIGN_TOP = "align_top"
    ALIGN_BOTTOM = "align_bottom"
    CENTER = "center"
    
    # Pattern operations
    REPEAT_PATTERN = "repeat_pattern"
    EXTEND_PATTERN = "extend_pattern"
    MIRROR_PATTERN = "mirror_pattern"
    COMPLETE_SYMMETRY = "complete_symmetry"
    
    # Grid operations
    CROP = "crop"
    PAD = "pad"
    RESIZE = "resize"
    SPLIT_GRID = "split_grid"
    MERGE_GRIDS = "merge_grids"
    
    # Logical operations
    AND = "and"
    OR = "or"
    XOR = "xor"
    NOT = "not"
    
    # Arithmetic operations
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    
    # Movement operations
    MOVE = "move"
    SWAP = "swap"
    ROTATE_AROUND = "rotate_around"
    
    # Conditional operations
    IF_COLOR = "if_color"
    IF_SHAPE = "if_shape"
    IF_SIZE = "if_size"
    IF_POSITION = "if_position"


@dataclass
class ProgramNode:
    """Node in a program tree"""
    primitive: ProgramPrimitive
    params: Dict[str, any]
    children: List['ProgramNode'] = None
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute this program node on a grid"""
        return execute_primitive(self.primitive, grid, self.params, self.children)


class NeuralProgramPredictor(nn.Module):
    """Neural network that predicts program structures"""
    
    def __init__(self, hidden_dim: int = 512, num_primitives: int = len(ProgramPrimitive)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_primitives = num_primitives
        
        # Grid encoder
        self.conv1 = nn.Conv2d(10, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(7)
        
        # Relation encoder (for input-output pairs)
        self.relation_fc = nn.Linear(256 * 7 * 7 * 2, hidden_dim)
        
        # Program decoder
        self.primitive_head = nn.Linear(hidden_dim, num_primitives)
        self.param_heads = nn.ModuleDict({
            'color': nn.Linear(hidden_dim, 10),  # For color parameters
            'direction': nn.Linear(hidden_dim, 8),  # For direction parameters
            'size': nn.Linear(hidden_dim, 30),  # For size parameters
            'position': nn.Linear(hidden_dim, 900),  # For position parameters (30x30 grid)
        })
        
        # Program composition predictor
        self.composition_head = nn.Linear(hidden_dim, 3)  # Sequential, parallel, conditional
        self.depth_head = nn.Linear(hidden_dim, 5)  # Max program depth
        
        # Confidence predictor
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a grid into features"""
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.flatten(1)
        
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict program structure from input-output pair"""
        # Encode both grids
        input_features = self.encode_grid(input_grid)
        output_features = self.encode_grid(output_grid)
        
        # Combine features
        combined = torch.cat([input_features, output_features], dim=1)
        hidden = F.relu(self.relation_fc(combined))
        
        # Predict program components
        primitive_logits = self.primitive_head(hidden)
        
        params = {
            'color': self.param_heads['color'](hidden),
            'direction': self.param_heads['direction'](hidden),
            'size': self.param_heads['size'](hidden),
            'position': self.param_heads['position'](hidden),
        }
        
        composition = self.composition_head(hidden)
        depth = self.depth_head(hidden)
        confidence = torch.sigmoid(self.confidence_head(hidden))
        
        return {
            'primitives': primitive_logits,
            'params': params,
            'composition': composition,
            'depth': depth,
            'confidence': confidence
        }


class ProgramSynthesizer:
    """Synthesize programs using neural guidance and search"""
    
    def __init__(self, neural_predictor: NeuralProgramPredictor):
        self.neural_predictor = neural_predictor
        self.primitive_executors = self._initialize_executors()
        
    def synthesize(self, 
                   input_grid: np.ndarray, 
                   output_grid: np.ndarray,
                   examples: List[Tuple[np.ndarray, np.ndarray]] = None,
                   max_programs: int = 1000,
                   beam_width: int = 10) -> Optional[ProgramNode]:
        """Synthesize a program that transforms input to output"""
        
        # Get neural guidance
        with torch.no_grad():
            input_tensor = self._grid_to_tensor(input_grid)
            output_tensor = self._grid_to_tensor(output_grid)
            predictions = self.neural_predictor(input_tensor.unsqueeze(0), output_tensor.unsqueeze(0))
        
        # Extract top primitives
        primitive_probs = F.softmax(predictions['primitives'], dim=-1).squeeze(0)
        top_primitives = torch.topk(primitive_probs, k=min(20, len(ProgramPrimitive))).indices
        
        # Beam search for programs
        beam = [(0.0, [])]  # (score, program)
        
        for depth in range(int(predictions['depth'].argmax().item()) + 1):
            new_beam = []
            
            for score, program in beam:
                # Try extending with each primitive
                for prim_idx in top_primitives:
                    primitive = list(ProgramPrimitive)[prim_idx]
                    
                    # Generate parameter candidates based on neural predictions
                    param_candidates = self._generate_parameters(primitive, predictions['params'])
                    
                    for params in param_candidates[:5]:  # Try top 5 parameter sets
                        new_program = program + [(primitive, params)]
                        
                        # Execute and evaluate
                        try:
                            result = self._execute_program(input_grid, new_program)
                            if np.array_equal(result, output_grid):
                                # Found exact match!
                                return self._program_to_node(new_program)
                            
                            # Score based on similarity
                            similarity = self._grid_similarity(result, output_grid)
                            new_score = score + similarity * primitive_probs[prim_idx].item()
                            new_beam.append((new_score, new_program))
                            
                        except Exception:
                            continue
            
            # Keep top beam_width programs
            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]
            
            if not beam:
                break
        
        # Return best program found
        if beam:
            return self._program_to_node(beam[0][1])
        return None
    
    def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        """Convert numpy grid to tensor"""
        h, w = grid.shape
        tensor = torch.zeros(10, 30, 30)
        for i in range(h):
            for j in range(w):
                if i < 30 and j < 30:
                    tensor[grid[i, j], i, j] = 1
        return tensor
    
    def _generate_parameters(self, primitive: ProgramPrimitive, param_predictions: Dict) -> List[Dict]:
        """Generate parameter candidates for a primitive"""
        candidates = []
        
        if primitive in [ProgramPrimitive.REPLACE_COLOR, ProgramPrimitive.EXTRACT_COLOR]:
            # Color parameters
            color_probs = F.softmax(param_predictions['color'], dim=-1).squeeze(0)
            top_colors = torch.topk(color_probs, k=3).indices
            for src in top_colors:
                for dst in top_colors:
                    if src != dst:
                        candidates.append({'source_color': src.item(), 'target_color': dst.item()})
        
        elif primitive in [ProgramPrimitive.MOVE, ProgramPrimitive.ALIGN_LEFT, ProgramPrimitive.ALIGN_RIGHT]:
            # Direction parameters
            dir_probs = F.softmax(param_predictions['direction'], dim=-1).squeeze(0)
            top_dirs = torch.topk(dir_probs, k=3).indices
            for d in top_dirs:
                candidates.append({'direction': d.item()})
        
        else:
            # Default: no parameters
            candidates.append({})
        
        return candidates
    
    def _execute_program(self, grid: np.ndarray, program: List[Tuple[ProgramPrimitive, Dict]]) -> np.ndarray:
        """Execute a program sequence on a grid"""
        result = grid.copy()
        for primitive, params in program:
            result = self.primitive_executors[primitive](result, params)
        return result
    
    def _grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Compute similarity between two grids"""
        if grid1.shape != grid2.shape:
            return 0.0
        return np.mean(grid1 == grid2)
    
    def _program_to_node(self, program: List[Tuple[ProgramPrimitive, Dict]]) -> ProgramNode:
        """Convert program list to tree structure"""
        if not program:
            return None
        
        # For now, create a simple sequential program
        # TODO: Support more complex compositions
        nodes = []
        for primitive, params in program:
            nodes.append(ProgramNode(primitive, params))
        
        # Chain nodes together
        if len(nodes) == 1:
            return nodes[0]
        
        # Create sequential composition
        root = nodes[0]
        current = root
        for node in nodes[1:]:
            current.children = [node]
            current = node
        
        return root
    
    def _initialize_executors(self) -> Dict[ProgramPrimitive, callable]:
        """Initialize primitive execution functions"""
        return {
            ProgramPrimitive.ROTATE_90: lambda g, p: np.rot90(g, k=1),
            ProgramPrimitive.ROTATE_180: lambda g, p: np.rot90(g, k=2),
            ProgramPrimitive.ROTATE_270: lambda g, p: np.rot90(g, k=3),
            ProgramPrimitive.FLIP_HORIZONTAL: lambda g, p: np.fliplr(g),
            ProgramPrimitive.FLIP_VERTICAL: lambda g, p: np.flipud(g),
            ProgramPrimitive.TRANSPOSE: lambda g, p: g.T,
            # Add more executors...
        }


def execute_primitive(primitive: ProgramPrimitive, 
                     grid: np.ndarray, 
                     params: Dict[str, any],
                     children: List[ProgramNode] = None) -> np.ndarray:
    """Execute a primitive operation on a grid"""
    
    if primitive == ProgramPrimitive.ROTATE_90:
        return np.rot90(grid, k=1)
    elif primitive == ProgramPrimitive.ROTATE_180:
        return np.rot90(grid, k=2)
    elif primitive == ProgramPrimitive.ROTATE_270:
        return np.rot90(grid, k=3)
    elif primitive == ProgramPrimitive.FLIP_HORIZONTAL:
        return np.fliplr(grid)
    elif primitive == ProgramPrimitive.FLIP_VERTICAL:
        return np.flipud(grid)
    elif primitive == ProgramPrimitive.TRANSPOSE:
        return grid.T
    
    elif primitive == ProgramPrimitive.REPLACE_COLOR:
        result = grid.copy()
        src = params.get('source_color', 0)
        tgt = params.get('target_color', 0)
        result[grid == src] = tgt
        return result
    
    elif primitive == ProgramPrimitive.EXTRACT_COLOR:
        color = params.get('color', 1)
        result = np.zeros_like(grid)
        result[grid == color] = color
        return result
    
    elif primitive == ProgramPrimitive.FILL:
        color = params.get('color', 0)
        return np.full_like(grid, color)
    
    elif primitive == ProgramPrimitive.LARGEST_OBJECT:
        return extract_largest_object(grid)
    
    elif primitive == ProgramPrimitive.SMALLEST_OBJECT:
        return extract_smallest_object(grid)
    
    # Add more primitive implementations...
    
    else:
        # Default: return unchanged
        return grid.copy()


def extract_largest_object(grid: np.ndarray) -> np.ndarray:
    """Extract the largest connected component"""
    from scipy.ndimage import label
    result = np.zeros_like(grid)
    
    for color in range(1, 10):
        mask = (grid == color)
        if not mask.any():
            continue
            
        labeled, num_objects = label(mask)
        if num_objects == 0:
            continue
            
        sizes = [np.sum(labeled == i) for i in range(1, num_objects + 1)]
        largest_idx = np.argmax(sizes) + 1
        result[labeled == largest_idx] = color
    
    return result


def extract_smallest_object(grid: np.ndarray) -> np.ndarray:
    """Extract the smallest connected component"""
    from scipy.ndimage import label
    result = np.zeros_like(grid)
    
    for color in range(1, 10):
        mask = (grid == color)
        if not mask.any():
            continue
            
        labeled, num_objects = label(mask)
        if num_objects == 0:
            continue
            
        sizes = [np.sum(labeled == i) for i in range(1, num_objects + 1)]
        smallest_idx = np.argmin(sizes) + 1
        result[labeled == smallest_idx] = color
    
    return result


class ProgramVerifier:
    """Verify that synthesized programs work correctly"""
    
    @staticmethod
    def verify(program: ProgramNode, 
               examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Verify program on all examples"""
        if not program:
            return False
            
        for input_grid, expected_output in examples:
            try:
                result = program.execute(input_grid)
                if not np.array_equal(result, expected_output):
                    return False
            except Exception:
                return False
                
        return True
    
    @staticmethod
    def verify_properties(program: ProgramNode) -> Dict[str, bool]:
        """Check various program properties"""
        properties = {
            'deterministic': True,  # ARC programs should be deterministic
            'size_preserving': check_size_preserving(program),
            'color_preserving': check_color_preserving(program),
            'object_preserving': check_object_preserving(program),
        }
        return properties


def check_size_preserving(program: ProgramNode) -> bool:
    """Check if program preserves grid size"""
    # Some primitives change size
    size_changing = {
        ProgramPrimitive.CROP, ProgramPrimitive.PAD, 
        ProgramPrimitive.RESIZE, ProgramPrimitive.TRANSPOSE
    }
    
    def has_size_change(node):
        if node.primitive in size_changing:
            return True
        if node.children:
            return any(has_size_change(child) for child in node.children)
        return False
    
    return not has_size_change(program)


def check_color_preserving(program: ProgramNode) -> bool:
    """Check if program preserves colors"""
    color_changing = {
        ProgramPrimitive.REPLACE_COLOR, ProgramPrimitive.COLOR_MAP,
        ProgramPrimitive.FILL, ProgramPrimitive.FLOOD_FILL
    }
    
    def has_color_change(node):
        if node.primitive in color_changing:
            return True
        if node.children:
            return any(has_color_change(child) for child in node.children)
        return False
    
    return not has_color_change(program)


def check_object_preserving(program: ProgramNode) -> bool:
    """Check if program preserves objects"""
    object_changing = {
        ProgramPrimitive.EXTRACT_OBJECTS, ProgramPrimitive.LARGEST_OBJECT,
        ProgramPrimitive.SMALLEST_OBJECT, ProgramPrimitive.MERGE_GRIDS
    }
    
    def has_object_change(node):
        if node.primitive in object_changing:
            return True
        if node.children:
            return any(has_object_change(child) for child in node.children)
        return False
    
    return not has_object_change(program)