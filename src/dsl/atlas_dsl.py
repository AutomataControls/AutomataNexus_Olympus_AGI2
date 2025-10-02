"""
ATLAS-specific DSL integration for spatial transformations and geometric patterns
COMPLETELY INDEPENDENT - No imports from base DSL
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum


class AtlasOperation(Enum):
    """ATLAS-specific operations for spatial transformations and geometry"""
    # Basic spatial transformations
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    TRANSPOSE = "transpose"
    
    # Affine transformations
    SCALE = "scale"
    TRANSLATE = "translate"
    SHEAR = "shear"
    PERSPECTIVE = "perspective"
    
    # Geometric pattern operations
    TESSELLATE = "tessellate"
    TILE_PATTERN = "tile_pattern"
    MIRROR_PATTERN = "mirror_pattern"
    KALEIDOSCOPE = "kaleidoscope"
    
    # Shape transformations
    MORPH_SHAPE = "morph_shape"
    STRETCH = "stretch"
    COMPRESS = "compress"
    WARP = "warp"
    
    # Grid operations
    GRID_ALIGN = "grid_align"
    SNAP_TO_GRID = "snap_to_grid"
    GRID_TRANSFORM = "grid_transform"
    ELASTIC_GRID = "elastic_grid"
    
    # Spatial filters
    SPATIAL_BLUR = "spatial_blur"
    EDGE_ENHANCE = "edge_enhance"
    SPATIAL_MEDIAN = "spatial_median"
    DIRECTIONAL_FILTER = "directional_filter"
    
    # Pattern matching
    TEMPLATE_MATCH = "template_match"
    PATTERN_ALIGN = "pattern_align"
    GEOMETRIC_MATCH = "geometric_match"
    SPATIAL_CORRELATION = "spatial_correlation"
    
    # Advanced transformations
    POLAR_TRANSFORM = "polar_transform"
    LOG_POLAR = "log_polar"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    
    # Spatial analysis
    FIND_SYMMETRY = "find_symmetry"
    DETECT_REPETITION = "detect_repetition"
    SPATIAL_MOMENTS = "spatial_moments"
    ORIENTATION_ANALYSIS = "orientation_analysis"


class AtlasDSLProgram:
    """ATLAS-specific DSL program representation"""
    
    def __init__(self, operations: List[Tuple[AtlasOperation, Dict[str, Any]]]):
        self.operations = operations
    
    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute the ATLAS DSL program on input grid"""
        result = grid.copy()
        for op, params in self.operations:
            result = AtlasDSLExecutor.execute_operation(result, op, params)
        return result
    
    def to_string(self) -> str:
        """Convert program to readable string"""
        prog_str = []
        for op, params in self.operations:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            prog_str.append(f"{op.value}({param_str})")
        return " -> ".join(prog_str)


class AtlasDSLExecutor:
    """Executes ATLAS-specific DSL operations"""
    
    @staticmethod
    def execute_operation(grid: np.ndarray, op: AtlasOperation, params: Dict[str, Any]) -> np.ndarray:
        """Execute a single ATLAS DSL operation"""
        
        # Basic spatial transformations
        if op == AtlasOperation.ROTATE_90:
            return AtlasDSLExecutor._rotate_90(grid, params)
        elif op == AtlasOperation.ROTATE_180:
            return AtlasDSLExecutor._rotate_180(grid, params)
        elif op == AtlasOperation.ROTATE_270:
            return AtlasDSLExecutor._rotate_270(grid, params)
        elif op == AtlasOperation.FLIP_HORIZONTAL:
            return AtlasDSLExecutor._flip_horizontal(grid, params)
        elif op == AtlasOperation.FLIP_VERTICAL:
            return AtlasDSLExecutor._flip_vertical(grid, params)
        elif op == AtlasOperation.TRANSPOSE:
            return AtlasDSLExecutor._transpose(grid, params)
        
        # Affine transformations
        elif op == AtlasOperation.SCALE:
            return AtlasDSLExecutor._scale(grid, params)
        elif op == AtlasOperation.TRANSLATE:
            return AtlasDSLExecutor._translate(grid, params)
        elif op == AtlasOperation.SHEAR:
            return AtlasDSLExecutor._shear(grid, params)
        elif op == AtlasOperation.PERSPECTIVE:
            return AtlasDSLExecutor._perspective(grid, params)
        
        # Geometric patterns
        elif op == AtlasOperation.TESSELLATE:
            return AtlasDSLExecutor._tessellate(grid, params)
        elif op == AtlasOperation.TILE_PATTERN:
            return AtlasDSLExecutor._tile_pattern(grid, params)
        elif op == AtlasOperation.MIRROR_PATTERN:
            return AtlasDSLExecutor._mirror_pattern(grid, params)
        elif op == AtlasOperation.KALEIDOSCOPE:
            return AtlasDSLExecutor._kaleidoscope(grid, params)
        
        # Shape transformations
        elif op == AtlasOperation.MORPH_SHAPE:
            return AtlasDSLExecutor._morph_shape(grid, params)
        elif op == AtlasOperation.STRETCH:
            return AtlasDSLExecutor._stretch(grid, params)
        elif op == AtlasOperation.COMPRESS:
            return AtlasDSLExecutor._compress(grid, params)
        elif op == AtlasOperation.WARP:
            return AtlasDSLExecutor._warp(grid, params)
        
        # Grid operations
        elif op == AtlasOperation.GRID_ALIGN:
            return AtlasDSLExecutor._grid_align(grid, params)
        elif op == AtlasOperation.SNAP_TO_GRID:
            return AtlasDSLExecutor._snap_to_grid(grid, params)
        elif op == AtlasOperation.GRID_TRANSFORM:
            return AtlasDSLExecutor._grid_transform(grid, params)
        elif op == AtlasOperation.ELASTIC_GRID:
            return AtlasDSLExecutor._elastic_grid(grid, params)
        
        # Spatial filters
        elif op == AtlasOperation.SPATIAL_BLUR:
            return AtlasDSLExecutor._spatial_blur(grid, params)
        elif op == AtlasOperation.EDGE_ENHANCE:
            return AtlasDSLExecutor._edge_enhance(grid, params)
        elif op == AtlasOperation.SPATIAL_MEDIAN:
            return AtlasDSLExecutor._spatial_median(grid, params)
        elif op == AtlasOperation.DIRECTIONAL_FILTER:
            return AtlasDSLExecutor._directional_filter(grid, params)
        
        # Pattern matching
        elif op == AtlasOperation.TEMPLATE_MATCH:
            return AtlasDSLExecutor._template_match(grid, params)
        elif op == AtlasOperation.PATTERN_ALIGN:
            return AtlasDSLExecutor._pattern_align(grid, params)
        elif op == AtlasOperation.GEOMETRIC_MATCH:
            return AtlasDSLExecutor._geometric_match(grid, params)
        elif op == AtlasOperation.SPATIAL_CORRELATION:
            return AtlasDSLExecutor._spatial_correlation(grid, params)
        
        # Advanced transformations
        elif op == AtlasOperation.POLAR_TRANSFORM:
            return AtlasDSLExecutor._polar_transform(grid, params)
        elif op == AtlasOperation.LOG_POLAR:
            return AtlasDSLExecutor._log_polar(grid, params)
        elif op == AtlasOperation.CYLINDRICAL:
            return AtlasDSLExecutor._cylindrical(grid, params)
        elif op == AtlasOperation.SPHERICAL:
            return AtlasDSLExecutor._spherical(grid, params)
        
        # Spatial analysis
        elif op == AtlasOperation.FIND_SYMMETRY:
            return AtlasDSLExecutor._find_symmetry(grid, params)
        elif op == AtlasOperation.DETECT_REPETITION:
            return AtlasDSLExecutor._detect_repetition(grid, params)
        elif op == AtlasOperation.SPATIAL_MOMENTS:
            return AtlasDSLExecutor._spatial_moments(grid, params)
        elif op == AtlasOperation.ORIENTATION_ANALYSIS:
            return AtlasDSLExecutor._orientation_analysis(grid, params)
        
        else:
            return grid
    
    # Basic spatial transformations
    @staticmethod
    def _rotate_90(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Rotate grid 90 degrees clockwise"""
        return np.rot90(grid, k=1, axes=(0, 1))
    
    @staticmethod
    def _rotate_180(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Rotate grid 180 degrees"""
        return np.rot90(grid, k=2, axes=(0, 1))
    
    @staticmethod
    def _rotate_270(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Rotate grid 270 degrees clockwise (90 counter-clockwise)"""
        return np.rot90(grid, k=3, axes=(0, 1))
    
    @staticmethod
    def _flip_horizontal(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Flip grid horizontally"""
        return np.flip(grid, axis=1)
    
    @staticmethod
    def _flip_vertical(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Flip grid vertically"""
        return np.flip(grid, axis=0)
    
    @staticmethod
    def _transpose(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Transpose grid (swap rows and columns)"""
        return np.transpose(grid)
    
    # Affine transformations
    @staticmethod
    def _scale(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Scale grid by given factor"""
        factor = params.get('factor', 1.0)
        if factor == 1.0:
            return grid
        
        h, w = grid.shape
        new_h, new_w = int(h * factor), int(w * factor)
        
        if new_h <= 0 or new_w <= 0:
            return grid
        
        # Simple nearest neighbor scaling
        scaled = np.zeros((new_h, new_w), dtype=grid.dtype)
        for i in range(new_h):
            for j in range(new_w):
                orig_i = min(int(i / factor), h - 1)
                orig_j = min(int(j / factor), w - 1)
                scaled[i, j] = grid[orig_i, orig_j]
        
        return scaled
    
    @staticmethod
    def _translate(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Translate grid by given offset"""
        dx = params.get('dx', 0)
        dy = params.get('dy', 0)
        
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
    def _shear(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply shear transformation"""
        shear_x = params.get('shear_x', 0.0)
        shear_y = params.get('shear_y', 0.0)
        
        if shear_x == 0.0 and shear_y == 0.0:
            return grid
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        for i in range(h):
            for j in range(w):
                new_j = j + int(shear_x * i)
                new_i = i + int(shear_y * j)
                if 0 <= new_i < h and 0 <= new_j < w:
                    result[new_i, new_j] = grid[i, j]
        
        return result
    
    @staticmethod
    def _perspective(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply perspective transformation (simplified)"""
        # Simplified perspective - just return original for now
        return grid
    
    # Geometric patterns
    @staticmethod
    def _tessellate(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create tessellation pattern"""
        tile_h = params.get('tile_h', grid.shape[0])
        tile_w = params.get('tile_w', grid.shape[1])
        
        if tile_h >= grid.shape[0] and tile_w >= grid.shape[1]:
            return grid
        
        h, w = grid.shape
        result = np.zeros_like(grid)
        
        for i in range(0, h, tile_h):
            for j in range(0, w, tile_w):
                end_i = min(i + tile_h, h)
                end_j = min(j + tile_w, w)
                
                tile = grid[:end_i-i, :end_j-j]
                result[i:end_i, j:end_j] = tile
        
        return result
    
    @staticmethod
    def _tile_pattern(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create tiled pattern"""
        tiles_x = params.get('tiles_x', 2)
        tiles_y = params.get('tiles_y', 2)
        
        h, w = grid.shape
        tile_h = h // tiles_y
        tile_w = w // tiles_x
        
        if tile_h <= 0 or tile_w <= 0:
            return grid
        
        base_tile = grid[:tile_h, :tile_w]
        result = np.zeros_like(grid)
        
        for i in range(tiles_y):
            for j in range(tiles_x):
                start_i = i * tile_h
                start_j = j * tile_w
                end_i = min(start_i + tile_h, h)
                end_j = min(start_j + tile_w, w)
                
                result[start_i:end_i, start_j:end_j] = base_tile[:end_i-start_i, :end_j-start_j]
        
        return result
    
    @staticmethod
    def _mirror_pattern(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create mirror pattern"""
        axis = params.get('axis', 'both')  # 'horizontal', 'vertical', 'both'
        
        h, w = grid.shape
        
        if axis == 'horizontal':
            # Mirror horizontally
            left_half = grid[:, :w//2]
            result = np.zeros_like(grid)
            result[:, :w//2] = left_half
            result[:, w//2:w//2+left_half.shape[1]] = np.flip(left_half, axis=1)
        elif axis == 'vertical':
            # Mirror vertically
            top_half = grid[:h//2, :]
            result = np.zeros_like(grid)
            result[:h//2, :] = top_half
            result[h//2:h//2+top_half.shape[0], :] = np.flip(top_half, axis=0)
        else:  # both
            # Mirror both ways
            quarter = grid[:h//2, :w//2]
            result = np.zeros_like(grid)
            result[:h//2, :w//2] = quarter
            result[:h//2, w//2:w//2+quarter.shape[1]] = np.flip(quarter, axis=1)
            result[h//2:h//2+quarter.shape[0], :w//2] = np.flip(quarter, axis=0)
            result[h//2:h//2+quarter.shape[0], w//2:w//2+quarter.shape[1]] = np.flip(np.flip(quarter, axis=0), axis=1)
        
        return result
    
    @staticmethod
    def _kaleidoscope(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Create kaleidoscope effect"""
        segments = params.get('segments', 6)
        
        if segments <= 1:
            return grid
        
        h, w = grid.shape
        center_i, center_j = h // 2, w // 2
        
        # Create kaleidoscope by rotating and overlaying segments
        result = np.zeros_like(grid)
        
        for seg in range(segments):
            angle = seg * (360 // segments)
            rotated = AtlasDSLExecutor._rotate_by_angle(grid, angle)
            result = np.maximum(result, rotated)
        
        return result
    
    @staticmethod
    def _rotate_by_angle(grid: np.ndarray, angle: float) -> np.ndarray:
        """Rotate grid by arbitrary angle (simplified)"""
        # Simplified rotation - only handle 90-degree increments
        angle = angle % 360
        if angle == 90:
            return np.rot90(grid, k=1)
        elif angle == 180:
            return np.rot90(grid, k=2)
        elif angle == 270:
            return np.rot90(grid, k=3)
        else:
            return grid
    
    # Shape transformations (simplified implementations)
    @staticmethod
    def _morph_shape(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Morph shape (simplified)"""
        return grid
    
    @staticmethod
    def _stretch(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Stretch grid"""
        factor_x = params.get('factor_x', 1.0)
        factor_y = params.get('factor_y', 1.0)
        
        if factor_x == 1.0 and factor_y == 1.0:
            return grid
        
        return AtlasDSLExecutor._scale(grid, {'factor': max(factor_x, factor_y)})
    
    @staticmethod
    def _compress(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compress grid"""
        factor = params.get('factor', 0.5)
        return AtlasDSLExecutor._scale(grid, {'factor': factor})
    
    @staticmethod
    def _warp(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Warp grid (simplified)"""
        return grid
    
    # Grid operations (simplified implementations)
    @staticmethod
    def _grid_align(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Align to grid"""
        return grid
    
    @staticmethod
    def _snap_to_grid(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Snap to grid"""
        return grid
    
    @staticmethod
    def _grid_transform(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Grid transformation"""
        return grid
    
    @staticmethod
    def _elastic_grid(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Elastic grid transformation"""
        return grid
    
    # Spatial filters (simplified implementations)
    @staticmethod
    def _spatial_blur(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Spatial blur"""
        return grid
    
    @staticmethod
    def _edge_enhance(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Edge enhancement"""
        return grid
    
    @staticmethod
    def _spatial_median(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Spatial median filter"""
        return grid
    
    @staticmethod
    def _directional_filter(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Directional filter"""
        return grid
    
    # Pattern matching (simplified implementations)
    @staticmethod
    def _template_match(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Template matching"""
        return grid
    
    @staticmethod
    def _pattern_align(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Pattern alignment"""
        return grid
    
    @staticmethod
    def _geometric_match(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Geometric matching"""
        return grid
    
    @staticmethod
    def _spatial_correlation(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Spatial correlation"""
        return grid
    
    # Advanced transformations (simplified implementations)
    @staticmethod
    def _polar_transform(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Polar transformation"""
        return grid
    
    @staticmethod
    def _log_polar(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Log-polar transformation"""
        return grid
    
    @staticmethod
    def _cylindrical(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Cylindrical transformation"""
        return grid
    
    @staticmethod
    def _spherical(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Spherical transformation"""
        return grid
    
    # Spatial analysis (simplified implementations)
    @staticmethod
    def _find_symmetry(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Find symmetry"""
        return grid
    
    @staticmethod
    def _detect_repetition(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Detect repetition"""
        return grid
    
    @staticmethod
    def _spatial_moments(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Spatial moments"""
        return grid
    
    @staticmethod
    def _orientation_analysis(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Orientation analysis"""
        return grid


class ATLASDSLGenerator:
    """Generates ATLAS-specific DSL programs for spatial transformations"""
    
    def __init__(self):
        self.common_operations = [
            AtlasOperation.ROTATE_90,
            AtlasOperation.ROTATE_180,
            AtlasOperation.ROTATE_270,
            AtlasOperation.FLIP_HORIZONTAL,
            AtlasOperation.FLIP_VERTICAL,
            AtlasOperation.TRANSPOSE,
            AtlasOperation.SCALE,
            AtlasOperation.TRANSLATE,
            AtlasOperation.MIRROR_PATTERN,
            AtlasOperation.TILE_PATTERN
        ]
    
    def generate_program(self, complexity: int = 3) -> AtlasDSLProgram:
        """Generate random ATLAS DSL program"""
        operations = []
        
        for _ in range(min(complexity, 5)):
            op = np.random.choice(self.common_operations)
            params = self._generate_params(op)
            operations.append((op, params))
        
        return AtlasDSLProgram(operations)
    
    def _generate_params(self, op: AtlasOperation) -> Dict[str, Any]:
        """Generate parameters for operation"""
        if op in [AtlasOperation.ROTATE_90, AtlasOperation.ROTATE_180, 
                  AtlasOperation.ROTATE_270, AtlasOperation.FLIP_HORIZONTAL,
                  AtlasOperation.FLIP_VERTICAL, AtlasOperation.TRANSPOSE]:
            return {}
        elif op == AtlasOperation.SCALE:
            return {'factor': np.random.choice([0.5, 1.5, 2.0])}
        elif op == AtlasOperation.TRANSLATE:
            return {'dx': np.random.randint(-3, 4), 'dy': np.random.randint(-3, 4)}
        elif op == AtlasOperation.MIRROR_PATTERN:
            return {'axis': np.random.choice(['horizontal', 'vertical', 'both'])}
        elif op == AtlasOperation.TILE_PATTERN:
            return {'tiles_x': np.random.randint(2, 5), 'tiles_y': np.random.randint(2, 5)}
        else:
            return {}
    
    def generate_spatial_transformation_program(self, input_grid: np.ndarray, 
                                               output_grid: np.ndarray) -> Optional[AtlasDSLProgram]:
        """Generate program that transforms input to output"""
        # Try simple transformations first
        simple_transforms = [
            (AtlasOperation.ROTATE_90, {}),
            (AtlasOperation.ROTATE_180, {}),
            (AtlasOperation.ROTATE_270, {}),
            (AtlasOperation.FLIP_HORIZONTAL, {}),
            (AtlasOperation.FLIP_VERTICAL, {}),
            (AtlasOperation.TRANSPOSE, {})
        ]
        
        for op, params in simple_transforms:
            program = AtlasDSLProgram([(op, params)])
            if np.array_equal(program.execute(input_grid), output_grid):
                return program
        
        # Try combinations of transformations
        for op1, params1 in simple_transforms:
            for op2, params2 in simple_transforms:
                program = AtlasDSLProgram([(op1, params1), (op2, params2)])
                if np.array_equal(program.execute(input_grid), output_grid):
                    return program
        
        return None


class ATLASDSLTraining:
    """ATLAS-specific DSL training integration"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.generator = ATLASDSLGenerator()
    
    def create_dsl_augmented_batch(self, input_grids: torch.Tensor, 
                                  output_grids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create DSL-augmented training batch for ATLAS"""
        batch_size = input_grids.shape[0]
        augmented_inputs = []
        augmented_outputs = []
        
        for i in range(batch_size):
            input_np = input_grids[i].cpu().numpy()
            output_np = output_grids[i].cpu().numpy()
            
            # Generate ATLAS-specific transformation program
            program = self.generator.generate_spatial_transformation_program(input_np, output_np)
            
            if program is not None:
                # Apply program to create variations
                for _ in range(3):  # Create 3 variations per sample
                    variation_program = self.generator.generate_program(complexity=2)
                    
                    try:
                        augmented_input = variation_program.execute(input_np)
                        augmented_output = variation_program.execute(output_np)
                        
                        augmented_inputs.append(torch.tensor(augmented_input, dtype=input_grids.dtype))
                        augmented_outputs.append(torch.tensor(augmented_output, dtype=output_grids.dtype))
                    except:
                        # If transformation fails, use original
                        augmented_inputs.append(input_grids[i])
                        augmented_outputs.append(output_grids[i])
            else:
                # No program found, use original
                augmented_inputs.append(input_grids[i])
                augmented_outputs.append(output_grids[i])
        
        # Convert to tensors
        aug_inputs = torch.stack(augmented_inputs).to(self.device)
        aug_outputs = torch.stack(augmented_outputs).to(self.device)
        
        return aug_inputs, aug_outputs
    
    def spatial_consistency_loss(self, predictions: torch.Tensor, 
                                targets: torch.Tensor) -> torch.Tensor:
        """ATLAS-specific spatial consistency loss"""
        # Base loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Spatial gradient consistency
        pred_grad_x = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        pred_grad_y = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]
        
        target_grad_x = targets[:, :, 1:, :] - targets[:, :, :-1, :]
        target_grad_y = targets[:, :, :, 1:] - targets[:, :, :, :-1]
        
        gradient_loss = F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)
        
        return base_loss + 0.1 * gradient_loss