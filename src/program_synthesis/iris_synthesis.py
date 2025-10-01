"""
IRIS-Specific Program Synthesis Module
Tailored for color pattern transformations and recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import random
from dataclasses import dataclass


@dataclass
class ColorProgram:
    """Represents a color transformation program"""
    operation: str  # Type of color operation
    parameters: Dict[str, Any]  # Operation-specific parameters
    description: str  # Human-readable description
    confidence: float  # Confidence in this program


class IRISProgramSynthesizer:
    """Program synthesis specifically for IRIS color patterns"""
    
    def __init__(self):
        # Color-specific operations
        self.color_operations = {
            'replace_color': self._replace_color,
            'swap_colors': self._swap_colors,
            'color_gradient': self._color_gradient,
            'color_propagate': self._color_propagate,
            'color_boundary': self._color_boundary,
            'color_fill': self._color_fill,
            'color_invert': self._color_invert,
            'color_pattern': self._apply_color_pattern,
            'color_mask': self._apply_color_mask,
            'color_blend': self._color_blend
        }
        
        # Pattern recognition templates
        self.color_patterns = {
            'solid': self._detect_solid,
            'stripes': self._detect_stripes,
            'checkerboard': self._detect_checkerboard,
            'gradient': self._detect_gradient,
            'regions': self._detect_regions,
            'boundaries': self._detect_boundaries
        }
        
        # Program library for successful transformations
        self.program_library = defaultdict(list)
        self.success_count = defaultdict(int)
        
    def synthesize_program(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> List[ColorProgram]:
        """Synthesize color transformation programs from input-output examples"""
        programs = []
        
        # Convert to indices if needed
        if input_grid.dim() == 4:
            input_grid = input_grid.argmax(dim=1).squeeze(0)
        if output_grid.dim() == 4:
            output_grid = output_grid.argmax(dim=1).squeeze(0)
        
        # Analyze color changes
        color_analysis = self._analyze_color_transformation(input_grid, output_grid)
        
        # Try different synthesis strategies
        # 1. Direct color mapping
        if color_analysis['is_color_mapping']:
            program = self._synthesize_color_mapping(input_grid, output_grid, color_analysis)
            if program:
                programs.append(program)
        
        # 2. Pattern-based transformation
        pattern_program = self._synthesize_pattern_transform(input_grid, output_grid)
        if pattern_program:
            programs.append(pattern_program)
        
        # 3. Region-based transformation
        region_program = self._synthesize_region_transform(input_grid, output_grid)
        if region_program:
            programs.append(region_program)
        
        # 4. Boundary-based transformation
        boundary_program = self._synthesize_boundary_transform(input_grid, output_grid)
        if boundary_program:
            programs.append(boundary_program)
        
        # Sort by confidence
        programs.sort(key=lambda p: p.confidence, reverse=True)
        
        return programs
    
    def _analyze_color_transformation(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict:
        """Analyze the color transformation between input and output"""
        analysis = {}
        
        # Get unique colors
        input_colors = torch.unique(input_grid).cpu().numpy()
        output_colors = torch.unique(output_grid).cpu().numpy()
        
        analysis['input_colors'] = input_colors
        analysis['output_colors'] = output_colors
        analysis['num_input_colors'] = len(input_colors)
        analysis['num_output_colors'] = len(output_colors)
        
        # Check for simple color mapping
        if len(input_colors) == len(output_colors):
            # Try to find consistent mapping
            color_map = {}
            is_mapping = True
            
            for in_color in input_colors:
                mask = (input_grid == in_color)
                out_values = output_grid[mask]
                unique_out = torch.unique(out_values)
                
                if len(unique_out) == 1:
                    color_map[int(in_color)] = int(unique_out[0])
                else:
                    is_mapping = False
                    break
            
            analysis['is_color_mapping'] = is_mapping
            analysis['color_map'] = color_map
        else:
            analysis['is_color_mapping'] = False
            analysis['color_map'] = {}
        
        # Analyze spatial patterns
        analysis['has_gradient'] = self._detect_gradient(output_grid) is not None
        analysis['has_regions'] = self._detect_regions(output_grid) is not None
        analysis['has_pattern'] = self._detect_pattern_type(output_grid) is not None
        
        return analysis
    
    def _synthesize_color_mapping(self, input_grid: torch.Tensor, output_grid: torch.Tensor, 
                                 analysis: Dict) -> Optional[ColorProgram]:
        """Synthesize a simple color replacement program"""
        if not analysis['is_color_mapping']:
            return None
        
        color_map = analysis['color_map']
        
        # Verify the mapping works
        test_output = input_grid.clone()
        for in_color, out_color in color_map.items():
            test_output[input_grid == in_color] = out_color
        
        if torch.equal(test_output, output_grid):
            return ColorProgram(
                operation='replace_color',
                parameters={'color_map': color_map},
                description=f"Replace colors: {color_map}",
                confidence=1.0
            )
        
        return None
    
    def _synthesize_pattern_transform(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Optional[ColorProgram]:
        """Synthesize pattern-based color transformations"""
        # Detect output pattern type
        pattern_type = self._detect_pattern_type(output_grid)
        
        if pattern_type == 'stripes':
            direction = self._detect_stripe_direction(output_grid)
            colors = self._extract_stripe_colors(output_grid, direction)
            
            return ColorProgram(
                operation='color_pattern',
                parameters={
                    'pattern': 'stripes',
                    'direction': direction,
                    'colors': colors
                },
                description=f"Apply {direction} stripes with colors {colors}",
                confidence=0.9
            )
        
        elif pattern_type == 'checkerboard':
            colors = self._extract_checker_colors(output_grid)
            
            return ColorProgram(
                operation='color_pattern',
                parameters={
                    'pattern': 'checkerboard',
                    'colors': colors
                },
                description=f"Apply checkerboard pattern with colors {colors}",
                confidence=0.9
            )
        
        elif pattern_type == 'gradient':
            gradient_info = self._extract_gradient_info(output_grid)
            
            return ColorProgram(
                operation='color_gradient',
                parameters=gradient_info,
                description=f"Apply color gradient",
                confidence=0.85
            )
        
        return None
    
    def _synthesize_region_transform(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Optional[ColorProgram]:
        """Synthesize region-based color transformations"""
        # Detect regions in output
        regions = self._detect_color_regions(output_grid)
        
        if not regions:
            return None
        
        # Check if regions correspond to input features
        input_regions = self._detect_color_regions(input_grid)
        
        if len(regions) == len(input_regions):
            # Try to map input regions to output regions
            region_map = self._match_regions(input_regions, regions)
            
            if region_map:
                return ColorProgram(
                    operation='color_fill',
                    parameters={
                        'region_map': region_map,
                        'fill_type': 'region'
                    },
                    description="Fill regions with specific colors",
                    confidence=0.8
                )
        
        return None
    
    def _synthesize_boundary_transform(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Optional[ColorProgram]:
        """Synthesize boundary-based color transformations"""
        # Detect if output highlights boundaries
        boundaries = self._detect_boundaries(output_grid)
        
        if boundaries is not None:
            # Check what type of boundaries are highlighted
            boundary_colors = torch.unique(output_grid[boundaries])
            
            if len(boundary_colors) <= 2:  # Simple boundary coloring
                return ColorProgram(
                    operation='color_boundary',
                    parameters={
                        'boundary_color': int(boundary_colors[-1]),
                        'background_color': int(boundary_colors[0]) if len(boundary_colors) > 1 else 0
                    },
                    description="Color object boundaries",
                    confidence=0.85
                )
        
        return None
    
    def execute_program(self, program: ColorProgram, input_grid: torch.Tensor) -> torch.Tensor:
        """Execute a color transformation program"""
        if program.operation not in self.color_operations:
            raise ValueError(f"Unknown operation: {program.operation}")
        
        # Ensure 2D grid
        if input_grid.dim() == 4:
            input_grid = input_grid.argmax(dim=1).squeeze(0)
        elif input_grid.dim() == 3:
            input_grid = input_grid.squeeze(0)
        
        # Execute the operation
        output = self.color_operations[program.operation](input_grid, program.parameters)
        
        return output
    
    # Color transformation operations
    def _replace_color(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Replace colors according to mapping"""
        output = grid.clone()
        color_map = params['color_map']
        
        for old_color, new_color in color_map.items():
            output[grid == old_color] = new_color
        
        return output
    
    def _swap_colors(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Swap two colors"""
        output = grid.clone()
        color1 = params['color1']
        color2 = params['color2']
        
        mask1 = (grid == color1)
        mask2 = (grid == color2)
        
        output[mask1] = color2
        output[mask2] = color1
        
        return output
    
    def _color_gradient(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Apply color gradient"""
        output = torch.zeros_like(grid)
        direction = params.get('direction', 'horizontal')
        colors = params.get('colors', [0, 1, 2, 3, 4])
        
        h, w = grid.shape
        
        if direction == 'horizontal':
            for i in range(h):
                color_idx = int((i / h) * len(colors))
                color_idx = min(color_idx, len(colors) - 1)
                output[i, :] = colors[color_idx]
        
        elif direction == 'vertical':
            for j in range(w):
                color_idx = int((j / w) * len(colors))
                color_idx = min(color_idx, len(colors) - 1)
                output[:, j] = colors[color_idx]
        
        elif direction == 'diagonal':
            for i in range(h):
                for j in range(w):
                    progress = (i + j) / (h + w - 2)
                    color_idx = int(progress * len(colors))
                    color_idx = min(color_idx, len(colors) - 1)
                    output[i, j] = colors[color_idx]
        
        return output
    
    def _color_propagate(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Propagate color from seed points"""
        output = grid.clone()
        color = params['color']
        direction = params.get('direction', 'all')
        
        # Find seed points (non-zero in original)
        seeds = (grid == color).nonzero(as_tuple=False)
        
        if direction == 'horizontal':
            for seed in seeds:
                output[seed[0], :] = color
        elif direction == 'vertical':
            for seed in seeds:
                output[:, seed[1]] = color
        elif direction == 'all':
            # Flood fill from seeds
            output = self._flood_fill_multi(grid, seeds, color)
        
        return output
    
    def _color_boundary(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Color boundaries between different regions"""
        output = grid.clone()
        boundary_color = params['boundary_color']
        
        # Detect boundaries
        h, w = grid.shape
        for i in range(h):
            for j in range(w):
                # Check neighbors
                is_boundary = False
                current = grid[i, j]
                
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if grid[ni, nj] != current:
                            is_boundary = True
                            break
                
                if is_boundary:
                    output[i, j] = boundary_color
        
        return output
    
    def _color_fill(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Fill regions with colors"""
        output = grid.clone()
        fill_type = params.get('fill_type', 'flood')
        
        if fill_type == 'flood':
            seed = params['seed']
            color = params['color']
            output = self._flood_fill(output, seed, color)
        
        elif fill_type == 'region':
            region_map = params['region_map']
            for region_id, color in region_map.items():
                output[grid == region_id] = color
        
        return output
    
    def _color_invert(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Invert colors based on a mapping"""
        output = grid.clone()
        max_color = params.get('max_color', 9)
        
        # Simple inversion
        output = max_color - grid
        output = torch.clamp(output, 0, max_color)
        
        return output
    
    def _apply_color_pattern(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Apply a color pattern"""
        pattern = params['pattern']
        
        if pattern == 'stripes':
            return self._create_stripes(grid.shape, params)
        elif pattern == 'checkerboard':
            return self._create_checkerboard(grid.shape, params)
        elif pattern == 'rings':
            return self._create_rings(grid.shape, params)
        
        return grid
    
    def _apply_color_mask(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Apply color based on a mask"""
        output = grid.clone()
        mask = params['mask']
        color = params['color']
        
        output[mask] = color
        
        return output
    
    def _color_blend(self, grid: torch.Tensor, params: Dict) -> torch.Tensor:
        """Blend colors in regions"""
        output = grid.clone()
        blend_type = params.get('blend_type', 'average')
        
        if blend_type == 'average':
            # Simple averaging in 3x3 neighborhoods
            kernel = torch.ones(3, 3) / 9.0
            if grid.is_cuda:
                kernel = kernel.cuda()
            
            grid_float = grid.float().unsqueeze(0).unsqueeze(0)
            blended = F.conv2d(grid_float, kernel.unsqueeze(0).unsqueeze(0), padding=1)
            output = blended.squeeze().round().long()
            output = torch.clamp(output, 0, 9)
        
        return output
    
    # Pattern detection helpers
    def _detect_solid(self, grid: torch.Tensor) -> Optional[int]:
        """Detect if grid is solid color"""
        unique_colors = torch.unique(grid)
        if len(unique_colors) == 1:
            return int(unique_colors[0])
        return None
    
    def _detect_stripes(self, grid: torch.Tensor) -> Optional[str]:
        """Detect stripe pattern"""
        h, w = grid.shape
        
        # Check horizontal stripes
        is_h_stripes = True
        for i in range(h):
            if not torch.all(grid[i, :] == grid[i, 0]):
                is_h_stripes = False
                break
        
        if is_h_stripes:
            return 'horizontal'
        
        # Check vertical stripes
        is_v_stripes = True
        for j in range(w):
            if not torch.all(grid[:, j] == grid[0, j]):
                is_v_stripes = False
                break
        
        if is_v_stripes:
            return 'vertical'
        
        return None
    
    def _detect_checkerboard(self, grid: torch.Tensor) -> Optional[bool]:
        """Detect checkerboard pattern"""
        h, w = grid.shape
        
        # Check if alternating pattern
        is_checker = True
        for i in range(h-1):
            for j in range(w-1):
                if grid[i,j] == grid[i+1,j] or grid[i,j] == grid[i,j+1]:
                    if grid[i,j] != grid[i+1,j+1]:
                        is_checker = False
                        break
            if not is_checker:
                break
        
        return is_checker
    
    def _detect_gradient(self, grid: torch.Tensor) -> Optional[str]:
        """Detect gradient pattern"""
        h, w = grid.shape
        
        # Check horizontal gradient
        h_gradient = True
        for i in range(1, h):
            if not torch.all(grid[i, :] >= grid[i-1, :]):
                h_gradient = False
                break
        
        if h_gradient:
            return 'horizontal'
        
        # Check vertical gradient
        v_gradient = True
        for j in range(1, w):
            if not torch.all(grid[:, j] >= grid[:, j-1]):
                v_gradient = False
                break
        
        if v_gradient:
            return 'vertical'
        
        return None
    
    def _detect_regions(self, grid: torch.Tensor) -> Optional[Dict]:
        """Detect color regions"""
        unique_colors = torch.unique(grid)
        if len(unique_colors) <= 1:
            return None
        
        regions = {}
        for color in unique_colors:
            mask = (grid == color)
            if mask.sum() > 0:
                regions[int(color)] = mask
        
        return regions
    
    def _detect_boundaries(self, grid: torch.Tensor) -> Optional[torch.Tensor]:
        """Detect boundaries between different colors"""
        h, w = grid.shape
        boundaries = torch.zeros_like(grid, dtype=torch.bool)
        
        for i in range(h):
            for j in range(w):
                current = grid[i, j]
                
                # Check 4-neighbors
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if grid[ni, nj] != current:
                            boundaries[i, j] = True
                            break
        
        if boundaries.sum() > 0:
            return boundaries
        return None
    
    def _detect_pattern_type(self, grid: torch.Tensor) -> Optional[str]:
        """Detect the type of pattern in the grid"""
        if self._detect_stripes(grid):
            return 'stripes'
        elif self._detect_checkerboard(grid):
            return 'checkerboard'
        elif self._detect_gradient(grid):
            return 'gradient'
        return None
    
    def _detect_stripe_direction(self, grid: torch.Tensor) -> str:
        """Detect stripe direction"""
        stripe_type = self._detect_stripes(grid)
        return stripe_type if stripe_type else 'none'
    
    def _extract_stripe_colors(self, grid: torch.Tensor, direction: str) -> List[int]:
        """Extract colors from stripe pattern"""
        if direction == 'horizontal':
            colors = [int(grid[i, 0]) for i in range(grid.shape[0])]
        elif direction == 'vertical':
            colors = [int(grid[0, j]) for j in range(grid.shape[1])]
        else:
            colors = []
        
        # Remove duplicates while preserving order
        seen = set()
        unique_colors = []
        for c in colors:
            if c not in seen:
                seen.add(c)
                unique_colors.append(c)
        
        return unique_colors
    
    def _extract_checker_colors(self, grid: torch.Tensor) -> Tuple[int, int]:
        """Extract two colors from checkerboard pattern"""
        color1 = int(grid[0, 0])
        color2 = int(grid[0, 1]) if grid[0, 1] != color1 else int(grid[1, 0])
        return (color1, color2)
    
    def _extract_gradient_info(self, grid: torch.Tensor) -> Dict:
        """Extract gradient information"""
        unique_colors = torch.unique(grid).cpu().numpy().tolist()
        
        # Determine direction
        direction = self._detect_gradient(grid)
        
        return {
            'direction': direction if direction else 'horizontal',
            'colors': unique_colors
        }
    
    def _detect_color_regions(self, grid: torch.Tensor) -> List[torch.Tensor]:
        """Detect connected color regions"""
        regions = []
        visited = torch.zeros_like(grid, dtype=torch.bool)
        
        h, w = grid.shape
        
        for i in range(h):
            for j in range(w):
                if not visited[i, j]:
                    # Start flood fill from this point
                    region = torch.zeros_like(grid, dtype=torch.bool)
                    color = grid[i, j]
                    
                    # DFS to find connected region
                    stack = [(i, j)]
                    while stack:
                        ci, cj = stack.pop()
                        if visited[ci, cj]:
                            continue
                        
                        if grid[ci, cj] == color:
                            visited[ci, cj] = True
                            region[ci, cj] = True
                            
                            # Add neighbors
                            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                                ni, nj = ci + di, cj + dj
                                if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj]:
                                    stack.append((ni, nj))
                    
                    if region.sum() > 0:
                        regions.append(region)
        
        return regions
    
    def _match_regions(self, input_regions: List[torch.Tensor], 
                      output_regions: List[torch.Tensor]) -> Optional[Dict]:
        """Match input regions to output regions"""
        if len(input_regions) != len(output_regions):
            return None
        
        # Simple matching based on overlap
        region_map = {}
        
        for i, in_region in enumerate(input_regions):
            best_match = -1
            best_overlap = 0
            
            for j, out_region in enumerate(output_regions):
                overlap = (in_region & out_region).sum().item()
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = j
            
            if best_match >= 0:
                region_map[i] = best_match
        
        return region_map if len(region_map) == len(input_regions) else None
    
    def _flood_fill(self, grid: torch.Tensor, seed: Tuple[int, int], color: int) -> torch.Tensor:
        """Flood fill from seed point"""
        output = grid.clone()
        h, w = grid.shape
        
        if not (0 <= seed[0] < h and 0 <= seed[1] < w):
            return output
        
        target_color = grid[seed[0], seed[1]]
        if target_color == color:
            return output
        
        stack = [seed]
        while stack:
            y, x = stack.pop()
            
            if output[y, x] != target_color:
                continue
            
            output[y, x] = color
            
            # Add neighbors
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and output[ny, nx] == target_color:
                    stack.append((ny, nx))
        
        return output
    
    def _flood_fill_multi(self, grid: torch.Tensor, seeds: torch.Tensor, color: int) -> torch.Tensor:
        """Flood fill from multiple seed points"""
        output = grid.clone()
        
        for seed in seeds:
            output = self._flood_fill(output, tuple(seed.tolist()), color)
        
        return output
    
    def _create_stripes(self, shape: Tuple[int, int], params: Dict) -> torch.Tensor:
        """Create stripe pattern"""
        h, w = shape
        output = torch.zeros((h, w), dtype=torch.long)
        
        direction = params.get('direction', 'horizontal')
        colors = params.get('colors', [0, 1])
        
        if direction == 'horizontal':
            for i in range(h):
                output[i, :] = colors[i % len(colors)]
        else:  # vertical
            for j in range(w):
                output[:, j] = colors[j % len(colors)]
        
        return output
    
    def _create_checkerboard(self, shape: Tuple[int, int], params: Dict) -> torch.Tensor:
        """Create checkerboard pattern"""
        h, w = shape
        output = torch.zeros((h, w), dtype=torch.long)
        
        colors = params.get('colors', (0, 1))
        
        for i in range(h):
            for j in range(w):
                output[i, j] = colors[(i + j) % 2]
        
        return output
    
    def _create_rings(self, shape: Tuple[int, int], params: Dict) -> torch.Tensor:
        """Create concentric rings pattern"""
        h, w = shape
        output = torch.zeros((h, w), dtype=torch.long)
        
        center_y, center_x = h // 2, w // 2
        colors = params.get('colors', [0, 1, 2])
        
        for i in range(h):
            for j in range(w):
                dist = int(np.sqrt((i - center_y)**2 + (j - center_x)**2))
                output[i, j] = colors[dist % len(colors)]
        
        return output


def create_iris_synthesis_system() -> Dict:
    """Create IRIS-specific synthesis components"""
    synthesizer = IRISProgramSynthesizer()
    
    return {
        'synthesizer': synthesizer,
        'program_library': synthesizer.program_library
    }