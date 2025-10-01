"""
IRIS-specific PRISM (Program Reasoning through Inductive Synthesis) System
Focuses on color transformation rules and perceptual color reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import itertools


class IrisColorTransformation:
    """Represents a color transformation rule for IRIS"""
    
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.confidence = 1.0
    
    def apply(self, color: int) -> int:
        """Apply transformation to a single color"""
        if self.name == "shift":
            return (color + self.params['offset']) % 10
        elif self.name == "invert":
            return 9 - color
        elif self.name == "map":
            mapping = self.params.get('mapping', {})
            return mapping.get(color, color)
        elif self.name == "threshold":
            threshold = self.params.get('threshold', 5)
            return 0 if color < threshold else 9
        elif self.name == "modulo":
            mod = self.params.get('mod', 3)
            return color % mod
        elif self.name == "scale":
            factor = self.params.get('factor', 2)
            return min(9, color * factor)
        elif self.name == "mix":
            # Mix with another color
            other = self.params.get('other_color', 0)
            return (color + other) // 2
        else:
            return color
    
    def to_string(self) -> str:
        """String representation of transformation"""
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({param_str})"


class IrisProgram:
    """Represents a synthesized program for IRIS color transformations"""
    
    def __init__(self, transformations: List[IrisColorTransformation]):
        self.transformations = transformations
        self.confidence = 1.0
        self.complexity = len(transformations)
    
    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """Execute program on input grid"""
        output = input_grid.copy()
        
        for transform in self.transformations:
            if transform.name in ['gradient', 'wave', 'radial']:
                # Spatial color transformations
                output = self._apply_spatial_transform(output, transform)
            else:
                # Per-pixel color transformations
                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        output[i, j] = transform.apply(output[i, j])
        
        return output
    
    def _apply_spatial_transform(self, grid: np.ndarray, transform: IrisColorTransformation) -> np.ndarray:
        """Apply spatial color transformations"""
        h, w = grid.shape
        output = grid.copy()
        
        if transform.name == 'gradient':
            # Apply color gradient
            direction = transform.params.get('direction', 'horizontal')
            start_color = transform.params.get('start', 0)
            end_color = transform.params.get('end', 9)
            
            if direction == 'horizontal':
                for j in range(w):
                    color = start_color + (end_color - start_color) * j // max(1, w - 1)
                    output[:, j] = color
            else:
                for i in range(h):
                    color = start_color + (end_color - start_color) * i // max(1, h - 1)
                    output[i, :] = color
        
        elif transform.name == 'wave':
            # Apply wave pattern
            frequency = transform.params.get('frequency', 1)
            amplitude = transform.params.get('amplitude', 3)
            
            for i in range(h):
                for j in range(w):
                    wave = np.sin(i * frequency * np.pi / h) + np.sin(j * frequency * np.pi / w)
                    color = int((wave + 2) * amplitude) % 10
                    output[i, j] = color
        
        elif transform.name == 'radial':
            # Apply radial gradient
            center_h, center_w = h // 2, w // 2
            max_dist = np.sqrt(center_h**2 + center_w**2)
            
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                    color = int(dist / max_dist * 9)
                    output[i, j] = color
        
        return output
    
    def to_string(self) -> str:
        """String representation of program"""
        return " -> ".join(t.to_string() for t in self.transformations)


class IrisProgramSynthesizer:
    """Synthesizes color transformation programs for IRIS"""
    
    def __init__(self):
        self.transformation_library = self._build_transformation_library()
        self.synthesis_cache = {}
        self.color_analysis_cache = {}
    
    def _build_transformation_library(self) -> List[IrisColorTransformation]:
        """Build library of color transformations"""
        library = []
        
        # Basic color shifts
        for offset in range(-5, 6):
            if offset != 0:
                library.append(IrisColorTransformation("shift", {"offset": offset}))
        
        # Color inversion
        library.append(IrisColorTransformation("invert", {}))
        
        # Threshold operations
        for threshold in [2, 3, 4, 5, 6, 7]:
            library.append(IrisColorTransformation("threshold", {"threshold": threshold}))
        
        # Modulo operations
        for mod in [2, 3, 4, 5]:
            library.append(IrisColorTransformation("modulo", {"mod": mod}))
        
        # Scaling operations
        for factor in [2, 3]:
            library.append(IrisColorTransformation("scale", {"factor": factor}))
        
        # Gradient operations
        for direction in ['horizontal', 'vertical']:
            for start in [0, 1, 2]:
                for end in [7, 8, 9]:
                    library.append(IrisColorTransformation("gradient", {
                        "direction": direction,
                        "start": start,
                        "end": end
                    }))
        
        # Wave patterns
        for freq in [1, 2, 3]:
            for amp in [2, 3, 4]:
                library.append(IrisColorTransformation("wave", {
                    "frequency": freq,
                    "amplitude": amp
                }))
        
        # Radial patterns
        library.append(IrisColorTransformation("radial", {}))
        
        return library
    
    def synthesize(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                   max_depth: int = 3) -> Optional[IrisProgram]:
        """Synthesize a color transformation program"""
        # Check cache
        cache_key = (input_grid.tobytes(), output_grid.tobytes())
        if cache_key in self.synthesis_cache:
            return self.synthesis_cache[cache_key]
        
        # Analyze color properties
        input_colors = self._analyze_colors(input_grid)
        output_colors = self._analyze_colors(output_grid)
        
        # Try single transformations first
        for transform in self.transformation_library:
            test_output = self._apply_transform_to_grid(input_grid, transform)
            if np.array_equal(test_output, output_grid):
                program = IrisProgram([transform])
                self.synthesis_cache[cache_key] = program
                return program
        
        # Try combinations of transformations
        if max_depth > 1:
            best_program = self._beam_search_synthesis(
                input_grid, output_grid, max_depth,
                input_colors, output_colors
            )
            if best_program:
                self.synthesis_cache[cache_key] = best_program
                return best_program
        
        # Try to learn custom mapping
        custom_transform = self._learn_color_mapping(input_grid, output_grid)
        if custom_transform:
            program = IrisProgram([custom_transform])
            self.synthesis_cache[cache_key] = program
            return program
        
        return None
    
    def _analyze_colors(self, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze color properties of a grid"""
        flat = grid.flatten()
        unique_colors = np.unique(flat)
        
        # Color histogram
        hist = np.bincount(flat, minlength=10)[:10]
        
        # Color transitions
        h_transitions = np.sum(grid[:, :-1] != grid[:, 1:])
        v_transitions = np.sum(grid[:-1, :] != grid[1:, :])
        
        # Dominant color
        dominant = int(np.argmax(hist))
        
        # Color distribution entropy
        probs = hist / (np.sum(hist) + 1e-8)
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        
        return {
            'unique': unique_colors,
            'histogram': hist,
            'dominant': dominant,
            'transitions': h_transitions + v_transitions,
            'entropy': entropy,
            'has_gradient': self._detect_gradient(grid),
            'has_pattern': self._detect_pattern(grid)
        }
    
    def _detect_gradient(self, grid: np.ndarray) -> bool:
        """Detect if grid has gradient pattern"""
        # Check horizontal gradient
        for row in grid:
            if np.all(row[:-1] <= row[1:]) or np.all(row[:-1] >= row[1:]):
                return True
        
        # Check vertical gradient
        for col in grid.T:
            if np.all(col[:-1] <= col[1:]) or np.all(col[:-1] >= col[1:]):
                return True
        
        return False
    
    def _detect_pattern(self, grid: np.ndarray) -> bool:
        """Detect if grid has repeating pattern"""
        h, w = grid.shape
        
        # Check for 2x2 pattern
        if h >= 4 and w >= 4:
            pattern = grid[:2, :2]
            for i in range(0, h-1, 2):
                for j in range(0, w-1, 2):
                    if not np.array_equal(grid[i:i+2, j:j+2], pattern):
                        return False
            return True
        
        return False
    
    def _apply_transform_to_grid(self, grid: np.ndarray, 
                                transform: IrisColorTransformation) -> np.ndarray:
        """Apply a transformation to entire grid"""
        dummy_program = IrisProgram([transform])
        return dummy_program.execute(grid)
    
    def _beam_search_synthesis(self, input_grid: np.ndarray, output_grid: np.ndarray,
                              max_depth: int, input_colors: Dict, 
                              output_colors: Dict) -> Optional[IrisProgram]:
        """Use beam search to find transformation sequence"""
        beam_size = 10
        beam = [(input_grid, [])]  # (current_grid, transformations)
        
        for depth in range(max_depth):
            new_beam = []
            
            for current_grid, transforms in beam:
                # Skip if already matches
                if np.array_equal(current_grid, output_grid):
                    return IrisProgram(transforms)
                
                # Try each transformation
                for transform in self._get_promising_transforms(
                    current_grid, output_grid, input_colors, output_colors
                ):
                    new_grid = self._apply_transform_to_grid(current_grid, transform)
                    new_transforms = transforms + [transform]
                    
                    # Calculate score
                    score = self._grid_similarity_score(new_grid, output_grid)
                    new_beam.append((score, new_grid, new_transforms))
            
            # Keep top k
            new_beam.sort(key=lambda x: x[0], reverse=True)
            beam = [(grid, transforms) for _, grid, transforms in new_beam[:beam_size]]
            
            # Check if any matches
            for grid, transforms in beam:
                if np.array_equal(grid, output_grid):
                    return IrisProgram(transforms)
        
        return None
    
    def _get_promising_transforms(self, current_grid: np.ndarray, 
                                 target_grid: np.ndarray,
                                 input_colors: Dict, output_colors: Dict) -> List[IrisColorTransformation]:
        """Get promising transformations based on color analysis"""
        transforms = []
        
        current_unique = set(np.unique(current_grid))
        target_unique = set(np.unique(target_grid))
        
        # If gradient detected in target
        if output_colors['has_gradient']:
            transforms.extend([t for t in self.transformation_library 
                             if t.name == 'gradient'])
        
        # If need to reduce colors
        if len(target_unique) < len(current_unique):
            transforms.extend([t for t in self.transformation_library 
                             if t.name in ['threshold', 'modulo']])
        
        # If need color shift
        current_max = max(current_unique)
        target_max = max(target_unique)
        if target_max != current_max:
            offset = target_max - current_max
            transforms.append(IrisColorTransformation("shift", {"offset": offset}))
        
        # Add some general transforms
        transforms.extend(self.transformation_library[:5])
        
        return transforms
    
    def _grid_similarity_score(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Calculate similarity between two grids"""
        if grid1.shape != grid2.shape:
            return 0.0
        
        # Exact match score
        exact_match = np.mean(grid1 == grid2)
        
        # Color distribution similarity
        hist1 = np.bincount(grid1.flatten(), minlength=10)[:10]
        hist2 = np.bincount(grid2.flatten(), minlength=10)[:10]
        
        hist1_norm = hist1 / (np.sum(hist1) + 1e-8)
        hist2_norm = hist2 / (np.sum(hist2) + 1e-8)
        
        hist_similarity = 1.0 - np.sum(np.abs(hist1_norm - hist2_norm)) / 2
        
        # Combined score
        return 0.7 * exact_match + 0.3 * hist_similarity
    
    def _learn_color_mapping(self, input_grid: np.ndarray, 
                            output_grid: np.ndarray) -> Optional[IrisColorTransformation]:
        """Learn a direct color mapping"""
        if input_grid.shape != output_grid.shape:
            return None
        
        # Build mapping
        mapping = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = int(input_grid[i, j])
                out_color = int(output_grid[i, j])
                
                if in_color in mapping and mapping[in_color] != out_color:
                    # Inconsistent mapping
                    return None
                
                mapping[in_color] = out_color
        
        return IrisColorTransformation("map", {"mapping": mapping})
    
    def extract_color_rules(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> List[IrisProgram]:
        """Extract common color transformation rules from examples"""
        programs = []
        
        for input_grid, output_grid in examples:
            program = self.synthesize(input_grid, output_grid)
            if program:
                programs.append(program)
        
        # Find common patterns
        if len(programs) >= 2:
            common_rules = self._find_common_rules(programs)
            return common_rules
        
        return programs
    
    def _find_common_rules(self, programs: List[IrisProgram]) -> List[IrisProgram]:
        """Find common transformation patterns across programs"""
        # Count transformation frequencies
        transform_counts = defaultdict(int)
        
        for program in programs:
            for transform in program.transformations:
                key = (transform.name, tuple(sorted(transform.params.items())))
                transform_counts[key] += 1
        
        # Create programs from common transforms
        common_programs = []
        for (name, params_tuple), count in transform_counts.items():
            if count >= len(programs) * 0.5:  # At least 50% of programs
                params = dict(params_tuple)
                transform = IrisColorTransformation(name, params)
                common_programs.append(IrisProgram([transform]))
        
        return common_programs


class IrisProgramLibrary:
    """Library of successful IRIS color transformation programs"""
    
    def __init__(self, max_programs: int = 5000):
        self.max_programs = max_programs
        self.programs = {}
        self.program_usage = defaultdict(int)
        self.program_success = defaultdict(float)
        self.color_specific_programs = defaultdict(list)
    
    def add_program(self, program: IrisProgram, input_example: np.ndarray, 
                   output_example: np.ndarray, success_score: float = 1.0):
        """Add successful program to library"""
        program_key = program.to_string()
        
        if program_key not in self.programs:
            self.programs[program_key] = {
                'program': program,
                'examples': [],
                'success_rate': success_score,
                'usage_count': 0
            }
        
        # Store example
        self.programs[program_key]['examples'].append({
            'input': input_example,
            'output': output_example
        })
        
        # Update stats
        self.program_usage[program_key] += 1
        self.program_success[program_key] = (
            self.program_success[program_key] * 0.9 + success_score * 0.1
        )
        
        # Categorize by color characteristics
        input_colors = set(np.unique(input_example))
        output_colors = set(np.unique(output_example))
        
        color_key = f"{len(input_colors)}->{len(output_colors)}"
        self.color_specific_programs[color_key].append(program_key)
        
        # Evict if over capacity
        if len(self.programs) > self.max_programs:
            self._evict_least_useful()
    
    def _evict_least_useful(self):
        """Remove least useful programs"""
        # Score programs
        program_scores = {}
        for key, data in self.programs.items():
            usage = self.program_usage.get(key, 1)
            success = self.program_success.get(key, 0.5)
            complexity = 1.0 / (data['program'].complexity + 1)
            
            score = usage * success * complexity
            program_scores[key] = score
        
        # Remove bottom 10%
        sorted_programs = sorted(program_scores.items(), key=lambda x: x[1])
        to_remove = sorted_programs[:len(sorted_programs) // 10]
        
        for key, _ in to_remove:
            del self.programs[key]
            # Clean up references
            for color_key in list(self.color_specific_programs.keys()):
                if key in self.color_specific_programs[color_key]:
                    self.color_specific_programs[color_key].remove(key)
    
    def find_similar_programs(self, input_grid: np.ndarray, 
                             output_grid: np.ndarray, k: int = 5) -> List[IrisProgram]:
        """Find similar programs that might work"""
        if not self.programs:
            return []
        
        # Get color characteristics
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        color_key = f"{len(input_colors)}->{len(output_colors)}"
        
        # Look for programs with similar color transformations
        candidate_keys = self.color_specific_programs.get(color_key, [])
        
        if not candidate_keys:
            # Fallback to all programs
            candidate_keys = list(self.programs.keys())
        
        # Score candidates
        scored_programs = []
        for key in candidate_keys[:50]:  # Limit search
            program_data = self.programs[key]
            program = program_data['program']
            
            # Test on current input
            try:
                test_output = program.execute(input_grid)
                similarity = self._output_similarity(test_output, output_grid)
                scored_programs.append((similarity, program))
            except:
                continue
        
        # Return top k
        scored_programs.sort(key=lambda x: x[0], reverse=True)
        return [prog for _, prog in scored_programs[:k]]
    
    def _output_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Calculate similarity between outputs"""
        if grid1.shape != grid2.shape:
            return 0.0
        
        # Exact match percentage
        exact = np.mean(grid1 == grid2)
        
        # Color distribution similarity
        hist1 = np.bincount(grid1.flatten(), minlength=10)[:10]
        hist2 = np.bincount(grid2.flatten(), minlength=10)[:10]
        
        hist1 = hist1 / (np.sum(hist1) + 1e-8)
        hist2 = hist2 / (np.sum(hist2) + 1e-8)
        
        hist_sim = 1.0 - np.sum(np.abs(hist1 - hist2)) / 2
        
        return 0.8 * exact + 0.2 * hist_sim


def create_iris_prism_system(hidden_dim: int = 256) -> Dict:
    """Create IRIS-specific PRISM components"""
    synthesizer = IrisProgramSynthesizer()
    library = IrisProgramLibrary()
    
    return {
        'synthesizer': synthesizer,
        'library': library,
        'hidden_dim': hidden_dim
    }


class IrisProgramPredictor(nn.Module):
    """Neural network that predicts color transformation programs for IRIS"""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Color analysis encoder
        self.color_encoder = nn.Sequential(
            nn.Linear(20, hidden_dim),  # 10 colors x 2 (input/output histograms)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Spatial pattern encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Program prediction head
        self.program_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),  # Transformation space
            nn.ReLU(),
            nn.Linear(128, 64)  # Program embedding
        )
        
        # Transformation type classifier
        self.transform_classifier = nn.Linear(64, 10)  # 10 transformation types
        
        # Parameter predictors
        self.param_predictor = nn.Linear(64, 20)  # Various parameters
    
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict program for color transformation"""
        B = input_grid.shape[0]
        
        # Color analysis
        input_hist = self._compute_color_histogram(input_grid)
        output_hist = self._compute_color_histogram(output_grid)
        color_features = torch.cat([input_hist, output_hist], dim=1)
        color_encoded = self.color_encoder(color_features)
        
        # Spatial analysis
        spatial_diff = (output_grid.float() - input_grid.float()).unsqueeze(1)
        spatial_features = self.spatial_encoder(spatial_diff).squeeze(-1).squeeze(-1)
        
        # Combine features
        combined = torch.cat([color_encoded, spatial_features], dim=1)
        
        # Predict program
        program_embedding = self.program_head(combined)
        
        # Predict transformation type and parameters
        transform_logits = self.transform_classifier(program_embedding)
        param_values = self.param_predictor(program_embedding)
        
        return {
            'transform_type': transform_logits,
            'parameters': param_values,
            'program_embedding': program_embedding
        }
    
    def _compute_color_histogram(self, grid: torch.Tensor) -> torch.Tensor:
        """Compute color histogram for batch of grids"""
        B = grid.shape[0]
        histograms = []
        
        for b in range(B):
            hist = torch.histc(grid[b].float(), bins=10, min=0, max=9)
            hist = hist / hist.sum()
            histograms.append(hist)
        
        return torch.stack(histograms)