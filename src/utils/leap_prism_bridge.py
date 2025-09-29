"""
LEAP-PRISM Bridge: Enhances LEAP pattern detection using PRISM's meta-programs
Helps LEAP patterns achieve better success rates by leveraging program synthesis insights
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging

class LEAPPRISMBridge:
    """Bridges LEAP pattern detection with PRISM synthesis insights"""
    
    def __init__(self, leap_system=None, prism_synthesizer=None):
        self.leap_system = leap_system
        self.prism_synthesizer = prism_synthesizer
        self.pattern_synthesis_map = defaultdict(list)
        self.successful_patterns = defaultdict(int)
        self.pattern_meta_program_affinity = defaultdict(lambda: defaultdict(float))
        
    def analyze_failed_leap_pattern(self, 
                                   pattern_type: str,
                                   input_grid: np.ndarray,
                                   expected_output: np.ndarray,
                                   leap_output: Optional[np.ndarray] = None) -> Dict:
        """Analyze why a LEAP pattern failed and suggest improvements"""
        
        analysis = {
            'pattern_type': pattern_type,
            'failure_reasons': [],
            'suggested_meta_programs': [],
            'synthesis_hint': None
        }
        
        # Try PRISM synthesis to understand the transformation
        if self.prism_synthesizer:
            try:
                program = self.prism_synthesizer.synthesize(
                    input_grid, expected_output, time_limit=2.0  # Give more time for synthesis
                )
                if program:
                    # Record successful synthesis
                    self.pattern_synthesis_map[pattern_type].append(program.meta_program)
                    analysis['synthesis_hint'] = program.meta_program
                    analysis['suggested_meta_programs'] = [program.meta_program]
                    
                    # Update affinity scores
                    self.pattern_meta_program_affinity[pattern_type][str(program.meta_program)] += 1.0
            except:
                pass
        
        # Analyze failure reasons
        if leap_output is not None:
            # Check common failure modes
            if np.array_equal(leap_output, input_grid):
                analysis['failure_reasons'].append('identity_transformation')
            elif np.all(leap_output == 0):
                analysis['failure_reasons'].append('null_output')
            elif leap_output.shape != expected_output.shape:
                analysis['failure_reasons'].append('shape_mismatch')
            
            # Check pattern-specific issues
            if pattern_type == 'symmetry' and not self._has_symmetry(expected_output):
                analysis['failure_reasons'].append('no_symmetry_in_output')
            elif pattern_type == 'repetition' and not self._has_repetition(expected_output):
                analysis['failure_reasons'].append('no_repetition_in_output')
        
        return analysis
    
    def enhance_leap_pattern(self, pattern_type: str, pattern_model: nn.Module) -> nn.Module:
        """Enhance a LEAP pattern model with PRISM insights"""
        
        # Get most successful meta-programs for this pattern
        best_meta_programs = sorted(
            self.pattern_meta_program_affinity[pattern_type].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if best_meta_programs:
            # Create enhanced model with meta-program guidance
            return PRISMGuidedPatternModel(
                base_model=pattern_model,
                meta_programs=best_meta_programs,
                pattern_type=pattern_type
            )
        
        return pattern_model
    
    def suggest_pattern_improvements(self, pattern_stats: Dict) -> List[Dict]:
        """Suggest improvements based on pattern performance statistics"""
        
        suggestions = []
        
        for pattern_type, stats in pattern_stats.items():
            if stats.get('success_rate', 0) < 0.1:  # Less than 10% success
                suggestion = {
                    'pattern_type': pattern_type,
                    'current_success_rate': stats.get('success_rate', 0),
                    'recommendations': []
                }
                
                # Get affinity scores
                affinities = self.pattern_meta_program_affinity[pattern_type]
                
                if not affinities:
                    suggestion['recommendations'].append(
                        f"No PRISM synthesis data available. Try running synthesis on {pattern_type} examples."
                    )
                else:
                    top_meta = max(affinities.items(), key=lambda x: x[1])
                    suggestion['recommendations'].append(
                        f"Align {pattern_type} with {top_meta[0]} meta-program (affinity: {top_meta[1]:.2f})"
                    )
                
                # Pattern-specific recommendations
                if pattern_type == 'rotation':
                    suggestion['recommendations'].extend([
                        "Add explicit rotation angle detection",
                        "Consider OBJECT_TRANSFORM meta-program for isolated objects"
                    ])
                elif pattern_type == 'symmetry':
                    suggestion['recommendations'].extend([
                        "Use SYMMETRY_COMPLETE meta-program",
                        "Add axis detection (horizontal/vertical/diagonal)"
                    ])
                elif pattern_type == 'repetition':
                    suggestion['recommendations'].extend([
                        "Use PATTERN_PROPAGATE meta-program",
                        "Detect pattern size before propagation"
                    ])
                elif pattern_type == 'progression':
                    suggestion['recommendations'].extend([
                        "Use HIERARCHICAL_COMPOSE for step-wise transformations",
                        "Track intermediate states"
                    ])
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _has_symmetry(self, grid: np.ndarray) -> bool:
        """Check if grid has any symmetry"""
        h, w = grid.shape
        
        # Horizontal symmetry
        if np.array_equal(grid[:, :w//2], np.fliplr(grid[:, w//2:w//2*2])):
            return True
        
        # Vertical symmetry
        if np.array_equal(grid[:h//2, :], np.flipud(grid[h//2:h//2*2, :])):
            return True
        
        return False
    
    def _has_repetition(self, grid: np.ndarray) -> bool:
        """Check if grid has repeating patterns"""
        h, w = grid.shape
        
        for size in [2, 3, 4]:
            if h >= size * 2 and w >= size * 2:
                pattern = grid[:size, :size]
                # Check tiling
                matches = 0
                total = 0
                for i in range(0, h-size+1, size):
                    for j in range(0, w-size+1, size):
                        total += 1
                        if np.array_equal(grid[i:i+size, j:j+size], pattern):
                            matches += 1
                if total > 0 and matches / total > 0.7:  # 70% match threshold
                    return True
        
        return False


class PRISMGuidedPatternModel(nn.Module):
    """Enhanced pattern model guided by PRISM meta-programs"""
    
    def __init__(self, base_model: nn.Module, meta_programs: List[Tuple[str, float]], pattern_type: str):
        super().__init__()
        self.base_model = base_model
        self.meta_programs = meta_programs
        self.pattern_type = pattern_type
        
        # Add meta-program embedding
        self.meta_embedding = nn.Embedding(8, 64)  # 8 meta-programs, 64-dim embedding
        self.meta_attention = nn.MultiheadAttention(64, 4, batch_first=True)
        
        # Meta-program to index mapping
        self.meta_to_idx = {
            'MetaProgram.RECURSIVE_DECOMPOSE': 0,
            'MetaProgram.SYMMETRY_COMPLETE': 1,
            'MetaProgram.OBJECT_TRANSFORM': 2,
            'MetaProgram.PATTERN_PROPAGATE': 3,
            'MetaProgram.CONSTRAINT_SOLVE': 4,
            'MetaProgram.ANALOGY_APPLY': 5,
            'MetaProgram.INVARIANT_PRESERVE': 6,
            'MetaProgram.HIERARCHICAL_COMPOSE': 7
        }
        
    def forward(self, x, *args, **kwargs):
        # Get base model output
        base_output = self.base_model(x, *args, **kwargs)
        
        # Apply meta-program guidance if we have embeddings
        if hasattr(base_output, 'shape') and len(base_output.shape) >= 3:
            batch_size = base_output.shape[0]
            
            # Get relevant meta-program embeddings
            meta_indices = []
            meta_weights = []
            for meta_name, weight in self.meta_programs:
                if meta_name in self.meta_to_idx:
                    meta_indices.append(self.meta_to_idx[meta_name])
                    meta_weights.append(weight)
            
            if meta_indices:
                # Create weighted embeddings
                indices_tensor = torch.tensor(meta_indices, device=x.device)
                weights_tensor = torch.tensor(meta_weights, device=x.device).unsqueeze(1)
                
                meta_embeds = self.meta_embedding(indices_tensor)  # [num_meta, 64]
                weighted_embeds = meta_embeds * weights_tensor  # Weight by affinity
                
                # Expand for batch
                meta_embeds_batch = weighted_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Apply attention if base_output has compatible shape
                if base_output.shape[-1] == 64:  # Check feature dimension
                    enhanced_output, _ = self.meta_attention(
                        base_output, meta_embeds_batch, meta_embeds_batch
                    )
                    return enhanced_output
        
        return base_output


def create_leap_prism_bridge(leap_system=None, prism_synthesizer=None):
    """Create a LEAP-PRISM bridge instance"""
    return LEAPPRISMBridge(leap_system, prism_synthesizer)


class LEAPPatternEnhancer:
    """Specific enhancements for each LEAP pattern type"""
    
    @staticmethod
    def enhance_symmetry_pattern(model: nn.Module) -> nn.Module:
        """Add symmetry-specific enhancements"""
        class SymmetryEnhancedModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.axis_detector = nn.Sequential(
                    nn.Conv2d(10, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 3, 1),  # 3 axes: horizontal, vertical, diagonal
                    nn.Softmax(dim=1)
                )
                
            def forward(self, x, *args, **kwargs):
                # Detect symmetry axis
                if len(x.shape) == 4:  # Batch processing
                    axis_probs = self.axis_detector(x)
                    axis_probs_pooled = axis_probs.mean(dim=[2, 3])  # Global axis probability
                    
                    # Apply base model
                    output = self.base_model(x, *args, **kwargs)
                    
                    # Apply symmetry enforcement based on detected axis
                    if hasattr(output, 'shape') and output.shape == x.shape:
                        batch_size, channels, h, w = output.shape
                        for b in range(batch_size):
                            axis = axis_probs_pooled[b].argmax()
                            if axis == 0:  # Horizontal symmetry
                                output[b, :, :, w//2:] = torch.flip(output[b, :, :, :w//2], dims=[2])
                            elif axis == 1:  # Vertical symmetry
                                output[b, :, h//2:, :] = torch.flip(output[b, :, :h//2, :], dims=[1])
                    
                    return output
                    
                return self.base_model(x, *args, **kwargs)
        
        return SymmetryEnhancedModel(model)
    
    @staticmethod
    def enhance_rotation_pattern(model: nn.Module) -> nn.Module:
        """Add rotation-specific enhancements"""
        class RotationEnhancedModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.angle_detector = nn.Sequential(
                    nn.AdaptiveAvgPool2d(8),
                    nn.Flatten(),
                    nn.Linear(8 * 8 * 10, 128),
                    nn.ReLU(),
                    nn.Linear(128, 4),  # 0°, 90°, 180°, 270°
                    nn.Softmax(dim=1)
                )
                
            def forward(self, x, *args, **kwargs):
                if len(x.shape) == 4:
                    # Detect rotation angle
                    angle_probs = self.angle_detector(x)
                    
                    # Apply base model
                    output = self.base_model(x, *args, **kwargs)
                    
                    # Apply rotation based on detected angle
                    if hasattr(output, 'shape') and output.shape == x.shape:
                        batch_size = output.shape[0]
                        for b in range(batch_size):
                            angle_idx = angle_probs[b].argmax()
                            if angle_idx > 0:  # Apply rotation
                                output[b] = torch.rot90(output[b], k=angle_idx, dims=[1, 2])
                    
                    return output
                    
                return self.base_model(x, *args, **kwargs)
        
        return RotationEnhancedModel(model)
    
    @staticmethod
    def enhance_repetition_pattern(model: nn.Module) -> nn.Module:
        """Add repetition-specific enhancements"""
        class RepetitionEnhancedModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.pattern_size_detector = nn.Sequential(
                    nn.Conv2d(10, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(16, 4),  # Pattern sizes: 2x2, 3x3, 4x4, 5x5
                    nn.Softmax(dim=1)
                )
                
            def forward(self, x, *args, **kwargs):
                if len(x.shape) == 4:
                    # Detect pattern size
                    size_probs = self.pattern_size_detector(x)
                    
                    # Apply base model
                    output = self.base_model(x, *args, **kwargs)
                    
                    # Apply tiling based on detected size
                    if hasattr(output, 'shape') and output.shape == x.shape:
                        batch_size, channels, h, w = output.shape
                        for b in range(batch_size):
                            size_idx = size_probs[b].argmax()
                            pattern_size = size_idx + 2  # 2, 3, 4, or 5
                            
                            # Extract and tile pattern
                            if pattern_size < min(h, w):
                                pattern = output[b, :, :pattern_size, :pattern_size]
                                for i in range(0, h, pattern_size):
                                    for j in range(0, w, pattern_size):
                                        end_i = min(i + pattern_size, h)
                                        end_j = min(j + pattern_size, w)
                                        output[b, :, i:end_i, j:end_j] = pattern[:, :end_i-i, :end_j-j]
                    
                    return output
                    
                return self.base_model(x, *args, **kwargs)
        
        return RepetitionEnhancedModel(model)