"""
PROMETHEUS-specific DSL (Domain-Specific Language) System
Specialized for meta-learning and ensemble coordination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import defaultdict

# Try to import base DSL system
try:
    from .base_dsl import DSLTraining, DSLPattern
    BASE_DSL_AVAILABLE = True
except ImportError:
    BASE_DSL_AVAILABLE = False


class PrometheusMetaDSLTraining:
    """PROMETHEUS-specific DSL training for meta-learning patterns"""
    
    @staticmethod
    def create_prometheus_dsl_samples(curriculum_stage: int = 0, num_samples: int = 200) -> List[Dict]:
        """Create PROMETHEUS DSL training samples focused on meta-learning and ensemble coordination"""
        
        samples = []
        stage_configs = {
            0: {'size': 6, 'complexity': 0.3},   # 6x6 grids, low complexity
            1: {'size': 8, 'complexity': 0.4},   # 8x8 grids
            2: {'size': 10, 'complexity': 0.5},  # 10x10 grids
            3: {'size': 12, 'complexity': 0.6},  # 12x12 grids
            4: {'size': 15, 'complexity': 0.7},  # 15x15 grids
            5: {'size': 19, 'complexity': 0.8},  # 19x19 grids
            6: {'size': 25, 'complexity': 0.9},  # 25x25 grids
            7: {'size': 30, 'complexity': 1.0},  # 30x30 grids, full complexity
        }
        
        config = stage_configs.get(curriculum_stage, stage_configs[0])
        grid_size = config['size']
        complexity = config['complexity']
        
        # Generate meta-learning specific patterns
        meta_patterns = PrometheusMetaDSLTraining._generate_meta_learning_dsl_patterns(
            num_samples // 3, grid_size, complexity
        )
        samples.extend(meta_patterns)
        
        # Generate ensemble coordination patterns
        ensemble_patterns = PrometheusMetaDSLTraining._generate_ensemble_coordination_patterns(
            num_samples // 3, grid_size, complexity
        )
        samples.extend(ensemble_patterns)
        
        # Generate adaptation patterns
        adaptation_patterns = PrometheusMetaDSLTraining._generate_adaptation_patterns(
            num_samples - len(samples), grid_size, complexity
        )
        samples.extend(adaptation_patterns)
        
        print(f"🔧 Generated {len(samples)} PROMETHEUS DSL samples for stage {curriculum_stage}")
        return samples
    
    @staticmethod
    def _generate_meta_learning_dsl_patterns(num_samples: int, grid_size: int, complexity: float) -> List[Dict]:
        """Generate patterns that require meta-learning capabilities"""
        patterns = []
        
        for i in range(num_samples):
            # Meta-learning pattern types
            pattern_type = random.choice([
                'few_shot_analogy', 'cross_domain_transfer', 'abstract_reasoning',
                'pattern_completion', 'rule_extraction', 'compositional_learning'
            ])
            
            if pattern_type == 'few_shot_analogy':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_analogy_pattern(grid_size, complexity)
            elif pattern_type == 'cross_domain_transfer':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_transfer_pattern(grid_size, complexity)
            elif pattern_type == 'abstract_reasoning':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_abstract_pattern(grid_size, complexity)
            elif pattern_type == 'pattern_completion':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_completion_pattern(grid_size, complexity)
            elif pattern_type == 'rule_extraction':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_rule_pattern(grid_size, complexity)
            else:  # compositional_learning
                input_grid, output_grid = PrometheusMetaDSLTraining._create_compositional_pattern(grid_size, complexity)
            
            patterns.append({
                'inputs': input_grid.astype(np.int64),
                'outputs': output_grid.astype(np.int64),
                'meta_info': {
                    'pattern_type': pattern_type,
                    'requires_meta_learning': True,
                    'complexity': complexity,
                    'stage': f'meta_learning_{pattern_type}'
                }
            })
        
        return patterns
    
    @staticmethod
    def _create_analogy_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create A:B :: C:D analogy patterns"""
        input_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        output_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        
        # Create A:B relationship in top half
        quarter_size = grid_size // 4
        if quarter_size > 0:
            # A pattern (top-left)
            a_pattern = np.random.randint(1, int(3 + complexity * 3), (quarter_size, quarter_size))
            input_grid[:quarter_size, :quarter_size] = a_pattern
            
            # B pattern (top-right) - transformed A
            b_pattern = np.rot90(a_pattern)  # Simple transformation
            input_grid[:quarter_size, quarter_size:2*quarter_size] = b_pattern
            
            # C pattern (bottom-left) - given
            c_pattern = np.random.randint(1, int(3 + complexity * 3), (quarter_size, quarter_size))
            input_grid[2*quarter_size:3*quarter_size, :quarter_size] = c_pattern
            
            # D pattern (bottom-right) - apply same transformation as A->B to C
            d_pattern = np.rot90(c_pattern)
            output_grid[2*quarter_size:3*quarter_size, quarter_size:2*quarter_size] = d_pattern
            
            # Copy known parts to output
            output_grid[:quarter_size, :quarter_size] = a_pattern
            output_grid[:quarter_size, quarter_size:2*quarter_size] = b_pattern
            output_grid[2*quarter_size:3*quarter_size, :quarter_size] = c_pattern
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_transfer_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create patterns requiring knowledge transfer between domains"""
        input_grid = np.random.randint(0, int(2 + complexity * 4), (grid_size, grid_size))
        output_grid = np.zeros_like(input_grid)
        
        # Transfer pattern: spatial relationships preserved but colors change
        color_mapping = {}
        unique_colors = np.unique(input_grid)
        for color in unique_colors:
            if color == 0:
                color_mapping[color] = 0  # Keep background
            else:
                color_mapping[color] = (color + int(complexity * 3)) % 6 + 1
        
        # Apply spatial transformation + color transfer
        transformed = np.flip(input_grid, axis=0)  # Vertical flip
        for old_color, new_color in color_mapping.items():
            output_grid[transformed == old_color] = new_color
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_abstract_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create abstract reasoning patterns"""
        input_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        output_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        
        # Abstract pattern: symmetry + progression
        center = grid_size // 2
        
        for i in range(center):
            for j in range(center):
                if i + j < center * complexity:
                    value = int((i + j) % (3 + int(complexity * 2))) + 1
                    # Create symmetric pattern
                    input_grid[i, j] = value
                    input_grid[i, grid_size-1-j] = value
                    input_grid[grid_size-1-i, j] = value
                    input_grid[grid_size-1-i, grid_size-1-j] = value
                    
                    # Output: increment pattern
                    out_value = (value % (3 + int(complexity * 2))) + 1
                    output_grid[i, j] = out_value
                    output_grid[i, grid_size-1-j] = out_value
                    output_grid[grid_size-1-i, j] = out_value
                    output_grid[grid_size-1-i, grid_size-1-j] = out_value
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_completion_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create pattern completion tasks"""
        # Start with a complete pattern, then mask part for input
        complete_pattern = np.zeros((grid_size, grid_size), dtype=np.int64)
        
        # Create repeating motif
        motif_size = max(2, int(grid_size * 0.2))
        motif = np.random.randint(1, int(3 + complexity * 3), (motif_size, motif_size))
        
        # Tile the motif
        for i in range(0, grid_size, motif_size):
            for j in range(0, grid_size, motif_size):
                end_i = min(i + motif_size, grid_size)
                end_j = min(j + motif_size, grid_size)
                complete_pattern[i:end_i, j:end_j] = motif[:end_i-i, :end_j-j]
        
        # Input: mask some regions
        input_grid = complete_pattern.copy()
        mask_prob = 0.3 + complexity * 0.2
        mask = np.random.random((grid_size, grid_size)) < mask_prob
        input_grid[mask] = 0
        
        output_grid = complete_pattern
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_rule_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create rule extraction patterns"""
        input_grid = np.random.randint(0, int(3 + complexity * 2), (grid_size, grid_size))
        output_grid = np.zeros_like(input_grid)
        
        # Rule: cells become next color if they have specific neighbors
        for i in range(grid_size):
            for j in range(grid_size):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            neighbors.append(input_grid[ni, nj])
                
                # Rule: if majority of neighbors are same color, adopt that color
                if neighbors:
                    neighbor_counts = {}
                    for n in neighbors:
                        neighbor_counts[n] = neighbor_counts.get(n, 0) + 1
                    
                    most_common = max(neighbor_counts.items(), key=lambda x: x[1])
                    if most_common[1] >= len(neighbors) * (0.4 + complexity * 0.2):
                        output_grid[i, j] = most_common[0]
                    else:
                        output_grid[i, j] = input_grid[i, j]
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_compositional_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create compositional learning patterns"""
        input_grid = np.random.randint(0, int(4 + complexity * 2), (grid_size, grid_size))
        output_grid = input_grid.copy()
        
        # Compose multiple simple transformations
        num_transformations = int(1 + complexity * 2)
        
        for _ in range(num_transformations):
            transform = random.choice(['rotate', 'shift', 'color_map', 'local_rule'])
            
            if transform == 'rotate':
                output_grid = np.rot90(output_grid, k=random.randint(1, 3))
            elif transform == 'shift':
                shift = random.randint(1, max(1, grid_size // 4))
                axis = random.randint(0, 1)
                output_grid = np.roll(output_grid, shift, axis=axis)
            elif transform == 'color_map':
                old_color = random.randint(1, int(3 + complexity * 2))
                new_color = random.randint(1, int(3 + complexity * 2))
                output_grid[output_grid == old_color] = new_color
            else:  # local_rule
                # Apply local transformation to random regions
                mask_size = max(1, grid_size // 4)
                start_i = random.randint(0, max(0, grid_size - mask_size))
                start_j = random.randint(0, max(0, grid_size - mask_size))
                region = output_grid[start_i:start_i+mask_size, start_j:start_j+mask_size]
                output_grid[start_i:start_i+mask_size, start_j:start_j+mask_size] = (region + 1) % 6
        
        return input_grid, output_grid
    
    @staticmethod
    def _generate_ensemble_coordination_patterns(num_samples: int, grid_size: int, complexity: float) -> List[Dict]:
        """Generate patterns requiring ensemble coordination"""
        patterns = []
        
        for i in range(num_samples):
            # Multi-model coordination scenarios
            scenario = random.choice([
                'multi_aspect_analysis', 'consensus_building', 'conflict_resolution',
                'complementary_skills', 'hierarchical_reasoning'
            ])
            
            if scenario == 'multi_aspect_analysis':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_multi_aspect_pattern(grid_size, complexity)
            elif scenario == 'consensus_building':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_consensus_pattern(grid_size, complexity)
            elif scenario == 'conflict_resolution':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_conflict_resolution_pattern(grid_size, complexity)
            elif scenario == 'complementary_skills':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_complementary_pattern(grid_size, complexity)
            else:  # hierarchical_reasoning
                input_grid, output_grid = PrometheusMetaDSLTraining._create_hierarchical_pattern(grid_size, complexity)
            
            patterns.append({
                'inputs': input_grid.astype(np.int64),
                'outputs': output_grid.astype(np.int64),
                'ensemble_context': {
                    'scenario': scenario,
                    'requires_ensemble': True,
                    'complexity': complexity,
                    'coordination_type': scenario
                }
            })
        
        return patterns
    
    @staticmethod
    def _create_multi_aspect_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pattern requiring multiple model perspectives"""
        input_grid = np.random.randint(0, int(5 + complexity * 2), (grid_size, grid_size))
        output_grid = np.zeros_like(input_grid)
        
        # Aspect 1: Color-based transformation (IRIS strength)
        color_mask = input_grid == 1
        output_grid[color_mask] = 5
        
        # Aspect 2: Spatial pattern (MINERVA strength)
        for i in range(grid_size):
            for j in range(grid_size):
                if input_grid[i, j] == 2:
                    # Spatial relationship: move to opposite corner
                    new_i = grid_size - 1 - i
                    new_j = grid_size - 1 - j
                    if 0 <= new_i < grid_size and 0 <= new_j < grid_size:
                        output_grid[new_i, new_j] = 3
        
        # Aspect 3: Sequential pattern (CHRONOS strength)
        sequence_cells = np.where(input_grid == 3)
        for idx, (i, j) in enumerate(zip(sequence_cells[0], sequence_cells[1])):
            output_grid[i, j] = (idx % 4) + 1
        
        # Aspect 4: Object-based (ATLAS strength)
        object_mask = input_grid >= 4
        output_grid[object_mask] = input_grid[object_mask] - 1
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_consensus_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pattern requiring consensus among models"""
        input_grid = np.random.randint(0, int(4 + complexity * 2), (grid_size, grid_size))
        output_grid = input_grid.copy()
        
        # Multiple possible transformations, output should be consensus
        # Transform 1: Rotation
        rot_grid = np.rot90(input_grid)
        
        # Transform 2: Color shift
        color_grid = (input_grid + 1) % 5
        
        # Transform 3: Flip
        flip_grid = np.flip(input_grid, axis=0)
        
        # Consensus: majority vote per cell
        for i in range(grid_size):
            for j in range(grid_size):
                candidates = [input_grid[i, j], rot_grid[i, j], color_grid[i, j], flip_grid[i, j]]
                # Find most common value
                counts = {}
                for val in candidates:
                    counts[val] = counts.get(val, 0) + 1
                consensus = max(counts.items(), key=lambda x: x[1])[0]
                output_grid[i, j] = consensus
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_conflict_resolution_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pattern with conflicting signals requiring resolution"""
        input_grid = np.random.randint(0, int(3 + complexity * 3), (grid_size, grid_size))
        output_grid = np.zeros_like(input_grid)
        
        # Create conflicting transformations in different regions
        mid = grid_size // 2
        
        # Top half: one rule (increment colors)
        top_region = input_grid[:mid, :]
        output_grid[:mid, :] = (top_region + 1) % 5
        
        # Bottom half: different rule (decrement colors)  
        bottom_region = input_grid[mid:, :]
        output_grid[mid:, :] = np.maximum(bottom_region - 1, 0)
        
        # Middle region: resolution (average)
        if mid > 0:
            boundary = max(0, mid - 1)
            for j in range(grid_size):
                top_val = output_grid[boundary, j]
                bottom_val = output_grid[min(mid, grid_size-1), j]
                output_grid[mid-1:mid+1, j] = (top_val + bottom_val) // 2
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_complementary_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pattern requiring complementary model skills"""
        input_grid = np.random.randint(0, 6, (grid_size, grid_size))
        output_grid = np.zeros_like(input_grid)
        
        # Each model contributes to different aspects
        # IRIS: Color analysis (colors 1-2)
        iris_mask = (input_grid >= 1) & (input_grid <= 2)
        output_grid[iris_mask] = input_grid[iris_mask] + 3
        
        # MINERVA: Spatial reasoning (geometric shapes)
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                if input_grid[i, j] == 3:
                    # Create cross pattern
                    output_grid[i-1:i+2, j] = 4
                    output_grid[i, j-1:j+2] = 4
        
        # CHRONOS: Sequential patterns
        sequence_mask = input_grid == 4
        coords = np.where(sequence_mask)
        for idx, (i, j) in enumerate(zip(coords[0], coords[1])):
            output_grid[i, j] = 1 + (idx % 3)
        
        # ATLAS: Object manipulation
        object_mask = input_grid == 5
        # Move objects to center
        center_i, center_j = grid_size // 2, grid_size // 2
        if np.any(object_mask):
            output_grid[center_i, center_j] = 5
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_hierarchical_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pattern requiring hierarchical reasoning"""
        input_grid = np.random.randint(0, int(4 + complexity), (grid_size, grid_size))
        output_grid = np.zeros_like(input_grid)
        
        # Level 1: Local patterns
        for i in range(0, grid_size-1, 2):
            for j in range(0, grid_size-1, 2):
                # 2x2 local pattern
                block = input_grid[i:i+2, j:j+2]
                if np.all(block == block[0, 0]) and block[0, 0] > 0:
                    # Uniform block -> special marker
                    output_grid[i:i+2, j:j+2] = block[0, 0] + 1
        
        # Level 2: Global pattern
        # Check for larger structures
        quarter = grid_size // 4
        if quarter > 0:
            for i in range(0, grid_size, quarter):
                for j in range(0, grid_size, quarter):
                    region = input_grid[i:i+quarter, j:j+quarter]
                    if np.std(region) < complexity:  # Low variance = structured
                        output_grid[i:i+quarter, j:j+quarter] = np.mean(region).astype(int)
        
        return input_grid, output_grid
    
    @staticmethod
    def _generate_adaptation_patterns(num_samples: int, grid_size: int, complexity: float) -> List[Dict]:
        """Generate patterns requiring adaptation"""
        patterns = []
        
        for i in range(num_samples):
            # Adaptation scenarios
            scenario = random.choice([
                'domain_shift', 'distribution_change', 'context_switch',
                'scale_variation', 'noise_robustness'
            ])
            
            if scenario == 'domain_shift':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_domain_shift_pattern(grid_size, complexity)
            elif scenario == 'distribution_change':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_distribution_change_pattern(grid_size, complexity)
            elif scenario == 'context_switch':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_context_switch_pattern(grid_size, complexity)
            elif scenario == 'scale_variation':
                input_grid, output_grid = PrometheusMetaDSLTraining._create_scale_variation_pattern(grid_size, complexity)
            else:  # noise_robustness
                input_grid, output_grid = PrometheusMetaDSLTraining._create_noise_robustness_pattern(grid_size, complexity)
            
            patterns.append({
                'inputs': input_grid.astype(np.int64),
                'outputs': output_grid.astype(np.int64),
                'adaptation_context': {
                    'scenario': scenario,
                    'requires_adaptation': True,
                    'complexity': complexity,
                    'adaptation_type': scenario
                }
            })
        
        return patterns
    
    @staticmethod
    def _create_domain_shift_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create domain shift adaptation pattern"""
        # Same logical operation but different visual domain
        input_grid = np.random.randint(0, int(3 + complexity * 2), (grid_size, grid_size))
        
        # Domain shift: colors represent different concepts
        # Original domain: colors are spatial markers
        # New domain: colors are object types
        output_grid = np.zeros_like(input_grid)
        
        # Adaptation: maintain logical relationship despite domain change
        for i in range(grid_size):
            for j in range(grid_size):
                if input_grid[i, j] > 0:
                    # Transform based on position (spatial) not color (object type)
                    if i < grid_size // 2:
                        output_grid[i, j] = (input_grid[i, j] + 1) % 4 + 1
                    else:
                        output_grid[i, j] = max(1, input_grid[i, j] - 1)
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_distribution_change_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create distribution change pattern"""
        # Change in color distribution but same underlying rule
        base_colors = int(2 + complexity * 2)
        
        # Skewed distribution
        if random.random() < 0.5:
            # Sparse pattern
            input_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
            num_non_zero = max(1, int(grid_size * grid_size * 0.1 * complexity))
            positions = random.sample(range(grid_size * grid_size), num_non_zero)
            for pos in positions:
                i, j = pos // grid_size, pos % grid_size
                input_grid[i, j] = random.randint(1, base_colors)
        else:
            # Dense pattern
            input_grid = np.random.randint(1, base_colors + 1, (grid_size, grid_size))
            # Add some zeros
            zero_positions = random.sample(range(grid_size * grid_size), 
                                         max(1, int(grid_size * grid_size * 0.2)))
            for pos in zero_positions:
                i, j = pos // grid_size, pos % grid_size
                input_grid[i, j] = 0
        
        # Same transformation rule regardless of distribution
        output_grid = (input_grid + 1) % (base_colors + 1)
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_context_switch_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create context switch pattern"""
        input_grid = np.random.randint(0, int(4 + complexity), (grid_size, grid_size))
        output_grid = np.zeros_like(input_grid)
        
        # Context switch: rule changes based on global context
        global_sum = np.sum(input_grid)
        
        if global_sum % 2 == 0:
            # Even context: increment rule
            output_grid = (input_grid + 1) % 5
        else:
            # Odd context: rotation rule
            output_grid = np.rot90(input_grid)
            if output_grid.shape != input_grid.shape:
                # Handle size mismatch
                min_size = min(grid_size, output_grid.shape[0], output_grid.shape[1])
                temp = np.zeros((grid_size, grid_size), dtype=np.int64)
                temp[:min_size, :min_size] = output_grid[:min_size, :min_size]
                output_grid = temp
        
        return input_grid, output_grid
    
    @staticmethod
    def _create_scale_variation_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create scale variation pattern"""
        input_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        output_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        
        # Create patterns at different scales
        scales = [2, 4, 8] if grid_size >= 8 else [2]
        scale = random.choice([s for s in scales if s <= grid_size // 2])
        
        # Create base pattern at chosen scale
        pattern_size = grid_size // scale
        base_pattern = np.random.randint(1, int(3 + complexity), (pattern_size, pattern_size))
        
        # Tile pattern
        for i in range(scale):
            for j in range(scale):
                start_i, start_j = i * pattern_size, j * pattern_size
                end_i = min(start_i + pattern_size, grid_size)
                end_j = min(start_j + pattern_size, grid_size)
                input_grid[start_i:end_i, start_j:end_j] = base_pattern[:end_i-start_i, :end_j-start_j]
        
        # Output: same pattern but scaled differently
        if scale > 2:
            # Downscale
            new_scale = scale // 2
            new_pattern_size = grid_size // new_scale
            for i in range(new_scale):
                for j in range(new_scale):
                    start_i, start_j = i * new_pattern_size, j * new_pattern_size
                    end_i = min(start_i + new_pattern_size, grid_size)
                    end_j = min(start_j + new_pattern_size, grid_size)
                    # Use average of base pattern
                    avg_val = int(np.mean(base_pattern)) if base_pattern.size > 0 else 1
                    output_grid[start_i:end_i, start_j:end_j] = avg_val
        else:
            # Upscale
            output_grid = input_grid.copy()
            
        return input_grid, output_grid
    
    @staticmethod
    def _create_noise_robustness_pattern(grid_size: int, complexity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create noise robustness pattern"""
        # Clean pattern
        clean_grid = np.random.randint(0, int(3 + complexity), (grid_size, grid_size))
        
        # Add noise
        noise_level = complexity * 0.2
        noise_mask = np.random.random((grid_size, grid_size)) < noise_level
        input_grid = clean_grid.copy()
        input_grid[noise_mask] = np.random.randint(0, int(3 + complexity), np.sum(noise_mask))
        
        # Output: denoised version with transformation
        output_grid = clean_grid.copy()
        # Apply transformation to clean version
        output_grid = (output_grid + 1) % int(3 + complexity)
        
        return input_grid, output_grid


def create_prometheus_dsl_system() -> Dict:
    """Create PROMETHEUS-specific DSL system"""
    
    return {
        'training': PrometheusMetaDSLTraining(),
        'meta_learning_enabled': True,
        'ensemble_coordination_enabled': True,
        'adaptation_enabled': True,
        'pattern_types': [
            'meta_learning', 'ensemble_coordination', 'adaptation',
            'few_shot_analogy', 'cross_domain_transfer', 'abstract_reasoning',
            'pattern_completion', 'rule_extraction', 'compositional_learning',
            'multi_aspect_analysis', 'consensus_building', 'conflict_resolution',
            'complementary_skills', 'hierarchical_reasoning', 'domain_shift',
            'distribution_change', 'context_switch', 'scale_variation', 'noise_robustness'
        ]
    }