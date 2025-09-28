"""
DSL-based training integration for Stage 0
Generates exact deterministic outputs for better exact match training
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from .arc_dsl import DSLProgram, DSLProgramGenerator, DSLExecutor, Operation


class DSLAugmentedDataset:
    """Augments Stage 0 dataset with DSL-generated examples"""
    
    def __init__(self, base_samples: List[Dict[str, np.ndarray]]):
        self.base_samples = base_samples
        self.dsl_programs = DSLProgramGenerator.generate_simple_programs()
        self.augmented_samples = []
        self._generate_dsl_samples()
    
    def _generate_dsl_samples(self):
        """Generate training samples using DSL programs"""
        
        # For each base sample, apply various DSL programs
        for sample in self.base_samples[:100]:  # Use first 100 samples
            input_grid = sample['input']
            
            # Skip if grid is too large or too complex
            if input_grid.shape[0] > 10 or input_grid.shape[1] > 10:
                continue
            
            # Apply each DSL program
            for program in self.dsl_programs:
                try:
                    output_grid = program.execute(input_grid)
                    
                    # Only add if output is valid and different
                    if output_grid.shape == input_grid.shape and not np.array_equal(output_grid, input_grid):
                        self.augmented_samples.append({
                            'input': input_grid.copy(),
                            'output': output_grid,
                            'program': program.to_string(),
                            'is_dsl': True
                        })
                except Exception:
                    # Skip failed transformations
                    continue
        
        # Add pure DSL examples with simple patterns
        self._add_pure_dsl_examples()
    
    def _add_pure_dsl_examples(self):
        """Add examples that are purely DSL-generated"""
        
        # Simple patterns
        patterns = [
            # Checkerboard
            np.array([[0, 1], [1, 0]]),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            
            # Lines
            np.array([[1, 1, 1], [0, 0, 0], [2, 2, 2]]),
            np.array([[1, 0, 2], [1, 0, 2], [1, 0, 2]]),
            
            # Corners
            np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
            np.array([[2, 2, 0], [2, 0, 0], [0, 0, 0]]),
            
            # Small objects
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[1, 1], [1, 1]]),
            
            # Multi-color
            np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
        ]
        
        # Apply programs to patterns
        for pattern in patterns:
            for program in self.dsl_programs[:20]:  # Use subset
                try:
                    output = program.execute(pattern)
                    if output.shape == pattern.shape:
                        self.augmented_samples.append({
                            'input': pattern.copy(),
                            'output': output,
                            'program': program.to_string(),
                            'is_dsl': True
                        })
                except Exception:
                    continue
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all augmented samples"""
        return self.augmented_samples


class DSLTrainingIntegration:
    """Integrates DSL program synthesis with neural network training"""
    
    @staticmethod
    def create_stage0_dsl_samples(curriculum_stage: int = 0) -> List[Dict[str, np.ndarray]]:
        """Create DSL-based samples for Stage 0"""
        samples = []
        
        if curriculum_stage == 0:
            # Stage 0: Simple deterministic transformations
            base_grids = DSLTrainingIntegration._create_simple_grids()
            programs = DSLProgramGenerator.generate_simple_programs()
            
            for grid in base_grids:
                for program in programs[:30]:  # Use first 30 programs
                    try:
                        output = program.execute(grid)
                        if output.shape == grid.shape:
                            samples.append({
                                'input': grid,
                                'output': output,
                                'program': program
                            })
                    except Exception:
                        continue
        
        return samples
    
    @staticmethod
    def _create_simple_grids() -> List[np.ndarray]:
        """Create simple base grids for DSL transformation"""
        grids = []
        
        # Single color grids
        for size in [(3, 3), (4, 4), (5, 5)]:
            for color in range(3):
                grid = np.full(size, color)
                grids.append(grid)
        
        # Two-color patterns
        grids.extend([
            # Horizontal stripes
            np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
            np.array([[1, 1], [0, 0]]),
            
            # Vertical stripes
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
            np.array([[1, 0], [1, 0]]),
            
            # Diagonal
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            
            # Center dot
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
            
            # L-shape
            np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),
            
            # Cross
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            
            # Corners
            np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
        ])
        
        # Three-color patterns
        grids.extend([
            np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
            np.array([[0, 1], [2, 0]]),
            np.array([[0, 0, 1], [0, 2, 1], [2, 2, 1]]),
        ])
        
        return grids
    
    @staticmethod
    def dsl_loss(pred_programs: Dict[str, torch.Tensor], 
                 target_programs: List[DSLProgram],
                 input_grids: torch.Tensor,
                 target_grids: torch.Tensor) -> torch.Tensor:
        """Loss function for DSL program prediction"""
        
        # For now, use standard reconstruction loss
        # In future, could add program-specific loss
        reconstruction_loss = torch.nn.functional.mse_loss(
            pred_programs.get('output', input_grids),
            target_grids
        )
        
        return reconstruction_loss
    
    @staticmethod
    def augment_batch_with_dsl(batch: Dict[str, torch.Tensor], 
                              dsl_ratio: float = 0.3) -> Dict[str, torch.Tensor]:
        """Augment training batch with DSL-generated examples"""
        
        B = batch['input'].shape[0]
        num_dsl = int(B * dsl_ratio)
        
        if num_dsl > 0:
            # Generate DSL examples
            dsl_samples = DSLTrainingIntegration.create_stage0_dsl_samples(0)
            
            if dsl_samples:
                # Replace some batch samples with DSL examples
                for i in range(min(num_dsl, len(dsl_samples))):
                    idx = i % len(dsl_samples)
                    batch['input'][i] = torch.tensor(dsl_samples[idx]['input'])
                    batch['output'][i] = torch.tensor(dsl_samples[idx]['output'])
        
        return batch