# ARC Data Synthesis Module
# Uses PyTorch to generate synthetic training data from ARC datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict

class ARCDataSynthesizer:
    """
    Synthesizes new training data by applying learned transformations
    and augmentations to existing ARC datasets
    """
    
    def __init__(self, arc_data_path: str = '/content/AutomataNexus_Olympus_AGI2/data'):
        self.arc_data_path = arc_data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ARC datasets
        self.training_data = self._load_arc_data('training')
        self.evaluation_data = self._load_arc_data('evaluation')
        
        # Transformation primitives
        self.transformations = {
            'rotate_90': self._rotate_90,
            'rotate_180': self._rotate_180,
            'rotate_270': self._rotate_270,
            'flip_horizontal': self._flip_horizontal,
            'flip_vertical': self._flip_vertical,
            'transpose': self._transpose,
            'color_swap': self._color_swap,
            'color_extract': self._color_extract,
            'scale_up': self._scale_up,
            'scale_down': self._scale_down,
            'translate': self._translate,
            'mirror_pattern': self._mirror_pattern,
            'complete_symmetry': self._complete_symmetry,
            'extract_objects': self._extract_objects,
            'fill_pattern': self._fill_pattern
        }
        
        # Pattern templates for exact match training
        self.exact_patterns = {
            'identity': self._generate_identity,
            'simple_rotation': self._generate_simple_rotation,
            'color_mapping': self._generate_color_mapping,
            'binary_logic': self._generate_binary_logic,
            'object_isolation': self._generate_object_isolation,
            'pattern_completion': self._generate_pattern_completion,
            'size_filtering': self._generate_size_filtering,
            'position_based': self._generate_position_based
        }
        
    def _load_arc_data(self, split: str) -> List[Dict]:
        """Load ARC dataset from JSON files"""
        data = []
        data_dir = os.path.join(self.arc_data_path, split)
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(data_dir, filename), 'r') as f:
                        data.append(json.load(f))
        
        return data
    
    def generate_synthetic_batch(self, 
                                batch_size: int,
                                stage: int = 0,
                                exact_match_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of synthetic training data
        
        Args:
            batch_size: Number of samples to generate
            stage: Curriculum stage (0=easy, 1=medium, 2=hard)
            exact_match_ratio: Ratio of exact-match friendly patterns
        """
        samples = []
        
        # Determine mix based on stage
        if stage == 0:
            # Stage 0: Focus on exact patterns
            n_exact = int(batch_size * 0.8)
            n_transform = int(batch_size * 0.15)
            n_arc = batch_size - n_exact - n_transform
        elif stage == 1:
            # Stage 1: Balanced mix
            n_exact = int(batch_size * 0.4)
            n_transform = int(batch_size * 0.3)
            n_arc = batch_size - n_exact - n_transform
        else:
            # Stage 2: Focus on complex ARC patterns
            n_exact = int(batch_size * 0.2)
            n_transform = int(batch_size * 0.3)
            n_arc = batch_size - n_exact - n_transform
        
        # Generate exact match patterns
        for _ in range(n_exact):
            pattern_type = random.choice(list(self.exact_patterns.keys()))
            sample = self.exact_patterns[pattern_type](stage)
            samples.append(sample)
        
        # Generate transformation-based patterns
        for _ in range(n_transform):
            sample = self._generate_transformation_pattern(stage)
            samples.append(sample)
        
        # Generate ARC-derived patterns
        for _ in range(n_arc):
            sample = self._generate_arc_derived_pattern(stage)
            samples.append(sample)
        
        # Convert to PyTorch tensors
        return self._samples_to_tensors(samples)
    
    def _generate_identity(self, stage: int) -> Dict:
        """Generate identity transformation (output = input)"""
        size = random.choice([3, 4, 5]) if stage == 0 else random.choice([5, 6, 7, 8])
        n_colors = 2 if stage == 0 else random.choice([2, 3, 4])
        
        grid = np.random.randint(0, n_colors, (size, size))
        return {'input': grid, 'output': grid.copy()}
    
    def _generate_simple_rotation(self, stage: int) -> Dict:
        """Generate simple rotation patterns"""
        size = random.choice([3, 4, 5]) if stage == 0 else random.choice([5, 6, 7])
        n_colors = 2 if stage == 0 else 3
        
        # Create a simple L or T shape
        grid = np.zeros((size, size), dtype=int)
        if random.random() < 0.5:
            # L shape
            grid[0, :size//2] = 1
            grid[:size//2, 0] = 1
        else:
            # T shape
            grid[0, :] = 1
            grid[:size//2, size//2] = 1
        
        rotation = random.choice(['rotate_90', 'rotate_180', 'rotate_270'])
        output = self.transformations[rotation](grid)
        
        return {'input': grid, 'output': output}
    
    def _generate_color_mapping(self, stage: int) -> Dict:
        """Generate color mapping patterns"""
        size = random.choice([3, 4, 5]) if stage == 0 else random.choice([5, 6, 7])
        
        # Create simple pattern with 2-3 colors
        grid = np.random.randint(0, 3, (size, size))
        
        # Create color mapping
        color_map = {0: 0, 1: 2, 2: 1}  # Swap colors 1 and 2
        output = np.vectorize(color_map.get)(grid)
        
        return {'input': grid, 'output': output}
    
    def _generate_binary_logic(self, stage: int) -> Dict:
        """Generate binary logic patterns"""
        size = random.choice([3, 4, 5]) if stage == 0 else random.choice([5, 6, 7])
        
        grid = np.random.randint(0, 3, (size, size))
        
        # Apply binary logic
        if random.random() < 0.5:
            # Convert to binary
            output = (grid > 0).astype(int)
        else:
            # Invert binary
            output = (grid == 0).astype(int)
        
        return {'input': grid, 'output': output}
    
    def _generate_object_isolation(self, stage: int) -> Dict:
        """Generate object isolation patterns"""
        size = random.choice([5, 6, 7]) if stage <= 1 else random.choice([7, 8, 9])
        
        grid = np.zeros((size, size), dtype=int)
        
        # Add 2-3 small objects
        n_objects = 2 if stage == 0 else 3
        for i in range(n_objects):
            color = i + 1
            obj_size = 2 if stage == 0 else random.choice([2, 3])
            x = random.randint(0, size - obj_size)
            y = random.randint(0, size - obj_size)
            grid[x:x+obj_size, y:y+obj_size] = color
        
        # Extract specific color
        target_color = random.randint(1, n_objects)
        output = np.where(grid == target_color, target_color, 0)
        
        return {'input': grid, 'output': output}
    
    def _generate_pattern_completion(self, stage: int) -> Dict:
        """Generate pattern completion tasks"""
        size = 6 if stage == 0 else 8
        
        grid = np.zeros((size, size), dtype=int)
        
        # Create partial pattern
        if stage == 0:
            # Simple checkerboard
            grid[::2, ::2] = 1
            grid[1::2, 1::2] = 1
            # Remove some squares
            mask = np.random.random((size, size)) > 0.7
            grid[mask] = 0
            output = np.zeros((size, size), dtype=int)
            output[::2, ::2] = 1
            output[1::2, 1::2] = 1
        else:
            # More complex pattern
            for i in range(size):
                for j in range(size):
                    if (i + j) % 3 == 0:
                        grid[i, j] = 1
                    elif (i + j) % 3 == 1:
                        grid[i, j] = 2
            mask = np.random.random((size, size)) > 0.6
            grid[mask] = 0
            output = grid.copy()
            for i in range(size):
                for j in range(size):
                    if grid[i, j] == 0:
                        if (i + j) % 3 == 0:
                            output[i, j] = 1
                        elif (i + j) % 3 == 1:
                            output[i, j] = 2
        
        return {'input': grid, 'output': output}
    
    def _generate_size_filtering(self, stage: int) -> Dict:
        """Generate size-based filtering patterns"""
        size = 8 if stage <= 1 else 10
        
        grid = np.zeros((size, size), dtype=int)
        
        # Add objects of different sizes
        objects = []
        for i in range(3):
            obj_size = random.choice([1, 2, 3, 4])
            color = i + 1
            x = random.randint(0, max(1, size - obj_size))
            y = random.randint(0, max(1, size - obj_size))
            grid[x:x+obj_size, y:y+obj_size] = color
            objects.append((obj_size, color, x, y))
        
        # Keep only largest object
        largest = max(objects, key=lambda x: x[0])
        output = np.zeros((size, size), dtype=int)
        obj_size, color, x, y = largest
        output[x:x+obj_size, y:y+obj_size] = color
        
        return {'input': grid, 'output': output}
    
    def _generate_position_based(self, stage: int) -> Dict:
        """Generate position-based transformation patterns"""
        size = 5 if stage == 0 else 7
        
        grid = np.random.randint(0, 3, (size, size))
        output = grid.copy()
        
        # Apply position-based rule
        if stage == 0:
            # Simple: Clear bottom half
            output[size//2:, :] = 0
        else:
            # Complex: Keep only diagonal elements
            mask = np.zeros((size, size), dtype=bool)
            np.fill_diagonal(mask, True)
            np.fill_diagonal(np.fliplr(mask), True)
            output[~mask] = 0
        
        return {'input': grid, 'output': output}
    
    def _generate_transformation_pattern(self, stage: int) -> Dict:
        """Generate pattern using transformation primitives"""
        size = random.choice([4, 5, 6]) if stage == 0 else random.choice([6, 7, 8])
        n_colors = 3 if stage <= 1 else 4
        
        # Create base pattern
        grid = np.random.randint(0, n_colors, (size, size))
        
        # Apply 1-2 transformations
        n_transforms = 1 if stage == 0 else random.choice([1, 2])
        output = grid.copy()
        
        for _ in range(n_transforms):
            transform = random.choice(list(self.transformations.keys())[:8])  # Basic transforms
            output = self.transformations[transform](output)
        
        return {'input': grid, 'output': output}
    
    def _generate_arc_derived_pattern(self, stage: int) -> Dict:
        """Generate pattern derived from actual ARC tasks"""
        # Select appropriate dataset
        dataset = self.training_data if random.random() < 0.8 else self.evaluation_data
        
        if not dataset:
            # Fallback to generated pattern
            return self._generate_transformation_pattern(stage)
        
        # Select a task
        task = random.choice(dataset)
        examples = task.get('train', [])
        
        if not examples:
            return self._generate_transformation_pattern(stage)
        
        # Select an example and apply modifications
        example = random.choice(examples)
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Apply stage-appropriate modifications
        if stage == 0:
            # Simplify for stage 0
            if input_grid.shape[0] > 8:
                # Crop to smaller size
                crop_size = random.choice([5, 6, 7])
                input_grid = input_grid[:crop_size, :crop_size]
                output_grid = output_grid[:crop_size, :crop_size]
            
            # Reduce colors
            unique_colors = np.unique(np.concatenate([input_grid.flat, output_grid.flat]))
            if len(unique_colors) > 3:
                color_map = {c: i for i, c in enumerate(unique_colors[:3])}
                input_grid = np.vectorize(lambda x: color_map.get(x, 0))(input_grid)
                output_grid = np.vectorize(lambda x: color_map.get(x, 0))(output_grid)
        
        elif stage == 1:
            # Moderate modifications
            if random.random() < 0.3:
                # Add noise
                noise_mask = np.random.random(input_grid.shape) < 0.1
                input_grid[noise_mask] = 0
        
        # Apply additional augmentation
        if random.random() < 0.5:
            aug_type = random.choice(['rotate', 'flip', 'transpose'])
            if aug_type == 'rotate':
                k = random.choice([1, 2, 3])
                input_grid = np.rot90(input_grid, k)
                output_grid = np.rot90(output_grid, k)
            elif aug_type == 'flip':
                if random.random() < 0.5:
                    input_grid = np.fliplr(input_grid)
                    output_grid = np.fliplr(output_grid)
                else:
                    input_grid = np.flipud(input_grid)
                    output_grid = np.flipud(output_grid)
            else:
                input_grid = input_grid.T
                output_grid = output_grid.T
        
        return {'input': input_grid, 'output': output_grid}
    
    # Transformation primitives
    def _rotate_90(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=3)
    
    def _rotate_180(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)
    
    def _rotate_270(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=1)
    
    def _flip_horizontal(self, grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)
    
    def _flip_vertical(self, grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)
    
    def _transpose(self, grid: np.ndarray) -> np.ndarray:
        return grid.T
    
    def _color_swap(self, grid: np.ndarray) -> np.ndarray:
        unique_colors = np.unique(grid)
        if len(unique_colors) < 2:
            return grid
        
        c1, c2 = random.sample(list(unique_colors), 2)
        output = grid.copy()
        output[grid == c1] = c2
        output[grid == c2] = c1
        return output
    
    def _color_extract(self, grid: np.ndarray) -> np.ndarray:
        unique_colors = np.unique(grid)
        non_zero_colors = unique_colors[unique_colors > 0]
        if len(non_zero_colors) == 0:
            return grid
        
        target_color = random.choice(non_zero_colors)
        return np.where(grid == target_color, target_color, 0)
    
    def _scale_up(self, grid: np.ndarray) -> np.ndarray:
        scale = 2
        output = np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)
        return output
    
    def _scale_down(self, grid: np.ndarray) -> np.ndarray:
        if grid.shape[0] < 4 or grid.shape[1] < 4:
            return grid
        return grid[::2, ::2]
    
    def _translate(self, grid: np.ndarray) -> np.ndarray:
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        output = np.zeros_like(grid)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ni, nj = i + dx, j + dy
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    output[ni, nj] = grid[i, j]
        
        return output
    
    def _mirror_pattern(self, grid: np.ndarray) -> np.ndarray:
        half = grid.shape[1] // 2
        output = grid.copy()
        output[:, half:] = np.fliplr(output[:, :half])[:, :output.shape[1]-half]
        return output
    
    def _complete_symmetry(self, grid: np.ndarray) -> np.ndarray:
        output = grid.copy()
        # Complete horizontal symmetry
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1] // 2):
                if output[i, j] != 0 and output[i, -(j+1)] == 0:
                    output[i, -(j+1)] = output[i, j]
                elif output[i, -(j+1)] != 0 and output[i, j] == 0:
                    output[i, j] = output[i, -(j+1)]
        return output
    
    def _extract_objects(self, grid: np.ndarray) -> np.ndarray:
        # Simple connected component extraction
        # For now, just extract non-zero regions
        return (grid > 0).astype(int)
    
    def _fill_pattern(self, grid: np.ndarray) -> np.ndarray:
        # Fill enclosed regions
        output = grid.copy()
        # Simple flood fill - fill all 0s that are completely enclosed
        # This is a simplified version
        if np.sum(grid > 0) > 0:
            most_common = np.bincount(grid.flat).argmax()
            if most_common == 0 and len(np.unique(grid)) > 1:
                most_common = np.bincount(grid[grid > 0].flat).argmax()
            output[grid == 0] = most_common
        return output
    
    def _samples_to_tensors(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert samples to PyTorch tensors"""
        max_h = max(s['input'].shape[0] for s in samples)
        max_w = max(s['input'].shape[1] for s in samples)
        
        inputs = []
        outputs = []
        
        for sample in samples:
            # Pad to max size
            input_padded = np.zeros((max_h, max_w), dtype=np.int64)
            output_padded = np.zeros((max_h, max_w), dtype=np.int64)
            
            h, w = sample['input'].shape
            input_padded[:h, :w] = sample['input']
            output_padded[:h, :w] = sample['output']
            
            inputs.append(input_padded)
            outputs.append(output_padded)
        
        return {
            'inputs': torch.tensor(np.stack(inputs), device=self.device),
            'outputs': torch.tensor(np.stack(outputs), device=self.device)
        }


class ARCDataAugmenter:
    """
    Advanced augmentation techniques specifically for ARC tasks
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def augment_batch(self, inputs: torch.Tensor, outputs: torch.Tensor, 
                     augment_prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to a batch"""
        batch_size = inputs.shape[0]
        augmented_inputs = []
        augmented_outputs = []
        
        for i in range(batch_size):
            if random.random() < augment_prob:
                aug_input, aug_output = self._augment_sample(
                    inputs[i].cpu().numpy(),
                    outputs[i].cpu().numpy()
                )
                # Ensure augmented tensors have same shape as original
                if aug_input.shape != inputs[i].shape:
                    # Skip augmentation if shape changed (e.g., from transpose)
                    augmented_inputs.append(inputs[i])
                    augmented_outputs.append(outputs[i])
                else:
                    augmented_inputs.append(torch.tensor(aug_input.copy()))
                    augmented_outputs.append(torch.tensor(aug_output.copy()))
            else:
                augmented_inputs.append(inputs[i])
                augmented_outputs.append(outputs[i])
        
        return torch.stack(augmented_inputs).to(self.device), \
               torch.stack(augmented_outputs).to(self.device)
    
    def _augment_sample(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to a single sample"""
        # Check if grid is square for rotation
        is_square = input_grid.shape[0] == input_grid.shape[1]
        
        if is_square:
            aug_type = random.choice(['rotate', 'flip', 'color_permute', 'add_noise'])
        else:
            # For non-square grids, avoid rotations
            aug_type = random.choice(['flip', 'color_permute', 'add_noise'])
        
        if aug_type == 'rotate' and is_square:
            k = random.choice([1, 2, 3])
            return np.rot90(input_grid, k).copy(), np.rot90(output_grid, k).copy()
        
        elif aug_type == 'flip':
            if random.random() < 0.5:
                return np.fliplr(input_grid).copy(), np.fliplr(output_grid).copy()
            else:
                return np.flipud(input_grid).copy(), np.flipud(output_grid).copy()
        
        elif aug_type == 'color_permute':
            # Create random color permutation
            unique_colors = np.unique(np.concatenate([input_grid.flat, output_grid.flat]))
            color_perm = {c: c for c in unique_colors}
            
            # Shuffle non-zero colors
            non_zero_colors = [c for c in unique_colors if c > 0]
            if len(non_zero_colors) > 1:
                shuffled = non_zero_colors.copy()
                random.shuffle(shuffled)
                for orig, new in zip(non_zero_colors, shuffled):
                    color_perm[orig] = new
            
            aug_input = np.vectorize(color_perm.get)(input_grid)
            aug_output = np.vectorize(color_perm.get)(output_grid)
            return aug_input, aug_output
        
        elif aug_type == 'add_noise':
            # Add sparse noise to input only
            aug_input = input_grid.copy()
            noise_mask = np.random.random(input_grid.shape) < 0.05
            aug_input[noise_mask] = 0
            return aug_input, output_grid
        
        return input_grid, output_grid