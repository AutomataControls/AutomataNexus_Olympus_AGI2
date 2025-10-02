"""
ATLAS-specific PRISM (Program Reasoning through Inductive Synthesis) System
Specialized for spatial transformation program synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import random
import itertools
from dataclasses import dataclass


@dataclass
class AtlasSpatialProgram:
    """Represents a spatial transformation program for ATLAS"""
    operations: List[Tuple[str, Dict[str, Any]]]
    confidence: float
    execution_time: float
    success_rate: float
    spatial_features: Dict[str, float]
    
    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """Execute the spatial program"""
        result = input_grid.copy()
        
        for operation, params in self.operations:
            result = AtlasProgramExecutor.execute_operation(result, operation, params)
        
        return result
    
    def to_string(self) -> str:
        """Convert program to string representation"""
        op_strings = []
        for op, params in self.operations:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            op_strings.append(f"{op}({param_str})")
        return " -> ".join(op_strings)


class AtlasProgramExecutor:
    """Executes spatial transformation programs"""
    
    @staticmethod
    def execute_operation(grid: np.ndarray, operation: str, params: Dict[str, Any]) -> np.ndarray:
        """Execute a single spatial operation"""
        
        if operation == "rotate_90":
            return np.rot90(grid, k=1, axes=(0, 1))
        elif operation == "rotate_180":
            return np.rot90(grid, k=2, axes=(0, 1))
        elif operation == "rotate_270":
            return np.rot90(grid, k=3, axes=(0, 1))
        elif operation == "flip_horizontal":
            return np.flip(grid, axis=1)
        elif operation == "flip_vertical":
            return np.flip(grid, axis=0)
        elif operation == "transpose":
            return np.transpose(grid)
        elif operation == "scale":
            return AtlasProgramExecutor._scale_grid(grid, params.get('factor', 1.0))
        elif operation == "translate":
            dx = params.get('dx', 0)
            dy = params.get('dy', 0)
            return AtlasProgramExecutor._translate_grid(grid, dx, dy)
        elif operation == "mirror_horizontal":
            return AtlasProgramExecutor._mirror_horizontal(grid)
        elif operation == "mirror_vertical":
            return AtlasProgramExecutor._mirror_vertical(grid)
        elif operation == "tile":
            tiles_x = params.get('tiles_x', 2)
            tiles_y = params.get('tiles_y', 2)
            return AtlasProgramExecutor._tile_grid(grid, tiles_x, tiles_y)
        else:
            return grid
    
    @staticmethod
    def _scale_grid(grid: np.ndarray, factor: float) -> np.ndarray:
        """Scale grid by factor"""
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
    def _translate_grid(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate grid by dx, dy"""
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
    def _mirror_horizontal(grid: np.ndarray) -> np.ndarray:
        """Create horizontal mirror pattern"""
        h, w = grid.shape
        left_half = grid[:, :w//2]
        result = np.zeros_like(grid)
        result[:, :w//2] = left_half
        result[:, w//2:w//2+left_half.shape[1]] = np.flip(left_half, axis=1)
        return result
    
    @staticmethod
    def _mirror_vertical(grid: np.ndarray) -> np.ndarray:
        """Create vertical mirror pattern"""
        h, w = grid.shape
        top_half = grid[:h//2, :]
        result = np.zeros_like(grid)
        result[:h//2, :] = top_half
        result[h//2:h//2+top_half.shape[0], :] = np.flip(top_half, axis=0)
        return result
    
    @staticmethod
    def _tile_grid(grid: np.ndarray, tiles_x: int, tiles_y: int) -> np.ndarray:
        """Create tiled pattern"""
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


class AtlasProgramSynthesizer:
    """Synthesizes spatial transformation programs for ATLAS"""
    
    def __init__(self, max_program_length: int = 4):
        self.max_program_length = max_program_length
        self.program_cache = {}
        self.success_patterns = deque(maxlen=1000)
        
        # Basic spatial operations
        self.basic_operations = [
            "rotate_90", "rotate_180", "rotate_270",
            "flip_horizontal", "flip_vertical", "transpose"
        ]
        
        # Parameterized operations
        self.parameterized_operations = [
            ("scale", [{"factor": 0.5}, {"factor": 2.0}]),
            ("translate", [{"dx": 1, "dy": 0}, {"dx": 0, "dy": 1}, {"dx": -1, "dy": 0}, {"dx": 0, "dy": -1}]),
            ("tile", [{"tiles_x": 2, "tiles_y": 2}, {"tiles_x": 3, "tiles_y": 3}])
        ]
        
        # Complex operations
        self.complex_operations = [
            "mirror_horizontal", "mirror_vertical"
        ]
    
    def synthesize_program(self, input_grid: np.ndarray, 
                          output_grid: np.ndarray,
                          max_attempts: int = 100) -> Optional[AtlasSpatialProgram]:
        """Synthesize spatial transformation program"""
        
        # Create cache key
        cache_key = self._create_cache_key(input_grid, output_grid)
        if cache_key in self.program_cache:
            return self.program_cache[cache_key]
        
        best_program = None
        best_score = float('inf')
        
        # Try progressively longer programs
        for length in range(1, self.max_program_length + 1):
            for attempt in range(max_attempts // self.max_program_length):
                program = self._generate_random_program(length)
                
                try:
                    result = program.execute(input_grid)
                    score = self._evaluate_program(result, output_grid)
                    
                    if score < best_score:
                        best_score = score
                        program.confidence = 1.0 / (1.0 + score)
                        program.success_rate = 1.0 if score < 0.01 else 0.0
                        best_program = program
                        
                        # Early termination for exact match
                        if score < 0.01:
                            break
                            
                except Exception:
                    continue
            
            # Stop if we found a good solution
            if best_program and best_program.success_rate > 0.5:
                break
        
        # Cache the result
        if best_program:
            self.program_cache[cache_key] = best_program
            if best_program.success_rate > 0.5:
                self.success_patterns.append(best_program)
        
        return best_program
    
    def _generate_random_program(self, length: int) -> AtlasSpatialProgram:
        """Generate random spatial transformation program"""
        operations = []
        
        for _ in range(length):
            # Choose operation type
            op_type = random.choice(['basic', 'parameterized', 'complex'])
            
            if op_type == 'basic':
                operation = random.choice(self.basic_operations)
                params = {}
            elif op_type == 'parameterized':
                operation, param_options = random.choice(self.parameterized_operations)
                params = random.choice(param_options)
            else:  # complex
                operation = random.choice(self.complex_operations)
                params = {}
            
            operations.append((operation, params))
        
        return AtlasSpatialProgram(
            operations=operations,
            confidence=0.0,
            execution_time=0.0,
            success_rate=0.0,
            spatial_features={}
        )
    
    def _create_cache_key(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Create cache key for grid pair"""
        input_hash = hash(input_grid.tobytes())
        output_hash = hash(output_grid.tobytes())
        return f"{input_hash}_{output_hash}"
    
    def _evaluate_program(self, result: np.ndarray, target: np.ndarray) -> float:
        """Evaluate program output against target"""
        if result.shape != target.shape:
            return float('inf')
        
        # Mean squared error
        mse = np.mean((result - target) ** 2)
        
        # Structural similarity penalty
        structural_penalty = self._compute_structural_penalty(result, target)
        
        return mse + 0.1 * structural_penalty
    
    def _compute_structural_penalty(self, result: np.ndarray, target: np.ndarray) -> float:
        """Compute structural similarity penalty"""
        # Edge preservation
        result_edges = self._detect_edges(result)
        target_edges = self._detect_edges(target)
        edge_penalty = np.mean((result_edges - target_edges) ** 2)
        
        # Pattern consistency
        pattern_penalty = self._compute_pattern_penalty(result, target)
        
        return edge_penalty + pattern_penalty
    
    def _detect_edges(self, grid: np.ndarray) -> np.ndarray:
        """Simple edge detection"""
        edges = np.zeros_like(grid)
        h, w = grid.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                gx = grid[i+1, j] - grid[i-1, j]
                gy = grid[i, j+1] - grid[i, j-1]
                edges[i, j] = np.sqrt(gx*gx + gy*gy)
        
        return edges
    
    def _compute_pattern_penalty(self, result: np.ndarray, target: np.ndarray) -> float:
        """Compute pattern consistency penalty"""
        # Simple pattern metric based on autocorrelation
        result_autocorr = self._compute_autocorrelation(result)
        target_autocorr = self._compute_autocorrelation(target)
        
        return abs(result_autocorr - target_autocorr)
    
    def _compute_autocorrelation(self, grid: np.ndarray) -> float:
        """Compute simple autocorrelation measure"""
        h, w = grid.shape
        if h < 2 or w < 2:
            return 0.0
        
        # Compute correlation with shifted version
        shifted = np.roll(grid, 1, axis=0)
        correlation = np.corrcoef(grid.flatten(), shifted.flatten())[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def synthesize_from_examples(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[AtlasSpatialProgram]:
        """Synthesize program from multiple input-output examples"""
        if not examples:
            return None
        
        # Find program that works for all examples
        candidate_programs = []
        
        # Generate candidates from first example
        input_grid, output_grid = examples[0]
        first_program = self.synthesize_program(input_grid, output_grid)
        
        if first_program is None:
            return None
        
        candidate_programs.append(first_program)
        
        # Test on remaining examples
        for input_grid, output_grid in examples[1:]:
            works = False
            
            for program in candidate_programs:
                try:
                    result = program.execute(input_grid)
                    score = self._evaluate_program(result, output_grid)
                    if score < 0.01:
                        works = True
                        break
                except Exception:
                    continue
            
            if not works:
                # Try to synthesize new program for this example
                new_program = self.synthesize_program(input_grid, output_grid)
                if new_program:
                    candidate_programs.append(new_program)
        
        # Return the program with highest success rate across all examples
        best_program = None
        best_overall_score = float('inf')
        
        for program in candidate_programs:
            total_score = 0.0
            valid_examples = 0
            
            for input_grid, output_grid in examples:
                try:
                    result = program.execute(input_grid)
                    score = self._evaluate_program(result, output_grid)
                    total_score += score
                    valid_examples += 1
                except Exception:
                    total_score += 10.0  # Heavy penalty for failure
                    valid_examples += 1
            
            avg_score = total_score / valid_examples if valid_examples > 0 else float('inf')
            
            if avg_score < best_overall_score:
                best_overall_score = avg_score
                best_program = program
        
        return best_program
    
    def get_program_statistics(self) -> Dict[str, Any]:
        """Get synthesis statistics"""
        return {
            'cache_size': len(self.program_cache),
            'success_patterns': len(self.success_patterns),
            'avg_program_length': self._compute_avg_program_length(),
            'operation_frequency': self._compute_operation_frequency()
        }
    
    def _compute_avg_program_length(self) -> float:
        """Compute average program length"""
        if not self.success_patterns:
            return 0.0
        
        total_length = sum(len(p.operations) for p in self.success_patterns)
        return total_length / len(self.success_patterns)
    
    def _compute_operation_frequency(self) -> Dict[str, int]:
        """Compute operation frequency in successful programs"""
        frequency = defaultdict(int)
        
        for program in self.success_patterns:
            for operation, _ in program.operations:
                frequency[operation] += 1
        
        return dict(frequency)


class AtlasPRISMIntegration:
    """Integration layer for ATLAS PRISM system"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.synthesizer = AtlasProgramSynthesizer()
        self.program_bank = deque(maxlen=5000)
        self.success_rate_tracker = defaultdict(list)
        
    def enhanced_training_step(self, input_batch: torch.Tensor,
                              target_batch: torch.Tensor,
                              optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Training step with PRISM enhancement"""
        
        # Standard forward pass
        self.model.train()
        optimizer.zero_grad()
        
        outputs = self.model(input_batch, target_batch, mode='training')
        predictions = outputs['predicted_output']
        
        # Synthesize programs for batch
        programs = self._synthesize_batch_programs(input_batch, target_batch)
        
        # Compute PRISM-enhanced loss
        loss = self._compute_prism_loss(predictions, target_batch, outputs, programs)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update program bank
        self._update_program_bank(programs, predictions, target_batch)
        
        return {
            'loss': loss.item(),
            'synthesized_programs': len(programs),
            'program_bank_size': len(self.program_bank),
            'avg_program_confidence': self._compute_avg_confidence(programs)
        }
    
    def _synthesize_batch_programs(self, input_batch: torch.Tensor,
                                  target_batch: torch.Tensor) -> List[Optional[AtlasSpatialProgram]]:
        """Synthesize programs for entire batch"""
        programs = []
        
        for i in range(input_batch.shape[0]):
            input_np = input_batch[i].cpu().numpy()
            target_np = target_batch[i].cpu().numpy()
            
            # Convert to 2D if needed (take first channel)
            if len(input_np.shape) > 2:
                input_np = input_np[0]
            if len(target_np.shape) > 2:
                target_np = target_np[0]
            
            program = self.synthesizer.synthesize_program(input_np, target_np, max_attempts=20)
            programs.append(program)
        
        return programs
    
    def _compute_prism_loss(self, predictions: torch.Tensor,
                           targets: torch.Tensor,
                           model_outputs: Dict[str, torch.Tensor],
                           programs: List[Optional[AtlasSpatialProgram]]) -> torch.Tensor:
        """Compute PRISM-enhanced loss"""
        
        # Base prediction loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Program consistency loss
        program_loss = self._compute_program_consistency_loss(predictions, targets, programs)
        
        # Spatial transformation loss
        spatial_loss = self._compute_spatial_transformation_loss(model_outputs, programs)
        
        total_loss = base_loss + 0.2 * program_loss + 0.1 * spatial_loss
        
        return total_loss
    
    def _compute_program_consistency_loss(self, predictions: torch.Tensor,
                                         targets: torch.Tensor,
                                         programs: List[Optional[AtlasSpatialProgram]]) -> torch.Tensor:
        """Compute program consistency loss"""
        
        consistency_loss = 0.0
        valid_programs = 0
        
        for i, program in enumerate(programs):
            if program is None or program.confidence < 0.5:
                continue
            
            try:
                # Execute program on input
                input_np = predictions[i].detach().cpu().numpy()
                if len(input_np.shape) > 2:
                    input_np = input_np[0]
                
                program_result = program.execute(input_np)
                program_tensor = torch.tensor(program_result, dtype=predictions.dtype, device=predictions.device)
                
                # Ensure shapes match
                if program_tensor.shape == targets[i].shape:
                    consistency_loss += F.mse_loss(program_tensor, targets[i])
                    valid_programs += 1
                    
            except Exception:
                continue
        
        if valid_programs > 0:
            consistency_loss /= valid_programs
        
        return consistency_loss
    
    def _compute_spatial_transformation_loss(self, model_outputs: Dict[str, torch.Tensor],
                                           programs: List[Optional[AtlasSpatialProgram]]) -> torch.Tensor:
        """Compute spatial transformation consistency loss"""
        
        transformation_loss = 0.0
        
        # Rotation consistency
        if 'rotation_logits' in model_outputs:
            rotation_loss = self._compute_rotation_program_consistency(
                model_outputs['rotation_logits'], programs
            )
            transformation_loss += rotation_loss
        
        # Reflection consistency
        if 'reflection_logits' in model_outputs:
            reflection_loss = self._compute_reflection_program_consistency(
                model_outputs['reflection_logits'], programs
            )
            transformation_loss += reflection_loss
        
        return transformation_loss
    
    def _compute_rotation_program_consistency(self, rotation_logits: torch.Tensor,
                                            programs: List[Optional[AtlasSpatialProgram]]) -> torch.Tensor:
        """Compute rotation-program consistency"""
        
        consistency_loss = 0.0
        valid_programs = 0
        
        for i, program in enumerate(programs):
            if program is None:
                continue
            
            # Check if program contains rotation operations
            rotation_ops = [op for op, _ in program.operations if 'rotate' in op]
            
            if rotation_ops:
                # Expected rotation from program
                total_rotation = 0
                for op in rotation_ops:
                    if op == 'rotate_90':
                        total_rotation += 90
                    elif op == 'rotate_180':
                        total_rotation += 180
                    elif op == 'rotate_270':
                        total_rotation += 270
                
                total_rotation = (total_rotation // 90) % 4
                
                # Model prediction
                rotation_pred = torch.softmax(rotation_logits[i], dim=0)
                target_rotation = torch.zeros_like(rotation_pred)
                target_rotation[total_rotation] = 1.0
                
                consistency_loss += F.kl_div(torch.log(rotation_pred + 1e-8), target_rotation, reduction='sum')
                valid_programs += 1
        
        if valid_programs > 0:
            consistency_loss /= valid_programs
        
        return consistency_loss
    
    def _compute_reflection_program_consistency(self, reflection_logits: torch.Tensor,
                                              programs: List[Optional[AtlasSpatialProgram]]) -> torch.Tensor:
        """Compute reflection-program consistency"""
        
        consistency_loss = 0.0
        valid_programs = 0
        
        for i, program in enumerate(programs):
            if program is None:
                continue
            
            # Check for reflection operations
            has_horizontal = any('flip_horizontal' in op for op, _ in program.operations)
            has_vertical = any('flip_vertical' in op for op, _ in program.operations)
            
            # Expected reflection from program
            if has_horizontal and has_vertical:
                expected_reflection = 2  # Both
            elif has_horizontal:
                expected_reflection = 1  # Horizontal
            elif has_vertical:
                expected_reflection = 2  # Vertical
            else:
                expected_reflection = 0  # None
            
            # Model prediction
            reflection_pred = torch.softmax(reflection_logits[i], dim=0)
            target_reflection = torch.zeros_like(reflection_pred)
            target_reflection[expected_reflection] = 1.0
            
            consistency_loss += F.kl_div(torch.log(reflection_pred + 1e-8), target_reflection, reduction='sum')
            valid_programs += 1
        
        if valid_programs > 0:
            consistency_loss /= valid_programs
        
        return consistency_loss
    
    def _update_program_bank(self, programs: List[Optional[AtlasSpatialProgram]],
                           predictions: torch.Tensor,
                           targets: torch.Tensor):
        """Update program bank with new programs"""
        
        for i, program in enumerate(programs):
            if program is None:
                continue
            
            # Evaluate program performance
            prediction_loss = F.mse_loss(predictions[i], targets[i]).item()
            success = prediction_loss < 0.1
            
            # Update program statistics
            program.success_rate = 1.0 if success else 0.0
            program.spatial_features = self._extract_spatial_features(
                predictions[i], targets[i]
            )
            
            # Add to bank if successful
            if success:
                self.program_bank.append(program)
                
                # Track success rate by operation
                for operation, _ in program.operations:
                    self.success_rate_tracker[operation].append(1.0)
            else:
                # Track failure
                for operation, _ in program.operations:
                    self.success_rate_tracker[operation].append(0.0)
    
    def _extract_spatial_features(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Extract spatial features from prediction-target pair"""
        features = {}
        
        # Basic statistics
        features['mse'] = float(F.mse_loss(prediction, target))
        features['correlation'] = float(self._compute_correlation(prediction, target))
        
        # Spatial properties
        features['edge_similarity'] = float(self._compute_edge_similarity(prediction, target))
        features['pattern_similarity'] = float(self._compute_pattern_similarity(prediction, target))
        
        return features
    
    def _compute_correlation(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute correlation between prediction and target"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
        return correlation.item() if not torch.isnan(correlation) else 0.0
    
    def _compute_edge_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute edge similarity between prediction and target"""
        # Simple edge detection using gradients
        pred_grad_x = pred[..., 1:, :] - pred[..., :-1, :]
        pred_grad_y = pred[..., :, 1:] - pred[..., :, :-1]
        
        target_grad_x = target[..., 1:, :] - target[..., :-1, :]
        target_grad_y = target[..., :, 1:] - target[..., :, :-1]
        
        edge_sim_x = 1.0 - F.mse_loss(pred_grad_x, target_grad_x)
        edge_sim_y = 1.0 - F.mse_loss(pred_grad_y, target_grad_y)
        
        return float((edge_sim_x + edge_sim_y) / 2.0)
    
    def _compute_pattern_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute pattern similarity"""
        # Simple autocorrelation-based pattern similarity
        pred_autocorr = self._compute_tensor_autocorrelation(pred)
        target_autocorr = self._compute_tensor_autocorrelation(target)
        
        return 1.0 - abs(pred_autocorr - target_autocorr)
    
    def _compute_tensor_autocorrelation(self, tensor: torch.Tensor) -> float:
        """Compute autocorrelation for tensor"""
        flat = tensor.flatten()
        if len(flat) < 2:
            return 0.0
        
        shifted = torch.roll(flat, 1)
        correlation = torch.corrcoef(torch.stack([flat, shifted]))[0, 1]
        
        return correlation.item() if not torch.isnan(correlation) else 0.0
    
    def _compute_avg_confidence(self, programs: List[Optional[AtlasSpatialProgram]]) -> float:
        """Compute average confidence of programs"""
        valid_programs = [p for p in programs if p is not None]
        if not valid_programs:
            return 0.0
        
        return sum(p.confidence for p in valid_programs) / len(valid_programs)


def create_atlas_prism_system(model: nn.Module, device: str = 'cuda') -> Dict[str, Any]:
    """Create complete ATLAS PRISM system"""
    
    prism_integration = AtlasPRISMIntegration(model, device)
    
    def atlas_prism_train_step(input_batch: torch.Tensor,
                              target_batch: torch.Tensor,
                              optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """ATLAS PRISM training step"""
        return prism_integration.enhanced_training_step(input_batch, target_batch, optimizer)
    
    def atlas_prism_inference(input_batch: torch.Tensor) -> torch.Tensor:
        """ATLAS PRISM enhanced inference"""
        model.eval()
        with torch.no_grad():
            outputs = model(input_batch, mode='inference')
            return outputs['predicted_output']
    
    def synthesize_program_for_example(input_grid: np.ndarray, 
                                      output_grid: np.ndarray) -> Optional[AtlasSpatialProgram]:
        """Synthesize program for single example"""
        return prism_integration.synthesizer.synthesize_program(input_grid, output_grid)
    
    return {
        'integration': prism_integration,
        'synthesizer': prism_integration.synthesizer,
        'train_step': atlas_prism_train_step,
        'inference': atlas_prism_inference,
        'synthesize_program': synthesize_program_for_example,
        'program_bank': prism_integration.program_bank,
        'name': 'ATLAS_PRISM_v1'
    }