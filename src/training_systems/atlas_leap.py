"""
ATLAS-specific LEAP (Learning Enhancement through Adaptive Patterns) System
Specialized for spatial transformation pattern adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import random
import math


class AtlasSpatialPatternGenerator:
    """Generates adaptive spatial patterns for ATLAS"""
    
    def __init__(self, max_grid_size: int = 30):
        self.max_grid_size = max_grid_size
        self.pattern_history = deque(maxlen=1000)
        self.success_patterns = []
        self.failure_patterns = []
        
        # Spatial transformation types
        self.transformations = [
            'rotation_90', 'rotation_180', 'rotation_270',
            'flip_horizontal', 'flip_vertical', 'transpose',
            'scale_up', 'scale_down', 'translate'
        ]
        
    def generate_spatial_pattern(self, base_pattern: torch.Tensor, 
                                difficulty: float = 0.5) -> Dict[str, Any]:
        """Generate adaptive spatial pattern based on difficulty"""
        
        pattern_data = {
            'input': base_pattern.clone(),
            'transformation': self._select_transformation(difficulty),
            'parameters': {},
            'difficulty': difficulty
        }
        
        # Apply transformation
        transformed = self._apply_transformation(
            base_pattern, pattern_data['transformation'], difficulty
        )
        
        pattern_data['output'] = transformed
        pattern_data['spatial_features'] = self._extract_spatial_features(base_pattern, transformed)
        
        return pattern_data
    
    def _select_transformation(self, difficulty: float) -> str:
        """Select transformation based on difficulty"""
        if difficulty < 0.3:
            # Easy transformations
            return random.choice(['rotation_90', 'rotation_180', 'flip_horizontal', 'flip_vertical'])
        elif difficulty < 0.7:
            # Medium transformations
            return random.choice(['rotation_270', 'transpose', 'scale_up', 'scale_down'])
        else:
            # Hard transformations (combinations)
            return random.choice(self.transformations)
    
    def _apply_transformation(self, pattern: torch.Tensor, 
                            transformation: str, difficulty: float) -> torch.Tensor:
        """Apply spatial transformation to pattern"""
        
        result = pattern.clone()
        
        if transformation == 'rotation_90':
            result = torch.rot90(result, k=1, dims=[-2, -1])
        elif transformation == 'rotation_180':
            result = torch.rot90(result, k=2, dims=[-2, -1])
        elif transformation == 'rotation_270':
            result = torch.rot90(result, k=3, dims=[-2, -1])
        elif transformation == 'flip_horizontal':
            result = torch.flip(result, dims=[-1])
        elif transformation == 'flip_vertical':
            result = torch.flip(result, dims=[-2])
        elif transformation == 'transpose':
            if len(result.shape) >= 2:
                result = torch.transpose(result, -2, -1)
        elif transformation == 'scale_up':
            result = self._scale_pattern(result, factor=2.0)
        elif transformation == 'scale_down':
            result = self._scale_pattern(result, factor=0.5)
        elif transformation == 'translate':
            dx, dy = self._generate_translation(difficulty)
            result = self._translate_pattern(result, dx, dy)
        
        return result
    
    def _scale_pattern(self, pattern: torch.Tensor, factor: float) -> torch.Tensor:
        """Scale pattern by given factor"""
        if factor == 1.0:
            return pattern
        
        # Use interpolation for scaling
        if len(pattern.shape) >= 3:  # Has channel dimension
            scaled = F.interpolate(
                pattern.unsqueeze(0), 
                scale_factor=factor, 
                mode='nearest'
            ).squeeze(0)
        else:
            # Add channel dimension temporarily
            scaled = F.interpolate(
                pattern.unsqueeze(0).unsqueeze(0), 
                scale_factor=factor, 
                mode='nearest'
            ).squeeze(0).squeeze(0)
        
        return scaled
    
    def _generate_translation(self, difficulty: float) -> Tuple[int, int]:
        """Generate translation parameters based on difficulty"""
        max_shift = int(3 * difficulty + 1)
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        return dx, dy
    
    def _translate_pattern(self, pattern: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        """Translate pattern by dx, dy"""
        if dx == 0 and dy == 0:
            return pattern
        
        result = torch.zeros_like(pattern)
        h, w = pattern.shape[-2:]
        
        # Calculate source and destination regions
        src_y_start = max(0, -dy)
        src_y_end = min(h, h - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(w, w - dx)
        
        dst_y_start = max(0, dy)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, dx)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        if len(pattern.shape) >= 3:  # Has channel dimension
            result[..., dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                pattern[..., src_y_start:src_y_end, src_x_start:src_x_end]
        else:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                pattern[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return result
    
    def _extract_spatial_features(self, input_pattern: torch.Tensor, 
                                 output_pattern: torch.Tensor) -> Dict[str, float]:
        """Extract spatial features from pattern pair"""
        features = {}
        
        # Basic statistics
        features['input_mean'] = float(torch.mean(input_pattern))
        features['output_mean'] = float(torch.mean(output_pattern))
        features['correlation'] = float(self._compute_correlation(input_pattern, output_pattern))
        
        # Spatial properties
        features['symmetry_change'] = self._compute_symmetry_change(input_pattern, output_pattern)
        features['orientation_change'] = self._compute_orientation_change(input_pattern, output_pattern)
        features['scale_change'] = self._compute_scale_change(input_pattern, output_pattern)
        
        return features
    
    def _compute_correlation(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """Compute correlation between two patterns"""
        if pattern1.shape != pattern2.shape:
            return 0.0
        
        p1_flat = pattern1.flatten()
        p2_flat = pattern2.flatten()
        
        correlation = torch.corrcoef(torch.stack([p1_flat, p2_flat]))[0, 1]
        return float(correlation) if not torch.isnan(correlation) else 0.0
    
    def _compute_symmetry_change(self, input_pattern: torch.Tensor, 
                               output_pattern: torch.Tensor) -> float:
        """Compute change in symmetry"""
        input_sym = self._calculate_symmetry(input_pattern)
        output_sym = self._calculate_symmetry(output_pattern)
        return float(abs(output_sym - input_sym))
    
    def _calculate_symmetry(self, pattern: torch.Tensor) -> float:
        """Calculate overall symmetry of pattern"""
        h_sym = 1.0 - torch.mean(torch.abs(pattern - torch.flip(pattern, dims=[-1])))
        v_sym = 1.0 - torch.mean(torch.abs(pattern - torch.flip(pattern, dims=[-2])))
        return float((h_sym + v_sym) / 2.0)
    
    def _compute_orientation_change(self, input_pattern: torch.Tensor, 
                                  output_pattern: torch.Tensor) -> float:
        """Compute orientation change between patterns"""
        # Simple rotation detection
        rotations = [0, 1, 2, 3]  # 0°, 90°, 180°, 270°
        min_diff = float('inf')
        
        for k in rotations:
            rotated_input = torch.rot90(input_pattern, k=k, dims=[-2, -1])
            if rotated_input.shape == output_pattern.shape:
                diff = torch.mean(torch.abs(rotated_input - output_pattern))
                min_diff = min(min_diff, float(diff))
        
        return min_diff
    
    def _compute_scale_change(self, input_pattern: torch.Tensor, 
                            output_pattern: torch.Tensor) -> float:
        """Compute scale change between patterns"""
        input_size = input_pattern.shape[-1] * input_pattern.shape[-2]
        output_size = output_pattern.shape[-1] * output_pattern.shape[-2]
        
        if input_size == 0:
            return 0.0
        
        scale_ratio = output_size / input_size
        return float(abs(math.log(scale_ratio)))
    
    def update_pattern_success(self, pattern_data: Dict[str, Any], success: bool):
        """Update pattern success statistics"""
        if success:
            self.success_patterns.append(pattern_data)
            if len(self.success_patterns) > 500:
                self.success_patterns.pop(0)
        else:
            self.failure_patterns.append(pattern_data)
            if len(self.failure_patterns) > 200:
                self.failure_patterns.pop(0)
    
    def get_adaptive_difficulty(self, current_accuracy: float) -> float:
        """Get adaptive difficulty based on current performance"""
        # Increase difficulty as accuracy improves
        if current_accuracy < 0.3:
            return 0.2
        elif current_accuracy < 0.6:
            return 0.4
        elif current_accuracy < 0.8:
            return 0.6
        else:
            return 0.8


class AtlasLEAPTrainer:
    """ATLAS-specific LEAP trainer for spatial transformations"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.pattern_generator = AtlasSpatialPatternGenerator()
        self.adaptation_history = []
        self.performance_tracker = defaultdict(list)
        
    def adaptive_training_step(self, input_batch: torch.Tensor, 
                              target_batch: torch.Tensor,
                              optimizer: torch.optim.Optimizer,
                              current_accuracy: float) -> Dict[str, Any]:
        """Perform adaptive training step with LEAP"""
        
        # Generate adaptive patterns
        adaptive_difficulty = self.pattern_generator.get_adaptive_difficulty(current_accuracy)
        
        # Create enhanced batch with adaptive patterns
        enhanced_inputs, enhanced_targets = self._create_adaptive_batch(
            input_batch, target_batch, adaptive_difficulty
        )
        
        # Forward pass
        self.model.train()
        optimizer.zero_grad()
        
        outputs = self.model(enhanced_inputs, enhanced_targets, mode='training')
        predictions = outputs['predicted_output']
        
        # Compute adaptive loss
        loss = self._compute_adaptive_loss(predictions, enhanced_targets, outputs, adaptive_difficulty)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update pattern success rates
        self._update_pattern_performance(predictions, enhanced_targets)
        
        return {
            'loss': loss.item(),
            'adaptive_difficulty': adaptive_difficulty,
            'enhanced_batch_size': enhanced_inputs.shape[0],
            'spatial_accuracy': self._compute_spatial_accuracy(predictions, enhanced_targets)
        }
    
    def _create_adaptive_batch(self, input_batch: torch.Tensor, 
                              target_batch: torch.Tensor, 
                              difficulty: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create enhanced batch with adaptive patterns"""
        
        batch_size = input_batch.shape[0]
        enhanced_inputs = [input_batch]
        enhanced_targets = [target_batch]
        
        # Generate adaptive patterns
        for i in range(batch_size):
            # Create 2 adaptive variations per sample
            for _ in range(2):
                pattern_data = self.pattern_generator.generate_spatial_pattern(
                    input_batch[i], difficulty
                )
                
                enhanced_inputs.append(pattern_data['input'].unsqueeze(0))
                enhanced_targets.append(pattern_data['output'].unsqueeze(0))
        
        # Concatenate all patterns
        all_inputs = torch.cat(enhanced_inputs, dim=0).to(self.device)
        all_targets = torch.cat(enhanced_targets, dim=0).to(self.device)
        
        return all_inputs, all_targets
    
    def _compute_adaptive_loss(self, predictions: torch.Tensor, 
                              targets: torch.Tensor,
                              model_outputs: Dict[str, torch.Tensor],
                              difficulty: float) -> torch.Tensor:
        """Compute adaptive loss for ATLAS LEAP"""
        
        # Base MSE loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Spatial consistency loss
        spatial_loss = self._compute_spatial_consistency_loss(predictions, targets)
        
        # Transformation-specific loss
        transformation_loss = self._compute_transformation_loss(model_outputs, targets)
        
        # Adaptive weighting based on difficulty
        spatial_weight = 0.2 + 0.3 * difficulty
        transformation_weight = 0.1 + 0.2 * difficulty
        
        total_loss = (base_loss + 
                     spatial_weight * spatial_loss + 
                     transformation_weight * transformation_loss)
        
        return total_loss
    
    def _compute_spatial_consistency_loss(self, predictions: torch.Tensor, 
                                        targets: torch.Tensor) -> torch.Tensor:
        """Compute spatial consistency loss"""
        
        # Gradient consistency
        pred_grad_x = predictions[..., 1:, :] - predictions[..., :-1, :]
        pred_grad_y = predictions[..., :, 1:] - predictions[..., :, :-1]
        
        target_grad_x = targets[..., 1:, :] - targets[..., :-1, :]
        target_grad_y = targets[..., :, 1:] - targets[..., :, :-1]
        
        grad_loss = (F.mse_loss(pred_grad_x, target_grad_x) + 
                    F.mse_loss(pred_grad_y, target_grad_y))
        
        # Edge preservation loss
        edge_loss = self._compute_edge_preservation_loss(predictions, targets)
        
        return grad_loss + 0.1 * edge_loss
    
    def _compute_edge_preservation_loss(self, predictions: torch.Tensor, 
                                      targets: torch.Tensor) -> torch.Tensor:
        """Compute edge preservation loss for spatial transformations"""
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=predictions.dtype, device=predictions.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=predictions.dtype, device=predictions.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        pred_edges_x = F.conv2d(predictions.sum(dim=1, keepdim=True), sobel_x, padding=1)
        pred_edges_y = F.conv2d(predictions.sum(dim=1, keepdim=True), sobel_y, padding=1)
        
        target_edges_x = F.conv2d(targets.sum(dim=1, keepdim=True), sobel_x, padding=1)
        target_edges_y = F.conv2d(targets.sum(dim=1, keepdim=True), sobel_y, padding=1)
        
        edge_loss = (F.mse_loss(pred_edges_x, target_edges_x) + 
                    F.mse_loss(pred_edges_y, target_edges_y))
        
        return edge_loss
    
    def _compute_transformation_loss(self, model_outputs: Dict[str, torch.Tensor], 
                                   targets: torch.Tensor) -> torch.Tensor:
        """Compute transformation-specific loss"""
        
        transformation_loss = 0.0
        
        # Rotation consistency
        if 'rotation_logits' in model_outputs:
            rotation_loss = self._compute_rotation_loss(model_outputs['rotation_logits'])
            transformation_loss += rotation_loss
        
        # Reflection consistency
        if 'reflection_logits' in model_outputs:
            reflection_loss = self._compute_reflection_loss(model_outputs['reflection_logits'])
            transformation_loss += reflection_loss
        
        # Affine parameter regularization
        if 'theta' in model_outputs:
            affine_loss = self._compute_affine_regularization(model_outputs['theta'])
            transformation_loss += affine_loss
        
        return transformation_loss
    
    def _compute_rotation_loss(self, rotation_logits: torch.Tensor) -> torch.Tensor:
        """Compute rotation-specific loss"""
        # Encourage confident rotation predictions
        rotation_probs = F.softmax(rotation_logits, dim=-1)
        entropy = -torch.sum(rotation_probs * torch.log(rotation_probs + 1e-8), dim=-1)
        return torch.mean(entropy) * 0.1
    
    def _compute_reflection_loss(self, reflection_logits: torch.Tensor) -> torch.Tensor:
        """Compute reflection-specific loss"""
        # Encourage confident reflection predictions
        reflection_probs = F.softmax(reflection_logits, dim=-1)
        entropy = -torch.sum(reflection_probs * torch.log(reflection_probs + 1e-8), dim=-1)
        return torch.mean(entropy) * 0.1
    
    def _compute_affine_regularization(self, theta: torch.Tensor) -> torch.Tensor:
        """Regularize affine transformation parameters"""
        # Encourage parameters close to identity
        identity = torch.tensor([[1, 0, 0], [0, 1, 0]], 
                               dtype=theta.dtype, device=theta.device)
        identity = identity.unsqueeze(0).expand(theta.shape[0], -1, -1)
        
        return F.mse_loss(theta, identity) * 0.05
    
    def _update_pattern_performance(self, predictions: torch.Tensor, 
                                   targets: torch.Tensor):
        """Update pattern performance tracking"""
        
        batch_size = predictions.shape[0]
        
        for i in range(batch_size):
            pattern_loss = F.mse_loss(predictions[i], targets[i])
            success = pattern_loss.item() < 0.1
            
            # Update pattern generator
            pattern_data = {
                'loss': pattern_loss.item(),
                'success': success
            }
            
            self.pattern_generator.update_pattern_success(pattern_data, success)
    
    def _compute_spatial_accuracy(self, predictions: torch.Tensor, 
                                 targets: torch.Tensor) -> float:
        """Compute spatial accuracy metric"""
        
        # Exact match accuracy
        exact_matches = torch.all(
            torch.abs(predictions - targets) < 0.1, 
            dim=list(range(1, len(predictions.shape)))
        )
        
        exact_accuracy = torch.mean(exact_matches.float()).item()
        
        # Spatial correlation accuracy
        batch_size = predictions.shape[0]
        correlation_sum = 0.0
        
        for i in range(batch_size):
            pred_flat = predictions[i].flatten()
            target_flat = targets[i].flatten()
            
            correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
            if not torch.isnan(correlation):
                correlation_sum += correlation.item()
        
        correlation_accuracy = correlation_sum / batch_size if batch_size > 0 else 0.0
        
        return (exact_accuracy + correlation_accuracy) / 2.0


def create_atlas_leap_system(model: nn.Module, device: str = 'cuda') -> Dict[str, Any]:
    """Create complete ATLAS LEAP system"""
    
    leap_trainer = AtlasLEAPTrainer(model, device)
    
    def atlas_leap_train_step(input_batch: torch.Tensor, 
                             target_batch: torch.Tensor,
                             optimizer: torch.optim.Optimizer,
                             current_accuracy: float = 0.5) -> Dict[str, Any]:
        """ATLAS LEAP training step"""
        return leap_trainer.adaptive_training_step(
            input_batch, target_batch, optimizer, current_accuracy
        )
    
    def atlas_leap_inference(input_batch: torch.Tensor) -> torch.Tensor:
        """ATLAS LEAP enhanced inference"""
        model.eval()
        with torch.no_grad():
            outputs = model(input_batch, mode='inference')
            return outputs['predicted_output']
    
    return {
        'trainer': leap_trainer,
        'pattern_generator': leap_trainer.pattern_generator,
        'train_step': atlas_leap_train_step,
        'inference': atlas_leap_inference,
        'name': 'ATLAS_LEAP_v1'
    }