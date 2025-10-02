"""
ATLAS-specific MEPT (Memory-Enhanced Pattern Training) System
Specialized for spatial transformation pattern learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import random


class AtlasSpatialMemory:
    """ATLAS-specific spatial transformation memory"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.spatial_patterns = deque(maxlen=capacity)
        self.transformation_history = deque(maxlen=capacity)
        self.success_rates = defaultdict(float)
        self.pattern_frequencies = defaultdict(int)
        
    def store_transformation(self, input_pattern: torch.Tensor, 
                           output_pattern: torch.Tensor,
                           transformation_type: str,
                           success: bool):
        """Store spatial transformation pattern"""
        pattern_data = {
            'input': input_pattern.detach().cpu(),
            'output': output_pattern.detach().cpu(),
            'transformation': transformation_type,
            'success': success,
            'spatial_features': self._extract_spatial_features(input_pattern)
        }
        
        self.spatial_patterns.append(pattern_data)
        self.transformation_history.append(transformation_type)
        
        if success:
            self.success_rates[transformation_type] = (
                self.success_rates[transformation_type] * 0.9 + 0.1
            )
        else:
            self.success_rates[transformation_type] *= 0.9
        
        self.pattern_frequencies[transformation_type] += 1
    
    def _extract_spatial_features(self, pattern: torch.Tensor) -> Dict[str, float]:
        """Extract spatial features for ATLAS"""
        features = {}
        
        if len(pattern.shape) >= 2:
            # Spatial statistics
            features['mean'] = float(torch.mean(pattern))
            features['std'] = float(torch.std(pattern))
            features['symmetry_h'] = float(self._calculate_symmetry(pattern, axis=0))
            features['symmetry_v'] = float(self._calculate_symmetry(pattern, axis=1))
            features['rotation_invariance'] = float(self._calculate_rotation_invariance(pattern))
            features['spatial_frequency'] = float(self._calculate_spatial_frequency(pattern))
        
        return features
    
    def _calculate_symmetry(self, pattern: torch.Tensor, axis: int) -> float:
        """Calculate symmetry score along axis"""
        flipped = torch.flip(pattern, dims=[axis])
        return float(1.0 - torch.mean(torch.abs(pattern - flipped)))
    
    def _calculate_rotation_invariance(self, pattern: torch.Tensor) -> float:
        """Calculate rotation invariance score"""
        rotated_90 = torch.rot90(pattern, k=1, dims=[-2, -1])
        return float(1.0 - torch.mean(torch.abs(pattern - rotated_90)))
    
    def _calculate_spatial_frequency(self, pattern: torch.Tensor) -> float:
        """Calculate spatial frequency content"""
        if len(pattern.shape) < 2:
            return 0.0
        
        # Simple gradient-based frequency estimation
        grad_x = torch.diff(pattern, dim=-1)
        grad_y = torch.diff(pattern, dim=-2)
        return float(torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y)))
    
    def retrieve_similar_patterns(self, query_pattern: torch.Tensor, 
                                 k: int = 5) -> List[Dict]:
        """Retrieve k most similar spatial patterns"""
        if not self.spatial_patterns:
            return []
        
        query_features = self._extract_spatial_features(query_pattern)
        similarities = []
        
        for pattern_data in self.spatial_patterns:
            similarity = self._calculate_feature_similarity(
                query_features, pattern_data['spatial_features']
            )
            similarities.append((similarity, pattern_data))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [data for _, data in similarities[:k]]
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], 
                                     features2: Dict[str, float]) -> float:
        """Calculate similarity between feature vectors"""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        similarity = 0.0
        for key in common_keys:
            diff = abs(features1[key] - features2[key])
            similarity += 1.0 / (1.0 + diff)
        
        return similarity / len(common_keys)


class AtlasPatternBank:
    """ATLAS-specific pattern bank for spatial transformations"""
    
    def __init__(self):
        self.rotation_patterns = []
        self.reflection_patterns = []
        self.scaling_patterns = []
        self.translation_patterns = []
        self.affine_patterns = []
        
    def add_pattern(self, pattern_type: str, input_pattern: torch.Tensor, 
                   output_pattern: torch.Tensor, transformation_params: Dict):
        """Add pattern to appropriate bank"""
        pattern_data = {
            'input': input_pattern.detach().cpu(),
            'output': output_pattern.detach().cpu(),
            'params': transformation_params,
            'success_count': 0,
            'usage_count': 0
        }
        
        if pattern_type == 'rotation':
            self.rotation_patterns.append(pattern_data)
        elif pattern_type == 'reflection':
            self.reflection_patterns.append(pattern_data)
        elif pattern_type == 'scaling':
            self.scaling_patterns.append(pattern_data)
        elif pattern_type == 'translation':
            self.translation_patterns.append(pattern_data)
        elif pattern_type == 'affine':
            self.affine_patterns.append(pattern_data)
    
    def get_relevant_patterns(self, pattern_type: str, max_patterns: int = 10) -> List[Dict]:
        """Get relevant patterns for given type"""
        if pattern_type == 'rotation':
            patterns = self.rotation_patterns
        elif pattern_type == 'reflection':
            patterns = self.reflection_patterns
        elif pattern_type == 'scaling':
            patterns = self.scaling_patterns
        elif pattern_type == 'translation':
            patterns = self.translation_patterns
        elif pattern_type == 'affine':
            patterns = self.affine_patterns
        else:
            return []
        
        # Sort by success rate and return top patterns
        sorted_patterns = sorted(patterns, 
                               key=lambda x: x['success_count'] / max(x['usage_count'], 1),
                               reverse=True)
        
        return sorted_patterns[:max_patterns]
    
    def update_pattern_performance(self, pattern_type: str, pattern_idx: int, success: bool):
        """Update pattern performance metrics"""
        patterns = getattr(self, f"{pattern_type}_patterns", [])
        if 0 <= pattern_idx < len(patterns):
            patterns[pattern_idx]['usage_count'] += 1
            if success:
                patterns[pattern_idx]['success_count'] += 1


class AtlasMEPTLoss(nn.Module):
    """ATLAS-specific MEPT loss function"""
    
    def __init__(self, base_weight: float = 1.0, 
                 spatial_weight: float = 0.3,
                 transformation_weight: float = 0.2,
                 memory_weight: float = 0.1):
        super().__init__()
        self.base_weight = base_weight
        self.spatial_weight = spatial_weight
        self.transformation_weight = transformation_weight
        self.memory_weight = memory_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                model_outputs: Dict[str, torch.Tensor],
                memory_context: Optional[Dict] = None) -> torch.Tensor:
        """Compute ATLAS-specific MEPT loss"""
        
        # Base prediction loss
        base_loss = F.mse_loss(predictions, targets)
        
        # Spatial consistency loss
        spatial_loss = self._compute_spatial_loss(predictions, targets)
        
        # Transformation consistency loss
        transformation_loss = self._compute_transformation_loss(model_outputs, targets)
        
        # Memory-enhanced loss
        memory_loss = self._compute_memory_loss(predictions, targets, memory_context)
        
        total_loss = (self.base_weight * base_loss +
                     self.spatial_weight * spatial_loss +
                     self.transformation_weight * transformation_loss +
                     self.memory_weight * memory_loss)
        
        return total_loss
    
    def _compute_spatial_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute spatial consistency loss for ATLAS"""
        # Gradient consistency
        pred_grad_x = predictions[..., 1:, :] - predictions[..., :-1, :]
        pred_grad_y = predictions[..., :, 1:] - predictions[..., :, :-1]
        
        target_grad_x = targets[..., 1:, :] - targets[..., :-1, :]
        target_grad_y = targets[..., :, 1:] - targets[..., :, :-1]
        
        grad_loss = (F.mse_loss(pred_grad_x, target_grad_x) + 
                    F.mse_loss(pred_grad_y, target_grad_y))
        
        # Structural similarity
        structure_loss = 1.0 - self._compute_ssim(predictions, targets)
        
        return grad_loss + 0.1 * structure_loss
    
    def _compute_transformation_loss(self, model_outputs: Dict[str, torch.Tensor], 
                                   targets: torch.Tensor) -> torch.Tensor:
        """Compute transformation-specific loss"""
        transformation_loss = 0.0
        
        # Rotation loss
        if 'rotation_logits' in model_outputs:
            rotation_loss = self._compute_rotation_consistency_loss(
                model_outputs['rotation_logits'], targets
            )
            transformation_loss += rotation_loss
        
        # Reflection loss
        if 'reflection_logits' in model_outputs:
            reflection_loss = self._compute_reflection_consistency_loss(
                model_outputs['reflection_logits'], targets
            )
            transformation_loss += reflection_loss
        
        # Affine transformation loss
        if 'theta' in model_outputs:
            affine_loss = self._compute_affine_consistency_loss(
                model_outputs['theta'], targets
            )
            transformation_loss += affine_loss
        
        return transformation_loss
    
    def _compute_memory_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                           memory_context: Optional[Dict]) -> torch.Tensor:
        """Compute memory-enhanced loss"""
        if memory_context is None:
            return torch.tensor(0.0, device=predictions.device)
        
        memory_loss = 0.0
        
        # Similar pattern consistency
        if 'similar_patterns' in memory_context:
            for pattern_data in memory_context['similar_patterns']:
                pattern_output = pattern_data['output'].to(predictions.device)
                if pattern_output.shape == predictions.shape:
                    pattern_similarity = F.cosine_similarity(
                        predictions.flatten(1), pattern_output.flatten(1), dim=1
                    ).mean()
                    memory_loss += (1.0 - pattern_similarity) * 0.1
        
        return memory_loss
    
    def _compute_ssim(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Simplified SSIM computation"""
        mu1 = torch.mean(predictions)
        mu2 = torch.mean(targets)
        sigma1 = torch.var(predictions)
        sigma2 = torch.var(targets)
        sigma12 = torch.mean((predictions - mu1) * (targets - mu2))
        
        c1 = 0.01
        c2 = 0.03
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        return torch.clamp(ssim, 0.0, 1.0)
    
    def _compute_rotation_consistency_loss(self, rotation_logits: torch.Tensor,
                                         targets: torch.Tensor) -> torch.Tensor:
        """Compute rotation consistency loss"""
        # Simple rotation consistency - ensure predictions are stable under rotation
        batch_size = targets.shape[0]
        rotation_loss = 0.0
        
        for i in range(batch_size):
            target = targets[i]
            rotation_pred = torch.softmax(rotation_logits[i], dim=0)
            
            # Penalize uncertain rotations
            entropy = -torch.sum(rotation_pred * torch.log(rotation_pred + 1e-8))
            rotation_loss += entropy * 0.1
        
        return rotation_loss / batch_size
    
    def _compute_reflection_consistency_loss(self, reflection_logits: torch.Tensor,
                                           targets: torch.Tensor) -> torch.Tensor:
        """Compute reflection consistency loss"""
        # Similar to rotation consistency
        batch_size = targets.shape[0]
        reflection_loss = 0.0
        
        for i in range(batch_size):
            target = targets[i]
            reflection_pred = torch.softmax(reflection_logits[i], dim=0)
            
            # Penalize uncertain reflections
            entropy = -torch.sum(reflection_pred * torch.log(reflection_pred + 1e-8))
            reflection_loss += entropy * 0.1
        
        return reflection_loss / batch_size
    
    def _compute_affine_consistency_loss(self, theta: torch.Tensor,
                                       targets: torch.Tensor) -> torch.Tensor:
        """Compute affine transformation consistency loss"""
        # Regularize affine parameters to prevent extreme transformations
        identity = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=theta.dtype, device=theta.device)
        identity = identity.unsqueeze(0).expand(theta.shape[0], -1, -1)
        
        affine_reg = F.mse_loss(theta, identity)
        return affine_reg * 0.1


def create_atlas_mept_system(model, device: str = 'cuda') -> Dict[str, Any]:
    """Create complete ATLAS MEPT system"""
    
    # Initialize ATLAS-specific components
    spatial_memory = AtlasSpatialMemory(capacity=10000)
    pattern_bank = AtlasPatternBank()
    mept_loss = AtlasMEPTLoss()
    
    def atlas_mept_train_step(input_batch: torch.Tensor, 
                             target_batch: torch.Tensor,
                             model: nn.Module,
                             optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """ATLAS-specific MEPT training step"""
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_batch, target_batch, mode='training')
        predictions = outputs['predicted_output']
        
        # Retrieve similar patterns from memory
        memory_context = {'similar_patterns': []}
        for i in range(input_batch.shape[0]):
            similar = spatial_memory.retrieve_similar_patterns(input_batch[i], k=3)
            memory_context['similar_patterns'].extend(similar)
        
        # Compute ATLAS MEPT loss
        loss = mept_loss(predictions, target_batch, outputs, memory_context)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Store patterns in memory
        with torch.no_grad():
            for i in range(input_batch.shape[0]):
                # Determine transformation type from model outputs
                if 'rotation_logits' in outputs:
                    rotation_idx = torch.argmax(outputs['rotation_logits'][i])
                    transformation_type = f"rotation_{rotation_idx.item() * 90}"
                else:
                    transformation_type = "spatial_transform"
                
                # Calculate success (simplified)
                pattern_loss = F.mse_loss(predictions[i], target_batch[i])
                success = pattern_loss.item() < 0.1
                
                spatial_memory.store_transformation(
                    input_batch[i], target_batch[i], transformation_type, success
                )
        
        return {
            'loss': loss.item(),
            'memory_size': len(spatial_memory.spatial_patterns),
            'transformation_types': len(spatial_memory.success_rates)
        }
    
    def atlas_mept_inference(input_batch: torch.Tensor,
                           model: nn.Module) -> torch.Tensor:
        """ATLAS-specific MEPT inference with memory enhancement"""
        
        model.eval()
        with torch.no_grad():
            # Get base prediction
            outputs = model(input_batch, mode='inference')
            base_predictions = outputs['predicted_output']
            
            # Enhance with memory patterns
            enhanced_predictions = base_predictions.clone()
            
            for i in range(input_batch.shape[0]):
                similar_patterns = spatial_memory.retrieve_similar_patterns(input_batch[i], k=3)
                
                if similar_patterns:
                    # Weighted combination with similar patterns
                    pattern_weight = 0.1
                    for pattern_data in similar_patterns:
                        if pattern_data['output'].shape == enhanced_predictions[i].shape:
                            pattern_output = pattern_data['output'].to(enhanced_predictions.device)
                            enhanced_predictions[i] = (
                                (1 - pattern_weight) * enhanced_predictions[i] +
                                pattern_weight * pattern_output
                            )
                            pattern_weight *= 0.5  # Decreasing weight
            
            return enhanced_predictions
    
    return {
        'spatial_memory': spatial_memory,
        'pattern_bank': pattern_bank,
        'loss_function': mept_loss,
        'train_step': atlas_mept_train_step,
        'inference': atlas_mept_inference,
        'name': 'ATLAS_MEPT_v1'
    }