"""
PROMETHEUS-specific MEPT (Memory-Enhanced Pattern Training) System
Specialized for meta-learning and ensemble coordination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict, deque

# Try to import base MEPT system
try:
    from .mept_system import MEPTReplayBuffer, PatternBank, MEPTLoss
    BASE_MEPT_AVAILABLE = True
except ImportError:
    BASE_MEPT_AVAILABLE = False


class PrometheusSpecializedReplayBuffer:
    """PROMETHEUS-specific replay buffer for meta-learning patterns"""
    
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.meta_patterns = deque(maxlen=capacity // 4)  # Meta-learning specific patterns
        self.ensemble_patterns = deque(maxlen=capacity // 4)  # Ensemble coordination patterns
        
    def add_pattern(self, input_grid: torch.Tensor, output_grid: torch.Tensor, 
                   meta_info: Dict = None, ensemble_context: Dict = None):
        """Add pattern with meta-learning context"""
        pattern = {
            'input': input_grid.clone(),
            'output': output_grid.clone(),
            'meta_info': meta_info or {},
            'ensemble_context': ensemble_context or {},
            'timestamp': len(self.buffer),
            'meta_learning_score': 0.0,
            'ensemble_coordination_score': 0.0
        }
        
        self.buffer.append(pattern)
        
        # Add to specialized buffers based on context
        if meta_info and meta_info.get('is_meta_pattern', False):
            self.meta_patterns.append(pattern)
        if ensemble_context and ensemble_context.get('is_ensemble_pattern', False):
            self.ensemble_patterns.append(pattern)
    
    def sample(self, batch_size: int, meta_bias: float = 0.3) -> List[Dict]:
        """Sample with bias toward meta-learning patterns"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        samples = []
        meta_count = int(batch_size * meta_bias)
        ensemble_count = int(batch_size * 0.2)
        regular_count = batch_size - meta_count - ensemble_count
        
        # Sample meta-learning patterns
        if len(self.meta_patterns) > 0:
            meta_samples = random.sample(list(self.meta_patterns), 
                                       min(meta_count, len(self.meta_patterns)))
            samples.extend(meta_samples)
        
        # Sample ensemble patterns
        if len(self.ensemble_patterns) > 0:
            ensemble_samples = random.sample(list(self.ensemble_patterns),
                                           min(ensemble_count, len(self.ensemble_patterns)))
            samples.extend(ensemble_samples)
        
        # Fill remaining with regular patterns
        remaining_needed = batch_size - len(samples)
        if remaining_needed > 0:
            regular_samples = random.sample(list(self.buffer), remaining_needed)
            samples.extend(regular_samples)
        
        return samples[:batch_size]
    
    def update_scores(self, patterns: List[Dict], meta_scores: List[float], 
                     ensemble_scores: List[float]):
        """Update meta-learning and ensemble scores for patterns"""
        for pattern, meta_score, ensemble_score in zip(patterns, meta_scores, ensemble_scores):
            pattern['meta_learning_score'] = meta_score
            pattern['ensemble_coordination_score'] = ensemble_score
    
    def get_meta_stats(self) -> Dict:
        """Get meta-learning statistics"""
        if not self.buffer:
            return {'size': 0, 'meta_patterns': 0, 'ensemble_patterns': 0}
        
        meta_scores = [p['meta_learning_score'] for p in self.buffer]
        ensemble_scores = [p['ensemble_coordination_score'] for p in self.buffer]
        
        return {
            'size': len(self.buffer),
            'meta_patterns': len(self.meta_patterns),
            'ensemble_patterns': len(self.ensemble_patterns),
            'avg_meta_score': np.mean(meta_scores) if meta_scores else 0.0,
            'avg_ensemble_score': np.mean(ensemble_scores) if ensemble_scores else 0.0
        }


class PrometheusMetaPatternBank:
    """PROMETHEUS-specific pattern bank for meta-learning"""
    
    def __init__(self, capacity: int = 10000, embedding_dim: int = 256):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.patterns = deque(maxlen=capacity)
        self.meta_embeddings = deque(maxlen=capacity)
        self.ensemble_strategies = deque(maxlen=capacity)
        
    def add_meta_pattern(self, pattern: torch.Tensor, meta_embedding: torch.Tensor,
                        ensemble_strategy: Dict = None):
        """Add meta-learning pattern with ensemble strategy"""
        self.patterns.append(pattern.clone())
        self.meta_embeddings.append(meta_embedding.clone())
        self.ensemble_strategies.append(ensemble_strategy or {})
    
    def find_similar_meta_patterns(self, query_embedding: torch.Tensor, 
                                  k: int = 5) -> List[Tuple[torch.Tensor, Dict]]:
        """Find similar meta-learning patterns"""
        if not self.meta_embeddings:
            return []
        
        embeddings = torch.stack(list(self.meta_embeddings))
        similarities = F.cosine_similarity(query_embedding.unsqueeze(0), embeddings, dim=1)
        top_k_indices = similarities.topk(min(k, len(similarities)))[1]
        
        results = []
        for idx in top_k_indices:
            pattern = self.patterns[idx]
            strategy = self.ensemble_strategies[idx]
            results.append((pattern, strategy))
        
        return results
    
    def get_ensemble_strategies(self) -> List[Dict]:
        """Get all ensemble coordination strategies"""
        return list(self.ensemble_strategies)


class PrometheusSpecializedLoss(nn.Module):
    """PROMETHEUS-specific loss function for meta-learning and ensemble coordination"""
    
    def __init__(self, base_weight: float = 1.0, meta_weight: float = 0.5, 
                 ensemble_weight: float = 0.3):
        super().__init__()
        self.base_weight = base_weight
        self.meta_weight = meta_weight
        self.ensemble_weight = ensemble_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                inputs: torch.Tensor, model_outputs: Dict,
                meta_context: Dict = None, ensemble_context: Dict = None) -> Dict:
        
        # Base cross-entropy loss
        base_loss = F.cross_entropy(predictions, targets)
        
        # Meta-learning consistency loss
        meta_loss = torch.tensor(0.0, device=predictions.device)
        if meta_context and 'meta_predictions' in model_outputs:
            meta_predictions = model_outputs['meta_predictions']
            # Encourage consistency across meta-learning iterations
            meta_loss = F.mse_loss(predictions, meta_predictions.detach())
        
        # Ensemble coordination loss
        ensemble_loss = torch.tensor(0.0, device=predictions.device)
        if ensemble_context and 'ensemble_weights' in model_outputs:
            ensemble_weights = model_outputs['ensemble_weights']
            # Encourage balanced ensemble coordination
            target_weights = torch.ones_like(ensemble_weights) / ensemble_weights.size(-1)
            ensemble_loss = F.kl_div(F.log_softmax(ensemble_weights, dim=-1),
                                   target_weights, reduction='batchmean')
        
        # Pattern diversity loss
        diversity_loss = torch.tensor(0.0, device=predictions.device)
        if 'pattern_features' in model_outputs:
            pattern_features = model_outputs['pattern_features']
            # Encourage diverse pattern representations
            if pattern_features.size(0) > 1:
                pairwise_sim = F.cosine_similarity(
                    pattern_features.unsqueeze(1), 
                    pattern_features.unsqueeze(0), 
                    dim=2
                )
                # Penalize high similarity between different patterns
                mask = ~torch.eye(pairwise_sim.size(0), dtype=torch.bool, device=pairwise_sim.device)
                diversity_loss = pairwise_sim[mask].mean()
        
        total_loss = (self.base_weight * base_loss + 
                     self.meta_weight * meta_loss + 
                     self.ensemble_weight * ensemble_loss + 
                     0.1 * diversity_loss)
        
        # Calculate exact matches for metrics
        pred_classes = predictions.argmax(dim=1)
        exact_matches = (pred_classes == targets).all(dim=[-2, -1]).float()
        
        return {
            'total': total_loss,
            'base_loss': base_loss,
            'meta_loss': meta_loss,
            'ensemble_loss': ensemble_loss,
            'diversity_loss': diversity_loss,
            'exact_count': exact_matches.sum(),
            'total_samples': targets.size(0)
        }


def create_prometheus_mept_system(capacity: int = 50000, pattern_bank_size: int = 10000,
                                 transformation_penalty: float = 0.5,
                                 exact_match_bonus: float = 2.0) -> Dict:
    """Create PROMETHEUS-specific MEPT system"""
    
    replay_buffer = PrometheusSpecializedReplayBuffer(capacity=capacity)
    pattern_bank = PrometheusMetaPatternBank(capacity=pattern_bank_size)
    loss_fn = PrometheusSpecializedLoss()
    
    return {
        'replay_buffer': replay_buffer,
        'pattern_bank': pattern_bank,
        'loss_function': loss_fn,  # Use 'loss_function' for PROMETHEUS
        'meta_learning_enabled': True,
        'ensemble_coordination_enabled': True
    }


# Meta-learning pattern generation for PROMETHEUS
class PrometheusMetaPatternGenerator:
    """Generate meta-learning specific patterns for PROMETHEUS"""
    
    @staticmethod
    def generate_meta_learning_patterns(num_patterns: int = 100) -> List[Dict]:
        """Generate patterns that require meta-learning"""
        patterns = []
        
        for i in range(num_patterns):
            size = random.choice([6, 8, 10, 12])
            
            # Meta-learning pattern types
            pattern_type = random.choice([
                'few_shot_learning', 'adaptation', 'multi_task', 'transfer_learning'
            ])
            
            if pattern_type == 'few_shot_learning':
                # Create patterns that require learning from few examples
                base_pattern = np.random.randint(0, 3, (size//2, size//2))
                input_grid = np.zeros((size, size), dtype=np.int64)
                output_grid = np.zeros((size, size), dtype=np.int64)
                
                # Place base pattern in different quadrants
                input_grid[:size//2, :size//2] = base_pattern
                output_grid[size//2:, size//2:] = base_pattern  # Move to opposite corner
                
            elif pattern_type == 'adaptation':
                # Patterns requiring adaptation to new rules
                input_grid = np.random.randint(0, 5, (size, size))
                # Apply complex transformation that requires adaptation
                output_grid = np.rot90(np.flip(input_grid, axis=0), k=2)
                
            elif pattern_type == 'multi_task':
                # Patterns combining multiple transformations
                input_grid = np.random.randint(0, 4, (size, size))
                output_grid = input_grid.copy()
                # Apply multiple transformations
                output_grid = np.rot90(output_grid)
                output_grid = np.where(output_grid > 2, 0, output_grid + 1)
                
            else:  # transfer_learning
                # Patterns that benefit from knowledge transfer
                input_grid = np.zeros((size, size), dtype=np.int64)
                output_grid = np.zeros((size, size), dtype=np.int64)
                
                # Create symmetric patterns
                for x in range(size//2):
                    for y in range(size//2):
                        val = random.randint(1, 4)
                        input_grid[x, y] = val
                        input_grid[size-1-x, y] = val
                        input_grid[x, size-1-y] = val
                        input_grid[size-1-x, size-1-y] = val
                        
                        # Output is input with colors shifted
                        output_grid[x, y] = (val % 4) + 1
                        output_grid[size-1-x, y] = (val % 4) + 1
                        output_grid[x, size-1-y] = (val % 4) + 1
                        output_grid[size-1-x, size-1-y] = (val % 4) + 1
            
            patterns.append({
                'inputs': input_grid,
                'outputs': output_grid,
                'meta_info': {
                    'pattern_type': pattern_type,
                    'requires_meta_learning': True,
                    'is_meta_pattern': True
                }
            })
        
        return patterns
    
    @staticmethod
    def generate_ensemble_coordination_patterns(num_patterns: int = 50) -> List[Dict]:
        """Generate patterns requiring ensemble coordination"""
        patterns = []
        
        for i in range(num_patterns):
            size = random.choice([8, 10, 12])
            
            # Create patterns that benefit from multiple model perspectives
            input_grid = np.random.randint(0, 6, (size, size))
            output_grid = np.zeros_like(input_grid)
            
            # Complex multi-step transformation requiring ensemble
            # Step 1: Color-based transformation (IRIS strength)
            mask1 = input_grid == 1
            output_grid[mask1] = 5
            
            # Step 2: Spatial transformation (MINERVA strength)
            mask2 = input_grid == 2
            coords = np.where(mask2)
            for x, y in zip(coords[0], coords[1]):
                if x < size//2 and y < size//2:
                    output_grid[x + size//2, y + size//2] = 3
            
            # Step 3: Sequential pattern (CHRONOS strength)
            mask3 = input_grid == 3
            coords = np.where(mask3)
            for i, (x, y) in enumerate(zip(coords[0], coords[1])):
                output_grid[x, y] = (i % 4) + 1
            
            # Step 4: Object-based transformation (ATLAS strength)
            mask4 = input_grid >= 4
            output_grid[mask4] = input_grid[mask4] - 1
            
            patterns.append({
                'inputs': input_grid,
                'outputs': output_grid,
                'ensemble_context': {
                    'requires_ensemble': True,
                    'model_contributions': ['IRIS', 'MINERVA', 'CHRONOS', 'ATLAS'],
                    'is_ensemble_pattern': True
                }
            })
        
        return patterns