"""
CHRONOS MEPT Training System
Memory-Enhanced Pattern Training - Sequence Memory and Replay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Deque
from collections import deque, defaultdict
import random
import numpy as np
from dataclasses import dataclass, field


@dataclass
class TemporalMemory:
    """Stores temporal sequence with metadata"""
    sequence: List[torch.Tensor]
    pattern_type: str
    timestamp: int
    access_count: int = 0
    importance_score: float = 1.0
    temporal_features: Dict[str, Any] = field(default_factory=dict)
    replay_priority: float = 1.0


@dataclass 
class SequenceTransition:
    """Represents a temporal state transition"""
    prev_state: torch.Tensor
    action: str
    next_state: torch.Tensor
    time_step: int
    reward: float = 0.0


class ChronosMEPT:
    """MEPT system specialized for CHRONOS temporal memory"""
    
    def __init__(self, memory_capacity: int = 10000, sequence_length: int = 10):
        self.memory_capacity = memory_capacity
        self.sequence_length = sequence_length
        self.temporal_memory = deque(maxlen=memory_capacity)
        self.pattern_index = defaultdict(list)
        self.transition_memory = deque(maxlen=memory_capacity * sequence_length)
        self.replay_buffer = []
        self.memory_timestamp = 0
        
        # Temporal pattern templates
        self.pattern_templates = self._init_pattern_templates()
        
        # Memory organization strategies
        self.memory_strategies = {
            'recency': self._recency_based_retrieval,
            'similarity': self._similarity_based_retrieval,
            'importance': self._importance_based_retrieval,
            'diversity': self._diversity_based_retrieval,
            'temporal_coherence': self._temporal_coherence_retrieval
        }
        
    def _init_pattern_templates(self) -> Dict[str, Dict]:
        """Initialize temporal pattern templates for memory organization"""
        return {
            'linear_motion': {
                'features': ['direction', 'speed', 'object_size'],
                'variations': ['horizontal', 'vertical', 'diagonal'],
                'complexity': 1
            },
            'circular_motion': {
                'features': ['radius', 'angular_velocity', 'center'],
                'variations': ['clockwise', 'counterclockwise', 'elliptical'],
                'complexity': 2
            },
            'oscillation': {
                'features': ['amplitude', 'frequency', 'phase'],
                'variations': ['sine', 'square', 'triangular'],
                'complexity': 2
            },
            'transformation': {
                'features': ['start_shape', 'end_shape', 'interpolation'],
                'variations': ['morph', 'grow', 'rotate', 'dissolve'],
                'complexity': 3
            },
            'periodic': {
                'features': ['period', 'duty_cycle', 'phase_shift'],
                'variations': ['regular', 'irregular', 'chaotic'],
                'complexity': 2
            },
            'composite': {
                'features': ['component_patterns', 'interaction_type'],
                'variations': ['sequential', 'parallel', 'interleaved'],
                'complexity': 4
            }
        }
    
    def store_sequence(self, sequence: List[torch.Tensor], pattern_type: str,
                      additional_features: Optional[Dict] = None) -> None:
        """Store temporal sequence in memory"""
        # Extract temporal features
        temporal_features = self._extract_temporal_features(sequence)
        if additional_features:
            temporal_features.update(additional_features)
            
        # Calculate importance score
        importance = self._calculate_importance(sequence, temporal_features)
        
        # Create memory entry
        memory = TemporalMemory(
            sequence=sequence,
            pattern_type=pattern_type,
            timestamp=self.memory_timestamp,
            importance_score=importance,
            temporal_features=temporal_features,
            replay_priority=importance
        )
        
        # Store in memory
        self.temporal_memory.append(memory)
        self.pattern_index[pattern_type].append(self.memory_timestamp)
        
        # Extract and store transitions
        self._extract_transitions(sequence, pattern_type)
        
        self.memory_timestamp += 1
        
        # Update replay buffer
        self._update_replay_buffer()
    
    def _extract_temporal_features(self, sequence: List[torch.Tensor]) -> Dict[str, Any]:
        """Extract features from temporal sequence"""
        features = {
            'length': len(sequence),
            'total_change': 0.0,
            'max_change': 0.0,
            'periodicity': None,
            'movement_type': None,
            'consistency': 0.0,
            'entropy': 0.0,
            'spatial_coverage': 0.0,
            'temporal_smoothness': 0.0
        }
        
        if len(sequence) < 2:
            return features
            
        # Calculate frame-to-frame changes
        changes = []
        for i in range(1, len(sequence)):
            change = torch.abs(sequence[i] - sequence[i-1]).sum().item()
            changes.append(change)
            
        features['total_change'] = sum(changes)
        features['max_change'] = max(changes) if changes else 0
        features['avg_change'] = np.mean(changes) if changes else 0
        features['change_variance'] = np.var(changes) if changes else 0
        
        # Detect periodicity
        features['periodicity'] = self._detect_periodicity(sequence)
        
        # Detect movement
        features['movement_type'] = self._classify_movement(sequence)
        
        # Calculate consistency
        features['consistency'] = self._calculate_consistency(changes)
        
        # Calculate entropy
        features['entropy'] = self._calculate_sequence_entropy(sequence)
        
        # Calculate spatial coverage
        features['spatial_coverage'] = self._calculate_spatial_coverage(sequence)
        
        # Calculate temporal smoothness
        features['temporal_smoothness'] = self._calculate_temporal_smoothness(changes)
        
        return features
    
    def _extract_transitions(self, sequence: List[torch.Tensor], pattern_type: str) -> None:
        """Extract state transitions from sequence"""
        for i in range(len(sequence) - 1):
            # Determine action based on change
            action = self._infer_action(sequence[i], sequence[i+1])
            
            transition = SequenceTransition(
                prev_state=sequence[i],
                action=action,
                next_state=sequence[i+1],
                time_step=i,
                reward=1.0  # Success reward for valid transitions
            )
            
            self.transition_memory.append(transition)
    
    def _infer_action(self, prev_state: torch.Tensor, next_state: torch.Tensor) -> str:
        """Infer action from state transition"""
        diff = next_state - prev_state
        
        # Check for movement
        if torch.abs(diff).sum() > 0:
            # Detect shift direction
            prev_pos = self._get_object_positions(prev_state)
            next_pos = self._get_object_positions(next_state)
            
            if prev_pos and next_pos:
                movement = self._calculate_movement_vector(prev_pos, next_pos)
                
                if movement['dx'] > 0:
                    return 'move_right'
                elif movement['dx'] < 0:
                    return 'move_left'
                elif movement['dy'] > 0:
                    return 'move_down'
                elif movement['dy'] < 0:
                    return 'move_up'
                elif movement['rotation'] != 0:
                    return f'rotate_{movement["rotation"]}'
                    
        # Check for transformation
        if self._is_transformation(prev_state, next_state):
            return 'transform'
            
        return 'maintain'
    
    def _get_object_positions(self, state: torch.Tensor) -> List[Tuple[int, int]]:
        """Get positions of objects in state"""
        positions = []
        active_pixels = (state > 0).nonzero(as_tuple=False)
        
        if len(active_pixels) > 0:
            # Group into objects (simplified)
            center_y = active_pixels[:, -2].float().mean().item()
            center_x = active_pixels[:, -1].float().mean().item()
            positions.append((int(center_y), int(center_x)))
            
        return positions
    
    def _calculate_movement_vector(self, prev_pos: List[Tuple], next_pos: List[Tuple]) -> Dict:
        """Calculate movement vector between positions"""
        if not prev_pos or not next_pos:
            return {'dx': 0, 'dy': 0, 'rotation': 0}
            
        # Simple calculation using first object
        dy = next_pos[0][0] - prev_pos[0][0]
        dx = next_pos[0][1] - prev_pos[0][1]
        
        # Simplified rotation detection
        rotation = 0
        if abs(dx) == abs(dy) and dx != 0:
            rotation = 90 if dx * dy > 0 else -90
            
        return {'dx': dx, 'dy': dy, 'rotation': rotation}
    
    def _is_transformation(self, prev_state: torch.Tensor, next_state: torch.Tensor) -> bool:
        """Check if transition represents a transformation"""
        # Check if number of active pixels changed significantly
        prev_active = (prev_state > 0).sum().item()
        next_active = (next_state > 0).sum().item()
        
        change_ratio = abs(next_active - prev_active) / max(prev_active, 1)
        return change_ratio > 0.2
    
    def retrieve_sequences(self, query_type: str = 'similarity',
                          query_sequence: Optional[List[torch.Tensor]] = None,
                          n_sequences: int = 5) -> List[TemporalMemory]:
        """Retrieve sequences from memory"""
        if query_type not in self.memory_strategies:
            query_type = 'recency'
            
        strategy = self.memory_strategies[query_type]
        return strategy(query_sequence, n_sequences)
    
    def _recency_based_retrieval(self, query: Optional[List[torch.Tensor]], 
                                n: int) -> List[TemporalMemory]:
        """Retrieve most recent sequences"""
        return list(self.temporal_memory)[-n:]
    
    def _similarity_based_retrieval(self, query: Optional[List[torch.Tensor]], 
                                   n: int) -> List[TemporalMemory]:
        """Retrieve similar sequences"""
        if not query or not self.temporal_memory:
            return self._recency_based_retrieval(query, n)
            
        # Extract query features
        query_features = self._extract_temporal_features(query)
        
        # Calculate similarities
        similarities = []
        for memory in self.temporal_memory:
            similarity = self._calculate_similarity(query_features, memory.temporal_features)
            similarities.append((similarity, memory))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        return [mem for _, mem in similarities[:n]]
    
    def _importance_based_retrieval(self, query: Optional[List[torch.Tensor]], 
                                   n: int) -> List[TemporalMemory]:
        """Retrieve most important sequences"""
        sorted_memories = sorted(self.temporal_memory, 
                               key=lambda x: x.importance_score, 
                               reverse=True)
        return sorted_memories[:n]
    
    def _diversity_based_retrieval(self, query: Optional[List[torch.Tensor]], 
                                  n: int) -> List[TemporalMemory]:
        """Retrieve diverse sequences"""
        if len(self.temporal_memory) <= n:
            return list(self.temporal_memory)
            
        selected = []
        remaining = list(self.temporal_memory)
        
        # Select first randomly
        first = random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        # Select diverse sequences
        while len(selected) < n and remaining:
            max_diversity = -1
            most_diverse = None
            
            for candidate in remaining:
                min_similarity = 1.0
                for selected_mem in selected:
                    sim = self._calculate_similarity(
                        candidate.temporal_features,
                        selected_mem.temporal_features
                    )
                    min_similarity = min(min_similarity, sim)
                    
                if min_similarity > max_diversity:
                    max_diversity = min_similarity
                    most_diverse = candidate
                    
            if most_diverse:
                selected.append(most_diverse)
                remaining.remove(most_diverse)
                
        return selected
    
    def _temporal_coherence_retrieval(self, query: Optional[List[torch.Tensor]], 
                                     n: int) -> List[TemporalMemory]:
        """Retrieve temporally coherent sequences"""
        if not query:
            return self._recency_based_retrieval(query, n)
            
        # Find sequences that could follow the query
        coherent_sequences = []
        
        for memory in self.temporal_memory:
            coherence_score = self._calculate_temporal_coherence(query, memory.sequence)
            if coherence_score > 0.5:
                coherent_sequences.append((coherence_score, memory))
                
        # Sort by coherence
        coherent_sequences.sort(key=lambda x: x[0], reverse=True)
        
        return [mem for _, mem in coherent_sequences[:n]]
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between feature sets"""
        similarity = 0.0
        count = 0
        
        # Compare numeric features
        numeric_features = ['total_change', 'consistency', 'entropy', 
                          'spatial_coverage', 'temporal_smoothness']
        
        for feature in numeric_features:
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                
                if max(val1, val2) > 0:
                    sim = 1 - abs(val1 - val2) / max(val1, val2)
                else:
                    sim = 1.0
                    
                similarity += sim
                count += 1
                
        # Compare categorical features
        if features1.get('movement_type') == features2.get('movement_type'):
            similarity += 1.0
            count += 1
            
        if features1.get('periodicity') == features2.get('periodicity'):
            similarity += 1.0
            count += 1
            
        return similarity / max(count, 1)
    
    def _calculate_temporal_coherence(self, seq1: List[torch.Tensor], 
                                    seq2: List[torch.Tensor]) -> float:
        """Calculate temporal coherence between sequences"""
        if not seq1 or not seq2:
            return 0.0
            
        # Check if seq2 could follow seq1
        last_frame_seq1 = seq1[-1]
        first_frame_seq2 = seq2[0]
        
        # Calculate transition probability
        transition_score = 1.0 - torch.abs(last_frame_seq1 - first_frame_seq2).mean().item()
        
        # Check movement continuity
        if len(seq1) > 1 and len(seq2) > 1:
            # Extract movement vectors
            mov1 = self._extract_movement_vector(seq1[-2], seq1[-1])
            mov2 = self._extract_movement_vector(seq2[0], seq2[1])
            
            # Compare movements
            if mov1 and mov2:
                dx_sim = 1.0 - abs(mov1['dx'] - mov2['dx']) / 5.0
                dy_sim = 1.0 - abs(mov1['dy'] - mov2['dy']) / 5.0
                movement_coherence = (dx_sim + dy_sim) / 2
                
                transition_score = (transition_score + movement_coherence) / 2
                
        return max(0, min(1, transition_score))
    
    def _extract_movement_vector(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Optional[Dict]:
        """Extract movement vector between frames"""
        pos1 = self._get_object_positions(frame1)
        pos2 = self._get_object_positions(frame2)
        
        if pos1 and pos2:
            return self._calculate_movement_vector(pos1, pos2)
        return None
    
    def replay_experiences(self, batch_size: int = 32) -> List[TemporalMemory]:
        """Replay experiences from memory"""
        if len(self.replay_buffer) < batch_size:
            self._update_replay_buffer()
            
        # Prioritized replay
        if len(self.replay_buffer) >= batch_size:
            # Sample with priority
            priorities = [mem.replay_priority for mem in self.replay_buffer]
            total_priority = sum(priorities)
            
            if total_priority > 0:
                probs = [p / total_priority for p in priorities]
                indices = np.random.choice(len(self.replay_buffer), 
                                         size=batch_size, 
                                         p=probs,
                                         replace=True)
                
                batch = [self.replay_buffer[i] for i in indices]
                
                # Update access counts
                for mem in batch:
                    mem.access_count += 1
                    
                return batch
                
        # Fallback to random sampling
        return random.sample(list(self.temporal_memory), 
                           min(batch_size, len(self.temporal_memory)))
    
    def _update_replay_buffer(self) -> None:
        """Update replay buffer with prioritized sequences"""
        # Calculate replay priorities
        for memory in self.temporal_memory:
            # Priority based on importance, recency, and access frequency
            recency_score = 1.0 / (self.memory_timestamp - memory.timestamp + 1)
            access_score = 1.0 / (memory.access_count + 1)
            
            memory.replay_priority = (
                memory.importance_score * 0.5 +
                recency_score * 0.3 +
                access_score * 0.2
            )
            
        # Select top sequences for replay buffer
        sorted_memories = sorted(self.temporal_memory,
                               key=lambda x: x.replay_priority,
                               reverse=True)
        
        self.replay_buffer = sorted_memories[:min(1000, len(sorted_memories))]
    
    def consolidate_memory(self) -> None:
        """Consolidate memory by merging similar sequences"""
        if len(self.temporal_memory) < self.memory_capacity * 0.9:
            return
            
        # Group similar sequences
        clusters = defaultdict(list)
        
        for memory in self.temporal_memory:
            # Find cluster
            found_cluster = False
            for cluster_key, cluster_members in clusters.items():
                if cluster_members:
                    representative = cluster_members[0]
                    similarity = self._calculate_similarity(
                        memory.temporal_features,
                        representative.temporal_features
                    )
                    
                    if similarity > 0.8:
                        clusters[cluster_key].append(memory)
                        found_cluster = True
                        break
                        
            if not found_cluster:
                clusters[len(clusters)] = [memory]
                
        # Keep best from each cluster
        consolidated = []
        for cluster_members in clusters.values():
            # Keep most important/recent
            best = max(cluster_members, key=lambda x: x.importance_score)
            consolidated.append(best)
            
        # Update memory
        self.temporal_memory = deque(consolidated, maxlen=self.memory_capacity)
        
    def _detect_periodicity(self, sequence: List[torch.Tensor]) -> Optional[int]:
        """Detect period in sequence"""
        if len(sequence) < 4:
            return None
            
        for period in range(2, len(sequence) // 2 + 1):
            is_periodic = True
            for i in range(period, len(sequence)):
                if not torch.allclose(sequence[i], sequence[i % period], atol=0.1):
                    is_periodic = False
                    break
                    
            if is_periodic:
                return period
                
        return None
    
    def _classify_movement(self, sequence: List[torch.Tensor]) -> Optional[str]:
        """Classify movement type in sequence"""
        if len(sequence) < 3:
            return None
            
        positions = []
        for frame in sequence:
            pos = self._get_object_positions(frame)
            if pos:
                positions.append(pos[0])
                
        if len(positions) < 3:
            return None
            
        # Analyze movement pattern
        movements = []
        for i in range(1, len(positions)):
            dy = positions[i][0] - positions[i-1][0]
            dx = positions[i][1] - positions[i-1][1]
            movements.append((dx, dy))
            
        # Classify based on movement pattern
        x_changes = [m[0] for m in movements]
        y_changes = [m[1] for m in movements]
        
        if all(x == x_changes[0] for x in x_changes) and all(y == y_changes[0] for y in y_changes):
            if x_changes[0] != 0 and y_changes[0] == 0:
                return 'horizontal'
            elif x_changes[0] == 0 and y_changes[0] != 0:
                return 'vertical'
            elif x_changes[0] != 0 and y_changes[0] != 0:
                return 'diagonal'
        
        # Check for circular movement
        if len(movements) > 4:
            angles = []
            for dx, dy in movements:
                if dx != 0 or dy != 0:
                    angle = np.arctan2(dy, dx)
                    angles.append(angle)
                    
            if len(angles) > 2:
                angle_changes = [angles[i] - angles[i-1] for i in range(1, len(angles))]
                if all(abs(ac - angle_changes[0]) < 0.1 for ac in angle_changes):
                    return 'circular'
                    
        return 'complex'
    
    def _calculate_consistency(self, changes: List[float]) -> float:
        """Calculate consistency of changes"""
        if not changes:
            return 1.0
            
        if len(changes) == 1:
            return 1.0
            
        mean_change = np.mean(changes)
        if mean_change == 0:
            return 1.0
            
        variance = np.var(changes)
        consistency = 1.0 - (variance / (mean_change ** 2))
        
        return max(0, min(1, consistency))
    
    def _calculate_sequence_entropy(self, sequence: List[torch.Tensor]) -> float:
        """Calculate entropy of sequence"""
        if not sequence:
            return 0.0
            
        # Calculate distribution of active pixels
        all_activations = []
        for frame in sequence:
            active = (frame > 0).float().mean().item()
            all_activations.append(active)
            
        if not all_activations:
            return 0.0
            
        # Normalize to probabilities
        total = sum(all_activations)
        if total == 0:
            return 0.0
            
        probs = [a / total for a in all_activations]
        
        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
                
        # Normalize by max entropy
        max_entropy = np.log2(len(sequence))
        if max_entropy > 0:
            entropy /= max_entropy
            
        return entropy
    
    def _calculate_spatial_coverage(self, sequence: List[torch.Tensor]) -> float:
        """Calculate spatial coverage of sequence"""
        if not sequence:
            return 0.0
            
        # Track all positions that were active
        all_active = torch.zeros_like(sequence[0])
        
        for frame in sequence:
            all_active = torch.maximum(all_active, frame)
            
        # Calculate coverage
        total_pixels = all_active.shape[-1] * all_active.shape[-2]
        active_pixels = (all_active > 0).sum().item()
        
        return active_pixels / total_pixels
    
    def _calculate_temporal_smoothness(self, changes: List[float]) -> float:
        """Calculate temporal smoothness"""
        if len(changes) < 2:
            return 1.0
            
        # Calculate second-order differences
        second_diffs = []
        for i in range(1, len(changes)):
            diff = abs(changes[i] - changes[i-1])
            second_diffs.append(diff)
            
        if not second_diffs:
            return 1.0
            
        # Lower values indicate smoother transitions
        avg_second_diff = np.mean(second_diffs)
        max_change = max(changes) if changes else 1.0
        
        if max_change == 0:
            return 1.0
            
        smoothness = 1.0 - (avg_second_diff / max_change)
        
        return max(0, min(1, smoothness))
    
    def _calculate_importance(self, sequence: List[torch.Tensor], 
                            features: Dict[str, Any]) -> float:
        """Calculate importance score for sequence"""
        importance = 0.0
        
        # Factor in complexity
        if features.get('movement_type') in ['circular', 'complex']:
            importance += 0.3
        elif features.get('movement_type') in ['horizontal', 'vertical', 'diagonal']:
            importance += 0.1
            
        # Factor in periodicity
        if features.get('periodicity'):
            importance += 0.2
            
        # Factor in consistency
        importance += features.get('consistency', 0) * 0.2
        
        # Factor in entropy (prefer moderate entropy)
        entropy = features.get('entropy', 0)
        if 0.3 < entropy < 0.7:
            importance += 0.2
            
        # Factor in spatial coverage
        importance += features.get('spatial_coverage', 0) * 0.1
        
        return min(1.0, importance)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory contents"""
        stats = {
            'total_sequences': len(self.temporal_memory),
            'total_transitions': len(self.transition_memory),
            'pattern_distribution': defaultdict(int),
            'avg_sequence_length': 0,
            'avg_importance': 0,
            'memory_utilization': len(self.temporal_memory) / self.memory_capacity
        }
        
        if self.temporal_memory:
            lengths = []
            importances = []
            
            for memory in self.temporal_memory:
                stats['pattern_distribution'][memory.pattern_type] += 1
                lengths.append(len(memory.sequence))
                importances.append(memory.importance_score)
                
            stats['avg_sequence_length'] = np.mean(lengths)
            stats['avg_importance'] = np.mean(importances)
            
        return stats


class ChronosMEPTLoss(nn.Module):
    """CHRONOS-specific MEPT loss function"""
    
    def __init__(self):
        super().__init__()
        import torch.nn as nn
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                input_grid: torch.Tensor, temporal_features: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Calculate CHRONOS-specific MEPT loss with temporal components"""
        B, C, H, W = pred.shape
        
        # Get predictions
        pred_indices = pred.argmax(dim=1)
        target_indices = target.argmax(dim=1)
        
        # Base cross-entropy loss
        ce_loss = F.cross_entropy(pred, target_indices, reduction='mean')
        
        # Temporal consistency loss
        if temporal_features and 'prev_prediction' in temporal_features:
            prev_pred = temporal_features['prev_prediction']
            temporal_consistency = F.mse_loss(pred, prev_pred) * 0.1
        else:
            temporal_consistency = torch.tensor(0.0).to(pred.device)
        
        # Movement preservation loss
        movement_loss = self._calculate_movement_loss(pred_indices, target_indices, input_grid.argmax(dim=1))
        
        # Exact match bonus
        exact_matches = (pred_indices == target_indices).all(dim=[1, 2]).float()
        exact_bonus = -exact_matches.mean() * 3.0  # Negative for bonus
        
        # Total loss
        total_loss = ce_loss + temporal_consistency + movement_loss * 0.2 + exact_bonus
        
        return {
            'total': total_loss,
            'reconstruction': ce_loss,
            'temporal_consistency': temporal_consistency,
            'movement': movement_loss,
            'exact_bonus': -exact_bonus,
            'exact_count': exact_matches.sum()
        }
    
    def _calculate_movement_loss(self, pred: torch.Tensor, target: torch.Tensor, input_grid: torch.Tensor) -> torch.Tensor:
        """Calculate loss for movement preservation"""
        # Detect movement direction from input to target
        input_com = self._center_of_mass(input_grid)
        target_com = self._center_of_mass(target)
        pred_com = self._center_of_mass(pred)
        
        if input_com is not None and target_com is not None and pred_com is not None:
            # Expected movement
            expected_movement = target_com - input_com
            # Actual movement
            actual_movement = pred_com - input_com
            # Movement error
            movement_error = F.mse_loss(actual_movement, expected_movement)
            return movement_error
        
        return torch.tensor(0.0).to(pred.device)
    
    def _center_of_mass(self, grid: torch.Tensor) -> Optional[torch.Tensor]:
        """Calculate center of mass for batch of grids"""
        B = grid.shape[0]
        centers = []
        
        for b in range(B):
            nonzero = (grid[b] > 0).nonzero(as_tuple=False)
            if len(nonzero) > 0:
                center = nonzero.float().mean(dim=0)
                centers.append(center)
            else:
                centers.append(None)
        
        # Stack if all grids have objects
        if all(c is not None for c in centers):
            return torch.stack(centers)
        return None


def create_chronos_mept_system(memory_capacity: int = 10000, 
                              transformation_penalty: float = 0.3,
                              exact_match_bonus: float = 2.5):
    """Factory function to create CHRONOS MEPT system"""
    # Create MEPT with specified capacity
    mept = ChronosMEPT(memory_capacity=memory_capacity)
    
    # Create loss with temporal parameters
    loss_fn = ChronosMEPTLoss()
    # Store params for later use if needed
    loss_fn.transformation_penalty = transformation_penalty
    loss_fn.exact_match_bonus = exact_match_bonus
    
    return {
        'memory_system': mept,  # Note: training script expects 'memory_system' not 'replay_buffer'
        'pattern_bank': None,  # CHRONOS uses integrated pattern memory
        'loss_fn': loss_fn,
        'description': 'CHRONOS Temporal Memory System'
    }