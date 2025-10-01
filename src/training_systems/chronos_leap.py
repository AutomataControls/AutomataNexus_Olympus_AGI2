"""
CHRONOS LEAP Training System
Learning Enhancement through Automated Patterns - Temporal Sequences
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import random
import numpy as np
from dataclasses import dataclass


@dataclass
class TemporalPattern:
    """Represents a temporal pattern with metadata"""
    name: str
    sequence: List[torch.Tensor]
    pattern_type: str
    complexity: int
    periodicity: Optional[int] = None
    movement_type: Optional[str] = None
    transformation_type: Optional[str] = None


class ChronosLEAP:
    """LEAP system specialized for CHRONOS temporal patterns"""
    
    def __init__(self, grid_size: int = 12, max_sequence_length: int = 10):
        self.grid_size = grid_size
        self.max_sequence_length = max_sequence_length
        self.pattern_generators = self._init_pattern_generators()
        self.temporal_rules = self._init_temporal_rules()
        
    def _init_pattern_generators(self) -> Dict[str, callable]:
        """Initialize temporal pattern generators"""
        return {
            # Movement patterns
            'linear_movement': self._generate_linear_movement,
            'circular_movement': self._generate_circular_movement,
            'bouncing_movement': self._generate_bouncing_movement,
            'zigzag_movement': self._generate_zigzag_movement,
            'spiral_movement': self._generate_spiral_movement,
            
            # Transformation patterns
            'growing_pattern': self._generate_growing_pattern,
            'morphing_pattern': self._generate_morphing_pattern,
            'pulsing_pattern': self._generate_pulsing_pattern,
            'rotating_pattern': self._generate_rotating_pattern,
            
            # Sequence patterns
            'repeating_cycle': self._generate_repeating_cycle,
            'accelerating_sequence': self._generate_accelerating_sequence,
            'decaying_sequence': self._generate_decaying_sequence,
            'oscillating_sequence': self._generate_oscillating_sequence,
            
            # Complex temporal patterns
            'wave_propagation': self._generate_wave_propagation,
            'cascade_effect': self._generate_cascade_effect,
            'temporal_symmetry': self._generate_temporal_symmetry,
            'phase_shift': self._generate_phase_shift
        }
    
    def _init_temporal_rules(self) -> Dict[str, Dict]:
        """Initialize temporal transformation rules"""
        return {
            'movement': {
                'types': ['linear', 'circular', 'zigzag', 'bounce', 'spiral'],
                'speeds': [1, 2, 3],
                'directions': ['horizontal', 'vertical', 'diagonal', 'radial']
            },
            'transformation': {
                'types': ['grow', 'shrink', 'morph', 'rotate', 'flip'],
                'rates': ['constant', 'accelerating', 'decelerating', 'oscillating']
            },
            'periodicity': {
                'periods': [2, 3, 4, 5],
                'phases': [0, 0.25, 0.5, 0.75]
            },
            'complexity': {
                'simple': 1,
                'moderate': 2,
                'complex': 3,
                'advanced': 4
            }
        }
    
    def generate_training_batch(self, batch_size: int = 32) -> List[TemporalPattern]:
        """Generate a batch of temporal training patterns"""
        batch = []
        
        # Ensure diversity in pattern types
        pattern_types = list(self.pattern_generators.keys())
        
        for _ in range(batch_size):
            # Select pattern type
            pattern_type = random.choice(pattern_types)
            
            # Generate pattern
            pattern = self.pattern_generators[pattern_type]()
            
            batch.append(pattern)
            
        return batch
    
    def generate_curriculum_batch(self, difficulty_level: int = 1) -> List[TemporalPattern]:
        """Generate patterns following curriculum learning"""
        batch_size = 32
        batch = []
        
        if difficulty_level == 1:
            # Basic movement patterns
            patterns = ['linear_movement', 'rotating_pattern', 'repeating_cycle']
        elif difficulty_level == 2:
            # Intermediate patterns
            patterns = ['circular_movement', 'growing_pattern', 'oscillating_sequence']
        elif difficulty_level == 3:
            # Advanced patterns
            patterns = ['zigzag_movement', 'morphing_pattern', 'wave_propagation']
        else:
            # Expert patterns
            patterns = ['spiral_movement', 'cascade_effect', 'temporal_symmetry']
            
        for _ in range(batch_size):
            pattern_type = random.choice(patterns)
            pattern = self.pattern_generators[pattern_type]()
            batch.append(pattern)
            
        return batch
    
    # Movement pattern generators
    def _generate_linear_movement(self) -> TemporalPattern:
        """Generate linear movement sequence"""
        sequence = []
        direction = random.choice(['right', 'left', 'up', 'down'])
        speed = random.randint(1, 2)
        
        # Create initial object
        grid = torch.zeros(10, self.grid_size, self.grid_size)
        obj_size = random.randint(2, 4)
        color = random.randint(1, 9)
        
        # Place object
        if direction in ['right', 'left']:
            start_x = 1 if direction == 'right' else self.grid_size - obj_size - 1
            start_y = random.randint(1, self.grid_size - obj_size - 1)
        else:
            start_x = random.randint(1, self.grid_size - obj_size - 1)
            start_y = 1 if direction == 'down' else self.grid_size - obj_size - 1
            
        # Generate movement sequence
        for t in range(self.max_sequence_length):
            frame = torch.zeros_like(grid)
            
            # Calculate position
            if direction == 'right':
                x = min(start_x + t * speed, self.grid_size - obj_size)
                y = start_y
            elif direction == 'left':
                x = max(start_x - t * speed, 0)
                y = start_y
            elif direction == 'down':
                x = start_x
                y = min(start_y + t * speed, self.grid_size - obj_size)
            else:  # up
                x = start_x
                y = max(start_y - t * speed, 0)
                
            # Draw object
            frame[color, y:y+obj_size, x:x+obj_size] = 1
            sequence.append(frame)
            
            # Stop if reached edge
            if (direction == 'right' and x >= self.grid_size - obj_size) or \
               (direction == 'left' and x <= 0) or \
               (direction == 'down' and y >= self.grid_size - obj_size) or \
               (direction == 'up' and y <= 0):
                break
                
        return TemporalPattern(
            name='linear_movement',
            sequence=sequence,
            pattern_type='movement',
            complexity=1,
            movement_type=direction
        )
    
    def _generate_circular_movement(self) -> TemporalPattern:
        """Generate circular movement sequence"""
        sequence = []
        radius = random.randint(3, min(5, self.grid_size // 3))
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        obj_size = 2
        color = random.randint(1, 9)
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            # Calculate position on circle
            angle = (t / self.max_sequence_length) * 2 * np.pi
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # Ensure within bounds
            x = max(0, min(x, self.grid_size - obj_size))
            y = max(0, min(y, self.grid_size - obj_size))
            
            # Draw object
            frame[color, y:y+obj_size, x:x+obj_size] = 1
            sequence.append(frame)
            
        return TemporalPattern(
            name='circular_movement',
            sequence=sequence,
            pattern_type='movement',
            complexity=2,
            movement_type='circular'
        )
    
    def _generate_bouncing_movement(self) -> TemporalPattern:
        """Generate bouncing movement sequence"""
        sequence = []
        obj_size = 3
        color = random.randint(1, 9)
        
        # Initial position and velocity
        x = random.randint(1, self.grid_size - obj_size - 1)
        y = random.randint(1, self.grid_size - obj_size - 1)
        vx = random.choice([-1, 1])
        vy = random.choice([-1, 1])
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            # Update position
            x += vx
            y += vy
            
            # Bounce off walls
            if x <= 0 or x >= self.grid_size - obj_size:
                vx = -vx
                x = max(0, min(x, self.grid_size - obj_size))
            if y <= 0 or y >= self.grid_size - obj_size:
                vy = -vy
                y = max(0, min(y, self.grid_size - obj_size))
            
            # Draw object
            frame[color, y:y+obj_size, x:x+obj_size] = 1
            sequence.append(frame)
            
        return TemporalPattern(
            name='bouncing_movement',
            sequence=sequence,
            pattern_type='movement',
            complexity=2,
            movement_type='bounce'
        )
    
    def _generate_zigzag_movement(self) -> TemporalPattern:
        """Generate zigzag movement sequence"""
        sequence = []
        obj_size = 2
        color = random.randint(1, 9)
        amplitude = 3
        
        # Start position
        x = 1
        y = self.grid_size // 2
        direction = 1
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            # Update position
            x += 1
            y += direction * amplitude
            
            # Change direction at limits
            if y <= amplitude or y >= self.grid_size - obj_size - amplitude:
                direction = -direction
                
            # Stop at right edge
            if x >= self.grid_size - obj_size:
                break
                
            # Draw object
            frame[color, y:y+obj_size, x:x+obj_size] = 1
            sequence.append(frame)
            
        return TemporalPattern(
            name='zigzag_movement',
            sequence=sequence,
            pattern_type='movement',
            complexity=2,
            movement_type='zigzag'
        )
    
    def _generate_spiral_movement(self) -> TemporalPattern:
        """Generate spiral movement sequence"""
        sequence = []
        obj_size = 2
        color = random.randint(1, 9)
        
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            # Spiral parameters
            angle = t * 0.5
            radius = t * 0.8
            
            # Calculate position
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # Check bounds
            if x < 0 or x >= self.grid_size - obj_size or \
               y < 0 or y >= self.grid_size - obj_size:
                break
                
            # Draw object
            frame[color, y:y+obj_size, x:x+obj_size] = 1
            sequence.append(frame)
            
        return TemporalPattern(
            name='spiral_movement',
            sequence=sequence,
            pattern_type='movement',
            complexity=3,
            movement_type='spiral'
        )
    
    # Transformation pattern generators
    def _generate_growing_pattern(self) -> TemporalPattern:
        """Generate growing pattern sequence"""
        sequence = []
        color = random.randint(1, 9)
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            # Calculate size
            size = min(t + 1, self.grid_size // 2)
            
            # Draw square of increasing size
            start_x = max(0, center_x - size)
            end_x = min(self.grid_size, center_x + size)
            start_y = max(0, center_y - size)
            end_y = min(self.grid_size, center_y + size)
            
            frame[color, start_y:end_y, start_x:end_x] = 1
            sequence.append(frame)
            
        return TemporalPattern(
            name='growing_pattern',
            sequence=sequence,
            pattern_type='transformation',
            complexity=1,
            transformation_type='grow'
        )
    
    def _generate_morphing_pattern(self) -> TemporalPattern:
        """Generate morphing pattern sequence"""
        sequence = []
        color = random.randint(1, 9)
        
        # Define start and end shapes
        shape_pairs = [
            ('square', 'circle'),
            ('horizontal_line', 'vertical_line'),
            ('cross', 'x_shape')
        ]
        start_shape, end_shape = random.choice(shape_pairs)
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            progress = t / (self.max_sequence_length - 1)
            
            # Create interpolated shape
            if start_shape == 'square' and end_shape == 'circle':
                # Morph from square to circle
                size = 5
                center = self.grid_size // 2
                
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        # Square distance
                        sq_dist = max(abs(i - center), abs(j - center))
                        # Circle distance
                        circ_dist = np.sqrt((i - center)**2 + (j - center)**2)
                        # Interpolate
                        dist = sq_dist * (1 - progress) + circ_dist * progress
                        
                        if dist <= size:
                            frame[color, i, j] = 1
                            
            elif start_shape == 'horizontal_line' and end_shape == 'vertical_line':
                # Rotate line
                center = self.grid_size // 2
                length = 6
                
                angle = progress * np.pi / 2
                for offset in range(-length//2, length//2 + 1):
                    x = int(center + offset * np.cos(angle))
                    y = int(center + offset * np.sin(angle))
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        frame[color, y, x] = 1
                        
            sequence.append(frame)
            
        return TemporalPattern(
            name='morphing_pattern',
            sequence=sequence,
            pattern_type='transformation',
            complexity=3,
            transformation_type='morph'
        )
    
    def _generate_pulsing_pattern(self) -> TemporalPattern:
        """Generate pulsing pattern sequence"""
        sequence = []
        base_color = random.randint(1, 5)
        pulse_color = base_color + 4
        
        # Create base shape
        base_shape = torch.zeros(10, self.grid_size, self.grid_size)
        size = 4
        center = self.grid_size // 2
        base_shape[base_color, center-size//2:center+size//2, center-size//2:center+size//2] = 1
        
        for t in range(self.max_sequence_length):
            frame = base_shape.clone()
            
            # Calculate pulse intensity
            intensity = (np.sin(t * 2 * np.pi / 5) + 1) / 2
            
            if intensity > 0.5:
                # Add pulse effect
                frame[pulse_color] = frame[base_color] * intensity
                
            sequence.append(frame)
            
        return TemporalPattern(
            name='pulsing_pattern',
            sequence=sequence,
            pattern_type='transformation',
            complexity=2,
            transformation_type='pulse'
        )
    
    def _generate_rotating_pattern(self) -> TemporalPattern:
        """Generate rotating pattern sequence"""
        sequence = []
        color = random.randint(1, 9)
        
        # Create asymmetric shape to show rotation
        shape = torch.zeros(10, self.grid_size, self.grid_size)
        center = self.grid_size // 2
        
        # L-shape
        shape[color, center-2:center+2, center-2:center] = 1
        shape[color, center:center+2, center-2:center+2] = 1
        
        for t in range(self.max_sequence_length):
            # Rotate 90 degrees each step
            rotations = t % 4
            frame = shape.clone()
            
            for _ in range(rotations):
                frame = torch.rot90(frame, k=1, dims=[-2, -1])
                
            sequence.append(frame)
            
        return TemporalPattern(
            name='rotating_pattern',
            sequence=sequence,
            pattern_type='transformation',
            complexity=1,
            transformation_type='rotate'
        )
    
    # Sequence pattern generators
    def _generate_repeating_cycle(self) -> TemporalPattern:
        """Generate repeating cycle sequence"""
        period = random.randint(2, 4)
        base_patterns = []
        
        # Create base patterns for cycle
        for i in range(period):
            pattern = torch.zeros(10, self.grid_size, self.grid_size)
            color = random.randint(1, 9)
            
            # Different pattern for each phase
            if i == 0:
                pattern[color, 2:4, 2:4] = 1
            elif i == 1:
                pattern[color, 2:4, 8:10] = 1
            elif i == 2:
                pattern[color, 8:10, 8:10] = 1
            else:
                pattern[color, 8:10, 2:4] = 1
                
            base_patterns.append(pattern)
        
        # Generate repeating sequence
        sequence = []
        for t in range(self.max_sequence_length):
            sequence.append(base_patterns[t % period].clone())
            
        return TemporalPattern(
            name='repeating_cycle',
            sequence=sequence,
            pattern_type='sequence',
            complexity=2,
            periodicity=period
        )
    
    def _generate_accelerating_sequence(self) -> TemporalPattern:
        """Generate accelerating movement sequence"""
        sequence = []
        color = random.randint(1, 9)
        obj_size = 3
        
        x = 1
        y = self.grid_size // 2
        velocity = 0.5
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            # Accelerate
            velocity = min(velocity * 1.2, 3)
            x += int(velocity)
            
            if x >= self.grid_size - obj_size:
                break
                
            # Draw object
            frame[color, y:y+obj_size, x:x+obj_size] = 1
            sequence.append(frame)
            
        return TemporalPattern(
            name='accelerating_sequence',
            sequence=sequence,
            pattern_type='sequence',
            complexity=2
        )
    
    def _generate_decaying_sequence(self) -> TemporalPattern:
        """Generate decaying/fading sequence"""
        sequence = []
        
        # Create initial pattern
        initial = torch.zeros(10, self.grid_size, self.grid_size)
        colors = list(range(1, 10))
        
        # Fill with random pattern
        for i in range(0, self.grid_size, 2):
            for j in range(0, self.grid_size, 2):
                color = random.choice(colors)
                initial[color, i:i+2, j:j+2] = 1
                
        for t in range(self.max_sequence_length):
            frame = initial.clone()
            
            # Apply decay
            decay_rate = 0.15 * t
            mask = torch.rand_like(frame) > decay_rate
            frame = frame * mask.float()
            
            sequence.append(frame)
            
        return TemporalPattern(
            name='decaying_sequence',
            sequence=sequence,
            pattern_type='sequence',
            complexity=2
        )
    
    def _generate_oscillating_sequence(self) -> TemporalPattern:
        """Generate oscillating sequence"""
        sequence = []
        color1 = random.randint(1, 5)
        color2 = color1 + 4
        
        # Create two states
        state1 = torch.zeros(10, self.grid_size, self.grid_size)
        state2 = torch.zeros(10, self.grid_size, self.grid_size)
        
        # Complementary patterns
        for i in range(0, self.grid_size, 2):
            for j in range(0, self.grid_size, 2):
                if (i + j) % 4 == 0:
                    state1[color1, i, j] = 1
                    state2[color2, i, j] = 1
                else:
                    state1[color2, i, j] = 1
                    state2[color1, i, j] = 1
                    
        for t in range(self.max_sequence_length):
            # Oscillate between states
            progress = (np.sin(t * np.pi / 2) + 1) / 2
            frame = state1 * (1 - progress) + state2 * progress
            sequence.append(frame)
            
        return TemporalPattern(
            name='oscillating_sequence',
            sequence=sequence,
            pattern_type='sequence',
            complexity=2,
            periodicity=4
        )
    
    # Complex temporal patterns
    def _generate_wave_propagation(self) -> TemporalPattern:
        """Generate wave propagation sequence"""
        sequence = []
        color = random.randint(1, 9)
        wave_speed = 1.5
        wave_width = 2
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            # Wave position
            wave_pos = int(t * wave_speed)
            
            # Draw wave
            for y in range(self.grid_size):
                # Sine wave shape
                offset = int(2 * np.sin(y * np.pi / 4))
                x = wave_pos + offset
                
                if 0 <= x < self.grid_size:
                    for w in range(wave_width):
                        if 0 <= x + w < self.grid_size:
                            frame[color, y, x + w] = 1
                            
            sequence.append(frame)
            
        return TemporalPattern(
            name='wave_propagation',
            sequence=sequence,
            pattern_type='complex',
            complexity=3
        )
    
    def _generate_cascade_effect(self) -> TemporalPattern:
        """Generate cascade effect sequence"""
        sequence = []
        colors = list(range(1, 10))
        
        # Initial state - full grid
        initial = torch.zeros(10, self.grid_size, self.grid_size)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                initial[random.choice(colors), i, j] = 1
                
        for t in range(self.max_sequence_length):
            if t == 0:
                frame = initial.clone()
            else:
                frame = sequence[-1].clone()
                
                # Cascade effect - clear diagonal line
                for k in range(self.grid_size):
                    if k + t < self.grid_size:
                        frame[:, k, k + t] = 0
                    if t - k >= 0 and t - k < self.grid_size:
                        frame[:, k, t - k] = 0
                        
            sequence.append(frame)
            
        return TemporalPattern(
            name='cascade_effect',
            sequence=sequence,
            pattern_type='complex',
            complexity=3
        )
    
    def _generate_temporal_symmetry(self) -> TemporalPattern:
        """Generate temporally symmetric sequence"""
        half_length = self.max_sequence_length // 2
        first_half = []
        
        # Generate first half
        color = random.randint(1, 9)
        for t in range(half_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            # Create evolving pattern
            size = t + 1
            center = self.grid_size // 2
            
            # Draw expanding square
            start = max(0, center - size)
            end = min(self.grid_size, center + size)
            frame[color, start:end, start:end] = 1
            
            # Clear center
            if size > 2:
                inner_start = start + 1
                inner_end = end - 1
                frame[color, inner_start:inner_end, inner_start:inner_end] = 0
                
            first_half.append(frame)
            
        # Create symmetric sequence
        sequence = first_half + list(reversed(first_half))
        
        return TemporalPattern(
            name='temporal_symmetry',
            sequence=sequence,
            pattern_type='complex',
            complexity=3
        )
    
    def _generate_phase_shift(self) -> TemporalPattern:
        """Generate phase shift sequence"""
        sequence = []
        num_objects = 3
        colors = random.sample(range(1, 10), num_objects)
        
        for t in range(self.max_sequence_length):
            frame = torch.zeros(10, self.grid_size, self.grid_size)
            
            for i, color in enumerate(colors):
                # Each object has different phase
                phase = (t + i * self.max_sequence_length // num_objects) % self.max_sequence_length
                
                # Calculate position based on phase
                x = int((phase / self.max_sequence_length) * (self.grid_size - 3))
                y = self.grid_size // 2 + i * 3 - 3
                
                # Draw object
                frame[color, y:y+2, x:x+2] = 1
                
            sequence.append(frame)
            
        return TemporalPattern(
            name='phase_shift',
            sequence=sequence,
            pattern_type='complex',
            complexity=4
        )
    
    def augment_sequence(self, pattern: TemporalPattern) -> TemporalPattern:
        """Augment a temporal pattern"""
        augmentation = random.choice(['speed_change', 'reverse', 'add_noise', 'color_shift'])
        
        if augmentation == 'speed_change':
            # Change temporal speed
            factor = random.choice([0.5, 2.0])
            if factor == 0.5:
                # Slow down - interpolate
                new_sequence = []
                for i in range(len(pattern.sequence) - 1):
                    new_sequence.append(pattern.sequence[i])
                    # Add interpolated frame
                    interp = (pattern.sequence[i] + pattern.sequence[i + 1]) / 2
                    new_sequence.append(interp)
                new_sequence.append(pattern.sequence[-1])
            else:
                # Speed up - skip frames
                new_sequence = pattern.sequence[::2]
                
            pattern.sequence = new_sequence[:self.max_sequence_length]
            
        elif augmentation == 'reverse':
            # Reverse temporal order
            pattern.sequence = list(reversed(pattern.sequence))
            
        elif augmentation == 'add_noise':
            # Add temporal noise
            for i in range(len(pattern.sequence)):
                noise = torch.rand_like(pattern.sequence[i]) < 0.05
                pattern.sequence[i] = torch.maximum(pattern.sequence[i], noise.float())
                
        elif augmentation == 'color_shift':
            # Shift colors
            shift = random.randint(1, 8)
            for i in range(len(pattern.sequence)):
                # Rotate color channels
                pattern.sequence[i] = torch.roll(pattern.sequence[i], shift, dims=0)
                
        return pattern
    
    def analyze_pattern(self, pattern: TemporalPattern) -> Dict[str, Any]:
        """Analyze temporal pattern properties"""
        analysis = {
            'name': pattern.name,
            'length': len(pattern.sequence),
            'complexity': pattern.complexity,
            'has_movement': self._detect_movement(pattern.sequence),
            'has_periodicity': self._detect_periodicity(pattern.sequence),
            'temporal_consistency': self._measure_consistency(pattern.sequence),
            'change_rate': self._calculate_change_rate(pattern.sequence)
        }
        
        if pattern.periodicity:
            analysis['period'] = pattern.periodicity
        if pattern.movement_type:
            analysis['movement_type'] = pattern.movement_type
        if pattern.transformation_type:
            analysis['transformation_type'] = pattern.transformation_type
            
        return analysis
    
    def _detect_movement(self, sequence: List[torch.Tensor]) -> bool:
        """Detect if sequence contains movement"""
        if len(sequence) < 2:
            return False
            
        for i in range(1, len(sequence)):
            diff = torch.abs(sequence[i] - sequence[i-1])
            if diff.sum() > 0:
                # Check if pattern moved (not just appeared/disappeared)
                prev_locs = (sequence[i-1] > 0).nonzero(as_tuple=False)
                curr_locs = (sequence[i] > 0).nonzero(as_tuple=False)
                
                if len(prev_locs) > 0 and len(curr_locs) > 0:
                    # Calculate center of mass movement
                    prev_center = prev_locs.float().mean(dim=0)
                    curr_center = curr_locs.float().mean(dim=0)
                    movement = torch.norm(curr_center - prev_center)
                    
                    if movement > 0.5:
                        return True
                        
        return False
    
    def _detect_periodicity(self, sequence: List[torch.Tensor]) -> Optional[int]:
        """Detect periodicity in sequence"""
        if len(sequence) < 4:
            return None
            
        for period in range(2, len(sequence) // 2):
            is_periodic = True
            for i in range(period, len(sequence)):
                if not torch.allclose(sequence[i], sequence[i - period], atol=1e-3):
                    is_periodic = False
                    break
                    
            if is_periodic:
                return period
                
        return None
    
    def _measure_consistency(self, sequence: List[torch.Tensor]) -> float:
        """Measure temporal consistency"""
        if len(sequence) < 2:
            return 1.0
            
        total_change = 0
        for i in range(1, len(sequence)):
            change = torch.abs(sequence[i] - sequence[i-1]).sum()
            total_change += change.item()
            
        # Normalize by sequence length and grid size
        avg_change = total_change / (len(sequence) - 1)
        max_possible_change = 10 * self.grid_size * self.grid_size * 2
        
        consistency = 1.0 - (avg_change / max_possible_change)
        return max(0, min(1, consistency))
    
    def _calculate_change_rate(self, sequence: List[torch.Tensor]) -> List[float]:
        """Calculate change rate over time"""
        rates = []
        
        for i in range(1, len(sequence)):
            change = torch.abs(sequence[i] - sequence[i-1]).sum().item()
            rates.append(change)
            
        return rates


class ChronosLEAPTrainer:
    """LEAP trainer wrapper for CHRONOS training integration"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.leap = ChronosLEAP()
        self.pattern_stats = {}
        
    def generate_leap_batch(self, batch_size: int = 32, stage: int = 0) -> Dict[str, torch.Tensor]:
        """Generate batch compatible with training script"""
        # Map stage to difficulty level
        difficulty_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4}
        difficulty = difficulty_map.get(stage, 1)
        
        # Map stage to grid size
        grid_sizes = {0: 6, 1: 9, 2: 12, 3: 15, 4: 18, 5: 21, 6: 24, 7: 30}
        grid_size = grid_sizes.get(stage, 12)
        
        # Update LEAP grid size
        self.leap.grid_size = grid_size
        
        # Generate patterns
        patterns = self.leap.generate_curriculum_batch(difficulty)
        
        # Convert to training format
        inputs = []
        outputs = []
        pattern_types = []
        
        for pattern in patterns[:batch_size]:
            if len(pattern.sequence) >= 2:
                # Use first and last frames
                input_frame = pattern.sequence[0].argmax(dim=0)  # Convert from one-hot
                output_frame = pattern.sequence[-1].argmax(dim=0)
                
                inputs.append(input_frame)
                outputs.append(output_frame)
                pattern_types.append(pattern.name)
                
        if not inputs:
            # Fallback empty batch
            inputs = [torch.zeros(grid_size, grid_size, dtype=torch.long) for _ in range(batch_size)]
            outputs = [torch.zeros(grid_size, grid_size, dtype=torch.long) for _ in range(batch_size)]
            pattern_types = ['empty'] * batch_size
            
        return {
            'inputs': torch.stack(inputs).to(self.device),
            'outputs': torch.stack(outputs).to(self.device),
            'pattern_types': pattern_types
        }
    
    def update_pattern_stats(self, pattern_types: List[str], predictions: torch.Tensor, 
                           targets: torch.Tensor):
        """Update statistics for pattern learning"""
        pred_indices = predictions.argmax(dim=1)
        target_indices = targets.argmax(dim=1)
        
        for i, pattern_type in enumerate(pattern_types):
            if pattern_type not in self.pattern_stats:
                self.pattern_stats[pattern_type] = {'total': 0, 'correct': 0}
                
            self.pattern_stats[pattern_type]['total'] += 1
            
            # Check if prediction captures temporal transformation
            is_correct = (pred_indices[i] == target_indices[i]).all()
            if is_correct:
                self.pattern_stats[pattern_type]['correct'] += 1
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        if not self.pattern_stats:
            return "No temporal patterns trained yet"
        
        reports = []
        for pattern, stats in sorted(self.pattern_stats.items()):
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total'] * 100
                if accuracy > 0:
                    reports.append(f"{pattern}: {accuracy:.1f}%")
        
        if reports:
            return "Temporal LEAP: " + ", ".join(reports[:3])  # Top 3
        return "Temporal LEAP: 0.0%"


def create_chronos_leap_system(grid_size: int = 30):
    """Factory function to create CHRONOS LEAP system"""
    trainer = ChronosLEAPTrainer()
    # Set initial grid size
    trainer.leap.grid_size = grid_size
    
    return {
        'trainer': trainer,  # Note: training script expects 'trainer'
        'generator': trainer,  # Same object handles generation
        'detector': None,  # CHRONOS doesn't use weak point detection
        'description': 'CHRONOS Temporal Sequence LEAP System'
    }