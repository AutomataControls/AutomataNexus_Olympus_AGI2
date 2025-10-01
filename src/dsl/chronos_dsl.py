"""
CHRONOS DSL - Domain Specific Language for Temporal Sequences
Focused on time-based patterns and sequence transformations
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import random
from dataclasses import dataclass


@dataclass
class TemporalOperation:
    """Represents a temporal transformation operation"""
    name: str
    params: Dict[str, Any]
    time_step: int
    duration: int = 1
    periodic: bool = False
    period: int = 1


class ChronosDSL:
    """Domain Specific Language for CHRONOS temporal patterns"""
    
    def __init__(self, max_time_steps: int = 10):
        self.max_time_steps = max_time_steps
        self.operations = self._define_temporal_operations()
        
    def _define_temporal_operations(self) -> Dict[str, callable]:
        """Define temporal-specific operations"""
        return {
            # Movement operations
            'shift_right': self._shift_right,
            'shift_left': self._shift_left,
            'shift_up': self._shift_up,
            'shift_down': self._shift_down,
            'rotate_clockwise': self._rotate_clockwise,
            'rotate_counterclockwise': self._rotate_counterclockwise,
            
            # Temporal transformations
            'fade_in': self._fade_in,
            'fade_out': self._fade_out,
            'pulse': self._pulse,
            'oscillate': self._oscillate,
            'grow': self._grow,
            'shrink': self._shrink,
            
            # Sequence operations
            'accumulate': self._accumulate,
            'trail': self._trail,
            'echo': self._echo,
            'reverse_time': self._reverse_time,
            'time_warp': self._time_warp,
            'periodic_change': self._periodic_change,
            
            # Pattern evolution
            'evolve_pattern': self._evolve_pattern,
            'morph': self._morph,
            'cascade': self._cascade,
            'ripple': self._ripple,
            'wave': self._wave,
            'spiral': self._spiral
        }
    
    def generate_temporal_sequence(self, initial_grid: torch.Tensor, 
                                 num_steps: int = 5) -> List[torch.Tensor]:
        """Generate a temporal sequence from initial grid"""
        sequence = [initial_grid]
        
        # Choose sequence type
        seq_type = random.choice(['movement', 'transformation', 'evolution', 'periodic'])
        
        if seq_type == 'movement':
            sequence = self._generate_movement_sequence(initial_grid, num_steps)
        elif seq_type == 'transformation':
            sequence = self._generate_transformation_sequence(initial_grid, num_steps)
        elif seq_type == 'evolution':
            sequence = self._generate_evolution_sequence(initial_grid, num_steps)
        else:
            sequence = self._generate_periodic_sequence(initial_grid, num_steps)
            
        return sequence
    
    def _generate_movement_sequence(self, grid: torch.Tensor, steps: int) -> List[torch.Tensor]:
        """Generate sequence with object movement"""
        sequence = [grid.clone()]
        current = grid.clone()
        
        # Choose movement pattern
        pattern = random.choice(['linear', 'circular', 'zigzag', 'bounce'])
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        for t in range(steps - 1):
            if pattern == 'linear':
                if direction == 'horizontal':
                    op = 'shift_right' if t % 2 == 0 else 'shift_left'
                else:
                    op = 'shift_down' if t % 2 == 0 else 'shift_up'
                current = self.operations[op](current, amount=1)
                
            elif pattern == 'circular':
                current = self.operations['rotate_clockwise'](current, angle=45)
                
            elif pattern == 'zigzag':
                if t % 4 == 0:
                    current = self.operations['shift_right'](current, amount=1)
                elif t % 4 == 1:
                    current = self.operations['shift_down'](current, amount=1)
                elif t % 4 == 2:
                    current = self.operations['shift_left'](current, amount=1)
                else:
                    current = self.operations['shift_up'](current, amount=1)
                    
            elif pattern == 'bounce':
                if t < steps // 2:
                    current = self.operations['shift_right'](current, amount=1)
                else:
                    current = self.operations['shift_left'](current, amount=1)
                    
            sequence.append(current.clone())
            
        return sequence
    
    def _generate_transformation_sequence(self, grid: torch.Tensor, steps: int) -> List[torch.Tensor]:
        """Generate sequence with gradual transformations"""
        sequence = [grid.clone()]
        current = grid.clone()
        
        transform_type = random.choice(['fade', 'morph', 'grow_shrink', 'pulse'])
        
        for t in range(steps - 1):
            progress = (t + 1) / steps
            
            if transform_type == 'fade':
                if progress < 0.5:
                    current = self.operations['fade_out'](current, intensity=progress * 2)
                else:
                    current = self.operations['fade_in'](current, intensity=(progress - 0.5) * 2)
                    
            elif transform_type == 'morph':
                current = self.operations['morph'](current, progress=progress)
                
            elif transform_type == 'grow_shrink':
                if progress < 0.5:
                    current = self.operations['grow'](current, factor=1 + progress)
                else:
                    current = self.operations['shrink'](current, factor=2 - progress)
                    
            elif transform_type == 'pulse':
                current = self.operations['pulse'](current, phase=progress * 2 * 3.14159)
                
            sequence.append(current.clone())
            
        return sequence
    
    def _generate_evolution_sequence(self, grid: torch.Tensor, steps: int) -> List[torch.Tensor]:
        """Generate sequence with pattern evolution"""
        sequence = [grid.clone()]
        current = grid.clone()
        
        evolution_type = random.choice(['cascade', 'ripple', 'wave', 'spiral'])
        
        for t in range(steps - 1):
            if evolution_type == 'cascade':
                current = self.operations['cascade'](current, step=t)
            elif evolution_type == 'ripple':
                current = self.operations['ripple'](current, time=t, center=(grid.shape[-2]//2, grid.shape[-1]//2))
            elif evolution_type == 'wave':
                current = self.operations['wave'](current, phase=t * 0.5)
            elif evolution_type == 'spiral':
                current = self.operations['spiral'](current, angle=t * 30)
                
            sequence.append(current.clone())
            
        return sequence
    
    def _generate_periodic_sequence(self, grid: torch.Tensor, steps: int) -> List[torch.Tensor]:
        """Generate sequence with periodic patterns"""
        sequence = [grid.clone()]
        
        period = random.randint(2, min(4, steps // 2))
        operation = random.choice(['oscillate', 'periodic_change', 'echo'])
        
        for t in range(steps - 1):
            phase = (t % period) / period
            
            if operation == 'oscillate':
                current = self.operations['oscillate'](grid, phase=phase)
            elif operation == 'periodic_change':
                current = self.operations['periodic_change'](grid, time=t, period=period)
            else:
                current = self.operations['echo'](grid, delay=t % period, decay=0.8)
                
            sequence.append(current.clone())
            
        return sequence
    
    # Movement operations
    def _shift_right(self, grid: torch.Tensor, amount: int = 1) -> torch.Tensor:
        """Shift pattern right"""
        shifted = torch.roll(grid, shifts=amount, dims=-1)
        if amount > 0:
            shifted[..., :amount] = 0
        return shifted
    
    def _shift_left(self, grid: torch.Tensor, amount: int = 1) -> torch.Tensor:
        """Shift pattern left"""
        shifted = torch.roll(grid, shifts=-amount, dims=-1)
        if amount > 0:
            shifted[..., -amount:] = 0
        return shifted
    
    def _shift_up(self, grid: torch.Tensor, amount: int = 1) -> torch.Tensor:
        """Shift pattern up"""
        shifted = torch.roll(grid, shifts=-amount, dims=-2)
        if amount > 0:
            shifted[..., -amount:, :] = 0
        return shifted
    
    def _shift_down(self, grid: torch.Tensor, amount: int = 1) -> torch.Tensor:
        """Shift pattern down"""
        shifted = torch.roll(grid, shifts=amount, dims=-2)
        if amount > 0:
            shifted[..., :amount, :] = 0
        return shifted
    
    def _rotate_clockwise(self, grid: torch.Tensor, angle: int = 90) -> torch.Tensor:
        """Rotate pattern clockwise"""
        if angle == 90:
            return torch.rot90(grid, k=1, dims=[-2, -1])
        elif angle == 180:
            return torch.rot90(grid, k=2, dims=[-2, -1])
        elif angle == 270:
            return torch.rot90(grid, k=3, dims=[-2, -1])
        else:
            # For other angles, use interpolation (simplified)
            return grid
    
    def _rotate_counterclockwise(self, grid: torch.Tensor, angle: int = 90) -> torch.Tensor:
        """Rotate pattern counterclockwise"""
        if angle == 90:
            return torch.rot90(grid, k=-1, dims=[-2, -1])
        elif angle == 180:
            return torch.rot90(grid, k=-2, dims=[-2, -1])
        elif angle == 270:
            return torch.rot90(grid, k=-3, dims=[-2, -1])
        else:
            return grid
    
    # Temporal transformations
    def _fade_in(self, grid: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Gradually increase visibility"""
        return grid * intensity
    
    def _fade_out(self, grid: torch.Tensor, intensity: float = 0.5) -> torch.Tensor:
        """Gradually decrease visibility"""
        return grid * (1 - intensity)
    
    def _pulse(self, grid: torch.Tensor, phase: float = 0) -> torch.Tensor:
        """Create pulsing effect"""
        intensity = (torch.sin(torch.tensor(phase)) + 1) / 2
        return grid * intensity
    
    def _oscillate(self, grid: torch.Tensor, phase: float = 0) -> torch.Tensor:
        """Oscillate between states"""
        mask = torch.rand_like(grid[:1]) > 0.5
        if phase < 0.5:
            return grid * mask
        else:
            return grid * (~mask)
    
    def _grow(self, grid: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
        """Grow pattern"""
        # Simple dilation effect
        kernel = torch.ones(1, 1, 3, 3) / 9
        if grid.dim() == 4:
            grown = F.conv2d(grid.float(), kernel, padding=1)
            return (grown > 0.3).float() * grid.max()
        return grid
    
    def _shrink(self, grid: torch.Tensor, factor: float = 0.7) -> torch.Tensor:
        """Shrink pattern"""
        # Simple erosion effect
        kernel = torch.ones(1, 1, 3, 3) / 9
        if grid.dim() == 4:
            shrunk = F.conv2d(grid.float(), kernel, padding=1)
            return (shrunk > 0.7).float() * grid.max()
        return grid
    
    # Sequence operations
    def _accumulate(self, sequence: List[torch.Tensor]) -> torch.Tensor:
        """Accumulate patterns over time"""
        if not sequence:
            return torch.zeros_like(sequence[0])
        accumulated = sequence[0].clone()
        for grid in sequence[1:]:
            accumulated = torch.maximum(accumulated, grid)
        return accumulated
    
    def _trail(self, grid: torch.Tensor, history: List[torch.Tensor], decay: float = 0.8) -> torch.Tensor:
        """Create trailing effect"""
        result = grid.clone()
        for i, hist in enumerate(reversed(history[-3:])):
            weight = decay ** (i + 1)
            result = torch.maximum(result, hist * weight)
        return result
    
    def _echo(self, grid: torch.Tensor, delay: int = 1, decay: float = 0.7) -> torch.Tensor:
        """Create echo effect"""
        if delay == 0:
            return grid
        shifted = torch.roll(grid, shifts=delay, dims=-1)
        return torch.maximum(grid, shifted * decay)
    
    def _reverse_time(self, sequence: List[torch.Tensor]) -> List[torch.Tensor]:
        """Reverse temporal order"""
        return list(reversed(sequence))
    
    def _time_warp(self, sequence: List[torch.Tensor], warp_factor: float = 2.0) -> List[torch.Tensor]:
        """Speed up or slow down time"""
        if warp_factor == 1.0:
            return sequence
        
        new_length = int(len(sequence) * warp_factor)
        if new_length < 2:
            return sequence[:2]
            
        indices = torch.linspace(0, len(sequence) - 1, new_length)
        warped = []
        for idx in indices:
            i = int(idx)
            if i >= len(sequence) - 1:
                warped.append(sequence[-1])
            else:
                # Simple interpolation
                alpha = idx - i
                interp = sequence[i] * (1 - alpha) + sequence[i + 1] * alpha
                warped.append(interp)
        return warped
    
    def _periodic_change(self, grid: torch.Tensor, time: int, period: int = 3) -> torch.Tensor:
        """Apply periodic transformations"""
        phase = time % period
        if phase == 0:
            return grid
        elif phase == 1:
            return self._shift_right(grid, amount=1)
        else:
            return self._rotate_clockwise(grid, angle=90)
    
    # Pattern evolution
    def _evolve_pattern(self, grid: torch.Tensor, rules: Optional[Dict] = None) -> torch.Tensor:
        """Evolve pattern based on rules"""
        if rules is None:
            # Simple cellular automaton-like evolution
            kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            if grid.dim() == 4:
                neighbors = F.conv2d(grid.float(), kernel, padding=1)
                # Birth: dead cell with 3 neighbors
                birth = (grid == 0) & (neighbors == 3)
                # Survival: live cell with 2 or 3 neighbors
                survival = (grid > 0) & ((neighbors == 2) | (neighbors == 3))
                return (birth | survival).float() * grid.max()
        return grid
    
    def _morph(self, grid: torch.Tensor, progress: float = 0.5) -> torch.Tensor:
        """Morph between shapes"""
        # Simple morphing by blending with transformed version
        target = self._rotate_clockwise(grid, angle=180)
        return grid * (1 - progress) + target * progress
    
    def _cascade(self, grid: torch.Tensor, step: int = 0) -> torch.Tensor:
        """Create cascading effect"""
        # Shift diagonally with each step
        shifted = torch.roll(grid, shifts=(step, step), dims=(-2, -1))
        # Clear wrapped areas
        if step > 0:
            shifted[..., :step, :] = 0
            shifted[..., :, :step] = 0
        return shifted
    
    def _ripple(self, grid: torch.Tensor, time: int, center: Tuple[int, int]) -> torch.Tensor:
        """Create ripple effect from center"""
        h, w = grid.shape[-2:]
        cy, cx = center
        
        # Create distance map
        y_coords = torch.arange(h).view(-1, 1).float()
        x_coords = torch.arange(w).view(1, -1).float()
        
        dist = torch.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
        
        # Create ripple mask
        wave_length = 3.0
        ripple = torch.sin(dist - time * 0.5) > 0
        
        return grid * ripple.float()
    
    def _wave(self, grid: torch.Tensor, phase: float = 0) -> torch.Tensor:
        """Create wave pattern"""
        h, w = grid.shape[-2:]
        x_coords = torch.arange(w).float()
        
        # Create sine wave
        wave = torch.sin(x_coords * 0.5 + phase) > 0
        wave = wave.view(1, -1).expand(h, -1)
        
        if grid.dim() == 4:
            wave = wave.unsqueeze(0).unsqueeze(0).expand_as(grid)
            
        return grid * wave.float()
    
    def _spiral(self, grid: torch.Tensor, angle: float = 0) -> torch.Tensor:
        """Create spiral transformation"""
        # Simplified spiral - rotate with distance-based scaling
        h, w = grid.shape[-2:]
        cy, cx = h // 2, w // 2
        
        rotated = self._rotate_clockwise(grid, angle=int(angle) % 360)
        
        # Add radial scaling
        y_coords = torch.arange(h).view(-1, 1).float()
        x_coords = torch.arange(w).view(1, -1).float()
        dist = torch.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
        scale = 1.0 - (dist / dist.max()) * 0.3
        
        if grid.dim() == 4:
            scale = scale.unsqueeze(0).unsqueeze(0).expand_as(grid)
            
        return rotated * scale
    
    def create_temporal_program(self, operations: List[TemporalOperation]) -> List[torch.Tensor]:
        """Execute a sequence of temporal operations"""
        if not operations:
            return []
            
        # Initialize with empty grid
        initial = torch.zeros(1, 10, 12, 12)
        sequence = [initial]
        
        for op in operations:
            current = sequence[-1].clone()
            
            if op.name in self.operations:
                if op.periodic and len(sequence) % op.period == 0:
                    current = self.operations[op.name](current, **op.params)
                elif not op.periodic:
                    current = self.operations[op.name](current, **op.params)
                    
            sequence.append(current)
            
            if len(sequence) >= self.max_time_steps:
                break
                
        return sequence
    
    def analyze_sequence(self, sequence: List[torch.Tensor]) -> Dict[str, Any]:
        """Analyze temporal sequence properties"""
        if not sequence:
            return {}
            
        analysis = {
            'length': len(sequence),
            'has_movement': False,
            'has_periodic': False,
            'has_evolution': False,
            'dominant_pattern': None,
            'temporal_features': []
        }
        
        # Check for movement
        for i in range(1, len(sequence)):
            diff = torch.abs(sequence[i] - sequence[i-1]).sum()
            if diff > 0:
                analysis['has_movement'] = True
                break
        
        # Check for periodicity
        if len(sequence) > 3:
            for period in range(2, len(sequence) // 2):
                is_periodic = True
                for i in range(period, len(sequence)):
                    if not torch.allclose(sequence[i], sequence[i - period], atol=1e-3):
                        is_periodic = False
                        break
                if is_periodic:
                    analysis['has_periodic'] = True
                    analysis['period'] = period
                    break
        
        # Extract temporal features
        for i in range(1, len(sequence)):
            feature = {
                'time_step': i,
                'change_magnitude': torch.abs(sequence[i] - sequence[i-1]).sum().item(),
                'active_pixels': (sequence[i] > 0).sum().item()
            }
            analysis['temporal_features'].append(feature)
            
        return analysis