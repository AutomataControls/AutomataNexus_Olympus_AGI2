"""
Transformation Reasoning Module - Core component for learning ARC transformations
Used by all OLYMPUS models to understand and apply transformation rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class TransformationRuleBank:
    """Bank of discovered transformation rules"""
    def __init__(self, max_rules: int = 1000):
        self.max_rules = max_rules
        self.rules = {}
        self.rule_embeddings = []
        self.rule_success_counts = {}
        
    def add_rule(self, input_pattern: np.ndarray, output_pattern: np.ndarray, 
                 transformation_type: str, success: bool = True):
        """Add a discovered transformation rule"""
        rule_key = f"{transformation_type}_{hash(input_pattern.tobytes())}_{hash(output_pattern.tobytes())}"
        
        if rule_key not in self.rules:
            self.rules[rule_key] = {
                'input': input_pattern,
                'output': output_pattern,
                'type': transformation_type,
                'successes': 0,
                'attempts': 0
            }
        
        self.rules[rule_key]['attempts'] += 1
        if success:
            self.rules[rule_key]['successes'] += 1
    
    def get_successful_rules(self, min_success_rate: float = 0.7) -> List[Dict]:
        """Get rules with high success rate"""
        successful = []
        for rule_key, rule in self.rules.items():
            if rule['attempts'] > 5:  # Need enough attempts
                success_rate = rule['successes'] / rule['attempts']
                if success_rate >= min_success_rate:
                    successful.append(rule)
        return successful


class TransformationDetector(nn.Module):
    """Detects what transformation was applied between input and output"""
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        
        # Transformation types we can detect
        self.transformation_types = [
            'identity', 'color_swap', 'rotation', 'reflection', 
            'translation', 'scaling', 'pattern_fill', 'extraction',
            'counting', 'grouping', 'symmetry', 'repetition',
            'logical_and', 'logical_or', 'logical_xor', 'inversion',
            'boundary', 'connectivity', 'morphology', 'composition'
        ]
        
        # Encoder for input/output pairs
        self.input_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, feature_dim)
        )
        
        self.output_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, feature_dim)
        )
        
        # Difference encoder
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 16, feature_dim)
        )
        
        # Transformation classifier
        self.transform_classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(self.transformation_types))
        )
        
        # Transformation embeddings for each type
        self.transform_embeddings = nn.Embedding(len(self.transformation_types), feature_dim)
        
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect transformation between input and output"""
        # Encode input and output
        input_features = self.input_encoder(input_grid)
        output_features = self.output_encoder(output_grid)
        
        # Compute difference
        diff = output_grid - input_grid
        diff_features = self.diff_encoder(diff)
        
        # Combine all features
        combined = torch.cat([input_features, output_features, diff_features], dim=1)
        
        # Classify transformation type
        transform_logits = self.transform_classifier(combined)
        transform_probs = F.softmax(transform_logits, dim=1)
        
        # Get transformation embeddings
        transform_idx = transform_logits.argmax(dim=1)
        transform_emb = self.transform_embeddings(transform_idx)
        
        return {
            'transform_logits': transform_logits,
            'transform_probs': transform_probs,
            'transform_embedding': transform_emb,
            'transform_type': transform_idx
        }


class TransformationReasoner(nn.Module):
    """Core reasoning module for understanding and applying transformations"""
    def __init__(self, feature_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Transformation detector
        self.transform_detector = TransformationDetector(feature_dim)
        
        # Rule memory bank
        self.rule_bank = TransformationRuleBank()
        
        # Compositional reasoning - can combine multiple transformations
        self.composition_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        
        # Rule application network
        self.rule_applicator = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Output predictor given transformation
        self.output_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 10 * 30 * 30)  # Max output size
        )
        
    def detect_transformation(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect what transformation was applied"""
        return self.transform_detector(input_grid, output_grid)
    
    def learn_from_example(self, input_grid: torch.Tensor, output_grid: torch.Tensor, 
                          success: bool = True):
        """Learn transformation rules from examples"""
        with torch.no_grad():
            # Detect transformation
            transform_info = self.detect_transformation(input_grid, output_grid)
            transform_type_idx = transform_info['transform_type'].item()
            transform_type = self.transform_detector.transformation_types[transform_type_idx]
            
            # Store rule in bank
            if input_grid.shape[0] == 1:  # Single example
                self.rule_bank.add_rule(
                    input_grid[0].cpu().numpy(),
                    output_grid[0].cpu().numpy(),
                    transform_type,
                    success
                )
    
    def apply_transformation(self, input_grid: torch.Tensor, transform_embedding: torch.Tensor) -> torch.Tensor:
        """Apply a transformation given its embedding"""
        B, C, H, W = input_grid.shape
        
        # Flatten input
        input_flat = input_grid.view(B, C * H * W)
        
        # Combine input with transformation
        combined = torch.cat([input_flat, transform_embedding], dim=1)
        
        # Predict output
        output_flat = self.output_predictor(combined)
        
        # Reshape to grid
        # Handle variable sizes by taking the relevant part
        output_size = C * H * W
        output = output_flat[:, :output_size].view(B, C, H, W)
        
        return output
    
    def reason_compositionally(self, input_grid: torch.Tensor, 
                              transform_sequence: List[torch.Tensor]) -> torch.Tensor:
        """Apply a sequence of transformations compositionally"""
        current = input_grid
        
        for transform_emb in transform_sequence:
            # Apply transformation
            current = self.apply_transformation(current, transform_emb)
            
            # Ensure valid range
            current = torch.clamp(current, 0, 9)
            
        return current
    
    def forward(self, input_grid: torch.Tensor, target_grid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for training or inference"""
        B = input_grid.shape[0]
        
        if target_grid is not None:
            # Training mode - learn the transformation
            transform_info = self.detect_transformation(input_grid, target_grid)
            
            # Apply the detected transformation
            predicted_output = self.apply_transformation(
                input_grid, 
                transform_info['transform_embedding']
            )
            
            # Learn from this example
            self.learn_from_example(input_grid, target_grid, success=True)
            
            return {
                'predicted_output': predicted_output,
                'transform_info': transform_info
            }
        else:
            # Inference mode - try to detect and apply best transformation
            # This is harder and requires either:
            # 1. Additional context/examples
            # 2. Trying multiple transformations
            # For now, return identity
            return {
                'predicted_output': input_grid,
                'transform_info': None
            }


class RuleBasedReasoner(nn.Module):
    """Explicit rule-based reasoning for common ARC patterns"""
    def __init__(self):
        super().__init__()
        
    def detect_pattern(self, grid: torch.Tensor) -> Dict[str, bool]:
        """Detect common patterns in a grid"""
        # Convert to numpy for easier manipulation
        if grid.dim() == 4:
            grid_np = grid[0].argmax(dim=0).cpu().numpy()
        else:
            grid_np = grid.cpu().numpy()
        
        patterns = {}
        
        # Check for symmetries
        patterns['horizontal_symmetry'] = np.array_equal(grid_np, np.flip(grid_np, axis=0))
        patterns['vertical_symmetry'] = np.array_equal(grid_np, np.flip(grid_np, axis=1))
        patterns['diagonal_symmetry'] = np.array_equal(grid_np, grid_np.T)
        
        # Check for patterns
        patterns['is_uniform'] = len(np.unique(grid_np)) == 1
        patterns['has_border'] = (
            np.any(grid_np[0, :] != 0) or np.any(grid_np[-1, :] != 0) or
            np.any(grid_np[:, 0] != 0) or np.any(grid_np[:, -1] != 0)
        )
        
        # Check for objects
        patterns['num_colors'] = len(np.unique(grid_np))
        patterns['has_objects'] = patterns['num_colors'] > 2
        
        return patterns
    
    def apply_rule(self, input_grid: torch.Tensor, rule_name: str) -> torch.Tensor:
        """Apply a specific transformation rule"""
        B, C, H, W = input_grid.shape
        grid_indices = input_grid.argmax(dim=1)
        
        if rule_name == 'identity':
            return input_grid
        
        elif rule_name == 'horizontal_flip':
            return input_grid.flip(dims=[2])
        
        elif rule_name == 'vertical_flip':
            return input_grid.flip(dims=[3])
        
        elif rule_name == 'rotate_90':
            return input_grid.rot90(1, dims=[2, 3])
        
        elif rule_name == 'invert_colors':
            # Swap non-zero colors
            output = input_grid.clone()
            for b in range(B):
                unique_colors = grid_indices[b].unique()
                if len(unique_colors) == 2:
                    c1, c2 = unique_colors[0], unique_colors[1]
                    mask1 = grid_indices[b] == c1
                    mask2 = grid_indices[b] == c2
                    output[b, c1, mask2] = 1
                    output[b, c1, mask1] = 0
                    output[b, c2, mask1] = 1
                    output[b, c2, mask2] = 0
            return output
        
        else:
            # Unknown rule, return identity
            return input_grid


# Create a global transformation reasoning system
_global_transformer = None

def get_transformation_reasoner() -> TransformationReasoner:
    """Get the global transformation reasoner instance"""
    global _global_transformer
    if _global_transformer is None:
        _global_transformer = TransformationReasoner()
    return _global_transformer


def create_transformation_aware_loss(base_loss: nn.Module) -> nn.Module:
    """Wrap a loss function to be transformation-aware"""
    class TransformationAwareLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_loss = base_loss
            self.transformer = get_transformation_reasoner()
            
        def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                    input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
            # Get base loss
            base_losses = self.base_loss(pred, target, input_grid)
            
            # Detect transformation
            with torch.no_grad():
                transform_info = self.transformer.detect_transformation(input_grid, target)
            
            # Add transformation consistency loss
            transform_pred = self.transformer.apply_transformation(
                input_grid, transform_info['transform_embedding']
            )
            
            transform_loss = F.mse_loss(pred, transform_pred) * 0.5
            
            base_losses['transformation_consistency'] = transform_loss
            base_losses['total'] = base_losses['total'] + transform_loss
            
            return base_losses
    
    return TransformationAwareLoss()