"""
IRIS Model - Color Pattern Recognition Expert
Part of the OLYMPUS AGI2 Ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EnhancedIrisNet(nn.Module):
    """Enhanced IRIS with color relationship learning"""
    def __init__(self, max_grid_size: int = 30):
        super().__init__()
        
        # Color embedding
        self.color_embed = nn.Embedding(10, 64)
        
        # Color attention mechanism
        self.color_attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        
        # Spatial color encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        
        # Color mapping predictor
        self.color_mapper = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10 * 10)  # 10x10 color mapping matrix
        )
        
        # Pattern-based color rules
        self.rule_encoder = nn.LSTM(64, 128, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(32, 10, 1)
        )
        
        # Initialize final layer for strong color changes
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=1.5)
        # Slightly varied bias for IRIS color specialization
        self.decoder[-1].bias.data = torch.tensor([0.0, 0.08, 0.06, 0.07, 0.05, 0.06, 0.08, 0.09, 0.07, 0.06])
        
        self.description = "Enhanced Color Pattern Recognition with Attention"
        
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        B, C, H, W = input_grid.shape
        
        # Get color distribution
        color_indices = input_grid.argmax(dim=1)  # B, H, W
        
        # Embed colors
        color_embeddings = self.color_embed(color_indices)  # B, H, W, 64
        color_flat = color_embeddings.view(B, -1, 64)
        
        # Color attention
        attended_colors, color_weights = self.color_attention(color_flat, color_flat, color_flat)
        attended_colors = attended_colors.view(B, H, W, 64).permute(0, 3, 1, 2)
        
        # Spatial encoding
        spatial_features = self.spatial_encoder(input_grid)
        
        # Combine color and spatial
        combined = spatial_features + attended_colors
        
        # Predict color mapping
        global_features = F.adaptive_avg_pool2d(combined, 1).squeeze(-1).squeeze(-1)
        color_map_logits = self.color_mapper(global_features).view(B, 10, 10)
        
        # Apply color mapping
        mapped_output = self._apply_color_mapping(input_grid, color_map_logits)
        
        # Final decode
        predicted_output = self.decoder(combined)
        
        # For IRIS, we don't add residual - it does color mapping
        # The mapped_output already contains the transformed colors
        
        return {
            'predicted_output': predicted_output,
            'color_map': F.softmax(color_map_logits, dim=-1),
            'color_attention': color_weights,
            'mapped_output': mapped_output
        }
    
    def _apply_color_mapping(self, input_grid: torch.Tensor, color_map: torch.Tensor) -> torch.Tensor:
        """Apply learned color mapping"""
        B, C, H, W = input_grid.shape
        
        # Get input colors
        color_indices = input_grid.argmax(dim=1, keepdim=True).float()  # B, 1, H, W
        
        # Apply mapping
        output = torch.zeros_like(input_grid)
        
        for b in range(B):
            for c_in in range(10):
                mask = (color_indices[b, 0] == c_in)
                if mask.any():
                    # Get mapped color probabilities
                    mapped_probs = color_map[b, c_in]
                    # Take argmax as new color
                    new_color = mapped_probs.argmax().item()
                    output[b, new_color, mask] = 1.0
        
        return output