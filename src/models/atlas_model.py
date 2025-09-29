"""
ATLAS Model - Spatial Transformation Specialist
Part of the OLYMPUS AGI2 Ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EnhancedAtlasNet(nn.Module):
    """Enhanced ATLAS with better spatial transformation learning"""
    def __init__(self, max_grid_size: int = 30):
        super().__init__()
        self.max_grid_size = max_grid_size
        
        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Spatial transformer network
        self.localization = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # Affine transformation parameters
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 2x3 affine matrix
        )
        
        # Rotation predictor
        self.rotation_head = nn.Linear(128, 4)  # 0°, 90°, 180°, 270°
        
        # Reflection predictor  
        self.reflection_head = nn.Linear(128, 3)  # None, Horizontal, Vertical
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(32, 10, 1)
        )
        
        # Initialize final layer with strong values
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=1.2)
        # Equal bias for all colors in ATLAS
        self.decoder[-1].bias.data = torch.ones(10) * 0.05
        
        # Initialize affine matrix to identity
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # Mix parameter - start at 0.05 to heavily favor transformations
        self.mix_param = nn.Parameter(torch.tensor(0.05))
        
        self.description = "Enhanced Spatial Transformer with Rotation/Reflection"
        
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        B = input_grid.shape[0]
        
        # Encode features
        features = self.encoder(input_grid)
        
        # Predict spatial transformation
        loc_features = self.localization(features)
        theta = self.fc_loc(loc_features.reshape(B, -1))
        theta = theta.view(-1, 2, 3)
        
        # Create sampling grid
        grid = F.affine_grid(theta, input_grid.size(), align_corners=False)
        
        # Apply spatial transformation
        transformed_input = F.grid_sample(input_grid, grid, align_corners=False)
        transformed_features = F.grid_sample(features, grid, align_corners=False)
        
        # Predict rotation and reflection
        pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        rotation_logits = self.rotation_head(pooled)
        reflection_logits = self.reflection_head(pooled)
        
        # Apply discrete transformations
        transformed_features = self._apply_discrete_transforms(
            transformed_features, rotation_logits, reflection_logits
        )
        
        # Decode to output
        predicted_output = self.decoder(transformed_features)
        
        # Add minimal residual for ATLAS to prevent collapse
        mix = torch.sigmoid(self.mix_param)
        if self.training:
            mix = mix * 0.2  # Use much less residual for ATLAS
        else:
            mix = mix * 0.3
        predicted_output = predicted_output * (1 - mix) + input_grid * mix
        
        return {
            'predicted_output': predicted_output,
            'theta': theta,
            'rotation_logits': rotation_logits,
            'reflection_logits': reflection_logits,
            'transformed_input': transformed_input
        }
    
    def _apply_discrete_transforms(self, features: torch.Tensor, 
                                  rotation_logits: torch.Tensor,
                                  reflection_logits: torch.Tensor) -> torch.Tensor:
        """Apply predicted rotations and reflections"""
        B = features.shape[0]
        
        # Get predictions
        rotation_idx = rotation_logits.argmax(dim=1)
        reflection_idx = reflection_logits.argmax(dim=1)
        
        output = features.clone()
        
        for i in range(B):
            # Apply rotation
            rot = rotation_idx[i].item()
            if rot > 0:
                output[i] = torch.rot90(output[i], k=rot, dims=[1, 2])
            
            # Apply reflection
            ref = reflection_idx[i].item()
            if ref == 1:  # Horizontal
                output[i] = torch.flip(output[i], dims=[1])
            elif ref == 2:  # Vertical
                output[i] = torch.flip(output[i], dims=[2])
        
        return output