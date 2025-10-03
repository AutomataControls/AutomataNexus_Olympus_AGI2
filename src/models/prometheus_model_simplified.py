"""
PROMETHEUS Model - Simplified Creative Pattern Generation
Part of the OLYMPUS AGI2 Ensemble - FIXED for numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SimplifiedPrometheusNet(nn.Module):
    """Simplified PROMETHEUS - Based on successful IRIS/ATLAS architecture"""
    def __init__(self, max_grid_size: int = 12):
        super().__init__()
        
        # Simple encoder like IRIS/ATLAS
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Pattern generation head
        self.pattern_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Rule generation head  
        self.rule_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Simple decoder like IRIS/ATLAS
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.ConvTranspose2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(32, 10, 1)
        )
        
        # Initialize final layer for creative patterns
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=1.2)
        # Slight bias variation for creativity
        self.decoder[-1].bias.data = torch.tensor([0.0, 0.05, 0.03, 0.04, 0.02, 0.04, 0.05, 0.06, 0.04, 0.03])
        
        # Creativity mixing parameter (learnable)
        self.creativity_mix = nn.Parameter(torch.tensor(0.1))
        
        self.description = "Simplified Creative Pattern Generation (No VAE)"
        
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        B, C, H, W = input_grid.shape
        
        # Encode features
        features = self.encoder(input_grid)
        
        # Global pattern analysis
        global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Generate pattern rules
        pattern_rules = self.pattern_head(global_features)
        transformation_rules = self.rule_head(global_features)
        
        # Creative feature enhancement
        enhanced_features = self._enhance_features(features, pattern_rules)
        
        # Decode
        predicted_output = self.decoder(enhanced_features)
        
        # Apply creativity mixing
        if mode == 'train' and output_grid is not None:
            # During training, mix with slight variations for creativity
            noise = torch.randn_like(predicted_output) * 0.1
            creative_output = predicted_output + self.creativity_mix * noise
        else:
            creative_output = predicted_output
        
        return {
            'predicted_output': creative_output,
            'raw_output': predicted_output,
            'pattern_rules': pattern_rules,
            'transform_rules': transformation_rules,
            'creativity_factor': self.creativity_mix
        }
    
    def _enhance_features(self, features: torch.Tensor, pattern_rules: torch.Tensor) -> torch.Tensor:
        """Enhance features using pattern rules"""
        B, C, H, W = features.shape
        
        # Expand pattern rules to spatial dimensions
        rules_spatial = pattern_rules.unsqueeze(-1).unsqueeze(-1).expand(B, 128, H, W)
        
        # Apply pattern-based modulation
        enhanced = features * (1.0 + 0.1 * torch.tanh(rules_spatial))
        
        return enhanced


# Alias for compatibility
EnhancedPrometheusNet = SimplifiedPrometheusNet