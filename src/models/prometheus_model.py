"""
PROMETHEUS Model - Creative Pattern Generation
Part of the OLYMPUS AGI2 Ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class EnhancedPrometheusNet(nn.Module):
    """Enhanced PROMETHEUS with better pattern generation"""
    def __init__(self, max_grid_size: int = 30, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Enhanced encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Object and pattern encoder
        self.pattern_encoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4)
        )
        
        # VAE components
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Pattern synthesis network
        self.synthesis_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4)
        )
        
        # Rule generator
        self.rule_generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(32, 10, 1)
        )
        
        # Initialize final layer for creative generation
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=2.0)
        # Small varied bias for PROMETHEUS creativity
        self.decoder[-1].bias.data = torch.tensor([0.0, 0.1, 0.08, 0.06, 0.09, 0.07, 0.11, 0.12, 0.1, 0.08])
        
        self.description = "Enhanced Creative Pattern Generation with VAE"
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution"""
        features = self.encoder(x)
        pattern_features = self.pattern_encoder(features)
        pattern_flat = pattern_features.view(pattern_features.shape[0], -1)
        
        mu = self.fc_mu(pattern_flat)
        log_var = self.fc_var(pattern_flat)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, input_shape: torch.Size) -> torch.Tensor:
        """Decode from latent space"""
        # Generate features
        features = self.synthesis_net(z)
        features = features.view(-1, 256, 4, 4)
        
        # Upsample to match input size
        B, _, H, W = input_shape
        
        # Decode
        output = self.decoder(features)
        
        # Ensure output matches input spatial size
        if output.shape[-2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output
    
    def forward(self, input_grid: torch.Tensor, target_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        # Encode
        mu, log_var = self.encode(input_grid)
        z = self.reparameterize(mu, log_var)
        
        # Generate transformation rules
        rules = self.rule_generator(z)
        
        # Decode
        predicted_output = self.decode(z, input_grid.shape)
        
        # Apply rules to refine output
        refined_output = self._apply_rules(predicted_output, rules, input_grid)
        
        # No additional residual here - already handled in _apply_rules
        
        outputs = {
            'predicted_output': refined_output,
            'raw_output': predicted_output,
            'mu': mu,
            'log_var': log_var,
            'latent': z,
            'rules': rules
        }
        
        return outputs
    
    def _apply_rules(self, generated: torch.Tensor, rules: torch.Tensor, input_grid: torch.Tensor) -> torch.Tensor:
        """Apply learned rules to refine generation"""
        # Simple rule application - enhance this
        B = generated.shape[0]
        
        # Use rules to modulate generation
        rule_weights = torch.sigmoid(rules[:, :10]).view(B, 10, 1, 1)
        
        # Blend with input based on rules
        refined = generated * rule_weights + input_grid * (1 - rule_weights)
        
        return refined