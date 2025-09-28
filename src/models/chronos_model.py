"""
CHRONOS Model - Temporal Sequence Analysis
Part of the OLYMPUS AGI2 Ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ObjectEncoder(nn.Module):
    """Extract and encode objects from grids"""
    def __init__(self, in_channels: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, hidden_dim, 1)
        
        # Object detection head
        self.object_conv = nn.Conv2d(hidden_dim, 1, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features with residual connections
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        features = self.conv3(h2)
        
        # Detect objects with stronger activation
        object_masks = torch.sigmoid(self.object_conv(features) * 2.0)  # Scale up for sharper masks
        
        # Masked features with residual
        object_features = features * object_masks + features * 0.2  # Keep some global features
        
        return object_features, object_masks


class EnhancedChronosNet(nn.Module):
    """Enhanced CHRONOS with sequence pattern learning"""
    def __init__(self, max_grid_size: int = 30, hidden_dim: int = 256):
        super().__init__()
        
        # Grid encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Object tracking
        self.object_encoder = ObjectEncoder(10, 128)
        
        # Temporal reasoning with attention  
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.hidden_dim = hidden_dim
        
        # Projection layer for feature dimension matching
        self.feature_proj = nn.Linear(128, hidden_dim) if hidden_dim != 128 else nn.Identity()
        
        # Sequence predictor
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # Movement predictor
        self.movement_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(64, 10, 1)
        )
        
        # Initialize final layer with strong values
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=1.3)
        # Equal small bias for CHRONOS
        self.decoder[-1].bias.data = torch.ones(10) * 0.05
        
        # Mix parameter - start at 0.05 to heavily favor transformations
        self.mix_param = nn.Parameter(torch.tensor(0.05))
        
        self.description = "Enhanced Temporal Sequence Analysis with Attention"
        
    def forward(self, sequence: List[torch.Tensor], target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if len(sequence) == 1:
            # Single frame - treat as static
            return self._forward_single(sequence[0])
        
        # Encode all frames
        encoded_sequence = []
        object_sequences = []
        
        for frame in sequence:
            features = self.encoder(frame)
            obj_features, obj_masks = self.object_encoder(frame)
            
            encoded_sequence.append(features)
            object_sequences.append(obj_features)
        
        # Stack sequences
        # Pool features to get fixed dimension
        seq_features = []
        for features in encoded_sequence:
            pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # B, 128
            # Project to hidden_dim if needed
            pooled = self.feature_proj(pooled)
            seq_features.append(pooled)
        
        seq_tensor = torch.stack(seq_features, dim=1)  # B, seq_len, hidden_dim
        
        # Temporal attention
        attended_seq, attention_weights = self.temporal_attention(seq_tensor, seq_tensor, seq_tensor)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(attended_seq)
        
        # Predict next movement
        movement_params = self.movement_head(lstm_out[:, -1])
        
        # Apply movement to last frame
        last_frame_features = encoded_sequence[-1]
        last_objects = object_sequences[-1]
        
        moved_features = self._apply_movement(last_objects, movement_params)
        
        # Decode
        combined = torch.cat([last_frame_features, moved_features], dim=1)
        predicted_output = self.decoder(combined)
        
        # Minimal residual from last frame
        mix = torch.sigmoid(self.mix_param)
        predicted_output = predicted_output * (1 - mix) + sequence[-1] * mix
        
        return {
            'predicted_output': predicted_output,
            'movement_params': movement_params,
            'attention_weights': attention_weights,
            'temporal_features': lstm_out
        }
    
    def _forward_single(self, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Handle single frame input"""
        features = self.encoder(input_grid)
        obj_features, obj_masks = self.object_encoder(input_grid)
        
        # Simple forward without temporal processing
        combined = torch.cat([features, obj_features], dim=1)
        predicted_output = self.decoder(combined)
        
        # Minimal residual
        mix = torch.sigmoid(self.mix_param)
        predicted_output = predicted_output * (1 - mix) + input_grid * mix
        
        return {
            'predicted_output': predicted_output,
            'movement_params': torch.zeros(input_grid.shape[0], 128).to(input_grid.device),
            'temporal_features': features
        }
    
    def _apply_movement(self, features: torch.Tensor, movement_params: torch.Tensor) -> torch.Tensor:
        """Apply predicted movement to features"""
        B, C, H, W = features.shape
        
        # Interpret movement parameters as displacement field
        # Simplified - enhance this with actual movement logic
        moved = features.clone()
        
        # Add movement-based modulation
        movement_field = movement_params[:, :C].view(B, C, 1, 1)
        moved = moved * torch.sigmoid(movement_field)
        
        return moved