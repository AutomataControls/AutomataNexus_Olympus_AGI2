"""
MINERVA Model - Strategic Pattern Analysis with Grid Reasoning
Part of the OLYMPUS AGI2 Ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GridAttention(nn.Module):
    """Grid-aware attention mechanism for ARC tasks"""
    def __init__(self, channels: int, grid_size: int = 30):
        super().__init__()
        self.channels = channels
        self.grid_size = grid_size
        
        # Learnable position embeddings for grid
        self.row_embed = nn.Parameter(torch.randn(grid_size, channels // 2))
        self.col_embed = nn.Parameter(torch.randn(grid_size, channels // 2))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        
        # Projection layers for dimension matching
        self.input_proj = None
        self.output_proj = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Initialize projection layers if needed
        if self.input_proj is None and C != self.channels:
            self.input_proj = nn.Linear(C, self.channels).to(x.device)
            self.output_proj = nn.Linear(self.channels, C).to(x.device)
        
        # Apply attention with position awareness
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # B, H*W, C
        
        # Project to attention dimension if needed
        if C != self.channels:
            x_flat = self.input_proj(x_flat)
        
        # Add positional embeddings
        row_emb = self.row_embed[:H, :].unsqueeze(1).expand(-1, W, -1)  # H, W, channels//2
        col_emb = self.col_embed[:W, :].unsqueeze(0).expand(H, -1, -1)  # H, W, channels//2
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # H, W, channels
        pos_emb = pos_emb.reshape(H*W, self.channels).unsqueeze(0).expand(B, -1, -1)  # B, H*W, channels
        
        x_with_pos = x_flat + pos_emb
        
        # Apply attention
        attended, _ = self.attention(x_with_pos, x_with_pos, x_with_pos)
        
        # Project back if needed
        if C != self.channels:
            attended = self.output_proj(attended)
        
        return attended.permute(0, 2, 1).view(B, C, H, W)


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


class RelationalReasoning(nn.Module):
    """Reason about relationships between grid elements"""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Relation network
        self.relation_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        
        # Use a more efficient approach - compute relations on downsampled features
        # Downsample to reduce memory usage
        if H > 8 or W > 8:
            features_small = F.adaptive_avg_pool2d(features, (8, 8))
            h_small, w_small = 8, 8
        else:
            features_small = features
            h_small, w_small = H, W
        
        # Flatten spatial dimensions
        features_flat = features_small.view(B, C, -1).permute(0, 2, 1)  # B, h*w, C
        N = features_flat.shape[1]
        
        # Instead of all pairs, use local neighborhood relations
        # Create a local attention pattern (e.g., 3x3 neighborhood)
        kernel_size = 3
        padding = kernel_size // 2
        
        # Unfold to get local neighborhoods
        features_unfold = F.unfold(features_small, kernel_size=kernel_size, padding=padding)  # B, C*k*k, h*w
        features_unfold = features_unfold.view(B, C, kernel_size*kernel_size, -1).permute(0, 3, 2, 1)  # B, h*w, k*k, C
        
        # Center features
        center_features = features_flat.unsqueeze(2)  # B, h*w, 1, C
        
        # Compute local relations
        local_relations = []
        for i in range(kernel_size * kernel_size):
            neighbor = features_unfold[:, :, i:i+1, :]  # B, h*w, 1, C
            pair = torch.cat([center_features, neighbor], dim=-1)  # B, h*w, 1, 2*C
            relation = self.relation_net(pair.view(-1, 2*C))  # (B*h*w, C)
            local_relations.append(relation.view(B, N, C))
        
        # Aggregate local relations
        aggregated = torch.stack(local_relations, dim=2).mean(dim=2)  # B, N, C
        aggregated = aggregated.permute(0, 2, 1).view(B, C, h_small, w_small)
        
        # Upsample back to original size if needed
        if H != h_small or W != w_small:
            aggregated = F.interpolate(aggregated, size=(H, W), mode='bilinear', align_corners=False)
        
        return aggregated


class TransformationPredictor(nn.Module):
    """Learn to predict transformations from examples"""
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # Encode input-output pairs
        self.pair_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Predict transformation parameters
        self.transform_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, input_features: torch.Tensor, output_features: torch.Tensor) -> torch.Tensor:
        # Global pooling
        input_global = F.adaptive_avg_pool2d(input_features, 1).squeeze(-1).squeeze(-1)
        output_global = F.adaptive_avg_pool2d(output_features, 1).squeeze(-1).squeeze(-1)
        
        # Encode transformation
        pair_features = torch.cat([input_global, output_global], dim=1)
        encoded = self.pair_encoder(pair_features)
        
        # Predict transformation
        transform_params = self.transform_head(encoded)
        
        return transform_params


class EnhancedMinervaNet(nn.Module):
    """Enhanced MINERVA with grid reasoning capabilities"""
    def __init__(self, max_grid_size: int = 30, hidden_dim: int = 256):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim
        
        # Grid encoder with object awareness
        self.object_encoder = ObjectEncoder(10, hidden_dim)
        
        # Grid attention
        self.grid_attention = GridAttention(hidden_dim, max_grid_size)
        
        # Relational reasoning
        self.relational = RelationalReasoning(hidden_dim)
        
        # Transformation learning
        self.transform_predictor = TransformationPredictor(hidden_dim)
        
        # Memory bank for pattern storage - initialize with more diverse patterns
        self.pattern_memory = nn.Parameter(torch.randn(200, hidden_dim) * 0.5)
        
        # Pattern attention for better matching
        self.pattern_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Transform projection layer
        self.transform_proj = nn.Linear(128, hidden_dim)
        
        # Simple mixing parameter - start at 0.05 to heavily favor transformations
        self.mix_param = nn.Parameter(torch.tensor(0.05))
        
        # Output decoder - predicts actual output grid
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Keep for compatibility with trained models
            nn.ConvTranspose2d(64, 10, 1),  # 10 colors
            # NO SOFTMAX - CrossEntropyLoss expects raw logits
        )
        
        # Initialize final layer with stronger values for more decisive predictions
        # Use xavier with higher gain for bolder outputs
        nn.init.xavier_uniform_(self.decoder[-1].weight, gain=2.0)
        # Initialize bias to encourage all colors equally
        # Give all colors including background equal chance
        bias_values = torch.ones(10) * 0.1  # Larger bias for stronger predictions
        self.decoder[-1].bias.data = bias_values
        
        self.description = "Enhanced Strategic Pattern Analysis with Grid Reasoning"
        
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None, 
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        
        # Extract object features
        input_features, input_objects = self.object_encoder(input_grid)
        
        # Apply grid attention
        attended_features = self.grid_attention(input_features)
        
        # Relational reasoning
        relational_features = self.relational(attended_features)
        
        # Combine features
        combined_features = attended_features + relational_features
        
        if mode == 'train' and output_grid is not None:
            # Learn transformation
            output_features, _ = self.object_encoder(output_grid)
            transform_params = self.transform_predictor(combined_features, output_features)
            
            # Apply transformation to predict output
            transformed = self._apply_transform(combined_features, transform_params)
            predicted_output = self.decoder(torch.cat([combined_features, transformed], dim=1))
            
            # Mix prediction with input using learnable parameter
            # Sigmoid to keep in [0, 1] range - but favor the prediction!
            mix = torch.sigmoid(self.mix_param)
            
            # During training, use even less input mixing to learn transformations
            if self.training:
                mix = mix * 0.2  # Reduce input contribution by 80% during training
                
            predicted_output = predicted_output * (1 - mix) + input_grid * mix  # Inverted to favor prediction
            
            return {
                'predicted_output': predicted_output,
                'transform_params': transform_params,
                'object_masks': input_objects,
                'features': combined_features
            }
        else:
            # Inference mode - find best matching pattern
            best_transform = self._find_best_pattern(combined_features)
            transformed = self._apply_transform(combined_features, best_transform)
            predicted_output = self.decoder(torch.cat([combined_features, transformed], dim=1))
            
            # Mix prediction with input
            mix = torch.sigmoid(self.mix_param)
            
            # During inference, still favor the transformation heavily
            if self.training:
                mix = mix * 0.2
            else:
                mix = mix * 0.3
                
            predicted_output = predicted_output * (1 - mix) + input_grid * mix  # Inverted to favor prediction
            
            return {
                'predicted_output': predicted_output,
                'transform_params': best_transform,
                'object_masks': input_objects
            }
    
    def _apply_transform(self, features: torch.Tensor, transform_params: torch.Tensor) -> torch.Tensor:
        """Apply learned transformation to features"""
        B, C, H, W = features.shape
        
        # Project transform params to match feature channels
        transform_params = self.transform_proj(transform_params)
        
        # Use transform params to modulate features
        transform_matrix = transform_params.view(B, C, 1, 1)
        
        # Apply EVEN STRONGER transformation - use tanh for wider range [-1, 1]
        # Then scale up to allow big changes
        transformed = features * (1.0 + 3.0 * torch.tanh(transform_matrix))
        
        return transformed
    
    def _find_best_pattern(self, features: torch.Tensor) -> torch.Tensor:
        """Find best matching pattern from memory using attention"""
        B = features.shape[0]
        
        # Pool features for comparison
        features_pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # B, hidden_dim
        
        # Use attention to find best patterns
        query = features_pooled.unsqueeze(1)  # B, 1, hidden_dim
        keys = self.pattern_memory.unsqueeze(0).expand(B, -1, -1)  # B, 200, hidden_dim
        
        # Apply pattern attention
        attended_patterns, attention_weights = self.pattern_attention(query, keys, keys)
        attended_patterns = attended_patterns.squeeze(1)  # B, hidden_dim
        
        # Also get top-k patterns for diversity
        similarity = F.cosine_similarity(features_pooled.unsqueeze(1), self.pattern_memory.unsqueeze(0), dim=2)
        top_k_values, top_k_idx = similarity.topk(3, dim=1)
        
        # Weighted combination of top patterns
        weighted_patterns = torch.zeros_like(attended_patterns)
        for i in range(3):
            weight = F.softmax(top_k_values, dim=1)[:, i:i+1]
            pattern_idx = top_k_idx[:, i]
            weighted_patterns += weight * self.pattern_memory[pattern_idx]
        
        # Combine attended and weighted patterns
        combined_patterns = 0.7 * attended_patterns + 0.3 * weighted_patterns
        
        # Return as transform parameters
        transform_params = torch.zeros(B, 128).to(features.device)
        transform_params[:, :min(128, combined_patterns.shape[1])] = combined_patterns[:, :min(128, combined_patterns.shape[1])]
        
        return transform_params