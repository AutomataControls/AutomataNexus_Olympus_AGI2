"""
ATLAS V4 Model - Advanced 2D Spatial Reasoning Transformer for ARC-AGI-2
Specialized in geometric transformations and spatial pattern understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List


class GeometricPositionalEncoding(nn.Module):
    """Enhanced 2D positional encoding with geometric awareness"""
    def __init__(self, d_model: int, max_grid_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        
        # Standard 2D positional encoding
        pe = torch.zeros(max_grid_size, max_grid_size, d_model)
        
        for h in range(max_grid_size):
            for w in range(max_grid_size):
                # Basic position encoding
                for i in range(0, d_model//2, 2):
                    pe[h, w, i] = math.sin(h / (10000 ** (i / d_model)))
                    pe[h, w, i + 1] = math.cos(h / (10000 ** (i / d_model)))
                    pe[h, w, i + d_model//2] = math.sin(w / (10000 ** (i / d_model)))
                    pe[h, w, i + d_model//2 + 1] = math.cos(w / (10000 ** (i / d_model)))
                
                # Add geometric features: distance from center, corners, etc.
                center_h, center_w = max_grid_size // 2, max_grid_size // 2
                dist_from_center = math.sqrt((h - center_h)**2 + (w - center_w)**2)
                
                # Encode geometric properties in remaining dimensions
                if d_model > 4:
                    pe[h, w, -4] = dist_from_center / max_grid_size  # Normalized distance
                    pe[h, w, -3] = h / max_grid_size  # Normalized row
                    pe[h, w, -2] = w / max_grid_size  # Normalized column
                    pe[h, w, -1] = (h + w) / (2 * max_grid_size)  # Diagonal position
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, d_model = x.shape
        return x + self.pe[:H, :W, :d_model].unsqueeze(0)


class SpatialRelationAttention(nn.Module):
    """Attention mechanism focused on spatial relationships"""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Spatial relationship encodings
        self.distance_embedding = nn.Embedding(60, num_heads)  # Max distance in 30x30 grid
        self.direction_embedding = nn.Embedding(8, num_heads)  # 8 cardinal directions
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, W, d_model = x.shape
        seq_len = H * W
        
        # Reshape to sequence format
        x_seq = x.view(B, seq_len, d_model)
        
        # Compute Q, K, V
        q = self.w_q(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Spatial attention with geometric bias
        attention = self._geometric_attention(q, k, v, H, W, mask)
        
        # Concatenate heads and apply output projection
        attention = attention.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        output = self.w_o(attention)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x_seq)
        
        return output.view(B, H, W, d_model)
    
    def _geometric_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           H: int, W: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, num_heads, seq_len, d_k = q.shape
        
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Add geometric bias
        geometric_bias = self._compute_geometric_bias(H, W, seq_len, num_heads)
        scores = scores + geometric_bias.unsqueeze(0)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _compute_geometric_bias(self, H: int, W: int, seq_len: int, num_heads: int) -> torch.Tensor:
        """Compute geometric bias based on spatial relationships"""
        bias = torch.zeros(num_heads, seq_len, seq_len)
        
        for i in range(seq_len):
            for j in range(seq_len):
                h1, w1 = i // W, i % W
                h2, w2 = j // W, j % W
                
                # Distance bias
                distance = int(math.sqrt((h1 - h2)**2 + (w1 - w2)**2))
                distance = min(distance, 59)  # Clamp to embedding size
                
                # Direction bias (8-way)
                if h2 != h1 or w2 != w1:
                    angle = math.atan2(h2 - h1, w2 - w1)
                    direction = int((angle + math.pi) / (2 * math.pi / 8)) % 8
                else:
                    direction = 0
                
                # Combine biases
                dist_bias = self.distance_embedding(torch.tensor(distance))
                dir_bias = self.direction_embedding(torch.tensor(direction))
                
                bias[:, i, j] = (dist_bias + dir_bias).squeeze()
        
        return bias.to(self.distance_embedding.weight.device)


class GeometricTransformationModule(nn.Module):
    """Neural module for learning geometric transformations"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Transformation detection heads
        self.rotation_head = nn.Linear(d_model, 4)  # 0°, 90°, 180°, 270°
        self.reflection_head = nn.Linear(d_model, 8)  # 8 reflection axes
        self.translation_head = nn.Linear(d_model, 2)  # x, y translation
        self.scaling_head = nn.Linear(d_model, 3)  # shrink, same, expand
        
        # Transformation application network
        self.transform_net = nn.Sequential(
            nn.Linear(d_model + 17, d_model * 2),  # +17 for all transformation params
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = features.shape
        
        # Global feature pooling for transformation detection
        global_features = features.mean(dim=[1, 2])
        
        # Predict transformations
        rotation_logits = self.rotation_head(global_features)
        reflection_logits = self.reflection_head(global_features)
        translation = torch.tanh(self.translation_head(global_features))
        scaling_logits = self.scaling_head(global_features)
        
        # Combine all transformation parameters
        transform_params = torch.cat([
            F.softmax(rotation_logits, dim=-1),
            F.softmax(reflection_logits, dim=-1),
            translation,
            F.softmax(scaling_logits, dim=-1)
        ], dim=-1)
        
        # Apply transformations to features
        features_flat = features.view(B, H * W, d_model)
        transform_input = torch.cat([
            features_flat,
            transform_params.unsqueeze(1).expand(-1, H * W, -1)
        ], dim=-1)
        
        transformed_features = self.transform_net(transform_input)
        transformed_features = transformed_features.view(B, H, W, d_model)
        
        # Residual connection
        output_features = features + transformed_features * 0.3
        
        return {
            'transformed_features': output_features,
            'rotation_logits': rotation_logits,
            'reflection_logits': reflection_logits,
            'translation': translation,
            'scaling_logits': scaling_logits,
            'transform_params': transform_params
        }


class SpatialReasoningBlock(nn.Module):
    """Enhanced transformer block with spatial reasoning"""
    def __init__(self, d_model: int, num_heads: int = 8, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model
        
        self.spatial_attention = SpatialRelationAttention(d_model, num_heads, dropout)
        self.geometric_transform = GeometricTransformationModule(d_model)
        
        # Enhanced feed-forward with geometric awareness
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        # Spatial attention with residual
        attn_out = self.spatial_attention(x, mask)
        
        # Geometric transformation
        transform_result = self.geometric_transform(attn_out)
        geometric_features = transform_result['transformed_features']
        
        # Feed-forward with residual
        B, H, W, d_model = geometric_features.shape
        ff_input = geometric_features.view(B * H * W, d_model)
        ff_out = self.ff(ff_input)
        ff_out = self.layer_norm2(ff_out + ff_input)
        
        output = ff_out.view(B, H, W, d_model)
        
        # Return both output and transformation info
        transform_info = {
            k: v for k, v in transform_result.items() 
            if k != 'transformed_features'
        }
        
        return output, transform_info


class AtlasV4Transformer(nn.Module):
    """ATLAS V4 with Advanced Spatial Reasoning Transformers"""
    def __init__(self, max_grid_size: int = 30, d_model: int = 320, num_layers: int = 8, 
                 num_heads: int = 8, enable_test_adaptation: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.enable_test_adaptation = enable_test_adaptation
        
        # Input embedding
        self.input_embedding = nn.Linear(10, d_model)
        
        # Enhanced geometric positional encoding
        self.pos_encoding = GeometricPositionalEncoding(d_model, max_grid_size)
        
        # Spatial reasoning transformer blocks
        self.spatial_blocks = nn.ModuleList([
            SpatialReasoningBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        # Multi-scale spatial processing
        self.multiscale_conv = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=k, padding=k//2) 
            for k in [1, 3, 5, 7]
        ])
        self.multiscale_fusion = nn.Linear(d_model * 4, d_model)
        
        # Output projection with geometric awareness
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 10)
        )
        
        # Test-time adaptation parameters
        self.adaptation_lr = 0.005
        self.adaptation_steps = 8
        
        self.description = "Advanced 2D Spatial Reasoning Transformer with Geometric Awareness"
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        B, C, H, W = input_grid.shape
        
        # Convert to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Reshape for transformer: B, C, H, W -> B, H, W, C
        x = input_grid.permute(0, 2, 3, 1)
        
        # Embed input
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Add geometric positional encoding
        x = self.pos_encoding(x)
        
        # Apply spatial reasoning blocks
        transformation_info = []
        for spatial_block in self.spatial_blocks:
            x, transform_info = spatial_block(x)
            transformation_info.append(transform_info)
        
        # Multi-scale processing
        x_conv = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        multiscale_features = []
        for conv in self.multiscale_conv:
            scale_features = F.relu(conv(x_conv))
            multiscale_features.append(scale_features)
        
        # Fuse multi-scale features
        multiscale_concat = torch.cat(multiscale_features, dim=1)  # B, 4*d_model, H, W
        multiscale_concat = multiscale_concat.permute(0, 2, 3, 1)  # B, H, W, 4*d_model
        
        fused_features = self.multiscale_fusion(multiscale_concat)  # B, H, W, d_model
        
        # Combine with transformer features
        combined_features = x + fused_features * 0.3
        
        # Output projection
        output = self.output_projection(combined_features)  # B, H, W, 10
        predicted_output = output.permute(0, 3, 1, 2)  # B, 10, H, W
        
        # Aggregate transformation info
        agg_transform_info = self._aggregate_transformations(transformation_info)
        
        result = {
            'predicted_output': predicted_output,
            'spatial_features': combined_features,
            'transformation_analysis': agg_transform_info,
            'multiscale_features': multiscale_features
        }
        
        return result
    
    def _aggregate_transformations(self, transformation_info: List[Dict]) -> Dict:
        """Aggregate transformation information across layers"""
        if not transformation_info:
            return {}
        
        # Average transformation predictions across layers
        agg_info = {}
        for key in transformation_info[0].keys():
            if torch.is_tensor(transformation_info[0][key]):
                stacked = torch.stack([info[key] for info in transformation_info])
                agg_info[f'avg_{key}'] = stacked.mean(dim=0)
                agg_info[f'std_{key}'] = stacked.std(dim=0)
        
        return agg_info
    
    def test_time_adapt(self, task_examples: List[Tuple], num_steps: int = None) -> None:
        """Enhanced test-time adaptation for spatial reasoning tasks"""
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Create optimizer focused on spatial reasoning components
        spatial_params = []
        for spatial_block in self.spatial_blocks:
            spatial_params.extend(list(spatial_block.parameters()))
        spatial_params.extend(list(self.output_projection.parameters()))
        
        optimizer = torch.optim.AdamW(spatial_params, lr=self.adaptation_lr, weight_decay=1e-5)
        
        print(f"\033[96mATLAS V4 Test-time adaptation starting: {num_steps} steps\033[0m")
        
        for step in range(num_steps):
            total_loss = 0
            spatial_loss = 0
            
            for input_grid, target_grid in task_examples:
                # Forward pass
                output = self(input_grid.unsqueeze(0), target_grid.unsqueeze(0), mode='adaptation')
                
                # Main prediction loss
                pred_output = output['predicted_output']
                main_loss = F.cross_entropy(pred_output, target_grid.argmax(dim=0))
                
                # Spatial consistency loss
                if 'transformation_analysis' in output and output['transformation_analysis']:
                    transform_info = output['transformation_analysis']
                    # Encourage consistent transformations
                    consistency_loss = 0
                    for key, value in transform_info.items():
                        if 'std_' in key and torch.is_tensor(value):
                            consistency_loss += value.mean()  # Minimize variation
                    spatial_loss += consistency_loss * 0.1
                
                total_loss += main_loss + spatial_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(spatial_params, max_norm=1.0)
            optimizer.step()
            
            if step % 2 == 0:
                print(f"\033[96m  Step {step}: Loss = {total_loss.item():.4f}\033[0m")
        
        print(f"\033[96mATLAS V4 adaptation completed!\033[0m")


# Compatibility alias
EnhancedAtlasV4Net = AtlasV4Transformer