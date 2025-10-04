"""
ATLAS V4 Enhanced Model - Advanced 2D Spatial Reasoning Expert for ARC-AGI-2
Enhanced with 2D transformers, geometric analysis, and OLYMPUS ensemble preparation
Preserves existing weights while adding advanced spatial intelligence capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

# Import existing ATLAS components for weight preservation
from src.models.atlas_model import EnhancedAtlasNet


class Geometric2DPositionalEncoding(nn.Module):
    """Advanced 2D positional encoding with geometric awareness"""
    def __init__(self, d_model: int, max_grid_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        
        # Create enhanced 2D positional encoding with geometric properties
        pe = torch.zeros(max_grid_size, max_grid_size, d_model)
        
        for h in range(max_grid_size):
            for w in range(max_grid_size):
                # Standard sinusoidal encoding
                for i in range(0, d_model//4, 2):
                    # Row encoding
                    pe[h, w, i] = math.sin(h / (10000 ** (i / d_model)))
                    pe[h, w, i + 1] = math.cos(h / (10000 ** (i / d_model)))
                    # Column encoding  
                    pe[h, w, i + d_model//2] = math.sin(w / (10000 ** (i / d_model)))
                    pe[h, w, i + d_model//2 + 1] = math.cos(w / (10000 ** (i / d_model)))
                
                # Geometric features
                if d_model > 4:
                    center_h, center_w = max_grid_size // 2, max_grid_size // 2
                    
                    # Distance from center
                    dist_center = math.sqrt((h - center_h)**2 + (w - center_w)**2)
                    pe[h, w, -4] = dist_center / max_grid_size
                    
                    # Distance from corners
                    corners = [(0, 0), (0, max_grid_size-1), (max_grid_size-1, 0), (max_grid_size-1, max_grid_size-1)]
                    min_corner_dist = min(math.sqrt((h - ch)**2 + (w - cw)**2) for ch, cw in corners)
                    pe[h, w, -3] = min_corner_dist / max_grid_size
                    
                    # Diagonal position
                    pe[h, w, -2] = (h + w) / (2 * max_grid_size)
                    
                    # Anti-diagonal position
                    pe[h, w, -1] = abs(h - w) / max_grid_size
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, d_model = x.shape
        return x + self.pe[:H, :W, :d_model].unsqueeze(0)


class SpatialRelationshipAttention(nn.Module):
    """Multi-head attention with spatial relationship awareness"""
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
        
        # Spatial relationship bias
        self.distance_bias = nn.Parameter(torch.zeros(60, num_heads))  # Max distance in 30x30
        self.direction_bias = nn.Parameter(torch.zeros(16, num_heads))  # 16 directional sectors
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, d_model = x.shape
        seq_len = H * W
        
        # Reshape to sequence format
        x_seq = x.view(B, seq_len, d_model)
        
        # Compute Q, K, V
        q = self.w_q(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Spatial attention with relationship bias
        attention = self._spatial_relationship_attention(q, k, v, H, W, mask)
        
        # Concatenate heads and apply output projection
        attention = attention.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        output = self.w_o(attention)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x_seq)
        
        # Get attention weights for visualization
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            attention_weights = F.softmax(scores, dim=-1).mean(dim=1)  # Average across heads
        
        return output.view(B, H, W, d_model), attention_weights
    
    def _spatial_relationship_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                      H: int, W: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, num_heads, seq_len, d_k = q.shape
        
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Add spatial relationship bias
        spatial_bias = self._compute_spatial_bias(H, W, seq_len, num_heads)
        scores = scores + spatial_bias.unsqueeze(0)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _compute_spatial_bias(self, H: int, W: int, seq_len: int, num_heads: int) -> torch.Tensor:
        """Compute spatial bias based on geometric relationships"""
        bias = torch.zeros(num_heads, seq_len, seq_len, device=self.distance_bias.device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                h1, w1 = i // W, i % W
                h2, w2 = j // W, j % W
                
                # Distance bias
                distance = int(math.sqrt((h1 - h2)**2 + (w1 - w2)**2))
                distance = min(distance, 59)  # Clamp to max
                
                # Direction bias
                if h2 != h1 or w2 != w1:
                    angle = math.atan2(h2 - h1, w2 - w1) + math.pi
                    direction = int(angle / (2 * math.pi) * 16) % 16
                else:
                    direction = 0
                
                # Combine biases
                dist_bias = self.distance_bias[distance]
                dir_bias = self.direction_bias[direction]
                
                bias[:, i, j] = dist_bias + dir_bias
        
        return bias


class GeometricTransformationAnalyzer(nn.Module):
    """Analyze and predict geometric transformations"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Transformation type detector
        self.transform_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 32)  # Various transformation types
        )
        
        # Specific transformation predictors
        self.rotation_predictor = nn.Linear(d_model, 4)  # 0°, 90°, 180°, 270°
        self.reflection_predictor = nn.Linear(d_model, 8)  # 8 axes
        self.scaling_predictor = nn.Linear(d_model, 5)  # Scale factors
        self.translation_predictor = nn.Linear(d_model, 2)  # x, y offset
        
        # Geometric invariant detector
        self.invariant_detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 16)  # Common geometric invariants
        )
        
        # Transformation confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model + 32 + 4 + 8 + 5 + 2 + 16, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = features.shape
        
        # Global geometric features
        global_features = features.mean(dim=[1, 2])  # B, d_model
        
        # Predict transformations
        transform_types = self.transform_detector(global_features)
        rotation_logits = self.rotation_predictor(global_features)
        reflection_logits = self.reflection_predictor(global_features)
        scaling_params = torch.tanh(self.scaling_predictor(global_features))
        translation_params = torch.tanh(self.translation_predictor(global_features))
        
        # Detect geometric invariants
        invariants = torch.sigmoid(self.invariant_detector(global_features))
        
        # Estimate confidence
        all_features = torch.cat([
            global_features, transform_types, rotation_logits, reflection_logits,
            scaling_params, translation_params, invariants
        ], dim=1)
        confidence = self.confidence_estimator(all_features)
        
        return {
            'transform_types': F.softmax(transform_types, dim=-1),
            'rotation_logits': rotation_logits,
            'reflection_logits': reflection_logits,
            'scaling_params': scaling_params,
            'translation_params': translation_params,
            'geometric_invariants': invariants,
            'transformation_confidence': confidence
        }


class Spatial2DTransformerBlock(nn.Module):
    """Enhanced transformer block for 2D spatial reasoning"""
    def __init__(self, d_model: int, num_heads: int = 8, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model
        
        self.spatial_attention = SpatialRelationshipAttention(d_model, num_heads, dropout)
        self.geometric_analyzer = GeometricTransformationAnalyzer(d_model)
        
        # Enhanced feedforward with geometric awareness
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        # Spatial attention
        attn_out, attention_weights = self.spatial_attention(x, mask)
        
        # Geometric analysis
        geometric_analysis = self.geometric_analyzer(attn_out)
        
        # Feedforward with residual
        B, H, W, d_model = attn_out.shape
        ff_input = attn_out.view(B * H * W, d_model)
        ff_out = self.ff(ff_input)
        ff_out = self.layer_norm(ff_out + ff_input)
        
        output = ff_out.view(B, H, W, d_model)
        
        analysis_info = {
            'attention_weights': attention_weights,
            'geometric_analysis': geometric_analysis
        }
        
        return output, analysis_info


class SpatialEnsembleInterface(nn.Module):
    """Interface for spatial reasoning coordination with other specialists"""
    def __init__(self, d_model: int, num_specialists: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_specialists = num_specialists
        
        # Spatial feature broadcaster for ensemble
        self.spatial_broadcaster = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Cross-specialist attention
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Spatial consensus mechanism
        self.consensus_network = nn.Sequential(
            nn.Linear(d_model * num_specialists, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spatial_features: torch.Tensor, 
                specialist_states: Optional[List] = None) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = spatial_features.shape
        
        # Global spatial features
        global_spatial = spatial_features.mean(dim=[1, 2])  # B, d_model
        
        # Broadcast spatial insights
        broadcast_features = self.spatial_broadcaster(global_spatial)
        
        # If we have other specialist states, perform cross-attention
        if specialist_states is not None and len(specialist_states) > 0:
            # Stack specialist states
            specialist_tensor = torch.stack(specialist_states, dim=1)  # B, num_specialists, d_model
            
            # Cross-attention with spatial features
            query = broadcast_features.unsqueeze(1)  # B, 1, d_model
            attended_features, cross_attention_weights = self.cross_attention(
                query, specialist_tensor, specialist_tensor
            )
            attended_features = attended_features.squeeze(1)  # B, d_model
            
            # Consensus calculation
            consensus_input = torch.cat([
                attended_features,
                specialist_tensor.view(B, -1)  # Flatten specialist states
            ], dim=1)
            consensus_score = self.consensus_network(consensus_input)
        else:
            attended_features = broadcast_features
            cross_attention_weights = None
            consensus_score = torch.ones(B, 1).to(spatial_features.device) * 0.8  # Default confidence
        
        return {
            'broadcast_features': broadcast_features,
            'attended_features': attended_features,
            'cross_attention_weights': cross_attention_weights,
            'spatial_consensus': consensus_score
        }


class AtlasV4Enhanced(nn.Module):
    """Enhanced ATLAS V4 with 2D spatial reasoning transformers and ensemble coordination"""
    def __init__(self, max_grid_size: int = 30, d_model: int = 256, num_layers: int = 6,
                 preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.preserve_weights = preserve_weights
        
        # PRESERVE: Original ATLAS components for weight loading
        self.original_atlas = EnhancedAtlasNet(max_grid_size)
        
        # ENHANCE: Input embedding for transformers
        self.input_embedding = nn.Linear(10, d_model)
        
        # ENHANCE: Geometric positional encoding
        self.pos_encoding = Geometric2DPositionalEncoding(d_model, max_grid_size)
        
        # ENHANCE: 2D spatial transformer layers
        self.spatial_transformer_layers = nn.ModuleList([
            Spatial2DTransformerBlock(d_model, num_heads=8) for _ in range(num_layers)
        ])
        
        # ENHANCE: Ensemble coordination interface
        self.ensemble_interface = SpatialEnsembleInterface(d_model, num_specialists=5)
        
        # ENHANCE: Advanced geometric reasoning
        self.pattern_matcher = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 64)  # Pattern encoding
        )
        
        self.transformation_composer = nn.Sequential(
            nn.Linear(d_model + 64, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 128)  # Composite transformation parameters
        )
        
        # Separate parameter for spatial memory (can't go in ModuleDict)
        self.spatial_memory = nn.Parameter(torch.randn(100, d_model) * 0.02)
        
        # ENHANCE: Multi-scale spatial processing
        self.multiscale_processor = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model//4)
            for k in [1, 3, 5, 7]
        ])
        self.multiscale_fusion = nn.Linear(d_model * 4, d_model)
        
        # ENHANCE: Advanced output decoder
        self.v4_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model + 128, d_model, 3, padding=1),  # Include original ATLAS features
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.ConvTranspose2d(d_model, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Dropout2d(0.02),
            nn.ConvTranspose2d(d_model // 2, 10, 1)
        )
        
        # Strategic mixing parameters
        self.spatial_mix = nn.Parameter(torch.tensor(0.4))  # Weight for enhanced features
        self.geometric_confidence = nn.Parameter(torch.tensor(0.7))
        
        # Test-time adaptation
        self.adaptation_lr = 0.008
        self.adaptation_steps = 6
        
        self.description = "Enhanced 2D Spatial Reasoning Expert with Geometric Transformers and OLYMPUS Preparation"
    
    def load_compatible_weights(self, checkpoint_path: str):
        """Load weights from existing ATLAS model while preserving architecture"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible weights into original_atlas
            model_dict = self.original_atlas.state_dict()
            compatible_params = {}
            
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_params[k] = v
            
            model_dict.update(compatible_params)
            self.original_atlas.load_state_dict(model_dict)
            
            print(f"\033[96mATLAS V4: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[96mATLAS V4: Could not load weights - {e}\033[0m")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # PRESERVE: Get features and predictions from original ATLAS
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_atlas(input_grid, output_grid, mode)
            base_prediction = original_output['predicted_output']
        
        # ENHANCE: Prepare input for transformers
        B, C, H, W = input_grid.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Reshape for transformer: B, C, H, W -> B, H, W, C
        x = input_grid.permute(0, 2, 3, 1)
        
        # Embed input tokens
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Add geometric positional encoding
        x = self.pos_encoding(x)
        
        # Apply spatial transformer layers
        spatial_analyses = []
        for layer in self.spatial_transformer_layers:
            x, analysis_info = layer(x)
            spatial_analyses.append(analysis_info)
        
        # Multi-scale spatial processing
        x_conv = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        multiscale_features = []
        for conv in self.multiscale_processor:
            scale_features = conv(x_conv)
            multiscale_features.append(scale_features)
        
        multiscale_concat = torch.cat(multiscale_features, dim=1)  # B, 4*d_model, H, W
        multiscale_concat = multiscale_concat.permute(0, 2, 3, 1)  # B, H, W, 4*d_model
        
        fused_features = self.multiscale_fusion(multiscale_concat)  # B, H, W, d_model
        
        # Combine with transformer features
        enhanced_features = x + fused_features * 0.3
        
        # Advanced geometric reasoning
        global_features = enhanced_features.mean(dim=[1, 2])  # B, d_model
        pattern_encoding = self.pattern_matcher(global_features)
        
        # Spatial memory matching
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.spatial_memory.unsqueeze(0), 
            dim=2
        )  # B, 100
        top_patterns = memory_similarity.topk(5, dim=1)[0].mean(dim=1, keepdim=True)  # B, 1
        
        # Compose transformations
        composition_input = torch.cat([global_features, pattern_encoding], dim=1)
        transformation_params = self.transformation_composer(composition_input)
        
        # Ensemble coordination
        ensemble_output = self.ensemble_interface(enhanced_features)
        
        # Combine enhanced features with transformation parameters
        enhanced_spatial = enhanced_features.permute(0, 3, 1, 2)  # B, d_model, H, W
        transform_spatial = transformation_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_features = torch.cat([enhanced_spatial, transform_spatial], dim=1)
        
        # Enhanced prediction
        enhanced_prediction = self.v4_decoder(combined_features)
        
        # Strategic mixing of original and enhanced predictions
        spatial_confidence = torch.sigmoid(self.geometric_confidence) * ensemble_output['spatial_consensus']
        spatial_weight = torch.sigmoid(self.spatial_mix) * spatial_confidence
        
        # Ensure predictions have matching dimensions
        if enhanced_prediction.shape != base_prediction.shape:
            # Resize base prediction to match enhanced prediction
            base_prediction = F.interpolate(
                base_prediction, 
                size=(enhanced_prediction.shape[2], enhanced_prediction.shape[3]),
                mode='bilinear', 
                align_corners=False
            )
        
        # Expand spatial_weight to match prediction dimensions
        spatial_weight_expanded = spatial_weight.unsqueeze(-1).unsqueeze(-1).expand_as(enhanced_prediction)
        
        final_prediction = (
            spatial_weight_expanded * enhanced_prediction + 
            (1 - spatial_weight_expanded) * base_prediction
        )
        
        # Comprehensive output for ensemble coordination
        result = {
            'predicted_output': final_prediction,
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'spatial_features': enhanced_features,
            'transformation_params': transformation_params,
            'pattern_encoding': pattern_encoding,
            'spatial_memory_similarity': top_patterns,
            'spatial_analyses': spatial_analyses,
            'ensemble_output': ensemble_output,
            'multiscale_features': multiscale_features,
            'spatial_confidence': spatial_confidence
        }
        
        # Add original outputs for compatibility
        result.update({
            'theta': original_output.get('theta'),
            'rotation_logits': original_output.get('rotation_logits'),
            'reflection_logits': original_output.get('reflection_logits'),
            'transformed_input': original_output.get('transformed_input')
        })
        
        return result
    
    def get_ensemble_state(self) -> Dict:
        """Get state for OLYMPUS ensemble coordination"""
        return {
            'model_type': 'ATLAS_V4',
            'spatial_confidence': self.geometric_confidence.detach(),
            'specialization': 'spatial_geometric_reasoning',
            'transformation_capabilities': ['rotation', 'reflection', 'affine', 'geometric'],
            'coordination_ready': True
        }
    
    def test_time_adapt(self, task_examples: List[Tuple], num_steps: int = None):
        """Spatial test-time adaptation"""
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Get adaptable spatial parameters
        spatial_params = []
        for layer in self.spatial_transformer_layers:
            spatial_params.extend(list(layer.parameters()))
        spatial_params.extend(list(self.v4_decoder.parameters()))
        
        optimizer = torch.optim.AdamW(spatial_params, lr=self.adaptation_lr, weight_decay=1e-6)
        
        print(f"\033[96mATLAS V4 spatial adaptation: {num_steps} steps\033[0m")
        
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
                if 'spatial_confidence' in output:
                    spatial_consistency = (1.0 - output['spatial_confidence']).mean()
                    spatial_loss += spatial_consistency * 0.1
                
                total_loss += main_loss + spatial_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(spatial_params, max_norm=1.0)
            optimizer.step()
            
            if step % 2 == 0:
                print(f"\033[96m  Spatial Step {step}: Loss = {total_loss.item():.4f}\033[0m")
        
        print(f"\033[96mATLAS V4 spatial adaptation complete!\033[0m")


# Compatibility alias
EnhancedAtlasV4Net = AtlasV4Enhanced