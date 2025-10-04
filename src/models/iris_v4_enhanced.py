"""
IRIS V4 Enhanced Model - Advanced Color Pattern Recognition Expert for ARC-AGI-2
Enhanced with chromatic transformers, color space reasoning, and OLYMPUS ensemble preparation
Preserves existing weights while adding sophisticated color intelligence capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

# Import existing IRIS components for weight preservation
from src.models.iris_model import EnhancedIrisNet


class ChromaticPositionalEncoding(nn.Module):
    """Advanced positional encoding with chromatic awareness"""
    def __init__(self, d_model: int, max_grid_size: int = 30, num_colors: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        
        # Standard 2D positional encoding
        self.spatial_pe = nn.Parameter(torch.zeros(max_grid_size, max_grid_size, d_model // 2))
        
        # Chromatic positional encoding
        self.chromatic_pe = nn.Parameter(torch.zeros(num_colors, d_model // 2))
        
        # Initialize with sinusoidal patterns
        self._init_positional_encodings()
        
    def _init_positional_encodings(self):
        """Initialize with sinusoidal positional encodings"""
        # Spatial encoding
        for h in range(self.max_grid_size):
            for w in range(self.max_grid_size):
                for i in range(0, self.d_model // 4, 2):
                    # Row encoding
                    self.spatial_pe.data[h, w, i] = math.sin(h / (10000 ** (i / (self.d_model // 2))))
                    self.spatial_pe.data[h, w, i + 1] = math.cos(h / (10000 ** (i / (self.d_model // 2))))
                    # Column encoding
                    self.spatial_pe.data[h, w, i + self.d_model // 4] = math.sin(w / (10000 ** (i / (self.d_model // 2))))
                    self.spatial_pe.data[h, w, i + self.d_model // 4 + 1] = math.cos(w / (10000 ** (i / (self.d_model // 2))))
        
        # Chromatic encoding (color frequency patterns)
        for c in range(self.num_colors):
            for i in range(0, self.d_model // 4, 2):
                self.chromatic_pe.data[c, i] = math.sin(c / (10000 ** (i / (self.d_model // 2))))
                self.chromatic_pe.data[c, i + 1] = math.cos(c / (10000 ** (i / (self.d_model // 2))))
    
    def forward(self, x: torch.Tensor, color_indices: torch.Tensor) -> torch.Tensor:
        """Apply spatial and chromatic positional encoding"""
        B, H, W, d_model = x.shape
        
        # Get spatial encoding
        spatial_enc = self.spatial_pe[:H, :W, :].unsqueeze(0).expand(B, -1, -1, -1)
        
        # Get chromatic encoding based on color indices
        chromatic_enc = torch.zeros(B, H, W, self.d_model // 2).to(x.device)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    color_idx = color_indices[b, h, w].long()
                    chromatic_enc[b, h, w, :] = self.chromatic_pe[color_idx, :]
        
        # Combine spatial and chromatic encodings
        combined_pe = torch.cat([spatial_enc, chromatic_enc], dim=-1)
        
        return x + combined_pe


class ColorRelationshipAttention(nn.Module):
    """Multi-head attention specialized for color relationships"""
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
        
        # Color relationship bias (10x10 color pair relationships)
        self.color_bias = nn.Parameter(torch.zeros(10, 10, num_heads))
        
        # Color harmony patterns
        self.harmony_patterns = nn.Parameter(torch.randn(16, num_heads) * 0.02)  # 16 harmony types
        
    def forward(self, x: torch.Tensor, color_indices: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, d_model = x.shape
        seq_len = H * W
        
        # Reshape to sequence format
        x_seq = x.view(B, seq_len, d_model)
        color_seq = color_indices.view(B, seq_len)
        
        # Compute Q, K, V
        q = self.w_q(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Color-aware attention with relationship bias
        attention = self._color_relationship_attention(q, k, v, color_seq, mask)
        
        # Concatenate heads and apply output projection
        attention = attention.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        output = self.w_o(attention)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x_seq)
        
        # Get attention weights for analysis
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            attention_weights = F.softmax(scores, dim=-1).mean(dim=1)  # Average across heads
        
        return output.view(B, H, W, d_model), attention_weights
    
    def _color_relationship_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                    color_indices: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, num_heads, seq_len, d_k = q.shape
        
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Add color relationship bias
        color_bias = self._compute_color_bias(color_indices, num_heads, seq_len)
        scores = scores + color_bias.unsqueeze(0)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _compute_color_bias(self, color_indices: torch.Tensor, num_heads: int, seq_len: int) -> torch.Tensor:
        """Compute color relationship bias matrix"""
        B = color_indices.shape[0]
        bias = torch.zeros(B, num_heads, seq_len, seq_len, device=color_indices.device)
        
        for b in range(B):
            for i in range(seq_len):
                for j in range(seq_len):
                    color_i = color_indices[b, i].long()
                    color_j = color_indices[b, j].long()
                    
                    # Color pair bias
                    if color_i < 10 and color_j < 10:
                        bias[b, :, i, j] = self.color_bias[color_i, color_j, :]
        
        return bias


class ColorSpaceTransformer(nn.Module):
    """Transformer for advanced color space reasoning"""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Color space analyzers
        self.hue_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 16)  # 16 hue categories
        )
        
        self.saturation_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 8)   # 8 saturation levels
        )
        
        self.brightness_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 8)   # 8 brightness levels
        )
        
        # Color harmony detector
        self.harmony_detector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 32),  # 32 harmony patterns
            nn.Softmax(dim=-1)
        )
        
        # Color transformation predictor
        self.color_transform_predictor = nn.Sequential(
            nn.Linear(d_model + 16 + 8 + 8 + 32, d_model),
            nn.GELU(),
            nn.Linear(d_model, 10 * 10)  # 10x10 color mapping
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = x.shape
        
        # Global color space features
        global_features = x.mean(dim=[1, 2])  # B, d_model
        
        # Analyze color space components
        hue_features = self.hue_analyzer(global_features)
        saturation_features = self.saturation_analyzer(global_features)
        brightness_features = self.brightness_analyzer(global_features)
        
        # Detect color harmony
        harmony_patterns = self.harmony_detector(global_features)
        
        # Predict color transformations
        combined_features = torch.cat([
            global_features, hue_features, saturation_features, brightness_features, harmony_patterns
        ], dim=1)
        color_transform = self.color_transform_predictor(combined_features)
        color_transform = color_transform.view(B, 10, 10)
        
        return {
            'hue_analysis': F.softmax(hue_features, dim=-1),
            'saturation_analysis': F.softmax(saturation_features, dim=-1),
            'brightness_analysis': F.softmax(brightness_features, dim=-1),
            'harmony_patterns': harmony_patterns,
            'color_transformation_matrix': F.softmax(color_transform, dim=-1)
        }


class ChromaticTransformerBlock(nn.Module):
    """Enhanced transformer block for chromatic reasoning"""
    def __init__(self, d_model: int, num_heads: int = 8, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model
        
        self.color_attention = ColorRelationshipAttention(d_model, num_heads, dropout)
        self.color_space_transformer = ColorSpaceTransformer(d_model, num_heads, dropout)
        
        # Enhanced feedforward with color awareness
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, color_indices: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        
        # Color relationship attention
        attn_out, attention_weights = self.color_attention(x, color_indices, mask)
        
        # Color space analysis
        color_analysis = self.color_space_transformer(attn_out)
        
        # Feedforward with residual
        B, H, W, d_model = attn_out.shape
        ff_input = attn_out.view(B * H * W, d_model)
        ff_out = self.ff(ff_input)
        ff_out = self.layer_norm(ff_out + ff_input)
        
        output = ff_out.view(B, H, W, d_model)
        
        chromatic_info = {
            'attention_weights': attention_weights,
            'color_analysis': color_analysis
        }
        
        return output, chromatic_info


class ColorEnsembleInterface(nn.Module):
    """Interface for color reasoning coordination with other specialists"""
    def __init__(self, d_model: int, num_specialists: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_specialists = num_specialists
        
        # Color feature broadcaster for ensemble
        self.color_broadcaster = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Cross-specialist color attention
        self.cross_color_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Color consensus mechanism
        self.color_consensus = nn.Sequential(
            nn.Linear(d_model * num_specialists, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Color expertise confidence
        self.color_expertise = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, color_features: torch.Tensor, 
                specialist_states: Optional[List] = None) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = color_features.shape
        
        # Global color features
        global_color = color_features.mean(dim=[1, 2])  # B, d_model
        
        # Broadcast color insights
        broadcast_features = self.color_broadcaster(global_color)
        
        # Color expertise confidence
        color_confidence = self.color_expertise(global_color)
        
        # Cross-attention with other specialists if available
        if specialist_states is not None and len(specialist_states) > 0:
            # Stack specialist states
            specialist_tensor = torch.stack(specialist_states, dim=1)  # B, num_specialists, d_model
            
            # Cross-attention
            query = broadcast_features.unsqueeze(1)  # B, 1, d_model
            attended_features, cross_attention_weights = self.cross_color_attention(
                query, specialist_tensor, specialist_tensor
            )
            attended_features = attended_features.squeeze(1)  # B, d_model
            
            # Color consensus
            consensus_input = torch.cat([
                attended_features,
                specialist_tensor.view(B, -1)
            ], dim=1)
            consensus_score = self.color_consensus(consensus_input)
        else:
            attended_features = broadcast_features
            cross_attention_weights = None
            consensus_score = color_confidence  # Use color expertise as consensus
        
        return {
            'broadcast_features': broadcast_features,
            'attended_features': attended_features,
            'cross_attention_weights': cross_attention_weights,
            'color_consensus': consensus_score,
            'color_expertise': color_confidence
        }


class IrisV4Enhanced(nn.Module):
    """Enhanced IRIS V4 with chromatic transformers and color intelligence"""
    def __init__(self, max_grid_size: int = 30, d_model: int = 256, num_layers: int = 6,
                 preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.preserve_weights = preserve_weights
        
        # PRESERVE: Original IRIS components for weight loading
        self.original_iris = EnhancedIrisNet(max_grid_size)
        
        # ENHANCE: Input embedding for transformers
        self.input_embedding = nn.Linear(10, d_model)
        
        # ENHANCE: Chromatic positional encoding
        self.pos_encoding = ChromaticPositionalEncoding(d_model, max_grid_size, num_colors=10)
        
        # ENHANCE: Chromatic transformer layers
        self.chromatic_transformer_layers = nn.ModuleList([
            ChromaticTransformerBlock(d_model, num_heads=8) for _ in range(num_layers)
        ])
        
        # ENHANCE: Ensemble coordination interface
        self.ensemble_interface = ColorEnsembleInterface(d_model, num_specialists=5)
        
        # ENHANCE: Advanced color reasoning
        self.advanced_color_reasoning = nn.ModuleDict({
            'color_pattern_memory': nn.Parameter(torch.randn(128, d_model) * 0.02),  # Color pattern memory
            'color_rule_extractor': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 64)  # Color rule encoding
            ),
            'color_transformation_composer': nn.Sequential(
                nn.Linear(d_model + 64, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 100)  # Color transformation parameters
            )
        })
        
        # ENHANCE: Multi-chromatic processing
        self.multichromatic_processor = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)  # Different color space processors
        ])
        self.chromatic_fusion = nn.Linear(d_model * 4, d_model)
        
        # ENHANCE: Advanced output decoder
        self.v4_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model + 64, d_model, 3, padding=1),  # Include original IRIS features
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Dropout2d(0.03),
            nn.ConvTranspose2d(d_model, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Dropout2d(0.01),
            nn.ConvTranspose2d(d_model // 2, 10, 1)
        )
        
        # Strategic mixing parameters
        self.chromatic_mix = nn.Parameter(torch.tensor(0.35))  # Weight for enhanced features
        self.color_confidence = nn.Parameter(torch.tensor(0.75))
        
        # Test-time adaptation
        self.adaptation_lr = 0.006
        self.adaptation_steps = 5
        
        self.description = "Enhanced Color Pattern Recognition Expert with Chromatic Transformers and OLYMPUS Preparation"
    
    def load_compatible_weights(self, checkpoint_path: str):
        """Load weights from existing IRIS model while preserving architecture"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible weights into original_iris
            model_dict = self.original_iris.state_dict()
            compatible_params = {}
            
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_params[k] = v
            
            model_dict.update(compatible_params)
            self.original_iris.load_state_dict(model_dict)
            
            print(f"\033[96mIRIS V4: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[96mIRIS V4: Could not load weights - {e}\033[0m")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # PRESERVE: Get features and predictions from original IRIS
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_iris(input_grid, output_grid, mode)
            base_prediction = original_output['predicted_output']
            original_color_map = original_output.get('color_map')
        
        # ENHANCE: Prepare input for transformers
        B, C, H, W = input_grid.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Get color indices for chromatic encoding
        color_indices = input_grid.argmax(dim=1)  # B, H, W
        
        # Reshape for transformer: B, C, H, W -> B, H, W, C
        x = input_grid.permute(0, 2, 3, 1)
        
        # Embed input tokens
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Add chromatic positional encoding
        x = self.pos_encoding(x, color_indices)
        
        # Apply chromatic transformer layers
        chromatic_analyses = []
        for layer in self.chromatic_transformer_layers:
            x, chromatic_info = layer(x, color_indices)
            chromatic_analyses.append(chromatic_info)
        
        # Multi-chromatic processing (different color space interpretations)
        x_flat = x.mean(dim=[1, 2])  # B, d_model
        multichromatic_features = []
        for processor in self.multichromatic_processor:
            chromatic_features = processor(x_flat)
            multichromatic_features.append(chromatic_features)
        
        multichromatic_concat = torch.cat(multichromatic_features, dim=1)  # B, 4*d_model
        fused_chromatic = self.chromatic_fusion(multichromatic_concat)  # B, d_model
        
        # Advanced color reasoning
        global_features = x.mean(dim=[1, 2])  # B, d_model
        color_rule_encoding = self.advanced_color_reasoning['color_rule_extractor'](global_features)
        
        # Color memory matching
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.advanced_color_reasoning['color_pattern_memory'].unsqueeze(0), 
            dim=2
        )  # B, 128
        top_color_patterns = memory_similarity.topk(8, dim=1)[0].mean(dim=1, keepdim=True)  # B, 1
        
        # Compose color transformations
        composition_input = torch.cat([global_features, color_rule_encoding], dim=1)
        color_transform_params = self.advanced_color_reasoning['color_transformation_composer'](composition_input)
        
        # Ensemble coordination
        ensemble_output = self.ensemble_interface(x)
        
        # Combine enhanced features with transformation parameters
        enhanced_chromatic = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        transform_spatial = color_transform_params[:, :64].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_features = torch.cat([enhanced_chromatic, transform_spatial], dim=1)
        
        # Enhanced prediction
        enhanced_prediction = self.v4_decoder(combined_features)
        
        # Strategic mixing of original and enhanced predictions
        color_expertise = ensemble_output['color_expertise']
        chromatic_weight = torch.sigmoid(self.chromatic_mix) * color_expertise
        
        final_prediction = (
            chromatic_weight * enhanced_prediction + 
            (1 - chromatic_weight) * base_prediction
        )
        
        # Comprehensive output for ensemble coordination
        result = {
            'predicted_output': final_prediction,
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'chromatic_features': x,
            'color_transform_params': color_transform_params,
            'color_rule_encoding': color_rule_encoding,
            'color_memory_similarity': top_color_patterns,
            'chromatic_analyses': chromatic_analyses,
            'ensemble_output': ensemble_output,
            'multichromatic_features': multichromatic_features,
            'color_expertise': color_expertise
        }
        
        # Add original outputs for compatibility
        result.update({
            'color_map': original_color_map,
            'color_attention': original_output.get('color_attention'),
            'mapped_output': original_output.get('mapped_output')
        })
        
        return result
    
    def get_ensemble_state(self) -> Dict:
        """Get state for OLYMPUS ensemble coordination"""
        return {
            'model_type': 'IRIS_V4',
            'color_expertise': self.color_confidence.detach(),
            'specialization': 'chromatic_pattern_recognition',
            'color_capabilities': ['color_mapping', 'chromatic_harmony', 'color_transformation'],
            'coordination_ready': True
        }
    
    def test_time_adapt(self, task_examples: List[Tuple], num_steps: int = None):
        """Chromatic test-time adaptation"""
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Get adaptable chromatic parameters
        chromatic_params = []
        for layer in self.chromatic_transformer_layers:
            chromatic_params.extend(list(layer.parameters()))
        chromatic_params.extend(list(self.v4_decoder.parameters()))
        
        optimizer = torch.optim.AdamW(chromatic_params, lr=self.adaptation_lr, weight_decay=1e-6)
        
        print(f"\033[96mIRIS V4 chromatic adaptation: {num_steps} steps\033[0m")
        
        for step in range(num_steps):
            total_loss = 0
            color_loss = 0
            
            for input_grid, target_grid in task_examples:
                # Forward pass
                output = self(input_grid.unsqueeze(0), target_grid.unsqueeze(0), mode='adaptation')
                
                # Main prediction loss
                pred_output = output['predicted_output']
                main_loss = F.cross_entropy(pred_output, target_grid.argmax(dim=0))
                
                # Color expertise loss
                if 'color_expertise' in output:
                    color_consistency = (1.0 - output['color_expertise']).mean()
                    color_loss += color_consistency * 0.1
                
                total_loss += main_loss + color_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(chromatic_params, max_norm=1.0)
            optimizer.step()
            
            if step % 2 == 0:
                print(f"\033[96m  Chromatic Step {step}: Loss = {total_loss.item():.4f}\033[0m")
        
        print(f"\033[96mIRIS V4 chromatic adaptation complete!\033[0m")


# Compatibility alias
EnhancedIrisV4Net = IrisV4Enhanced