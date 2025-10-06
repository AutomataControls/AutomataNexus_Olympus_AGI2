"""
ATLAS V5 Enhanced Model - Fast Spatial Pattern Recognition for ARC-AGI-2
Optimized version that loads V4 weights but runs much faster while maintaining spatial intelligence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from src.models.atlas_model import EnhancedAtlasNet

class FastSpatialTransformer(nn.Module):
    """Lightweight spatial transformer for speed"""
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, H, W, D = x.shape
        seq_len = H * W
        
        # Reshape for attention
        x_flat = x.view(B, seq_len, D)
        
        # Attention
        q = self.q(x_flat).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x_flat).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x_flat).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, seq_len, D)
        
        # Output projection and residual
        out = self.out(out)
        out = self.norm(out + x_flat)
        
        return out.view(B, H, W, D), {'attention_weights': attn.mean(dim=1)}

class EnhancedSpatialProcessor(nn.Module):
    """Enhanced spatial processing with more intelligence"""
    def __init__(self, d_model: int):
        super().__init__()
        self.spatial_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 96)  # More spatial features
        )
        self.geometry_detector = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 48),  # More geometry patterns
            nn.Softmax(dim=-1)
        )
        self.transform_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 144)  # 12x12 transformation matrix
        )
        self.relationship_analyzer = nn.Sequential(
            nn.Linear(d_model, 48),
            nn.ReLU(),
            nn.Linear(48, 16)
        )
        
    def forward(self, x):
        # x: B, d_model (global features)
        spatial_features = self.spatial_analyzer(x)
        geometry_patterns = self.geometry_detector(x)
        transforms = self.transform_predictor(x)
        relationships = self.relationship_analyzer(x)
        
        return {
            'spatial_features': spatial_features,
            'geometry_patterns': geometry_patterns,
            'transformation_matrix': F.softmax(transforms.view(-1, 12, 12), dim=-1),
            'spatial_relationships': relationships
        }

class AtlasV5Enhanced(nn.Module):
    """ATLAS V5 Enhanced - Fast but intelligent spatial pattern recognition"""
    
    def __init__(self, max_grid_size: int = 30, d_model: int = 128, num_layers: int = 2, preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Core ATLAS model
        self.original_atlas = EnhancedAtlasNet(max_grid_size)
        
        # V5 Enhancements - optimized for speed
        self.input_embedding = nn.Linear(10, d_model)
        
        # Fast spatial transformers with 8 attention heads (128/8=16 head_dim)
        self.spatial_layers = nn.ModuleList([
            FastSpatialTransformer(d_model, num_heads=8) for _ in range(num_layers)
        ])
        
        # Enhanced spatial processing
        self.spatial_processor = EnhancedSpatialProcessor(d_model)
        
        # Enhanced spatial memory (128 patterns for more intelligence)
        self.spatial_memory = nn.Parameter(torch.randn(128, d_model) * 0.02)
        
        # Enhanced pattern rule extractor
        self.rule_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 96)  # More spatial rules for intelligence
        )
        
        # Spatial pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(d_model, 48),
            nn.ReLU(),
            nn.Linear(48, 12),  # More spatial pattern types
            nn.Softmax(dim=-1)
        )
        
        # Enhanced decoder with more capacity
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model + 96, d_model, 3, padding=1),  # Updated for 96 rules
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.ConvTranspose2d(d_model, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(d_model // 2, 48, 3, padding=1),  # More spatial features
            nn.ReLU(),
            nn.ConvTranspose2d(48, 10, 1)
        )
        
        # Mixing parameters - Boost enhanced spatial intelligence
        self.spatial_weight = nn.Parameter(torch.tensor(0.7))  # Trust enhanced features more
        self.spatial_confidence = nn.Parameter(torch.tensor(0.9))  # Higher spatial confidence
        
    def load_compatible_weights(self, checkpoint_path: str):
        """Load V4 weights into core model"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible weights into original_atlas
            model_dict = self.original_atlas.state_dict()
            compatible_params = {}
            
            for name, param in state_dict.items():
                # Strip any prefix to match original atlas names
                clean_name = name.replace('original_atlas.', '')
                if clean_name in model_dict and model_dict[clean_name].shape == param.shape:
                    compatible_params[clean_name] = param
            
            model_dict.update(compatible_params)
            self.original_atlas.load_state_dict(model_dict)
            
            print(f"\033[96mATLAS V5: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            if len(compatible_params) < 10:
                print(f"\033[96mV5 Compatible params: {list(compatible_params.keys())}\033[0m")
            return len(compatible_params) > 0
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # Get original ATLAS prediction
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_atlas(input_grid, output_grid, mode)
            base_prediction = original_output['predicted_output']
        
        # V5 Enhanced processing
        B, C, H, W = input_grid.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Embed input for transformers
        x = input_grid.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Apply spatial transformers
        spatial_analyses = []
        for layer in self.spatial_layers:
            x, analysis = layer(x)
            spatial_analyses.append({'spatial_analysis': analysis})
        
        # Global spatial processing
        global_features = x.mean(dim=[1, 2])  # B, d_model
        spatial_analysis = self.spatial_processor(global_features)
        
        # Enhanced spatial memory matching
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.spatial_memory.unsqueeze(0), 
            dim=2
        )  # B, 64
        top_patterns = memory_similarity.topk(8, dim=1)[0].mean(dim=1, keepdim=True)
        
        # Enhanced rule extraction
        spatial_rules = self.rule_extractor(global_features)
        
        # Pattern classification
        pattern_types = self.pattern_classifier(global_features)
        
        # Enhanced prediction with more features
        enhanced_features = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        rule_spatial = spatial_rules.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_features = torch.cat([enhanced_features, rule_spatial], dim=1)
        
        enhanced_prediction = self.decoder(combined_features)
        
        # Adaptive strategic mixing based on spatial complexity
        spatial_expertise = torch.sigmoid(self.spatial_confidence)
        base_mix_weight = torch.sigmoid(self.spatial_weight)
        
        # Boost enhanced prediction weight for complex spatial patterns
        pattern_complexity = pattern_types.max(dim=1)[0]  # Max confidence in any pattern type
        spatial_complexity = top_patterns.squeeze(-1)     # Memory similarity strength
        complexity_boost = (pattern_complexity + spatial_complexity) / 2
        
        # Final adaptive mixing (higher complexity = more enhanced prediction)
        mix_weight = base_mix_weight * spatial_expertise * (1.0 + complexity_boost * 0.3)
        
        # Ensure same spatial dimensions
        if enhanced_prediction.shape != base_prediction.shape:
            base_prediction = F.interpolate(
                base_prediction, 
                size=(enhanced_prediction.shape[2], enhanced_prediction.shape[3]),
                mode='bilinear', 
                align_corners=False
            )
        
        # Expand mix weight to spatial dimensions
        mix_weight_expanded = mix_weight.unsqueeze(-1).unsqueeze(-1).expand_as(enhanced_prediction)
        
        final_prediction = (
            mix_weight_expanded * enhanced_prediction + 
            (1 - mix_weight_expanded) * base_prediction
        )
        
        # Enhanced comprehensive output
        result = {
            'predicted_output': final_prediction,
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'spatial_features': x,
            'spatial_transform_params': spatial_rules,
            'spatial_memory_similarity': top_patterns,
            'spatial_analyses': spatial_analyses,
            'ensemble_output': {
                'spatial_consensus': spatial_expertise,
                'spatial_expertise': spatial_expertise,
                'pattern_types': pattern_types
            },
            'multispatial_features': [spatial_analysis['spatial_features']],
            'spatial_expertise': spatial_expertise,
            'pattern_memory_similarity': top_patterns,
            'pattern_types': pattern_types,
            'geometry_patterns': spatial_analysis['geometry_patterns'],
            'transformation_matrix': spatial_analysis['transformation_matrix'],
            'spatial_relationships': spatial_analysis['spatial_relationships']
        }
        
        # Add original outputs for compatibility
        result.update({
            'spatial_map': original_output.get('spatial_map'),
            'spatial_attention': original_output.get('spatial_attention'),
            'transformed_output': original_output.get('transformed_output')
        })
        
        return result
    
    def get_ensemble_state(self) -> Dict:
        """Get ensemble state"""
        return {
            'model_type': 'ATLAS_V5',
            'spatial_expertise': self.spatial_confidence.detach(),
            'specialization': 'fast_spatial_pattern_recognition',
            'spatial_capabilities': ['spatial_mapping', 'geometry_detection', 'rule_extraction', 'memory_matching'],
            'coordination_ready': True
        }