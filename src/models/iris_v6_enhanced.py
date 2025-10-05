"""
IRIS V6 Enhanced Model - Intelligent Color Pattern Recognition for ARC-AGI-2
Brings back color intelligence while maintaining speed optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from src.models.iris_model import EnhancedIrisNet

class FastChromaticTransformer(nn.Module):
    """Lightweight chromatic transformer for speed"""
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
        
    def forward(self, x, color_indices):
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

class EnhancedColorProcessor(nn.Module):
    """Enhanced color space processing with more intelligence"""
    def __init__(self, d_model: int):
        super().__init__()
        self.color_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 64)  # More color features
        )
        self.harmony_detector = nn.Sequential(
            nn.Linear(d_model, 32),  # More harmony patterns
            nn.Softmax(dim=-1)
        )
        self.color_transform_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 100)  # Color transformation matrix
        )
        # Color relationship analyzer
        self.relationship_analyzer = nn.Sequential(
            nn.Linear(d_model, 48),
            nn.ReLU(),
            nn.Linear(48, 16)  # Color relationships
        )
        
    def forward(self, x):
        # x: B, d_model (global features)
        color_features = self.color_analyzer(x)
        harmony_patterns = self.harmony_detector(x)
        color_transforms = self.color_transform_predictor(x)
        relationships = self.relationship_analyzer(x)
        
        return {
            'color_features': color_features,
            'harmony_patterns': harmony_patterns,
            'color_transformation_matrix': F.softmax(color_transforms.view(-1, 10, 10), dim=-1),
            'color_relationships': relationships
        }

class IrisV6Enhanced(nn.Module):
    """IRIS V6 Enhanced - Intelligent but fast color pattern recognition"""
    
    def __init__(self, max_grid_size: int = 30, d_model: int = 128, num_layers: int = 3, preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Core IRIS model
        self.original_iris = EnhancedIrisNet(max_grid_size)
        
        # V6 Enhancements - optimized for speed
        self.input_embedding = nn.Linear(10, d_model)
        
        # Fast chromatic transformers with 8 attention heads (128/8=16 head_dim)
        self.chromatic_layers = nn.ModuleList([
            FastChromaticTransformer(d_model, num_heads=8) for _ in range(num_layers)
        ])
        
        # Enhanced color processing
        self.color_processor = EnhancedColorProcessor(d_model)
        
        # Enhanced color memory (64 patterns for more intelligence)
        self.color_memory = nn.Parameter(torch.randn(64, d_model) * 0.02)
        
        # Enhanced color rule extractor
        self.rule_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 64)  # More rule encoding
        )
        
        # Color pattern classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 8),  # 8 pattern types
            nn.Softmax(dim=-1)
        )
        
        # Enhanced decoder with more capacity
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model + 64, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.ConvTranspose2d(d_model, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(d_model // 2, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 10, 1)
        )
        
        # Mixing parameters
        self.chromatic_weight = nn.Parameter(torch.tensor(0.4))
        self.color_confidence = nn.Parameter(torch.tensor(0.8))
        
    def load_compatible_weights(self, checkpoint_path: str):
        """Load V4/V5 weights into core model"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible weights into original_iris
            model_dict = self.original_iris.state_dict()
            compatible_params = {}
            
            for name, param in state_dict.items():
                # Strip any prefix to match original iris names
                clean_name = name.replace('original_iris.', '')
                if clean_name in model_dict and model_dict[clean_name].shape == param.shape:
                    compatible_params[clean_name] = param
            
            model_dict.update(compatible_params)
            self.original_iris.load_state_dict(model_dict)
            
            print(f"\033[96mIRIS V6: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return len(compatible_params) > 0
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # Get original IRIS prediction
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_iris(input_grid, output_grid, mode)
            base_prediction = original_output['predicted_output']
        
        # V6 Enhanced processing
        B, C, H, W = input_grid.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Get color indices for analysis
        color_indices = input_grid.argmax(dim=1)  # B, H, W
        
        # Embed input for transformers
        x = input_grid.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Apply chromatic transformers
        chromatic_analyses = []
        for layer in self.chromatic_layers:
            x, analysis = layer(x, color_indices)
            chromatic_analyses.append({'color_analysis': analysis})
        
        # Global color processing
        global_features = x.mean(dim=[1, 2])  # B, d_model
        color_analysis = self.color_processor(global_features)
        
        # Enhanced color memory matching
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.color_memory.unsqueeze(0), 
            dim=2
        )  # B, 64
        top_patterns = memory_similarity.topk(8, dim=1)[0].mean(dim=1, keepdim=True)
        
        # Enhanced rule extraction
        color_rules = self.rule_extractor(global_features)
        
        # Pattern classification
        pattern_types = self.pattern_classifier(global_features)
        
        # Enhanced prediction with more features
        enhanced_features = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        rule_spatial = color_rules.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_features = torch.cat([enhanced_features, rule_spatial], dim=1)
        
        enhanced_prediction = self.decoder(combined_features)
        
        # Strategic mixing
        color_expertise = torch.sigmoid(self.color_confidence)
        mix_weight = torch.sigmoid(self.chromatic_weight) * color_expertise
        
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
            'chromatic_features': x,
            'color_transform_params': color_rules,
            'color_memory_similarity': top_patterns,
            'chromatic_analyses': chromatic_analyses,
            'ensemble_output': {
                'color_consensus': color_expertise,
                'color_expertise': color_expertise,
                'pattern_types': pattern_types
            },
            'multichromatic_features': [color_analysis['color_features']],
            'color_expertise': color_expertise,
            'creative_memory_similarity': top_patterns,
            'pattern_types': pattern_types,
            'color_harmony_patterns': color_analysis['harmony_patterns'],
            'color_transformation_matrix': color_analysis['color_transformation_matrix'],
            'color_relationships': color_analysis['color_relationships']
        }
        
        # Add original outputs for compatibility
        result.update({
            'color_map': original_output.get('color_map'),
            'color_attention': original_output.get('color_attention'),
            'mapped_output': original_output.get('mapped_output')
        })
        
        return result
    
    def get_ensemble_state(self) -> Dict:
        """Get ensemble state"""
        return {
            'model_type': 'IRIS_V6',
            'color_expertise': self.color_confidence.detach(),
            'specialization': 'intelligent_color_pattern_recognition',
            'color_capabilities': ['color_mapping', 'harmony_detection', 'rule_extraction', 'memory_matching'],
            'coordination_ready': True
        }