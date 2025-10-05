"""
PROMETHEUS V6 Enhanced Model - Fast Creative Pattern Generation for ARC-AGI-2
Optimized version that loads V4 weights but runs much faster while maintaining creativity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from src.models.prometheus_model import EnhancedPrometheusNet

class FastCreativeTransformer(nn.Module):
    """Lightweight creative transformer for speed"""
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

class EnhancedPatternProcessor(nn.Module):
    """Enhanced pattern processing with more intelligence"""
    def __init__(self, d_model: int):
        super().__init__()
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 64)
        )
        self.synthesis_detector = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Softmax(dim=-1)
        )
        self.creative_transform_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 100)
        )
        self.abstraction_analyzer = nn.Sequential(
            nn.Linear(d_model, 48),
            nn.ReLU(),
            nn.Linear(48, 16)
        )
        
    def forward(self, x):
        # x: B, d_model (global features)
        pattern_features = self.pattern_analyzer(x)
        synthesis_patterns = self.synthesis_detector(x)
        creative_transforms = self.creative_transform_predictor(x)
        abstractions = self.abstraction_analyzer(x)
        
        return {
            'pattern_features': pattern_features,
            'synthesis_patterns': synthesis_patterns,
            'creative_transformation_matrix': F.softmax(creative_transforms.view(-1, 10, 10), dim=-1),
            'abstraction_levels': abstractions
        }

class PrometheusV6Enhanced(nn.Module):
    """PROMETHEUS V6 Enhanced - Fast but intelligent creative pattern generation"""
    
    def __init__(self, max_grid_size: int = 30, d_model: int = 128, num_layers: int = 2, preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Core PROMETHEUS model
        self.original_prometheus = EnhancedPrometheusNet(max_grid_size)
        
        # V6 Enhancements - optimized for speed
        self.input_embedding = nn.Linear(10, d_model)
        
        # Fast creative transformers with 8 attention heads (128/8=16 head_dim)
        self.creative_layers = nn.ModuleList([
            FastCreativeTransformer(d_model, num_heads=8) for _ in range(num_layers)
        ])
        
        # Enhanced pattern processing
        self.pattern_processor = EnhancedPatternProcessor(d_model)
        
        # Enhanced creative memory (64 patterns for more intelligence)
        self.creative_memory = nn.Parameter(torch.randn(64, d_model) * 0.02)
        
        # Enhanced pattern rule extractor
        self.rule_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 64)
        )
        
        # Pattern type classifier
        self.pattern_classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
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
        self.creative_weight = nn.Parameter(torch.tensor(0.4))
        self.creative_confidence = nn.Parameter(torch.tensor(0.8))
        
    def load_compatible_weights(self, checkpoint_path: str):
        """Load V4 weights into core model"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible weights into original_prometheus
            model_dict = self.original_prometheus.state_dict()
            compatible_params = {}
            
            for name, param in state_dict.items():
                # Strip any prefix to match original prometheus names
                clean_name = name.replace('original_prometheus.', '')
                if clean_name in model_dict and model_dict[clean_name].shape == param.shape:
                    compatible_params[clean_name] = param
            
            model_dict.update(compatible_params)
            self.original_prometheus.load_state_dict(model_dict)
            
            print(f"\033[96mPROMETHEUS V6: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return len(compatible_params) > 0
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # Get original PROMETHEUS prediction
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_prometheus(input_grid, output_grid, mode)
            base_prediction = original_output['predicted_output']
        
        # V6 Enhanced processing
        B, C, H, W = input_grid.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Embed input for transformers
        x = input_grid.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Apply creative transformers
        creative_analyses = []
        for layer in self.creative_layers:
            x, analysis = layer(x)
            creative_analyses.append({'creative_analysis': analysis})
        
        # Global pattern processing
        global_features = x.mean(dim=[1, 2])  # B, d_model
        pattern_analysis = self.pattern_processor(global_features)
        
        # Enhanced creative memory matching
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.creative_memory.unsqueeze(0), 
            dim=2
        )  # B, 64
        top_patterns = memory_similarity.topk(8, dim=1)[0].mean(dim=1, keepdim=True)
        
        # Enhanced rule extraction
        creative_rules = self.rule_extractor(global_features)
        
        # Pattern classification
        pattern_types = self.pattern_classifier(global_features)
        
        # Enhanced prediction with more features
        enhanced_features = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        rule_spatial = creative_rules.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_features = torch.cat([enhanced_features, rule_spatial], dim=1)
        
        enhanced_prediction = self.decoder(combined_features)
        
        # Strategic mixing
        creative_expertise = torch.sigmoid(self.creative_confidence)
        mix_weight = torch.sigmoid(self.creative_weight) * creative_expertise
        
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
            'creative_features': x,
            'creative_transform_params': creative_rules,
            'creative_memory_similarity': top_patterns,
            'creative_analyses': creative_analyses,
            'ensemble_output': {
                'creative_consensus': creative_expertise,
                'creative_expertise': creative_expertise,
                'pattern_types': pattern_types
            },
            'creative_patterns': [pattern_analysis['pattern_features']],
            'creative_expertise': creative_expertise,
            'pattern_memory_similarity': top_patterns,
            'pattern_types': pattern_types,
            'synthesis_patterns': pattern_analysis['synthesis_patterns'],
            'creative_transformation_matrix': pattern_analysis['creative_transformation_matrix'],
            'abstraction_levels': pattern_analysis['abstraction_levels']
        }
        
        # Add original outputs for compatibility
        result.update({
            'pattern_map': original_output.get('pattern_map'),
            'creative_attention': original_output.get('creative_attention'),
            'synthesized_output': original_output.get('synthesized_output')
        })
        
        return result
    
    def get_ensemble_state(self) -> Dict:
        """Get ensemble state"""
        return {
            'model_type': 'PROMETHEUS_V6',
            'creative_expertise': self.creative_confidence.detach(),
            'specialization': 'fast_creative_pattern_generation',
            'creative_capabilities': ['pattern_synthesis', 'abstraction_detection', 'rule_extraction', 'memory_matching'],
            'coordination_ready': True
        }