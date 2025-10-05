"""
CHRONOS V7 Enhanced Model - Ultra-Fast Advanced Temporal Intelligence for ARC-AGI-2
Optimized version that loads V5 weights but runs at lightning speed for stages 11-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from src.models.chronos_v4_enhanced import ChronosV4Enhanced

class FastTemporalTransformer(nn.Module):
    """Ultra-lightweight temporal transformer for maximum speed"""
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

class AdvancedTemporalProcessor(nn.Module):
    """Advanced temporal processing for stages 11-18 intelligence"""
    def __init__(self, d_model: int):
        super().__init__()
        self.temporal_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 128)  # More temporal features
        )
        self.sequence_detector = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Softmax(dim=-1)
        )
        self.movement_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 120)  # Advanced movement patterns
        )
        # Advanced continuity analyzer
        self.continuity_analyzer = nn.Sequential(
            nn.Linear(d_model, 80),
            nn.ReLU(),
            nn.Linear(80, 32)
        )
        # Multi-temporal reasoning engine
        self.multitemporal_engine = nn.Sequential(
            nn.Linear(d_model, 96),
            nn.ReLU(),
            nn.Linear(96, 40),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: B, d_model (global features)
        temporal_features = self.temporal_analyzer(x)
        sequence_patterns = self.sequence_detector(x)
        movement_matrix = self.movement_predictor(x)
        continuity_analysis = self.continuity_analyzer(x)
        multitemporal_patterns = self.multitemporal_engine(x)
        
        return {
            'temporal_features': temporal_features,
            'sequence_patterns': sequence_patterns,
            'movement_matrix': F.softmax(movement_matrix.view(-1, 12, 10), dim=-1),
            'continuity_analysis': continuity_analysis,
            'multitemporal_patterns': multitemporal_patterns
        }

class ChronosV7Enhanced(nn.Module):
    """CHRONOS V7 Enhanced - Ultra-fast advanced temporal intelligence for stages 11-18"""
    
    def __init__(self, max_grid_size: int = 30, d_model: int = 128, num_layers: int = 2, preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Core CHRONOS model
        self.original_chronos = ChronosV4Enhanced(max_grid_size)
        
        # V7 Ultra-fast enhancements
        self.input_embedding = nn.Linear(10, d_model)
        
        # Fast temporal transformers with 8 attention heads (128/8=16 head_dim)
        self.temporal_layers = nn.ModuleList([
            FastTemporalTransformer(d_model, num_heads=8) for _ in range(num_layers)
        ])
        
        # Advanced temporal processing for stages 11-18
        self.temporal_processor = AdvancedTemporalProcessor(d_model)
        
        # Advanced temporal memory (128 patterns for stage 11-18 intelligence)
        self.temporal_memory = nn.Parameter(torch.randn(128, d_model) * 0.02)
        
        # Advanced sequence rule extractor
        self.sequence_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 96)  # More sequence rules
        )
        
        # Temporal genius classifier
        self.temporal_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Softmax(dim=-1)
        )
        
        # Ultra-fast decoder with enhanced capacity
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model + 96, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.ConvTranspose2d(d_model, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(d_model // 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 10, 1)
        )
        
        # Mixing parameters
        self.temporal_weight = nn.Parameter(torch.tensor(0.4))
        self.temporal_confidence = nn.Parameter(torch.tensor(0.8))
        
    def load_compatible_weights(self, checkpoint_path: str):
        """Load V5 weights into core model"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible weights into original_chronos
            model_dict = self.original_chronos.state_dict()
            compatible_params = {}
            
            for name, param in state_dict.items():
                # Strip any prefix to match original chronos names
                clean_name = name.replace('original_chronos.', '')
                if clean_name in model_dict and model_dict[clean_name].shape == param.shape:
                    compatible_params[clean_name] = param
            
            model_dict.update(compatible_params)
            self.original_chronos.load_state_dict(model_dict)
            
            print(f"\033[96mCHRONOS V7: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            if len(compatible_params) < 10:
                print(f"\033[96mV7 Compatible params: {list(compatible_params.keys())}\033[0m")
            return len(compatible_params) > 0
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # Get original CHRONOS prediction
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_chronos(input_grid, output_grid, mode)
            base_prediction = original_output['predicted_output']
        
        # V7 Ultra-fast processing
        B, C, H, W = input_grid.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Embed input for transformers
        x = input_grid.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Apply temporal transformers
        temporal_analyses = []
        for layer in self.temporal_layers:
            x, analysis = layer(x)
            temporal_analyses.append({'temporal_analysis': analysis})
        
        # Advanced temporal processing
        global_features = x.mean(dim=[1, 2])  # B, d_model
        temporal_analysis = self.temporal_processor(global_features)
        
        # Advanced temporal memory matching
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.temporal_memory.unsqueeze(0), 
            dim=2
        )  # B, 128
        top_patterns = memory_similarity.topk(12, dim=1)[0].mean(dim=1, keepdim=True)
        
        # Advanced sequence extraction
        temporal_rules = self.sequence_extractor(global_features)
        
        # Temporal genius classification
        temporal_types = self.temporal_classifier(global_features)
        
        # Enhanced prediction with more features
        enhanced_features = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        rule_spatial = temporal_rules.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_features = torch.cat([enhanced_features, rule_spatial], dim=1)
        
        enhanced_prediction = self.decoder(combined_features)
        
        # Strategic mixing
        temporal_expertise = torch.sigmoid(self.temporal_confidence)
        mix_weight = torch.sigmoid(self.temporal_weight) * temporal_expertise
        
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
            'temporal_features': x,
            'temporal_transform_params': temporal_rules,
            'temporal_memory_similarity': top_patterns,
            'temporal_analyses': temporal_analyses,
            'ensemble_output': {
                'temporal_consensus': temporal_expertise,
                'temporal_expertise': temporal_expertise,
                'temporal_types': temporal_types
            },
            'temporal_patterns': [temporal_analysis['temporal_features']],
            'temporal_expertise': temporal_expertise,
            'pattern_memory_similarity': top_patterns,
            'temporal_types': temporal_types,
            'sequence_patterns': temporal_analysis['sequence_patterns'],
            'movement_matrix': temporal_analysis['movement_matrix'],
            'continuity_analysis': temporal_analysis['continuity_analysis'],
            'multitemporal_patterns': temporal_analysis['multitemporal_patterns']
        }
        
        # Add original outputs for compatibility
        result.update({
            'temporal_map': original_output.get('temporal_map'),
            'sequence_attention': original_output.get('sequence_attention'),
            'predicted_sequence': original_output.get('predicted_sequence')
        })
        
        return result
    
    def get_ensemble_state(self) -> Dict:
        """Get ensemble state"""
        return {
            'model_type': 'CHRONOS_V7',
            'temporal_expertise': self.temporal_confidence.detach(),
            'specialization': 'ultra_fast_advanced_temporal_intelligence',
            'temporal_capabilities': ['sequence_analysis', 'movement_prediction', 'continuity_detection', 'multitemporal_reasoning'],
            'coordination_ready': True
        }