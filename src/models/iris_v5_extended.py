"""
IRIS V5 Extended Model - Extends IrisV4Enhanced Architecture
Loads V4 weights and adds additional layers/capabilities on top
Preserves V4 architecture completely while extending it
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from src.models.iris_v4_enhanced import IrisV4Enhanced

class IrisV5Extended(nn.Module):
    """
    IRIS V5 - Extends IrisV4Enhanced with additional capabilities
    Preserves V4 architecture and adds V5 extensions on top
    """
    def __init__(self, max_grid_size: int = 30, d_model: int = 256, num_layers: int = 6, preserve_weights: bool = True):
        super().__init__()
        
        # V4 Core - EXACT same architecture
        self.iris_v4_core = IrisV4Enhanced(
            max_grid_size=max_grid_size,
            d_model=d_model, 
            num_layers=num_layers,
            preserve_weights=preserve_weights
        )
        
        # V5 Extensions - Additional layers on top of V4
        self.v5_color_enhancer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.v5_pattern_memory = nn.ModuleDict({
            'memory_bank': nn.Linear(d_model, 512),
            'memory_retrieval': nn.Linear(512, d_model),
            'attention_gate': nn.Linear(d_model * 2, d_model)
        })
        
        self.v5_arc_specialization = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model)
        )
        
    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None, mode: str = 'train') -> Dict:
        """Forward pass - V4 core + V5 extensions"""
        
        # Get V4 core output
        v4_output = self.iris_v4_core(inputs, targets, mode)
        
        # V5 Extensions
        if 'features' in v4_output:
            features = v4_output['features']
            B, C, H, W = features.shape
            
            # Color enhancement
            features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
            enhanced_features = self.v5_color_enhancer(features_flat)
            enhanced_features = enhanced_features.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # Pattern memory (simplified for speed)
            memory_encoded = self.v5_pattern_memory['memory_bank'](features_flat)
            memory_retrieved = self.v5_pattern_memory['memory_retrieval'](memory_encoded)
            memory_features = memory_retrieved.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # ARC specialization
            arc_features = self.v5_arc_specialization(enhanced_features)
            
            # Combine V4 + V5
            final_features = features + enhanced_features * 0.3 + memory_features * 0.2 + arc_features * 0.1
            
            # Update output
            v4_output['features'] = final_features
            v4_output['v5_enhanced'] = True
            v4_output['v5_extensions'] = {
                'color_enhancement': enhanced_features,
                'pattern_memory': memory_features,
                'arc_specialization': arc_features
            }
        
        return v4_output
    
    def load_v4_weights(self, checkpoint_path: str) -> bool:
        """Load V4 weights into the V4 core"""
        return self.iris_v4_core.load_compatible_weights(checkpoint_path)
    
    def get_ensemble_state(self) -> Dict:
        """Get ensemble state from V4 core"""
        return self.iris_v4_core.get_ensemble_state()