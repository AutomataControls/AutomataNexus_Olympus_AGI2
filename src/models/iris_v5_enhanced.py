"""
IRIS V5 Enhanced Model - Lightweight Color Pattern Recognition for ARC-AGI-2
Fast and efficient version that loads V4 weights but runs much faster
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.models.iris_model import EnhancedIrisNet

class IrisV5Enhanced(nn.Module):
    """IRIS V5 Enhanced - loads V4 weights but runs efficiently"""
    
    def __init__(self, max_grid_size: int = 30, preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        
        # Core IRIS model
        self.original_iris = EnhancedIrisNet(max_grid_size)
        
        # Lightweight color enhancement
        self.color_enhancer = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 10, 3, padding=1)
        )
        
        # Simple mixing parameter
        self.mix_weight = nn.Parameter(torch.tensor(0.3))
        
    def load_compatible_weights(self, checkpoint_path: str):
        """Load V4 weights into core model"""
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
            
            print(f"\033[96mIRIS V5: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return len(compatible_params) > 0
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # Get original IRIS prediction
        original_output = self.original_iris(input_grid, output_grid, mode)
        base_prediction = original_output['predicted_output']
        
        # Simple color enhancement
        enhanced_prediction = self.color_enhancer(input_grid)
        
        # Mix predictions
        mix_weight = torch.sigmoid(self.mix_weight)
        final_prediction = mix_weight * enhanced_prediction + (1 - mix_weight) * base_prediction
        
        # Return comprehensive output
        result = {
            'predicted_output': final_prediction,
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'chromatic_features': torch.zeros(input_grid.shape[0], 64, input_grid.shape[2], input_grid.shape[3]).to(input_grid.device),
            'multichromatic_features': [torch.zeros(input_grid.shape[0], 64).to(input_grid.device)],
            'ensemble_output': {'color_expertise': torch.ones(input_grid.shape[0], 1).to(input_grid.device) * 0.7},
            'chromatic_analyses': [],
            'color_expertise': torch.ones(input_grid.shape[0], 1).to(input_grid.device) * 0.7
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
            'model_type': 'IRIS_V5',
            'color_expertise': self.mix_weight.detach(),
            'specialization': 'fast_color_pattern_recognition',
            'color_capabilities': ['fast_color_mapping', 'efficient_processing'],
            'coordination_ready': True
        }