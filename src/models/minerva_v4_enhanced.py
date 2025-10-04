"""
MINERVA V4 Enhanced Model - Strategic Ensemble Coordinator for ARC-AGI-2
Enhanced with 2D transformers, test-time adaptation, and OLYMPUS ensemble preparation
Preserves existing weights while adding advanced strategic reasoning capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

# Import existing MINERVA components for weight preservation
from src.models.minerva_model import (
    GridAttention, ObjectEncoder, RelationalReasoning, 
    TransformationPredictor, EnhancedMinervaNet
)


class EnsembleCoordinationModule(nn.Module):
    """Module for coordinating with other ensemble members (OLYMPUS preparation)"""
    def __init__(self, hidden_dim: int = 256, num_specialists: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_specialists = num_specialists
        
        # Specialist embeddings for ensemble coordination
        self.specialist_embeddings = nn.Embedding(num_specialists, hidden_dim)
        
        # Cross-attention for ensemble coordination
        self.ensemble_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        
        # Strategic decision network
        self.strategy_network = nn.Sequential(
            nn.Linear(hidden_dim * num_specialists, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64)  # Strategic parameters
        )
        
        # Confidence estimation for ensemble weighting
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, specialist_states: Optional[List] = None) -> Dict:
        B, C, H, W = features.shape
        
        # Global strategic features
        global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Get specialist embeddings
        specialist_ids = torch.arange(self.num_specialists).to(features.device)
        specialist_emb = self.specialist_embeddings(specialist_ids)  # num_specialists, hidden_dim
        specialist_emb = specialist_emb.unsqueeze(0).expand(B, -1, -1)  # B, num_specialists, hidden_dim
        
        # Cross-attention for coordination
        query = global_features.unsqueeze(1)  # B, 1, hidden_dim
        coordinated_features, attention_weights = self.ensemble_attention(
            query, specialist_emb, specialist_emb
        )
        coordinated_features = coordinated_features.squeeze(1)  # B, hidden_dim
        
        # Strategic decision making
        ensemble_input = torch.cat([
            specialist_emb.view(B, -1),  # Flatten specialist embeddings
        ], dim=1)
        strategic_params = self.strategy_network(ensemble_input)
        
        # Confidence estimation
        confidence = self.confidence_estimator(coordinated_features)
        
        return {
            'coordinated_features': coordinated_features,
            'strategic_params': strategic_params,
            'ensemble_attention': attention_weights,
            'confidence': confidence
        }


class Strategic2DTransformer(nn.Module):
    """2D-aware transformer for strategic pattern analysis"""
    def __init__(self, d_model: int = 256, num_heads: int = 8, max_grid_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_grid_size = max_grid_size
        
        # 2D positional encoding for strategic understanding
        self.pos_encoding_2d = nn.Parameter(
            torch.randn(max_grid_size, max_grid_size, d_model) * 0.02
        )
        
        # Multi-head attention with strategic bias
        self.strategic_attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        
        # Strategic reasoning feedforward
        self.strategic_ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Strategic pattern detection
        self.pattern_detector = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 32),  # Pattern type classification
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        B, C, H, W = x.shape
        
        # Convert to sequence format with 2D positional encoding
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # B, H*W, C
        
        # Add 2D positional encoding
        pos_enc = self.pos_encoding_2d[:H, :W, :C].reshape(H*W, C)
        x_with_pos = x_seq + pos_enc.unsqueeze(0)
        
        # Strategic attention
        attn_output, attn_weights = self.strategic_attention(
            x_with_pos, x_with_pos, x_with_pos, attn_mask=mask
        )
        x_seq = self.norm1(x_seq + attn_output)
        
        # Strategic feedforward
        ff_output = self.strategic_ff(x_seq)
        x_seq = self.norm2(x_seq + ff_output)
        
        # Pattern detection
        pattern_logits = self.pattern_detector(x_seq.mean(dim=1))  # Global pattern
        
        # Convert back to 2D
        x_out = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        strategic_info = {
            'attention_weights': attn_weights,
            'pattern_types': pattern_logits,
            'strategic_features': x_seq
        }
        
        return x_out, strategic_info


class TestTimeAdapter(nn.Module):
    """Test-time adaptation module for strategic learning"""
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Adaptation parameters
        self.adaptation_lr = 0.01
        self.adaptation_steps = 5
        
        # Fast adaptation network
        self.fast_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def adapt_to_task(self, model: nn.Module, examples: List[Tuple], num_steps: int = None):
        """Perform test-time adaptation on task examples"""
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Get adaptable parameters
        adaptable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(adaptable_params, lr=self.adaptation_lr)
        
        print(f"\033[96mMINERVA strategic adaptation: {num_steps} steps\033[0m")
        
        for step in range(num_steps):
            total_loss = 0
            
            for input_grid, target_grid in examples:
                # Forward pass
                output = model(input_grid.unsqueeze(0), target_grid.unsqueeze(0), mode='adaptation')
                
                # Strategic adaptation loss
                pred_output = output['predicted_output']
                loss = F.cross_entropy(pred_output, target_grid.argmax(dim=0))
                
                # Add strategic consistency loss
                if 'strategic_params' in output:
                    strategic_consistency = torch.var(output['strategic_params'], dim=1).mean()
                    loss += strategic_consistency * 0.1
                
                total_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adaptable_params, max_norm=1.0)
            optimizer.step()
        
        print(f"\033[96mMINERVA adaptation complete!\033[0m")


class MinervaV4Enhanced(nn.Module):
    """Enhanced MINERVA V4 with strategic coordination and ensemble preparation"""
    def __init__(self, max_grid_size: int = 30, hidden_dim: int = 256, 
                 preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim
        self.preserve_weights = preserve_weights
        
        # PRESERVE: Original MINERVA components for weight loading
        self.original_minerva = EnhancedMinervaNet(max_grid_size, hidden_dim)
        
        # ENHANCE: New V4 strategic components
        self.ensemble_coordinator = EnsembleCoordinationModule(hidden_dim, num_specialists=5)
        self.strategic_transformer = Strategic2DTransformer(hidden_dim, num_heads=8, max_grid_size=max_grid_size)
        self.test_time_adapter = TestTimeAdapter(hidden_dim)
        
        # Enhanced strategic reasoning
        self.strategic_reasoning = nn.ModuleDict({
            'pattern_analyzer': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 64)  # Strategic pattern encoding
            ),
            'decision_maker': nn.Sequential(
                nn.Linear(hidden_dim + 64, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 32),  # Decision parameters
                nn.Tanh()
            ),
            'confidence_calibrator': nn.Sequential(
                nn.Linear(hidden_dim + 32, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        })
        
        # OLYMPUS preparation: Ensemble integration points
        self.olympus_interface = nn.ModuleDict({
            'feature_broadcaster': nn.Linear(hidden_dim, hidden_dim),
            'consensus_aggregator': nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        })
        
        # Separate parameter for ensemble weights (can't go in ModuleDict)
        self.ensemble_weights = nn.Parameter(torch.ones(5) / 5)  # Equal initial weighting
        
        # Enhanced output decoder that can integrate ensemble features
        self.v4_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 3, hidden_dim, 3, padding=1),  # More input channels
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.ConvTranspose2d(hidden_dim // 2, 10, 1)
        )
        
        # Strategic mixing parameter
        self.strategic_mix = nn.Parameter(torch.tensor(0.3))
        
        self.description = "Enhanced Strategic Coordinator with Ensemble Integration and OLYMPUS Preparation"
    
    def load_compatible_weights(self, checkpoint_path: str):
        """Load weights from existing MINERVA model while preserving architecture"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Try to load into original_minerva first
            model_dict = self.original_minerva.state_dict()
            compatible_params = {}
            
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_params[k] = v
            
            # Always try direct load first (for full model compatibility)
            try:
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    self.load_state_dict(checkpoint, strict=False)
                compatible_params = state_dict
                print(f"\033[96mMINERVA V4: Loaded full model state dict\033[0m")
            except:
                # Fallback to original_minerva loading
                if len(compatible_params) > 0:
                    model_dict.update(compatible_params)
                    self.original_minerva.load_state_dict(model_dict)
            
            print(f"\033[96mMINERVA V4: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[96mMINERVA V4: Could not load weights - {e}\033[0m")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # PRESERVE: Get features from original MINERVA
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_minerva(input_grid, output_grid, mode)
            base_features = original_output['features']
            base_prediction = original_output['predicted_output']
        
        # ENHANCE: Apply strategic 2D transformer
        enhanced_features, strategic_info = self.strategic_transformer(base_features)
        
        # ENHANCE: Ensemble coordination
        coordination_output = self.ensemble_coordinator(enhanced_features)
        coordinated_features = coordination_output['coordinated_features']
        
        # ENHANCE: Strategic reasoning
        global_features = F.adaptive_avg_pool2d(enhanced_features, 1).squeeze(-1).squeeze(-1)
        pattern_encoding = self.strategic_reasoning['pattern_analyzer'](global_features)
        decision_params = self.strategic_reasoning['decision_maker'](
            torch.cat([global_features, pattern_encoding], dim=1)
        )
        confidence = self.strategic_reasoning['confidence_calibrator'](
            torch.cat([global_features, decision_params], dim=1)
        )
        
        # OLYMPUS: Prepare features for ensemble integration
        broadcast_features = self.olympus_interface['feature_broadcaster'](global_features)
        
        # Combine original and enhanced features
        B, C, H, W = enhanced_features.shape
        coordinated_spatial = coordinated_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        combined_features = torch.cat([
            base_features,
            enhanced_features,
            coordinated_spatial
        ], dim=1)
        
        # Enhanced prediction
        enhanced_prediction = self.v4_decoder(combined_features)
        
        # Strategic mixing of predictions
        strategic_weight = torch.sigmoid(self.strategic_mix) * confidence
        
        # Ensure predictions have same spatial dimensions
        if enhanced_prediction.shape != base_prediction.shape:
            # Resize base prediction to match enhanced prediction
            base_prediction_resized = F.interpolate(
                base_prediction, 
                size=(enhanced_prediction.shape[2], enhanced_prediction.shape[3]),
                mode='bilinear', 
                align_corners=False
            )
        else:
            base_prediction_resized = base_prediction
        
        # Expand strategic_weight to match spatial dimensions
        strategic_weight_expanded = strategic_weight.unsqueeze(-1).unsqueeze(-1).expand_as(enhanced_prediction)
        
        final_prediction = (
            strategic_weight_expanded * enhanced_prediction + 
            (1 - strategic_weight_expanded) * base_prediction_resized
        )
        
        # Comprehensive output for ensemble coordination
        result = {
            'predicted_output': final_prediction,
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'strategic_features': enhanced_features,
            'coordinated_features': coordinated_features,
            'pattern_encoding': pattern_encoding,
            'decision_params': decision_params,
            'confidence': confidence,
            'strategic_info': strategic_info,
            'coordination_output': coordination_output,
            'olympus_features': broadcast_features,  # For future ensemble integration
            'ensemble_weights': self.ensemble_weights
        }
        
        # Add original outputs for compatibility
        result.update({
            'transform_params': original_output.get('transform_params'),
            'object_masks': original_output.get('object_masks'),
            'features': enhanced_features  # Override with enhanced features
        })
        
        return result
    
    def get_ensemble_state(self) -> Dict:
        """Get state for OLYMPUS ensemble coordination"""
        return {
            'model_type': 'MINERVA_V4',
            'strategic_weights': self.ensemble_weights.detach(),
            'confidence_threshold': 0.7,
            'specialization': 'strategic_coordination',
            'coordination_ready': True
        }
    
    def test_time_adapt(self, task_examples: List[Tuple], num_steps: int = None):
        """Strategic test-time adaptation"""
        return self.test_time_adapter.adapt_to_task(self, task_examples, num_steps)


# Compatibility alias for easy integration
EnhancedMinervaV4Net = MinervaV4Enhanced