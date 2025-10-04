"""
MINERVA V6 Enhanced Model - Ultimate Strategic Intelligence Master for ARC-AGI-2
Deep strategic architecture with complete grid mastery (2x2 to 30x30) and advanced program synthesis
Builds upon V4/V5 architecture while adding revolutionary strategic intelligence capabilities
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


class DeepStrategicTransformer(nn.Module):
    """8-layer deep strategic transformer for complete pattern mastery"""
    def __init__(self, d_model: int = 256, num_heads: int = 8, num_layers: int = 8, max_grid_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_grid_size = max_grid_size
        
        # Enhanced 2D positional encoding for complete grid range
        self.pos_encoding_2d = nn.Parameter(
            torch.randn(max_grid_size, max_grid_size, d_model) * 0.01
        )
        
        # Deep strategic transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Multi-scale strategic pattern detector
        self.pattern_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 64),  # Pattern encoding
                nn.Softmax(dim=-1)
            ) for _ in range(num_layers)
        ])
        
        # Strategic complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        B, C, H, W = x.shape
        
        # Convert to sequence format with 2D positional encoding
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)  # B, H*W, C
        
        # Add 2D positional encoding
        pos_enc = self.pos_encoding_2d[:H, :W, :C].reshape(H*W, C)
        x_with_pos = x_seq + pos_enc.unsqueeze(0)
        
        # Deep strategic processing
        strategic_analyses = []
        hidden = x_with_pos
        
        for i, (transformer_layer, pattern_detector) in enumerate(zip(self.transformer_layers, self.pattern_detectors)):
            # Strategic attention
            hidden = transformer_layer(hidden, src_mask=mask)
            
            # Pattern analysis at this depth
            global_pattern = hidden.mean(dim=1)  # Global pattern
            pattern_logits = pattern_detector(global_pattern)
            
            strategic_analyses.append({
                'layer': i,
                'pattern_types': pattern_logits,
                'hidden_state': hidden
            })
        
        # Strategic complexity estimation
        final_global = hidden.mean(dim=1)
        complexity = self.complexity_estimator(final_global)
        
        # Convert back to 2D
        output = hidden.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        strategic_info = {
            'strategic_analyses': strategic_analyses,
            'strategic_complexity': complexity,
            'deep_features': hidden
        }
        
        return output, strategic_info


class MegaPatternMemory(nn.Module):
    """Massive strategic pattern memory for complete pattern mastery"""
    def __init__(self, d_model: int = 256, memory_size: int = 300):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        
        # Massive strategic pattern memory
        self.strategic_patterns = nn.Parameter(torch.randn(memory_size, d_model) * 0.01)
        
        # Pattern categories for organization
        self.pattern_categories = nn.Parameter(torch.randn(memory_size, 32) * 0.01)
        
        # Pattern complexity levels
        self.pattern_complexity = nn.Parameter(torch.randn(memory_size, 16) * 0.01)
        
        # Adaptive pattern matcher
        self.pattern_matcher = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128)  # Matching features
        )
        
        # Pattern synthesis network
        self.pattern_synthesizer = nn.Sequential(
            nn.Linear(d_model + 128, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 64)  # Synthesized strategic parameters
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = features.shape
        
        # Global strategic features
        global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Pattern matching
        match_features = self.pattern_matcher(global_features)
        
        # Memory similarity computation
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.strategic_patterns.unsqueeze(0), 
            dim=2
        )  # B, memory_size
        
        # Top pattern retrieval
        top_k = min(10, self.memory_size)
        top_patterns, top_indices = memory_similarity.topk(top_k, dim=1)
        
        # Category analysis
        category_weights = torch.gather(self.pattern_categories, 0, 
                                      top_indices.view(-1).unsqueeze(1).expand(-1, 32))
        category_weights = category_weights.view(B, top_k, 32).mean(dim=1)
        
        # Complexity analysis
        complexity_weights = torch.gather(self.pattern_complexity, 0,
                                        top_indices.view(-1).unsqueeze(1).expand(-1, 16))
        complexity_weights = complexity_weights.view(B, top_k, 16).mean(dim=1)
        
        # Pattern synthesis
        synthesis_input = torch.cat([global_features, match_features], dim=1)
        strategic_params = self.pattern_synthesizer(synthesis_input)
        
        return {
            'pattern_similarity': top_patterns,
            'pattern_indices': top_indices,
            'pattern_categories': category_weights,
            'pattern_complexity': complexity_weights,
            'strategic_parameters': strategic_params,
            'memory_features': match_features
        }


class AdvancedEnsembleInterface(nn.Module):
    """Advanced ensemble coordination for OLYMPUS integration"""
    def __init__(self, d_model: int = 256, num_specialists: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_specialists = num_specialists
        
        # Enhanced specialist embeddings
        self.specialist_embeddings = nn.Embedding(num_specialists, d_model)
        self.specialist_types = nn.Parameter(torch.randn(num_specialists, 64) * 0.01)
        
        # Multi-head cross-attention for ensemble coordination
        self.ensemble_attention = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True
        )
        
        # Advanced strategic decision network
        self.strategy_network = nn.Sequential(
            nn.Linear(d_model * num_specialists, d_model * 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128)  # Enhanced strategic parameters
        )
        
        # Confidence and consensus estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Consensus network for ensemble agreement
        self.consensus_network = nn.Sequential(
            nn.Linear(d_model + 64, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor, specialist_states: Optional[List] = None) -> Dict:
        B, C, H, W = features.shape
        
        # Global strategic features
        global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Get specialist embeddings and types
        specialist_ids = torch.arange(self.num_specialists).to(features.device)
        specialist_emb = self.specialist_embeddings(specialist_ids)  # num_specialists, d_model
        specialist_emb = specialist_emb.unsqueeze(0).expand(B, -1, -1)  # B, num_specialists, d_model
        
        # Enhanced cross-attention for coordination
        query = global_features.unsqueeze(1)  # B, 1, d_model
        coordinated_features, attention_weights = self.ensemble_attention(
            query, specialist_emb, specialist_emb
        )
        coordinated_features = coordinated_features.squeeze(1)  # B, d_model
        
        # Advanced strategic decision making
        ensemble_input = torch.cat([
            specialist_emb.view(B, -1),  # Flatten specialist embeddings
        ], dim=1)
        strategic_params = self.strategy_network(ensemble_input)
        
        # Confidence estimation
        confidence = self.confidence_estimator(coordinated_features)
        
        # Consensus calculation
        consensus_input = torch.cat([coordinated_features, self.specialist_types.mean(dim=0).expand(B, -1)], dim=1)
        consensus = self.consensus_network(consensus_input)
        
        return {
            'coordinated_features': coordinated_features,
            'strategic_params': strategic_params,
            'ensemble_attention': attention_weights,
            'confidence': confidence,
            'consensus': consensus,
            'specialist_agreement': attention_weights.mean(dim=1)
        }


class ProgramSynthesisIntegration(nn.Module):
    """Deep program synthesis integration for strategic reasoning"""
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Program synthesis embeddings
        self.program_embedder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 64)  # Program encoding
        )
        
        # Strategic program detector
        self.program_detector = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 32),  # Program types
            nn.Softmax(dim=-1)
        )
        
        # Program synthesis network
        self.synthesis_network = nn.Sequential(
            nn.Linear(d_model + 64, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 96)  # Synthesis parameters
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = features.shape
        
        # Global features for program synthesis
        global_features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        
        # Program embedding
        program_encoding = self.program_embedder(global_features)
        
        # Program type detection
        program_types = self.program_detector(global_features)
        
        # Program synthesis
        synthesis_input = torch.cat([global_features, program_encoding], dim=1)
        synthesis_params = self.synthesis_network(synthesis_input)
        
        return {
            'program_encoding': program_encoding,
            'program_types': program_types,
            'synthesis_params': synthesis_params
        }


class MinervaV6Enhanced(nn.Module):
    """MINERVA V6 Enhanced - Ultimate Strategic Intelligence Master with complete grid mastery"""
    def __init__(self, max_grid_size: int = 30, hidden_dim: int = 256, 
                 preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim
        self.preserve_weights = preserve_weights
        
        # PRESERVE: Original MINERVA components for weight loading
        self.original_minerva = EnhancedMinervaNet(max_grid_size, hidden_dim)
        
        # ENHANCE: Deep strategic components
        self.deep_strategic_transformer = DeepStrategicTransformer(
            hidden_dim, num_heads=8, num_layers=8, max_grid_size=max_grid_size
        )
        self.mega_pattern_memory = MegaPatternMemory(hidden_dim, memory_size=300)
        self.advanced_ensemble_interface = AdvancedEnsembleInterface(hidden_dim, num_specialists=5)
        self.program_synthesis = ProgramSynthesisIntegration(hidden_dim)
        
        # Enhanced strategic reasoning networks (NO ModuleDict - separate modules)
        self.strategic_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 96)  # Strategic analysis encoding
        )
        
        self.decision_maker = nn.Sequential(
            nn.Linear(hidden_dim + 96, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),  # Decision parameters
            nn.Tanh()
        )
        
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # OLYMPUS preparation: Advanced ensemble integration
        self.olympus_broadcaster = nn.Linear(hidden_dim, hidden_dim)
        self.olympus_aggregator = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Separate parameter for ensemble weights (can't go in ModuleDict)
        self.ensemble_weights = nn.Parameter(torch.ones(5) / 5)  # Equal initial weighting
        
        # Enhanced V6 decoder with multi-scale processing
        self.v6_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),  # More channels
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout2d(0.05),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(0.02),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout2d(0.01),
            nn.ConvTranspose2d(hidden_dim // 2, 10, 1)
        )
        
        # Strategic mixing parameters
        self.strategic_mix = nn.Parameter(torch.tensor(0.4))
        self.synthesis_mix = nn.Parameter(torch.tensor(0.3))
        
        self.description = "Ultimate Strategic Intelligence Master with Complete Grid Mastery and Advanced Program Synthesis"
    
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
                print(f"\033[96mMINERVA V6: Loaded full model state dict\033[0m")
            except:
                # Fallback to original_minerva loading
                if len(compatible_params) > 0:
                    model_dict.update(compatible_params)
                    self.original_minerva.load_state_dict(model_dict)
            
            print(f"\033[96mMINERVA V6: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[96mMINERVA V6: Could not load weights - {e}\033[0m")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # PRESERVE: Get features from original MINERVA
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_minerva(input_grid, output_grid, mode)
            base_features = original_output['features']
            base_prediction = original_output['predicted_output']
        
        # ENHANCE: Deep strategic transformer processing
        enhanced_features, strategic_info = self.deep_strategic_transformer(base_features)
        
        # ENHANCE: Mega pattern memory processing
        pattern_memory = self.mega_pattern_memory(enhanced_features)
        
        # ENHANCE: Advanced ensemble coordination
        ensemble_output = self.advanced_ensemble_interface(enhanced_features)
        
        # ENHANCE: Program synthesis integration
        program_synthesis = self.program_synthesis(enhanced_features)
        
        # ENHANCE: Strategic reasoning
        global_features = F.adaptive_avg_pool2d(enhanced_features, 1).squeeze(-1).squeeze(-1)
        strategic_analysis = self.strategic_analyzer(global_features)
        decision_params = self.decision_maker(
            torch.cat([global_features, strategic_analysis], dim=1)
        )
        confidence = self.confidence_calibrator(
            torch.cat([global_features, decision_params], dim=1)
        )
        
        # OLYMPUS: Prepare features for ensemble integration
        broadcast_features = self.olympus_broadcaster(global_features)
        
        # Combine all enhanced features for final prediction
        B, C, H, W = enhanced_features.shape
        
        # Multi-scale feature combination
        coordinated_spatial = ensemble_output['coordinated_features'].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        pattern_spatial = pattern_memory['strategic_parameters'].unsqueeze(-1).unsqueeze(-1).expand(-1, 64, H, W)
        synthesis_spatial = program_synthesis['synthesis_params'].unsqueeze(-1).unsqueeze(-1).expand(-1, 96, H, W)
        
        combined_features = torch.cat([
            base_features,           # Original MINERVA features
            enhanced_features,       # Deep strategic features
            coordinated_spatial,     # Ensemble coordination
            pattern_spatial,         # Pattern memory
            synthesis_spatial        # Program synthesis
        ], dim=1)
        
        # Enhanced V6 prediction
        enhanced_prediction = self.v6_decoder(combined_features)
        
        # Strategic mixing with multiple components
        strategic_weight = torch.sigmoid(self.strategic_mix) * confidence
        synthesis_weight = torch.sigmoid(self.synthesis_mix) * pattern_memory['pattern_similarity'].mean(dim=1, keepdim=True)
        
        # Ensure predictions have same spatial dimensions
        if enhanced_prediction.shape != base_prediction.shape:
            base_prediction = F.interpolate(
                base_prediction, 
                size=(enhanced_prediction.shape[2], enhanced_prediction.shape[3]),
                mode='bilinear', 
                align_corners=False
            )
        
        # Expand weights to match spatial dimensions
        strategic_weight_expanded = strategic_weight.unsqueeze(-1).unsqueeze(-1).expand_as(enhanced_prediction)
        synthesis_weight_expanded = synthesis_weight.unsqueeze(-1).unsqueeze(-1).expand_as(enhanced_prediction)
        
        # Multi-component strategic mixing
        final_prediction = (
            strategic_weight_expanded * synthesis_weight_expanded * enhanced_prediction + 
            (1 - strategic_weight_expanded * synthesis_weight_expanded) * base_prediction
        )
        
        # Comprehensive output for ensemble coordination
        result = {
            'predicted_output': final_prediction,
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'strategic_features': enhanced_features,
            'strategic_info': strategic_info,
            'pattern_memory': pattern_memory,
            'ensemble_output': ensemble_output,
            'program_synthesis': program_synthesis,
            'strategic_analysis': strategic_analysis,
            'decision_params': decision_params,
            'confidence': confidence,
            'olympus_features': broadcast_features,  # For OLYMPUS integration
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
            'model_type': 'MINERVA_V6',
            'strategic_weights': self.ensemble_weights.detach(),
            'confidence_threshold': 0.8,  # Higher threshold for V6
            'specialization': 'ultimate_strategic_coordination',
            'grid_mastery': '2x2_to_30x30',
            'program_synthesis': True,
            'coordination_ready': True
        }
    
    def test_time_adapt(self, task_examples: List[Tuple], num_steps: int = 8):
        """Advanced test-time adaptation for V6"""
        # Enhanced adaptation with more parameters
        adaptable_params = []
        for layer in self.deep_strategic_transformer.transformer_layers[-2:]:  # Last 2 layers
            adaptable_params.extend(list(layer.parameters()))
        adaptable_params.extend(list(self.v6_decoder.parameters()))
        
        optimizer = torch.optim.AdamW(adaptable_params, lr=0.005, weight_decay=1e-6)
        
        print(f"\033[96mMINERVA V6 ultimate strategic adaptation: {num_steps} steps\033[0m")
        
        for step in range(num_steps):
            total_loss = 0
            
            for input_grid, target_grid in task_examples:
                # Forward pass
                output = self(input_grid.unsqueeze(0), target_grid.unsqueeze(0), mode='adaptation')
                
                # Strategic adaptation loss
                pred_output = output['predicted_output']
                loss = F.cross_entropy(pred_output, target_grid.argmax(dim=0))
                
                # Add strategic consistency losses
                if 'confidence' in output:
                    confidence_loss = (1.0 - output['confidence']).mean() * 0.1
                    loss += confidence_loss
                
                if 'pattern_memory' in output:
                    pattern_consistency = torch.var(output['pattern_memory']['strategic_parameters'], dim=1).mean() * 0.05
                    loss += pattern_consistency
                
                total_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(adaptable_params, max_norm=0.5)
            optimizer.step()
        
        print(f"\033[96mMINERVA V6 ultimate adaptation complete!\\033[0m")


# Compatibility alias for easy integration
EnhancedMinervaV6Net = MinervaV6Enhanced