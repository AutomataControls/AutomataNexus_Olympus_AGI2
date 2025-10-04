"""
CHRONOS V4 Enhanced Model - Advanced Temporal Sequence Reasoning Expert for ARC-AGI-2
Enhanced with temporal transformers, sequence intelligence, and OLYMPUS ensemble preparation
Preserves existing weights while adding sophisticated temporal reasoning capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

# Import existing CHRONOS components for weight preservation
from src.models.chronos_model import EnhancedChronosNet


class TemporalPositionalEncoding(nn.Module):
    """Advanced temporal positional encoding with sequence awareness"""
    def __init__(self, d_model: int, max_sequence_length: int = 8, max_grid_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.max_grid_size = max_grid_size
        
        # Temporal positional encoding (time dimension) - adjust dimensions to sum to d_model
        temporal_dim = d_model // 3
        spatial_dim = d_model // 3  
        sequence_dim = d_model - temporal_dim - spatial_dim  # Remainder to ensure exact sum
        
        self.temporal_pe = nn.Parameter(torch.zeros(max_sequence_length, temporal_dim))
        
        # Spatial positional encoding (2D space)
        self.spatial_pe = nn.Parameter(torch.zeros(max_grid_size, max_grid_size, spatial_dim))
        
        # Sequence pattern encoding
        self.sequence_pe = nn.Parameter(torch.zeros(64, sequence_dim))  # 64 sequence patterns
        
        # Initialize with temporal sinusoidal patterns
        self._init_temporal_encodings()
        
    def _init_temporal_encodings(self):
        """Initialize with temporal sinusoidal patterns"""
        temporal_dim = self.d_model // 3
        spatial_dim = self.d_model // 3  
        sequence_dim = self.d_model - temporal_dim - spatial_dim
        
        # Temporal encoding (time steps)
        for t in range(self.max_sequence_length):
            for i in range(0, temporal_dim - 1, 2):
                freq = 1.0 / (10000 ** (i / temporal_dim))
                self.temporal_pe.data[t, i] = math.sin(t * freq)
                if i + 1 < temporal_dim:
                    self.temporal_pe.data[t, i + 1] = math.cos(t * freq)
        
        # Spatial encoding
        for h in range(self.max_grid_size):
            for w in range(self.max_grid_size):
                for i in range(0, spatial_dim - 1, 2):
                    freq = 1.0 / (10000 ** (i / spatial_dim))
                    self.spatial_pe.data[h, w, i] = math.sin(h * freq)
                    if i + 1 < spatial_dim:
                        self.spatial_pe.data[h, w, i + 1] = math.cos(w * freq)
        
        # Sequence pattern encoding
        for p in range(64):
            for i in range(0, sequence_dim - 1, 2):
                freq = 1.0 / (10000 ** (i / sequence_dim))
                self.sequence_pe.data[p, i] = math.sin(p * freq * 0.1)
                if i + 1 < sequence_dim:
                    self.sequence_pe.data[p, i + 1] = math.cos(p * freq * 0.1)
    
    def forward(self, x: torch.Tensor, temporal_step: int, sequence_pattern: torch.Tensor) -> torch.Tensor:
        """Apply temporal, spatial, and sequence positional encoding"""
        B, H, W, d_model = x.shape
        
        # Get temporal encoding
        temporal_enc = self.temporal_pe[temporal_step].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        temporal_enc = temporal_enc.expand(B, H, W, -1)
        
        # Get spatial encoding
        spatial_enc = self.spatial_pe[:H, :W, :].unsqueeze(0).expand(B, -1, -1, -1)
        
        # Get sequence pattern encoding
        sequence_dim = self.d_model - (self.d_model // 3) - (self.d_model // 3)
        seq_enc = torch.zeros(B, H, W, sequence_dim).to(x.device)
        for b in range(B):
            pattern_idx = sequence_pattern[b].long() % 64
            seq_enc[b] = self.sequence_pe[pattern_idx].unsqueeze(0).unsqueeze(0).expand(H, W, -1)
        
        # Combine all encodings
        combined_pe = torch.cat([temporal_enc, spatial_enc, seq_enc], dim=-1)
        
        return x + combined_pe


class SequenceRelationshipAttention(nn.Module):
    """Multi-head attention specialized for temporal sequence relationships"""
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
        
        # Temporal relationship bias (sequence position relationships)
        self.temporal_bias = nn.Parameter(torch.zeros(8, 8, num_heads))  # Max 8 time steps
        
        # Sequence continuity patterns
        self.continuity_patterns = nn.Parameter(torch.randn(16, num_heads) * 0.02)  # 16 continuity types
        
    def forward(self, x: torch.Tensor, temporal_step: int, sequence_info: Optional[Dict] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, d_model = x.shape
        seq_len = H * W
        
        # Reshape to sequence format
        x_seq = x.view(B, seq_len, d_model)
        
        # Compute Q, K, V
        q = self.w_q(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Temporal sequence-aware attention
        attention = self._temporal_sequence_attention(q, k, v, temporal_step, sequence_info, mask)
        
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
    
    def _temporal_sequence_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                   temporal_step: int, sequence_info: Optional[Dict], 
                                   mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, num_heads, seq_len, d_k = q.shape
        
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Add temporal bias for sequence continuity
        if temporal_step < 8:  # Within our temporal bias range
            temporal_bias = self._compute_temporal_bias(temporal_step, num_heads, seq_len)
            scores = scores + temporal_bias.unsqueeze(0)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _compute_temporal_bias(self, temporal_step: int, num_heads: int, seq_len: int) -> torch.Tensor:
        """Compute temporal relationship bias matrix"""
        bias = torch.zeros(num_heads, seq_len, seq_len, device=self.temporal_bias.device)
        
        # Apply temporal continuity bias
        # Encourage attention to previous time steps (causality)
        for i in range(min(seq_len, 64)):  # Limit for efficiency
            for j in range(min(seq_len, 64)):
                # Distance-based temporal bias
                distance_bias = 1.0 / (abs(i - j) + 1)  # Closer positions get higher bias
                bias[:, i, j] = distance_bias * 0.1
        
        return bias


class TemporalSequenceTransformer(nn.Module):
    """Advanced transformer for temporal sequence reasoning"""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Temporal pattern analyzers
        self.sequence_pattern_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 32)  # 32 sequence pattern types
        )
        
        self.movement_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 64)  # Movement parameters
        )
        
        self.sequence_continuity = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 16)  # Continuity patterns
        )
        
        # Temporal rule synthesis
        self.temporal_rule_synthesizer = nn.Sequential(
            nn.Linear(d_model + 32 + 64 + 16, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128)  # Temporal rule parameters
        )
        
    def forward(self, features: torch.Tensor, temporal_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = features.shape
        
        # Global temporal features
        global_features = features.mean(dim=[1, 2])  # B, d_model
        
        # Detect sequence patterns
        sequence_patterns = self.sequence_pattern_detector(global_features)
        patterns = F.softmax(sequence_patterns, dim=-1)
        
        # Predict movement/transformation
        movement_params = self.movement_predictor(global_features)
        movement = torch.tanh(movement_params)  # Bounded movement
        
        # Analyze sequence continuity
        continuity_features = self.sequence_continuity(global_features)
        continuity = torch.sigmoid(continuity_features)
        
        # Synthesize temporal rules
        combined_features = torch.cat([
            global_features, patterns, movement, continuity
        ], dim=1)
        temporal_rules = self.temporal_rule_synthesizer(combined_features)
        
        return {
            'sequence_patterns': patterns,
            'movement_prediction': movement,
            'continuity_analysis': continuity,
            'temporal_rules': temporal_rules,
            'temporal_confidence': torch.sigmoid(temporal_rules[:, :1])  # First param as confidence
        }


class TemporalTransformerBlock(nn.Module):
    """Enhanced transformer block for temporal reasoning"""
    def __init__(self, d_model: int, num_heads: int = 8, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model
        
        self.sequence_attention = SequenceRelationshipAttention(d_model, num_heads, dropout)
        self.temporal_transformer = TemporalSequenceTransformer(d_model, num_heads, dropout)
        
        # Enhanced feedforward with temporal awareness
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, temporal_step: int, sequence_info: Optional[Dict] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        
        # Temporal sequence attention
        attn_out, attention_weights = self.sequence_attention(x, temporal_step, sequence_info, mask)
        
        # Temporal analysis
        temporal_analysis = self.temporal_transformer(attn_out, sequence_info)
        
        # Feedforward with residual
        B, H, W, d_model = attn_out.shape
        ff_input = attn_out.view(B * H * W, d_model)
        ff_out = self.ff(ff_input)
        ff_out = self.layer_norm(ff_out + ff_input)
        
        output = ff_out.view(B, H, W, d_model)
        
        temporal_info = {
            'attention_weights': attention_weights,
            'temporal_analysis': temporal_analysis
        }
        
        return output, temporal_info


class TemporalEnsembleInterface(nn.Module):
    """Interface for temporal reasoning coordination with other specialists"""
    def __init__(self, d_model: int, num_specialists: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_specialists = num_specialists
        
        # Temporal feature broadcaster for ensemble
        self.temporal_broadcaster = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Cross-specialist temporal attention
        self.cross_temporal_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Temporal consensus mechanism
        self.temporal_consensus = nn.Sequential(
            nn.Linear(d_model * num_specialists, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Temporal expertise confidence
        self.temporal_expertise = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, temporal_features: torch.Tensor, 
                specialist_states: Optional[List] = None) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = temporal_features.shape
        
        # Global temporal features
        global_temporal = temporal_features.mean(dim=[1, 2])  # B, d_model
        
        # Broadcast temporal insights
        broadcast_features = self.temporal_broadcaster(global_temporal)
        
        # Temporal expertise confidence
        temporal_confidence = self.temporal_expertise(global_temporal)
        
        # Cross-attention with other specialists if available
        if specialist_states is not None and len(specialist_states) > 0:
            # Stack specialist states
            specialist_tensor = torch.stack(specialist_states, dim=1)  # B, num_specialists, d_model
            
            # Cross-attention
            query = broadcast_features.unsqueeze(1)  # B, 1, d_model
            attended_features, cross_attention_weights = self.cross_temporal_attention(
                query, specialist_tensor, specialist_tensor
            )
            attended_features = attended_features.squeeze(1)  # B, d_model
            
            # Temporal consensus
            consensus_input = torch.cat([
                attended_features,
                specialist_tensor.view(B, -1)
            ], dim=1)
            consensus_score = self.temporal_consensus(consensus_input)
        else:
            attended_features = broadcast_features
            cross_attention_weights = None
            consensus_score = temporal_confidence  # Use temporal expertise as consensus
        
        return {
            'broadcast_features': broadcast_features,
            'attended_features': attended_features,
            'cross_attention_weights': cross_attention_weights,
            'temporal_consensus': consensus_score,
            'temporal_expertise': temporal_confidence
        }


class ChronosV4Enhanced(nn.Module):
    """Enhanced CHRONOS V4 with temporal transformers and sequence intelligence"""
    def __init__(self, max_grid_size: int = 30, d_model: int = 256, num_layers: int = 6,
                 preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.preserve_weights = preserve_weights
        
        # PRESERVE: Original CHRONOS components for weight loading
        self.original_chronos = EnhancedChronosNet(max_grid_size)
        
        # ENHANCE: Input embedding for transformers
        self.input_embedding = nn.Linear(10, d_model)
        
        # ENHANCE: Temporal positional encoding
        self.pos_encoding = TemporalPositionalEncoding(d_model, max_sequence_length=8, max_grid_size=max_grid_size)
        
        # ENHANCE: Temporal transformer layers
        self.temporal_transformer_layers = nn.ModuleList([
            TemporalTransformerBlock(d_model, num_heads=8) for _ in range(num_layers)
        ])
        
        # ENHANCE: Ensemble coordination interface
        self.ensemble_interface = TemporalEnsembleInterface(d_model, num_specialists=5)
        
        # ENHANCE: Advanced temporal reasoning (Parameters separate from ModuleDict)
        self.sequence_memory = nn.Parameter(torch.randn(150, d_model) * 0.02)  # Temporal pattern memory
        
        self.advanced_temporal_reasoning = nn.ModuleDict({
            'temporal_rule_extractor': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 96)  # Temporal rule encoding
            ),
            'sequence_composer': nn.Sequential(
                nn.Linear(d_model + 96, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 160)  # Temporal sequence parameters
            )
        })
        
        # ENHANCE: Multi-temporal processing (different time scales)
        self.multitemporal_processor = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)  # Short, medium, long, global time scales
        ])
        self.temporal_fusion = nn.Linear(d_model * 4, d_model)
        
        # ENHANCE: Advanced output decoder
        self.v4_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model + 160, d_model, 3, padding=1),  # Include original CHRONOS features
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Dropout2d(0.06),
            nn.ConvTranspose2d(d_model, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Dropout2d(0.03),
            nn.ConvTranspose2d(d_model // 2, 10, 1)
        )
        
        # Strategic mixing parameters
        self.temporal_mix = nn.Parameter(torch.tensor(0.5))  # Weight for enhanced features
        self.sequence_confidence = nn.Parameter(torch.tensor(0.8))
        
        # Test-time adaptation
        self.adaptation_lr = 0.005
        self.adaptation_steps = 7
        
        self.description = "Enhanced Temporal Sequence Reasoning Expert with Temporal Transformers and OLYMPUS Preparation"
    
    def load_compatible_weights(self, checkpoint_path: str):
        """Load weights from existing CHRONOS model while preserving architecture"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible weights into original_chronos
            model_dict = self.original_chronos.state_dict()
            compatible_params = {}
            
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_params[k] = v
            
            model_dict.update(compatible_params)
            self.original_chronos.load_state_dict(model_dict)
            
            print(f"\033[96mCHRONOS V4: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[96mCHRONOS V4: Could not load weights - {e}\033[0m")
            return False
    
    def forward(self, input_sequence: List[torch.Tensor], output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # Handle single frame input (convert to sequence)
        if not isinstance(input_sequence, list):
            input_sequence = [input_sequence]
        
        # PRESERVE: Get features and predictions from original CHRONOS
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_chronos(input_sequence, output_grid)
            base_prediction = original_output['predicted_output']
            movement_params = original_output.get('movement_params')
            temporal_features = original_output.get('temporal_features')
        
        # Use the latest frame for transformer processing
        current_frame = input_sequence[-1]
        B, C, H, W = current_frame.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            current_frame = F.one_hot(current_frame.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Detect sequence patterns for positional encoding
        sequence_patterns = self._detect_sequence_patterns(input_sequence)
        
        # Reshape for transformer: B, C, H, W -> B, H, W, C
        x = current_frame.permute(0, 2, 3, 1)
        
        # Embed input tokens
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Add temporal positional encoding (using latest time step)
        temporal_step = len(input_sequence) - 1
        x = self.pos_encoding(x, temporal_step, sequence_patterns)
        
        # Apply temporal transformer layers
        temporal_analyses = []
        sequence_info = {'sequence_length': len(input_sequence), 'patterns': sequence_patterns}
        for layer in self.temporal_transformer_layers:
            x, temporal_info = layer(x, temporal_step, sequence_info)
            temporal_analyses.append(temporal_info)
        
        # Multi-temporal processing (different time scales)
        x_flat = x.mean(dim=[1, 2])  # B, d_model
        multitemporal_features = []
        for processor in self.multitemporal_processor:
            temporal_features = processor(x_flat)
            multitemporal_features.append(temporal_features)
        
        multitemporal_concat = torch.cat(multitemporal_features, dim=1)  # B, 4*d_model
        fused_temporal = self.temporal_fusion(multitemporal_concat)  # B, d_model
        
        # Advanced temporal reasoning
        global_features = x.mean(dim=[1, 2])  # B, d_model
        temporal_rule_encoding = self.advanced_temporal_reasoning['temporal_rule_extractor'](global_features)
        
        # Temporal sequence memory matching
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.sequence_memory.unsqueeze(0), 
            dim=2
        )  # B, 150
        top_temporal_patterns = memory_similarity.topk(10, dim=1)[0].mean(dim=1, keepdim=True)  # B, 1
        
        # Compose temporal sequences
        composition_input = torch.cat([global_features, temporal_rule_encoding], dim=1)
        temporal_sequence_params = self.advanced_temporal_reasoning['sequence_composer'](composition_input)
        
        # Ensemble coordination
        ensemble_output = self.ensemble_interface(x)
        
        # Combine enhanced features with sequence parameters
        enhanced_temporal = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        sequence_spatial = temporal_sequence_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_features = torch.cat([enhanced_temporal, sequence_spatial], dim=1)
        
        # Enhanced prediction
        enhanced_prediction = self.v4_decoder(combined_features)
        
        # Strategic mixing of original and enhanced predictions
        temporal_expertise = ensemble_output['temporal_expertise']
        sequence_weight = torch.sigmoid(self.temporal_mix) * temporal_expertise
        
        # Ensure predictions have same spatial dimensions
        if enhanced_prediction.shape != base_prediction.shape:
            base_prediction = F.interpolate(
                base_prediction, 
                size=(enhanced_prediction.shape[2], enhanced_prediction.shape[3]),
                mode='bilinear', 
                align_corners=False
            )
        
        # Expand weights to match spatial dimensions
        sequence_weight_expanded = sequence_weight.unsqueeze(-1).unsqueeze(-1).expand_as(enhanced_prediction)
        
        final_prediction = (
            sequence_weight_expanded * enhanced_prediction + 
            (1 - sequence_weight_expanded) * base_prediction
        )
        
        # Comprehensive output for ensemble coordination
        result = {
            'predicted_output': final_prediction,
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'temporal_features': x,
            'temporal_sequence_params': temporal_sequence_params,
            'temporal_rule_encoding': temporal_rule_encoding,
            'temporal_memory_similarity': top_temporal_patterns,
            'temporal_analyses': temporal_analyses,
            'ensemble_output': ensemble_output,
            'multitemporal_features': multitemporal_features,
            'temporal_expertise': temporal_expertise
        }
        
        # Add original outputs for compatibility
        result.update({
            'movement_params': movement_params,
            'attention_weights': original_output.get('attention_weights'),
            'temporal_features_original': temporal_features
        })
        
        return result
    
    def _detect_sequence_patterns(self, sequence: List[torch.Tensor]) -> torch.Tensor:
        """Detect temporal patterns in the sequence for positional encoding"""
        if len(sequence) <= 1:
            return torch.zeros(sequence[0].shape[0], dtype=torch.long, device=sequence[0].device)
        
        B = sequence[0].shape[0]
        pattern_indices = torch.zeros(B, dtype=torch.long, device=sequence[0].device)
        
        # Simple pattern detection based on frame differences
        for b in range(B):
            pattern_score = 0
            for i in range(1, len(sequence)):
                prev_frame = sequence[i-1][b].argmax(dim=0)
                curr_frame = sequence[i][b].argmax(dim=0)
                
                # Calculate change pattern
                change_ratio = (prev_frame != curr_frame).float().mean()
                pattern_score += change_ratio.item() * (i + 1)  # Weight later changes more
            
            # Convert to pattern index
            pattern_id = int(pattern_score * 16) % 64
            pattern_indices[b] = pattern_id
        
        return pattern_indices
    
    def get_ensemble_state(self) -> Dict:
        """Get state for OLYMPUS ensemble coordination"""
        return {
            'model_type': 'CHRONOS_V4',
            'temporal_confidence': self.sequence_confidence.detach(),
            'specialization': 'temporal_sequence_reasoning',
            'temporal_capabilities': ['sequence_analysis', 'movement_prediction', 'temporal_continuity'],
            'coordination_ready': True
        }
    
    def test_time_adapt(self, task_examples: List[Tuple], num_steps: int = None):
        """Temporal test-time adaptation"""
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Get adaptable temporal parameters
        temporal_params = []
        for layer in self.temporal_transformer_layers:
            temporal_params.extend(list(layer.parameters()))
        temporal_params.extend(list(self.v4_decoder.parameters()))
        
        optimizer = torch.optim.AdamW(temporal_params, lr=self.adaptation_lr, weight_decay=1e-6)
        
        print(f"\033[96mCHRONOS V4 temporal adaptation: {num_steps} steps\033[0m")
        
        for step in range(num_steps):
            total_loss = 0
            temporal_loss = 0
            
            for input_sequence, target_grid in task_examples:
                # Handle single frame or sequence input
                if not isinstance(input_sequence, list):
                    input_sequence = [input_sequence]
                
                # Forward pass
                output = self(input_sequence, target_grid, mode='adaptation')
                
                # Main prediction loss
                pred_output = output['predicted_output']
                main_loss = F.cross_entropy(pred_output, target_grid.argmax(dim=0))
                
                # Temporal expertise loss
                if 'temporal_expertise' in output:
                    temporal_consistency = (1.0 - output['temporal_expertise']).mean()
                    temporal_loss += temporal_consistency * 0.1
                
                total_loss += main_loss + temporal_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(temporal_params, max_norm=1.0)
            optimizer.step()
            
            if step % 2 == 0:
                print(f"\033[96m  Temporal Step {step}: Loss = {total_loss.item():.4f}\033[0m")
        
        print(f"\033[96mCHRONOS V4 temporal adaptation complete!\033[0m")


# Compatibility alias
EnhancedChronosV4Net = ChronosV4Enhanced