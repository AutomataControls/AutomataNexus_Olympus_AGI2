"""
PROMETHEUS V4 Enhanced Model - Advanced Creative Pattern Generation Expert for ARC-AGI-2
Enhanced with creative transformers, pattern synthesis reasoning, and OLYMPUS ensemble preparation
Preserves existing weights while adding sophisticated creative intelligence capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

# Import existing PROMETHEUS components for weight preservation
from src.models.prometheus_model import EnhancedPrometheusNet


class CreativePositionalEncoding(nn.Module):
    """Advanced positional encoding with creative pattern awareness"""
    def __init__(self, d_model: int, max_grid_size: int = 30, num_patterns: int = 64):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.num_patterns = num_patterns
        
        # Standard 2D positional encoding
        self.spatial_pe = nn.Parameter(torch.zeros(max_grid_size, max_grid_size, d_model // 2))
        
        # Pattern-based positional encoding
        self.pattern_pe = nn.Parameter(torch.zeros(num_patterns, d_model // 2))
        
        # Initialize with creative frequency patterns
        self._init_creative_encodings()
        
    def _init_creative_encodings(self):
        """Initialize with creative sinusoidal patterns"""
        # Spatial encoding with creative modulation
        for h in range(self.max_grid_size):
            for w in range(self.max_grid_size):
                for i in range(0, self.d_model // 4, 2):
                    # Creative row encoding (non-linear modulation)
                    freq_h = 1 + math.sin(h / 10.0) * 0.3  # Creative frequency modulation
                    self.spatial_pe.data[h, w, i] = math.sin(h * freq_h / (10000 ** (i / (self.d_model // 2))))
                    self.spatial_pe.data[h, w, i + 1] = math.cos(h * freq_h / (10000 ** (i / (self.d_model // 2))))
                    # Creative column encoding
                    freq_w = 1 + math.cos(w / 8.0) * 0.3
                    self.spatial_pe.data[h, w, i + self.d_model // 4] = math.sin(w * freq_w / (10000 ** (i / (self.d_model // 2))))
                    self.spatial_pe.data[h, w, i + self.d_model // 4 + 1] = math.cos(w * freq_w / (10000 ** (i / (self.d_model // 2))))
        
        # Pattern encoding (creative pattern frequencies)
        for p in range(self.num_patterns):
            for i in range(0, self.d_model // 4, 2):
                # Creative pattern frequency
                creative_freq = 1 + math.sin(p / 16.0) * 0.5
                self.pattern_pe.data[p, i] = math.sin(p * creative_freq / (10000 ** (i / (self.d_model // 2))))
                self.pattern_pe.data[p, i + 1] = math.cos(p * creative_freq / (10000 ** (i / (self.d_model // 2))))
    
    def forward(self, x: torch.Tensor, pattern_indices: torch.Tensor) -> torch.Tensor:
        """Apply spatial and creative pattern positional encoding"""
        B, H, W, d_model = x.shape
        
        # Get spatial encoding
        spatial_enc = self.spatial_pe[:H, :W, :].unsqueeze(0).expand(B, -1, -1, -1)
        
        # Get pattern encoding based on detected patterns
        pattern_enc = torch.zeros(B, H, W, self.d_model // 2).to(x.device)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    pattern_idx = pattern_indices[b, h, w].long() % self.num_patterns
                    pattern_enc[b, h, w, :] = self.pattern_pe[pattern_idx, :]
        
        # Combine spatial and pattern encodings
        combined_pe = torch.cat([spatial_enc, pattern_enc], dim=-1)
        
        return x + combined_pe


class PatternCreativityAttention(nn.Module):
    """Multi-head attention specialized for creative pattern relationships"""
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
        
        # Creative pattern relationship bias
        self.creativity_bias = nn.Parameter(torch.randn(64, 64, num_heads) * 0.02)  # Pattern x Pattern
        
        # Novelty encouragement patterns
        self.novelty_patterns = nn.Parameter(torch.randn(32, num_heads) * 0.02)  # 32 novelty types
        
    def forward(self, x: torch.Tensor, pattern_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, d_model = x.shape
        seq_len = H * W
        
        # Reshape to sequence format
        x_seq = x.view(B, seq_len, d_model)
        
        # Compute Q, K, V
        q = self.w_q(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Creative pattern-aware attention
        attention = self._creative_pattern_attention(q, k, v, pattern_features, mask)
        
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
    
    def _creative_pattern_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                  pattern_features: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, num_heads, seq_len, d_k = q.shape
        
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Add creative pattern bias
        if pattern_features is not None:
            pattern_bias = self._compute_creative_bias(pattern_features, num_heads, seq_len)
            scores = scores + pattern_bias.unsqueeze(0)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _compute_creative_bias(self, pattern_features: torch.Tensor, num_heads: int, seq_len: int) -> torch.Tensor:
        """Compute creative pattern relationship bias matrix"""
        B = pattern_features.shape[0]
        bias = torch.zeros(B, num_heads, seq_len, seq_len, device=pattern_features.device)
        
        # Use pattern features to determine creativity relationships
        pattern_similarity = torch.matmul(pattern_features, pattern_features.transpose(-2, -1))
        pattern_similarity = F.softmax(pattern_similarity, dim=-1)
        
        # Encourage creative pattern combinations (diversity bonus)
        diversity_bonus = 1.0 - pattern_similarity  # Reward dissimilar patterns
        
        for b in range(B):
            for i in range(min(seq_len, 64)):
                for j in range(min(seq_len, 64)):
                    # Add creativity encouragement
                    creativity_score = diversity_bonus[b, i, j] if i < 64 and j < 64 else 0.5
                    bias[b, :, i, j] = creativity_score * 0.1
        
        return bias


class CreativePatternSynthesis(nn.Module):
    """Advanced pattern synthesis for creative generation"""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Pattern generation networks
        self.pattern_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 64)  # 64 pattern types
        )
        
        self.novelty_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 32)  # 32 novelty categories
        )
        
        self.creativity_composer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 128)  # Creative composition parameters
        )
        
        # Creative rule synthesis
        self.rule_synthesizer = nn.Sequential(
            nn.Linear(d_model + 64 + 32 + 128, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 100)  # Creative rule parameters
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = features.shape
        
        # Global creative features
        global_features = features.mean(dim=[1, 2])  # B, d_model
        
        # Generate creative patterns
        pattern_logits = self.pattern_generator(global_features)
        patterns = F.softmax(pattern_logits, dim=-1)
        
        # Detect novelty potential
        novelty_features = self.novelty_detector(global_features)
        novelty_scores = F.softmax(novelty_features, dim=-1)
        
        # Compose creative elements
        creative_composition = self.creativity_composer(global_features)
        
        # Synthesize creative rules
        combined_features = torch.cat([
            global_features, patterns, novelty_scores, creative_composition
        ], dim=1)
        creative_rules = self.rule_synthesizer(combined_features)
        
        return {
            'generated_patterns': patterns,
            'novelty_scores': novelty_scores,
            'creative_composition': creative_composition,
            'creative_rules': creative_rules,
            'creativity_confidence': torch.sigmoid(creative_rules[:, :1])  # First param as confidence
        }


class CreativeTransformerBlock(nn.Module):
    """Enhanced transformer block for creative reasoning"""
    def __init__(self, d_model: int, num_heads: int = 8, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model
        
        self.pattern_attention = PatternCreativityAttention(d_model, num_heads, dropout)
        self.creative_synthesizer = CreativePatternSynthesis(d_model, num_heads, dropout)
        
        # Enhanced feedforward with creative awareness
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, pattern_features: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        
        # Creative pattern attention
        attn_out, attention_weights = self.pattern_attention(x, pattern_features, mask)
        
        # Creative synthesis analysis
        creative_analysis = self.creative_synthesizer(attn_out)
        
        # Feedforward with residual
        B, H, W, d_model = attn_out.shape
        ff_input = attn_out.view(B * H * W, d_model)
        ff_out = self.ff(ff_input)
        ff_out = self.layer_norm(ff_out + ff_input)
        
        output = ff_out.view(B, H, W, d_model)
        
        creative_info = {
            'attention_weights': attention_weights,
            'creative_analysis': creative_analysis
        }
        
        return output, creative_info


class CreativeEnsembleInterface(nn.Module):
    """Interface for creative reasoning coordination with other specialists"""
    def __init__(self, d_model: int, num_specialists: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_specialists = num_specialists
        
        # Creative feature broadcaster for ensemble
        self.creative_broadcaster = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Cross-specialist creative attention
        self.cross_creative_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Creative consensus mechanism
        self.creative_consensus = nn.Sequential(
            nn.Linear(d_model * num_specialists, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Creative expertise confidence
        self.creative_expertise = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, creative_features: torch.Tensor, 
                specialist_states: Optional[List] = None) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = creative_features.shape
        
        # Global creative features
        global_creative = creative_features.mean(dim=[1, 2])  # B, d_model
        
        # Broadcast creative insights
        broadcast_features = self.creative_broadcaster(global_creative)
        
        # Creative expertise confidence
        creative_confidence = self.creative_expertise(global_creative)
        
        # Cross-attention with other specialists if available
        if specialist_states is not None and len(specialist_states) > 0:
            # Stack specialist states
            specialist_tensor = torch.stack(specialist_states, dim=1)  # B, num_specialists, d_model
            
            # Cross-attention
            query = broadcast_features.unsqueeze(1)  # B, 1, d_model
            attended_features, cross_attention_weights = self.cross_creative_attention(
                query, specialist_tensor, specialist_tensor
            )
            attended_features = attended_features.squeeze(1)  # B, d_model
            
            # Creative consensus
            consensus_input = torch.cat([
                attended_features,
                specialist_tensor.view(B, -1)
            ], dim=1)
            consensus_score = self.creative_consensus(consensus_input)
        else:
            attended_features = broadcast_features
            cross_attention_weights = None
            consensus_score = creative_confidence  # Use creative expertise as consensus
        
        return {
            'broadcast_features': broadcast_features,
            'attended_features': attended_features,
            'cross_attention_weights': cross_attention_weights,
            'creative_consensus': consensus_score,
            'creative_expertise': creative_confidence
        }


class PrometheusV4Enhanced(nn.Module):
    """Enhanced PROMETHEUS V4 with creative transformers and pattern synthesis intelligence"""
    def __init__(self, max_grid_size: int = 30, d_model: int = 256, num_layers: int = 6,
                 preserve_weights: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.preserve_weights = preserve_weights
        
        # PRESERVE: Original PROMETHEUS components for weight loading
        self.original_prometheus = EnhancedPrometheusNet(max_grid_size)
        
        # ENHANCE: Input embedding for transformers
        self.input_embedding = nn.Linear(10, d_model)
        
        # ENHANCE: Creative positional encoding
        self.pos_encoding = CreativePositionalEncoding(d_model, max_grid_size, num_patterns=64)
        
        # ENHANCE: Creative transformer layers
        self.creative_transformer_layers = nn.ModuleList([
            CreativeTransformerBlock(d_model, num_heads=8) for _ in range(num_layers)
        ])
        
        # ENHANCE: Ensemble coordination interface
        self.ensemble_interface = CreativeEnsembleInterface(d_model, num_specialists=5)
        
        # ENHANCE: Advanced creative reasoning
        self.advanced_creative_reasoning = nn.ModuleDict({
            'pattern_memory': nn.Parameter(torch.randn(200, d_model) * 0.02),  # Creative pattern memory
            'innovation_engine': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 128)  # Innovation encoding
            ),
            'creative_rule_composer': nn.Sequential(
                nn.Linear(d_model + 128, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 150)  # Creative transformation parameters
            )
        })
        
        # ENHANCE: Multi-creative processing (different creative modes)
        self.multicreative_processor = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(4)  # Divergent, convergent, lateral, combinatorial
        ])
        self.creative_fusion = nn.Linear(d_model * 4, d_model)
        
        # ENHANCE: Advanced output decoder
        self.v4_decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model + 150, d_model, 3, padding=1),  # Include original PROMETHEUS features
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Dropout2d(0.08),
            nn.ConvTranspose2d(d_model, d_model // 2, 3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Dropout2d(0.04),
            nn.ConvTranspose2d(d_model // 2, 10, 1)
        )
        
        # Strategic mixing parameters
        self.creative_mix = nn.Parameter(torch.tensor(0.45))  # Weight for enhanced features
        self.innovation_confidence = nn.Parameter(torch.tensor(0.8))
        
        # Test-time adaptation
        self.adaptation_lr = 0.01
        self.adaptation_steps = 8
        
        self.description = "Enhanced Creative Pattern Generation Expert with Creative Transformers and OLYMPUS Preparation"
    
    def load_compatible_weights(self, checkpoint_path: str):
        """Load weights from existing PROMETHEUS model while preserving architecture"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load compatible weights into original_prometheus
            model_dict = self.original_prometheus.state_dict()
            compatible_params = {}
            
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_params[k] = v
            
            model_dict.update(compatible_params)
            self.original_prometheus.load_state_dict(model_dict)
            
            print(f"\033[96mPROMETHEUS V4: Loaded {len(compatible_params)}/{len(state_dict)} compatible parameters\033[0m")
            return True
            
        except Exception as e:
            print(f"\033[96mPROMETHEUS V4: Could not load weights - {e}\033[0m")
            return False
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference', ensemble_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        
        # PRESERVE: Get features and predictions from original PROMETHEUS
        with torch.no_grad() if mode == 'inference' else torch.enable_grad():
            original_output = self.original_prometheus(input_grid, output_grid, mode)
            base_prediction = original_output['predicted_output']
            creative_latent = original_output.get('latent')
            creative_rules = original_output.get('rules')
        
        # ENHANCE: Prepare input for transformers
        B, C, H, W = input_grid.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Detect creative patterns for positional encoding
        pattern_indices = self._detect_creative_patterns(input_grid)
        
        # Reshape for transformer: B, C, H, W -> B, H, W, C
        x = input_grid.permute(0, 2, 3, 1)
        
        # Embed input tokens
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Add creative positional encoding
        x = self.pos_encoding(x, pattern_indices)
        
        # Apply creative transformer layers
        creative_analyses = []
        pattern_features = None
        for layer in self.creative_transformer_layers:
            x, creative_info = layer(x, pattern_features)
            creative_analyses.append(creative_info)
            # Update pattern features based on creative analysis
            if 'creative_analysis' in creative_info:
                pattern_features = creative_info['creative_analysis'].get('generated_patterns')
        
        # Multi-creative processing (different creative thinking modes)
        x_flat = x.mean(dim=[1, 2])  # B, d_model
        multicreative_features = []
        for processor in self.multicreative_processor:
            creative_features = processor(x_flat)
            multicreative_features.append(creative_features)
        
        multicreative_concat = torch.cat(multicreative_features, dim=1)  # B, 4*d_model
        fused_creative = self.creative_fusion(multicreative_concat)  # B, d_model
        
        # Advanced creative reasoning
        global_features = x.mean(dim=[1, 2])  # B, d_model
        innovation_encoding = self.advanced_creative_reasoning['innovation_engine'](global_features)
        
        # Creative pattern memory matching
        memory_similarity = F.cosine_similarity(
            global_features.unsqueeze(1), 
            self.advanced_creative_reasoning['pattern_memory'].unsqueeze(0), 
            dim=2
        )  # B, 200
        top_creative_patterns = memory_similarity.topk(12, dim=1)[0].mean(dim=1, keepdim=True)  # B, 1
        
        # Compose creative transformations
        composition_input = torch.cat([global_features, innovation_encoding], dim=1)
        creative_transform_params = self.advanced_creative_reasoning['creative_rule_composer'](composition_input)
        
        # Ensemble coordination
        ensemble_output = self.ensemble_interface(x)
        
        # Combine enhanced features with transformation parameters
        enhanced_creative = x.permute(0, 3, 1, 2)  # B, d_model, H, W
        transform_spatial = creative_transform_params.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined_features = torch.cat([enhanced_creative, transform_spatial], dim=1)
        
        # Enhanced prediction
        enhanced_prediction = self.v4_decoder(combined_features)
        
        # Strategic mixing of original and enhanced predictions
        creative_expertise = ensemble_output['creative_expertise']
        innovation_weight = torch.sigmoid(self.creative_mix) * creative_expertise
        
        final_prediction = (
            innovation_weight * enhanced_prediction + 
            (1 - innovation_weight) * base_prediction
        )
        
        # Comprehensive output for ensemble coordination
        result = {
            'predicted_output': final_prediction,
            'base_prediction': base_prediction,
            'enhanced_prediction': enhanced_prediction,
            'creative_features': x,
            'creative_transform_params': creative_transform_params,
            'innovation_encoding': innovation_encoding,
            'creative_memory_similarity': top_creative_patterns,
            'creative_analyses': creative_analyses,
            'ensemble_output': ensemble_output,
            'multicreative_features': multicreative_features,
            'creative_expertise': creative_expertise
        }
        
        # Add original outputs for compatibility
        result.update({
            'raw_output': original_output.get('raw_output'),
            'mu': original_output.get('mu'),
            'log_var': original_output.get('log_var'),
            'latent': creative_latent,
            'rules': creative_rules
        })
        
        return result
    
    def _detect_creative_patterns(self, input_grid: torch.Tensor) -> torch.Tensor:
        """Detect creative patterns in input for positional encoding"""
        B, C, H, W = input_grid.shape
        
        # Simple pattern detection based on local neighborhoods
        pattern_indices = torch.zeros(B, H, W, dtype=torch.long, device=input_grid.device)
        
        # Get dominant color per cell
        dominant_colors = input_grid.argmax(dim=1)  # B, H, W
        
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    # Local 3x3 neighborhood pattern
                    h_min, h_max = max(0, h-1), min(H, h+2)
                    w_min, w_max = max(0, w-1), min(W, w+2)
                    
                    neighborhood = dominant_colors[b, h_min:h_max, w_min:w_max]
                    unique_colors = torch.unique(neighborhood)
                    
                    # Pattern encoding based on local complexity and symmetry
                    pattern_id = (len(unique_colors) * 8 + 
                                h % 4 * 2 + 
                                w % 4) % 64
                    
                    pattern_indices[b, h, w] = pattern_id
        
        return pattern_indices
    
    def get_ensemble_state(self) -> Dict:
        """Get state for OLYMPUS ensemble coordination"""
        return {
            'model_type': 'PROMETHEUS_V4',
            'creative_confidence': self.innovation_confidence.detach(),
            'specialization': 'creative_pattern_generation',
            'creative_capabilities': ['pattern_synthesis', 'novelty_generation', 'creative_combination'],
            'coordination_ready': True
        }
    
    def test_time_adapt(self, task_examples: List[Tuple], num_steps: int = None):
        """Creative test-time adaptation"""
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Get adaptable creative parameters
        creative_params = []
        for layer in self.creative_transformer_layers:
            creative_params.extend(list(layer.parameters()))
        creative_params.extend(list(self.v4_decoder.parameters()))
        
        optimizer = torch.optim.AdamW(creative_params, lr=self.adaptation_lr, weight_decay=1e-6)
        
        print(f"\033[96mPROMETHEUS V4 creative adaptation: {num_steps} steps\033[0m")
        
        for step in range(num_steps):
            total_loss = 0
            creative_loss = 0
            
            for input_grid, target_grid in task_examples:
                # Forward pass
                output = self(input_grid.unsqueeze(0), target_grid.unsqueeze(0), mode='adaptation')
                
                # Main prediction loss
                pred_output = output['predicted_output']
                main_loss = F.cross_entropy(pred_output, target_grid.argmax(dim=0))
                
                # Creative expertise loss
                if 'creative_expertise' in output:
                    creative_consistency = (1.0 - output['creative_expertise']).mean()
                    creative_loss += creative_consistency * 0.15
                
                total_loss += main_loss + creative_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(creative_params, max_norm=1.2)
            optimizer.step()
            
            if step % 2 == 0:
                print(f"\033[96m  Creative Step {step}: Loss = {total_loss.item():.4f}\033[0m")
        
        print(f"\033[96mPROMETHEUS V4 creative adaptation complete!\033[0m")


# Compatibility alias
EnhancedPrometheusV4Net = PrometheusV4Enhanced