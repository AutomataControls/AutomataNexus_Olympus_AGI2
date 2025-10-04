"""
PROMETHEUS V4 Model - 2D-Aware Transformer Architecture for ARC-AGI-2
Advanced spatial reasoning with test-time adaptation capability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class SpatialPositionalEncoding2D(nn.Module):
    """2D Positional Encoding for spatial grid understanding"""
    def __init__(self, d_model: int, max_grid_size: int = 30):
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        
        # Create 2D positional encoding
        pe = torch.zeros(max_grid_size, max_grid_size, d_model)
        
        # Position encodings for height and width
        for h in range(max_grid_size):
            for w in range(max_grid_size):
                for i in range(0, d_model, 4):
                    # Height encoding
                    pe[h, w, i] = math.sin(h / (10000 ** (i / d_model)))
                    pe[h, w, i + 1] = math.cos(h / (10000 ** (i / d_model)))
                    
                    # Width encoding
                    if i + 2 < d_model:
                        pe[h, w, i + 2] = math.sin(w / (10000 ** ((i + 2) / d_model)))
                    if i + 3 < d_model:
                        pe[h, w, i + 3] = math.cos(w / (10000 ** ((i + 2) / d_model)))
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, d_model)
        B, H, W, d_model = x.shape
        return x + self.pe[:H, :W, :d_model].unsqueeze(0)


class Spatial2DAttention(nn.Module):
    """2D-aware multi-head attention for spatial reasoning"""
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
        
        # Spatial bias for 2D relationships
        self.spatial_bias = nn.Parameter(torch.zeros(2 * 30 - 1, 2 * 30 - 1))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, W, d_model = x.shape
        seq_len = H * W
        
        # Reshape to sequence format
        x_seq = x.view(B, seq_len, d_model)
        
        # Apply attention
        q = self.w_q(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x_seq).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with spatial bias
        attention = self._spatial_attention(q, k, v, H, W, mask)
        
        # Concatenate heads and put through final linear layer
        attention = attention.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        output = self.w_o(attention)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x_seq)
        
        # Reshape back to 2D
        return output.view(B, H, W, d_model)
    
    def _spatial_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          H: int, W: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, num_heads, seq_len, d_k = q.shape
        
        # Standard attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Add spatial bias
        spatial_scores = self._get_spatial_bias(H, W, seq_len)
        scores = scores + spatial_scores.unsqueeze(0).unsqueeze(0)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, v)
    
    def _get_spatial_bias(self, H: int, W: int, seq_len: int) -> torch.Tensor:
        """Compute spatial bias based on 2D distance"""
        bias = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            for j in range(seq_len):
                h1, w1 = i // W, i % W
                h2, w2 = j // W, j % W
                
                # Manhattan distance bias
                distance = abs(h1 - h2) + abs(w1 - w2)
                if distance < len(self.spatial_bias):
                    bias[i, j] = self.spatial_bias[distance, 0]
        
        return bias.to(self.spatial_bias.device)


class SpatialTransformerBlock(nn.Module):
    """Transformer block with 2D spatial awareness"""
    def __init__(self, d_model: int, num_heads: int = 8, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * d_model
            
        self.attention = Spatial2DAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention with residual
        attn_out = self.attention(x, mask)
        
        # Feed-forward with residual
        B, H, W, d_model = attn_out.shape
        ff_input = attn_out.view(B * H * W, d_model)
        ff_out = self.ff(ff_input)
        ff_out = self.layer_norm(ff_out + ff_input)
        
        return ff_out.view(B, H, W, d_model)


class NeuralProgramSynthesis(nn.Module):
    """Neural-based program synthesis component"""
    def __init__(self, d_model: int, num_ops: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_ops = num_ops
        
        # Program operation embeddings
        self.op_embeddings = nn.Embedding(num_ops, d_model)
        
        # Program sequence generator
        self.program_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_ops * 8)  # Up to 8 operations
        )
        
        # Operation executor
        self.operation_executor = nn.ModuleList([
            self._create_operation_module() for _ in range(num_ops)
        ])
        
    def _create_operation_module(self) -> nn.Module:
        """Create a differentiable operation module"""
        return nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 10, 1),
            nn.Tanh()
        )
    
    def forward(self, features: torch.Tensor, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, H, W, d_model = features.shape
        
        # Generate program sequence
        global_features = features.mean(dim=[1, 2])  # Global pooling
        program_logits = self.program_generator(global_features)
        program_probs = F.softmax(program_logits.view(B, 8, self.num_ops), dim=-1)
        
        # Execute differentiable program
        current_grid = input_grid
        program_trace = []
        
        for step in range(8):
            step_probs = program_probs[:, step, :]  # B, num_ops
            
            # Weighted combination of all operations
            step_output = torch.zeros_like(current_grid)
            for op_idx, op_module in enumerate(self.operation_executor):
                op_weight = step_probs[:, op_idx].view(B, 1, 1, 1)
                op_result = op_module(current_grid)
                step_output += op_weight * op_result
            
            current_grid = step_output
            program_trace.append(step_output)
        
        return {
            'program_output': current_grid,
            'program_probs': program_probs,
            'program_trace': torch.stack(program_trace, dim=1)
        }


class PrometheusV4Transformer(nn.Module):
    """PROMETHEUS V4 with 2D-Aware Transformers and Program Synthesis"""
    def __init__(self, max_grid_size: int = 30, d_model: int = 256, num_layers: int = 6, 
                 num_heads: int = 8, enable_program_synthesis: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.enable_program_synthesis = enable_program_synthesis
        
        # Input embedding for grid values
        self.input_embedding = nn.Linear(10, d_model)
        
        # 2D positional encoding
        self.pos_encoding = SpatialPositionalEncoding2D(d_model, max_grid_size)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            SpatialTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        # Program synthesis component
        if enable_program_synthesis:
            self.program_synthesis = NeuralProgramSynthesis(d_model, num_ops=16)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 10)
        )
        
        # Test-time adaptation parameters
        self.adaptation_lr = 0.01
        self.adaptation_steps = 5
        
        self.description = "2D-Aware Transformer with Neural Program Synthesis"
    
    def forward(self, input_grid: torch.Tensor, output_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference') -> Dict[str, torch.Tensor]:
        B, C, H, W = input_grid.shape
        
        # Convert input to one-hot if needed
        if C == 1:
            input_grid = F.one_hot(input_grid.long().squeeze(1), num_classes=10).float().permute(0, 3, 1, 2)
        
        # Reshape to sequence format with spatial dimensions preserved
        # B, C, H, W -> B, H, W, C
        x = input_grid.permute(0, 2, 3, 1)
        
        # Embed input tokens
        x = self.input_embedding(x)  # B, H, W, d_model
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Program synthesis branch
        program_outputs = {}
        if self.enable_program_synthesis:
            program_result = self.program_synthesis(x, input_grid)
            program_outputs.update(program_result)
        
        # Main prediction branch
        output = self.output_projection(x)  # B, H, W, 10
        predicted_output = output.permute(0, 3, 1, 2)  # B, 10, H, W
        
        # Combine with program synthesis if enabled
        if self.enable_program_synthesis and 'program_output' in program_outputs:
            alpha = 0.7  # Learnable mixing parameter could be added
            predicted_output = alpha * predicted_output + (1 - alpha) * program_outputs['program_output']
        
        result = {
            'predicted_output': predicted_output,
            'spatial_features': x,
            'attention_maps': None,  # Could add attention visualization
        }
        
        if program_outputs:
            result.update(program_outputs)
        
        return result
    
    def test_time_adapt(self, task_examples: list, num_steps: int = None) -> None:
        """Test-time adaptation on specific task examples"""
        if num_steps is None:
            num_steps = self.adaptation_steps
        
        # Create optimizer for adaptation
        adaptation_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(adaptation_params, lr=self.adaptation_lr)
        
        # Adaptation loop
        for step in range(num_steps):
            total_loss = 0
            
            for input_grid, target_grid in task_examples:
                # Forward pass
                output = self(input_grid.unsqueeze(0), target_grid.unsqueeze(0), mode='adaptation')
                
                # Compute adaptation loss
                pred_output = output['predicted_output']
                loss = F.cross_entropy(pred_output, target_grid.argmax(dim=0))
                
                total_loss += loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"\033[96mTest-time adaptation completed: {num_steps} steps\033[0m")


# Compatibility alias
EnhancedPrometheusV4Net = PrometheusV4Transformer