"""
Tiny Grid Specialist - Ultra-lightweight model for 2x2 to 5x5 grids
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGridSpecialist(nn.Module):
    """Specialized model for tiny grids (2x2 to 5x5)"""
    
    def __init__(self, max_grid_size=5, hidden_dim=64):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim
        
        # Direct embedding for each position (max 25 positions for 5x5)
        self.position_embed = nn.Parameter(torch.randn(max_grid_size * max_grid_size, hidden_dim))
        
        # Simple color embedding
        self.color_embed = nn.Embedding(10, hidden_dim)
        
        # Tiny transformer - just 2 layers!
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Pattern memory bank - memorize common tiny patterns
        self.pattern_memory = nn.Parameter(torch.randn(100, max_grid_size * max_grid_size * hidden_dim))
        
        # Direct output prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # 10 colors
        )
        
    def forward(self, input_grid, target_grid=None):
        B, C, H, W = input_grid.shape
        
        # Flatten and embed
        input_flat = input_grid.argmax(dim=1).reshape(B, -1)  # B x (H*W)
        color_embeds = self.color_embed(input_flat)  # B x (H*W) x hidden_dim
        
        # Add position embeddings
        positions = self.position_embed[:H*W].unsqueeze(0).expand(B, -1, -1)
        x = color_embeds + positions
        
        # Apply tiny transformer
        x = self.transformer(x)  # B x (H*W) x hidden_dim
        
        # Compare with pattern memory
        x_flat = x.reshape(B, -1)  # B x (H*W*hidden_dim)
        pattern_scores = F.cosine_similarity(
            x_flat.unsqueeze(1),  # B x 1 x (H*W*hidden_dim)
            self.pattern_memory.unsqueeze(0),  # 1 x 100 x (H*W*hidden_dim)
            dim=2
        )  # B x 100
        
        # Use top patterns to influence output
        top_k = 5
        top_scores, top_indices = pattern_scores.topk(top_k, dim=1)
        top_patterns = self.pattern_memory[top_indices]  # B x top_k x (H*W*hidden_dim)
        
        # Weighted average of top patterns
        weights = F.softmax(top_scores, dim=1).unsqueeze(-1)  # B x top_k x 1
        pattern_influence = (top_patterns * weights).sum(dim=1)  # B x (H*W*hidden_dim)
        pattern_influence = pattern_influence.reshape(B, H*W, self.hidden_dim)
        
        # Combine transformer output with pattern memory
        x = x + 0.5 * pattern_influence
        
        # Predict output for each position
        output = self.output_head(x)  # B x (H*W) x 10
        output = output.reshape(B, H, W, 10).permute(0, 3, 1, 2)  # B x 10 x H x W
        
        return output


class TinyGridLoss(nn.Module):
    """Specialized loss for tiny grids"""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred, target):
        # Simple cross-entropy - no fancy stuff for tiny grids
        B, C, H, W = pred.shape
        
        # Get target indices
        if target.dim() == 4:  # One-hot
            target_idx = target.argmax(dim=1)
        else:
            target_idx = target
            
        # Reshape for loss
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target_idx.reshape(-1)
        
        # Basic CE loss
        ce_loss = self.ce_loss(pred_flat, target_flat)
        
        # Exact match bonus - HUGE for tiny grids
        pred_grid = pred.argmax(dim=1)
        exact_match = (pred_grid == target_idx).all(dim=(1,2)).float().mean()
        exact_bonus = exact_match * 100.0  # Massive bonus
        
        total_loss = ce_loss - exact_bonus
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'exact_match': exact_match,
            'exact_bonus': exact_bonus
        }