"""
Tiny Grid Memorizer - A completely different approach for 3x3-5x5 grids
Instead of using a massive ensemble, use a lightweight pattern memorization system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class TinyGridMemorizer(nn.Module):
    """
    For tiny grids, we don't need 102M parameters!
    This model:
    1. Encodes input grids into compact representations
    2. Stores a memory bank of input->output mappings
    3. Uses nearest neighbor matching with learned similarity
    """
    
    def __init__(self, max_grid_size=5, memory_size=50000):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.memory_size = memory_size
        
        # Tiny encoder - just 1M parameters total!
        self.encoder = nn.Sequential(
            # Input: 10 channels (one-hot) -> 64 channels
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            # Global pooling to fixed size
            nn.AdaptiveAvgPool2d((3, 3)),
            # Final encoding
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
        )
        
        # Flatten to vector
        self.flatten_size = 512 * 3 * 3  # 4608
        
        # Memory banks - store encoded inputs and their outputs
        self.register_buffer('memory_inputs', torch.zeros(memory_size, self.flatten_size))
        self.register_buffer('memory_outputs', torch.zeros(memory_size, max_grid_size, max_grid_size, dtype=torch.long))
        self.register_buffer('memory_valid', torch.zeros(memory_size, dtype=torch.bool))
        self.register_buffer('memory_ptr', torch.tensor(0, dtype=torch.long))
        
        # Learned similarity metric
        self.similarity_net = nn.Sequential(
            nn.Linear(self.flatten_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Output decoder - small and simple
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, max_grid_size * max_grid_size * 10),
        )
        
    def encode_input(self, x):
        """Encode input grid to compact representation"""
        # x: B x 10 x H x W
        encoded = self.encoder(x)  # B x 512 x 3 x 3
        return encoded.flatten(1)  # B x 4608
        
    def compute_similarity(self, query, keys):
        """Compute learned similarity between query and keys"""
        # query: B x D
        # keys: N x D
        B, D = query.shape
        N = keys.shape[0]
        
        # Expand for pairwise comparison
        query_exp = query.unsqueeze(1).expand(B, N, D)  # B x N x D
        keys_exp = keys.unsqueeze(0).expand(B, N, D)    # B x N x D
        
        # Concatenate and compute similarity
        combined = torch.cat([query_exp, keys_exp], dim=2)  # B x N x 2D
        similarity = self.similarity_net(combined).squeeze(-1)  # B x N
        
        return similarity
        
    def add_to_memory(self, inputs, outputs):
        """Add new examples to memory during training"""
        B = inputs.shape[0]
        with torch.no_grad():  # Don't track gradients for memory updates
            encoded = self.encode_input(inputs)
            
            for i in range(B):
                idx = self.memory_ptr % self.memory_size
                self.memory_inputs[idx] = encoded[i].detach()
                self.memory_outputs[idx] = outputs[i].detach()
                self.memory_valid[idx] = True
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
            
    def forward(self, x, target=None, mode='eval'):
        """
        Forward pass using memory-based nearest neighbor
        """
        B, C, H, W = x.shape
        
        # Encode input
        encoded = self.encode_input(x)
        
        # During training, also add to memory
        if mode == 'train' and target is not None:
            target_grid = target.argmax(dim=1) if target.dim() == 4 else target
            # Pass already encoded features to avoid recomputing
            with torch.no_grad():
                for i in range(x.shape[0]):
                    idx = self.memory_ptr % self.memory_size
                    self.memory_inputs[idx] = encoded[i].detach()
                    self.memory_outputs[idx] = target_grid[i].detach()
                    self.memory_valid[idx] = True
                    self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
        
        # Find k nearest neighbors in memory
        k = min(50, self.memory_valid.sum().item())
        
        if k > 0:
            # Get valid memory entries
            valid_memory = self.memory_inputs[self.memory_valid]
            valid_outputs = self.memory_outputs[self.memory_valid]
            
            # Compute similarities
            similarities = self.compute_similarity(encoded, valid_memory)  # B x N
            
            # Get top-k similar patterns
            top_k_sim, top_k_idx = similarities.topk(k, dim=1)  # B x k
            
            # Weighted average of top-k outputs
            weights = F.softmax(top_k_sim * 10.0, dim=1)  # B x k, temperature=0.1
            
            # Get corresponding outputs
            selected_outputs = valid_outputs[top_k_idx]  # B x k x H x W
            
            # Convert to one-hot for weighted average
            selected_oh = F.one_hot(selected_outputs, num_classes=10).float()  # B x k x H x W x 10
            selected_oh = selected_oh.permute(0, 1, 4, 2, 3)  # B x k x 10 x H x W
            
            # Weighted sum
            weights = weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # B x k x 1 x 1 x 1
            output = (selected_oh * weights).sum(dim=1)  # B x 10 x H x W
            
            # Add residual from decoder
            decoded = self.decoder(encoded).view(B, 10, H, W)
            output = output * 0.8 + decoded * 0.2
            
        else:
            # No memory yet, use decoder only
            output = self.decoder(encoded).view(B, 10, H, W)
            
        # Ensure output is proper shape
        if output.shape[2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='nearest')
            
        return output


def train_tiny_grid_memorizer(train_loader, val_loader=None, epochs=100, device='cuda'):
    """Simple training loop for the memorizer"""
    model = TinyGridMemorizer(max_grid_size=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"💾 Tiny Grid Memorizer initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   (vs 102M for OLYMPUS!)")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs, targets, mode='train')
            
            # Compute loss
            target_idx = targets.argmax(dim=1) if targets.dim() == 4 else targets
            loss = criterion(outputs.permute(0, 2, 3, 1).reshape(-1, 10), 
                           target_idx.reshape(-1))
            
            # Compute accuracy
            pred_grid = outputs.argmax(dim=1)
            correct += (pred_grid == target_idx).all(dim=(1,2)).sum().item()
            total += inputs.shape[0]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print progress
        accuracy = correct / total * 100
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        # Early stopping if we hit good accuracy
        if accuracy > 85:
            print(f"🎯 Hit {accuracy:.2f}% accuracy! Stopping early.")
            break
            
    return model