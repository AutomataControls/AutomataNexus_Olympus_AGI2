#!/usr/bin/env python3
"""
PROMETHEUS Specialized Training - Creative Pattern Generation
Following the existing OLYMPUS AGI2 training approach
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from collections import defaultdict
from tqdm import tqdm
import json

# Add src to path for imports
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')

# Model imports
from models.prometheus_model import EnhancedPrometheusNet

# Try to import training systems (use generic ones that exist)
try:
    from training_systems.mept_system import create_mept_system
    from training_systems.leap_system import create_leap_system  
    from training_systems.prism_system import create_prism_system
    from dsl.base_dsl import DSLTraining
    TRAINING_SYSTEMS_AVAILABLE = True
    print("✅ Training systems available")
except ImportError:
    TRAINING_SYSTEMS_AVAILABLE = False
    print("⚠️ Training systems not available")

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 PROMETHEUS Training on {device}")
print(f"Using device: {device}")

# PROMETHEUS Configuration
PROMETHEUS_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50,
    'epochs_per_stage': 25,
    'gradient_accumulation': 4,
    'gradient_clip': 1.0,
    'weight_decay': 1e-4,
    'transform_penalty': 0.5,
    'exact_match_bonus': 2.0,
    'vae_beta': 1.0,  # VAE loss weight
    'kl_weight': 0.1   # KL divergence weight
}

# 8-Stage Progressive Curriculum
STAGE_CONFIG = [
    {'stage': 0, 'max_grid_size': 6, 'lr_mult': 1.0},
    {'stage': 1, 'max_grid_size': 8, 'lr_mult': 0.9},
    {'stage': 2, 'max_grid_size': 10, 'lr_mult': 0.8},
    {'stage': 3, 'max_grid_size': 12, 'lr_mult': 0.7},
    {'stage': 4, 'max_grid_size': 15, 'lr_mult': 0.6},
    {'stage': 5, 'max_grid_size': 19, 'lr_mult': 0.5},
    {'stage': 6, 'max_grid_size': 25, 'lr_mult': 0.4},
    {'stage': 7, 'max_grid_size': 30, 'lr_mult': 0.3},
]

# Training flags
USE_MEPT = TRAINING_SYSTEMS_AVAILABLE
USE_LEAP = TRAINING_SYSTEMS_AVAILABLE
USE_PRISM = TRAINING_SYSTEMS_AVAILABLE
USE_DSL = TRAINING_SYSTEMS_AVAILABLE
USE_EXACT_BOOST = True

print("=" * 80)
print("PROMETHEUS Specialized Training - Creative Pattern Generation")
print("VAE-based Pattern Synthesis & Creative Generation")
print("=" * 80)


class PrometheusVAELoss(nn.Module):
    """PROMETHEUS-specific VAE loss function"""
    
    def __init__(self, base_weight=1.0, kl_weight=0.1, creativity_weight=0.2):
        super().__init__()
        self.base_weight = base_weight
        self.kl_weight = kl_weight
        self.creativity_weight = creativity_weight
        
    def forward(self, model_outputs, targets, inputs):
        
        pred_output = model_outputs['predicted_output']
        mu = model_outputs['mu']
        log_var = model_outputs['log_var']
        
        # Reconstruction loss
        recon_loss = F.cross_entropy(pred_output, targets)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / targets.size(0)
        
        # Creativity loss - encourage diverse patterns
        creativity_loss = torch.tensor(0.0, device=pred_output.device)
        if 'raw_output' in model_outputs:
            raw_output = model_outputs['raw_output']
            # Penalize too much similarity to input
            input_similarity = F.mse_loss(pred_output.softmax(dim=1), inputs)
            creativity_loss = -input_similarity  # Encourage dissimilarity for creativity
        
        total_loss = (self.base_weight * recon_loss + 
                     self.kl_weight * kl_loss + 
                     self.creativity_weight * creativity_loss)
        
        # Calculate exact matches
        pred_classes = pred_output.argmax(dim=1)
        exact_matches = (pred_classes == targets).all(dim=[-2, -1]).float()
        
        return {
            'total': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'creativity_loss': creativity_loss,
            'exact_count': exact_matches.sum(),
            'total_samples': targets.size(0)
        }


def custom_collate_fn(batch, stage):
    """Custom collate function for PROMETHEUS training"""
    inputs = []
    outputs = []
    
    target_sizes = {0: 6, 1: 8, 2: 10, 3: 12, 4: 15, 5: 19, 6: 25, 7: 30}
    target_size = target_sizes.get(stage, 12)  # PROMETHEUS max_grid_size is 12
    
    # Force target size to be reasonable for PROMETHEUS
    actual_target = min(target_size, 12)  # PROMETHEUS model can't handle grids larger than 12x12
    
    for item in batch:
        if isinstance(item, dict):
            input_grid = item['inputs']
            output_grid = item['outputs']
        else:
            if len(item) >= 2:
                input_grid, output_grid = item[0], item[1]
            else:
                continue
        
        # Convert to tensors
        if isinstance(input_grid, np.ndarray):
            input_grid = torch.from_numpy(input_grid.copy()).long()
        elif not isinstance(input_grid, torch.Tensor):
            input_grid = torch.tensor(input_grid, dtype=torch.long)
            
        if isinstance(output_grid, np.ndarray):
            output_grid = torch.from_numpy(output_grid.copy()).long()
        elif not isinstance(output_grid, torch.Tensor):
            output_grid = torch.tensor(output_grid, dtype=torch.long)
        
        # Ensure 2D
        while input_grid.dim() > 2:
            input_grid = input_grid.squeeze(0)
        while output_grid.dim() > 2:
            output_grid = output_grid.squeeze(0)
        if input_grid.dim() < 2:
            input_grid = input_grid.unsqueeze(0)
        if output_grid.dim() < 2:
            output_grid = output_grid.unsqueeze(0)
        
        # Resize to target size - all grids must be exactly actual_target size
        h, w = input_grid.shape
        if h != actual_target or w != actual_target:
            if h < actual_target or w < actual_target:
                pad_h = max(0, actual_target - h)
                pad_w = max(0, actual_target - w)
                input_grid = F.pad(input_grid, (0, pad_w, 0, pad_h), value=0)
                output_grid = F.pad(output_grid, (0, pad_w, 0, pad_h), value=0)
            elif h > actual_target or w > actual_target:
                input_grid = input_grid[:actual_target, :actual_target]
                output_grid = output_grid[:actual_target, :actual_target]
        
        inputs.append(input_grid)
        outputs.append(output_grid)
    
    if not inputs:
        empty_tensor = torch.zeros((1, actual_target, actual_target), dtype=torch.long)
        return {'inputs': empty_tensor, 'outputs': empty_tensor}
    
    # Ensure all tensors are exactly the same size before stacking
    final_inputs = []
    final_outputs = []
    
    for inp, out in zip(inputs, outputs):
        # Double-check sizes and pad if needed - everything should be actual_target size
        if inp.shape != (actual_target, actual_target):
            if inp.shape[0] < actual_target or inp.shape[1] < actual_target:
                pad_h = max(0, actual_target - inp.shape[0])
                pad_w = max(0, actual_target - inp.shape[1])
                inp = F.pad(inp, (0, pad_w, 0, pad_h), value=0)
            elif inp.shape[0] > actual_target or inp.shape[1] > actual_target:
                inp = inp[:actual_target, :actual_target]
        
        if out.shape != (actual_target, actual_target):
            if out.shape[0] < actual_target or out.shape[1] < actual_target:
                pad_h = max(0, actual_target - out.shape[0])
                pad_w = max(0, actual_target - out.shape[1])
                out = F.pad(out, (0, pad_w, 0, pad_h), value=0)
            elif out.shape[0] > actual_target or out.shape[1] > actual_target:
                out = out[:actual_target, :actual_target]
        
        final_inputs.append(inp)
        final_outputs.append(out)
    
    return {'inputs': torch.stack(final_inputs), 'outputs': torch.stack(final_outputs)}


def prometheus_exact_match_injection(model, device, num_epochs=100, target_accuracy=85.0):
    """PROMETHEUS exact match injection with creative pattern focus"""
    print("🎨 PROMETHEUS CREATIVE PATTERN INJECTION")
    print("=" * 60)
    print(f"  Target: {target_accuracy}% exact match")
    print(f"  Epochs: {num_epochs}")
    print("  Focus: Creative pattern generation")
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Generate creative patterns
    patterns = []
    
    for i in range(100):
        size = random.choice([6, 8, 10, 12])
        
        # Creative pattern types
        pattern_type = i % 5
        
        if pattern_type == 0:  # Color transformations
            input_grid = torch.randint(1, 6, (size, size))
            output_grid = (input_grid + 1) % 6 + 1  # Shift colors
        elif pattern_type == 1:  # Spatial transformations
            input_grid = torch.randint(1, 4, (size, size))
            output_grid = torch.rot90(input_grid, k=random.randint(1, 3))
        elif pattern_type == 2:  # Pattern completion
            input_grid = torch.zeros((size, size), dtype=torch.long)
            # Create partial pattern
            for x in range(size//2):
                for y in range(size//2):
                    input_grid[x, y] = random.randint(1, 4)
            # Complete pattern
            output_grid = input_grid.clone()
            output_grid[size//2:, size//2:] = input_grid[:size//2, :size//2]
        elif pattern_type == 3:  # Symmetry
            input_grid = torch.randint(1, 4, (size//2, size//2))
            # Create symmetric pattern
            full_input = torch.zeros((size, size), dtype=torch.long)
            full_input[:size//2, :size//2] = input_grid
            full_input[size//2:, :size//2] = input_grid.flip(0)
            full_input[:size//2, size//2:] = input_grid.flip(1)
            full_input[size//2:, size//2:] = input_grid.flip([0, 1])
            input_grid = full_input
            output_grid = input_grid.clone()
        else:  # Creative generation
            input_grid = torch.randint(0, 5, (size, size))
            output_grid = torch.randint(1, 6, (size, size))  # Completely different
        
        patterns.append({
            'inputs': input_grid.numpy(),
            'outputs': output_grid.numpy()
        })
    
    batch_size = 16
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        random.shuffle(patterns)
        
        correct = 0
        total = 0
        epoch_loss = 0.0
        
        for i in range(0, len(patterns), batch_size):
            batch_patterns = patterns[i:i + batch_size]
            
            if not batch_patterns:
                continue
            
            # Convert to tensors
            inputs_list = []
            outputs_list = []
            
            for p in batch_patterns:
                inp = torch.from_numpy(p['inputs']).long()
                out = torch.from_numpy(p['outputs']).long()
                inputs_list.append(inp)
                outputs_list.append(out)
            
            # Pad to same size
            max_h = max(inp.size(0) for inp in inputs_list)
            max_w = max(inp.size(1) for inp in inputs_list)
            
            padded_inputs = []
            padded_outputs = []
            for inp, out in zip(inputs_list, outputs_list):
                pad_h = max_h - inp.size(0)
                pad_w = max_w - inp.size(1)
                if pad_h > 0 or pad_w > 0:
                    inp = F.pad(inp, (0, pad_w, 0, pad_h), value=0)
                    out = F.pad(out, (0, pad_w, 0, pad_h), value=0)
                padded_inputs.append(inp)
                padded_outputs.append(out)
            
            inputs = torch.stack(padded_inputs).to(device)
            outputs = torch.stack(padded_outputs).to(device)
            
            input_oh = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
            target_oh = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            model_output = model(input_oh, target_oh, mode='inference')
            pred = model_output['predicted_output']
            
            # VAE Loss
            loss = F.cross_entropy(pred, outputs)
            if 'mu' in model_output and 'log_var' in model_output:
                mu = model_output['mu']
                log_var = model_output['log_var']
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / inputs.size(0)
                loss = loss + 0.1 * kl_loss
            
            # Check exact matches
            pred_idx = pred.argmax(dim=1)
            exact_matches = (pred_idx == outputs).all(dim=[1,2]).float()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            correct += exact_matches.sum().item()
            total += outputs.size(0)
            epoch_loss += loss.item()
        
        acc = correct / total * 100 if total > 0 else 0.0
        avg_loss = epoch_loss / max(1, len(patterns) // batch_size)
        
        if epoch % 20 == 0 or acc >= target_accuracy:
            print(f"Epoch {epoch+1}/{num_epochs}: {acc:.1f}% exact match | Loss: {avg_loss:.3f}")
        
        if acc > best_acc:
            best_acc = acc
        
        if acc >= target_accuracy:
            print(f"🏆 TARGET REACHED: {acc:.1f}% >= {target_accuracy}%")
            break
    
    print(f"📊 Final: {acc:.1f}% (best: {best_acc:.1f}%)")
    return model


def train_prometheus_specialized():
    """Main PROMETHEUS specialized training function"""
    print("🎨 Starting PROMETHEUS Specialized Training")
    print("=" * 70)
    print("📊 Creative Pattern Generation with VAE:")
    print("  • Variational Autoencoder for creative patterns")
    print("  • Rule-based pattern generation")
    print("  • Latent space exploration")
    print("  • Creative pattern synthesis")
    print("=" * 70)
    
    # Initialize model with your actual parameters
    model = EnhancedPrometheusNet(max_grid_size=12, latent_dim=128).to(device)
    
    print(f"🎨 PROMETHEUS Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize training systems if available
    systems = {}
    
    if USE_MEPT and TRAINING_SYSTEMS_AVAILABLE:
        try:
            mept_components = create_mept_system(
                capacity=20000,
                pattern_bank_size=5000,
                transformation_penalty=PROMETHEUS_CONFIG['transform_penalty'],
                exact_match_bonus=PROMETHEUS_CONFIG['exact_match_bonus']
            )
            systems['replay_buffer'] = mept_components['replay_buffer']
            systems['pattern_bank'] = mept_components['pattern_bank']
            systems['loss_fn'] = mept_components.get('loss_fn')
            print("✅ MEPT system initialized")
        except:
            print("⚠️ MEPT system failed to initialize")
    
    if USE_LEAP and TRAINING_SYSTEMS_AVAILABLE:
        try:
            leap_components = create_leap_system(device)
            systems['leap_trainer'] = leap_components['trainer']
            systems['pattern_generator'] = leap_components['pattern_generator']
            print("✅ LEAP system initialized")
        except:
            print("⚠️ LEAP system failed to initialize")
    
    if USE_PRISM and TRAINING_SYSTEMS_AVAILABLE:
        try:
            prism_components = create_prism_system()
            systems['prism_synthesizer'] = prism_components['synthesizer']
            print("✅ PRISM system initialized")
        except:
            print("⚠️ PRISM system failed to initialize")
    
    # Enhanced VAE loss function
    loss_fn = PrometheusVAELoss(
        base_weight=1.0,
        kl_weight=PROMETHEUS_CONFIG['kl_weight'],
        creativity_weight=0.2
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=PROMETHEUS_CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=PROMETHEUS_CONFIG['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    if device.type == 'cuda':
        scaler = GradScaler()
    else:
        scaler = GradScaler(enabled=False)
    
    # Check for existing best model
    models_dir = '/content/AutomataNexus_Olympus_AGI2/arc_models_v4'
    best_model_path = f'{models_dir}/prometheus_best.pt'
    
    best_exact = 0.0
    if os.path.exists(best_model_path):
        print(f"🔄 Loading best model from {best_model_path}")
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_exact = checkpoint.get('best_exact', 0.0)
            print(f"✅ Loaded best model with {best_exact:.2f}% exact match")
        except Exception as e:
            print(f"⚠️ Failed to load best model: {e}")
            print("🆕 Starting fresh training")
    else:
        print("🆕 No existing model found - starting fresh training")
    
    # 4-PHASE INJECTION
    if USE_EXACT_BOOST:
        print("\n" + "=" * 70)
        print("🎨 PROMETHEUS 4-PHASE CREATIVE INJECTION SEQUENCE")
        print("=" * 70)
        
        # Phase 1: Creative Pattern Exact Match
        print("\n🎨 PHASE 1: Creative Pattern Identity Mapping")
        model = prometheus_exact_match_injection(model, device, num_epochs=100, target_accuracy=80.0)
        
        # Phase 2: MEPT (if available)
        if USE_MEPT and 'replay_buffer' in systems:
            print("\n🎨 PHASE 2: Creative Pattern Memory (MEPT)")
            print("🎨 PROMETHEUS MEPT INJECTION")
            print("=" * 50)
            print(f"  Target: 80.0% pattern recall")
            
            # Simple MEPT injection
            for epoch in range(50):
                patterns = []
                for i in range(20):
                    size = random.choice([6, 8, 10, 12])
                    input_grid = torch.randint(1, 5, (size, size))
                    output_grid = torch.rot90(input_grid, k=1)  # Simple transformation
                    patterns.append({
                        'inputs': input_grid.numpy(),
                        'outputs': output_grid.numpy()
                    })
                
                total_recall = 0
                for pattern in patterns:
                    input_tensor = torch.from_numpy(pattern['inputs']).long().to(device)
                    output_tensor = torch.from_numpy(pattern['outputs']).long().to(device)
                    
                    systems['replay_buffer'].add(input_tensor, output_tensor)
                    
                    # Test recall
                    input_oh = F.one_hot(input_tensor.unsqueeze(0), num_classes=10).permute(0, 3, 1, 2).float()
                    target_oh = F.one_hot(output_tensor.unsqueeze(0), num_classes=10).permute(0, 3, 1, 2).float()
                    
                    with torch.no_grad():
                        pred = model(input_oh, target_oh, mode='inference')['predicted_output']
                        pred_idx = pred.argmax(dim=1)
                        exact_match = torch.equal(pred_idx[0], output_tensor)
                        total_recall += int(exact_match)
                
                recall_rate = total_recall / len(patterns) * 100
                if epoch % 10 == 0:
                    print(f"MEPT Epoch {epoch+1}/50: {recall_rate:.1f}% recall")
                
                if recall_rate >= 80.0:
                    print(f"🏆 MEPT TARGET REACHED: {recall_rate:.1f}%")
                    break
        
        print("\n✅ 4-PHASE CREATIVE INJECTION COMPLETE")
        print("=" * 70)
    
    # Data directory and proper dataset like other models
    DATA_DIR = '/content/AutomataNexus_Olympus_AGI2/data'
    
    # Import proper dataset and exact match injection like CHRONOS
    sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')
    from colab_training_v4_megascale_curriculum import CurriculumMegaScaleDataset, ExactMatchBoostDataset, inject_exact_match_training
    
    print(f"\n🎨 PROMETHEUS 8-Stage Progressive Curriculum Training")
    print("=" * 70)
    
    # 8-Stage Progressive Training (like CHRONOS)
    for stage in range(min(3, len(STAGE_CONFIG))):  # Start with first 3 stages
        stage_config = STAGE_CONFIG[stage]
        grid_size = stage_config['max_grid_size']
        
        print(f"\n🎨 PROMETHEUS Stage {stage}: {grid_size}x{grid_size} Creative Pattern Generation")
        print(f"   📏 Grid Size: {grid_size}x{grid_size}")
        print("=" * 60)
        
        # Create curriculum dataset with stage-specific grid size
        dataset = CurriculumMegaScaleDataset(
            DATA_DIR,
            curriculum_stage=min(stage, 2),  # Cap at stage 2 for compatibility
            use_arc_synthesis=True,
            synthesis_ratio=0.7  # High synthesis for creativity
        )
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=PROMETHEUS_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=PROMETHEUS_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: custom_collate_fn(batch, stage),
            drop_last=False
        )
        
        print(f"📚 Stage {stage} ({grid_size}x{grid_size}) - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        # Exact match injection for Stage 0 only
        exact_dataset = None
        if stage == 0 and USE_EXACT_BOOST:
            try:
                exact_dataset = ExactMatchBoostDataset(1300, fixed_size=grid_size)
                print(f"✅ Stage {stage} PROMETHEUS exact match injection dataset created ({grid_size}x{grid_size})")
            except Exception as e:
                print(f"⚠️ Could not create exact match dataset: {e}")
        
        # Stage training loop
        stage_epochs = PROMETHEUS_CONFIG['epochs_per_stage']
        for epoch in range(stage_epochs):
            
            # Exact match injection training (Stage 0 only, FIRST EPOCH ONLY)
            if exact_dataset and stage == 0 and epoch == 0:
                model = inject_exact_match_training(
                    model, device=device,
                    num_epochs=1,
                    target_accuracy=90.0  # High target for PROMETHEUS creative patterns
                )
                print(f"🎨 PROMETHEUS injection completed - Stage {stage}, Epoch {epoch+1}")
            model.train()
            train_metrics = defaultdict(float)
            
            pbar = tqdm(train_loader, desc=f"PROMETHEUS Stage {stage}, Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                inputs = batch['inputs'].to(device, non_blocking=True)
                outputs = batch['outputs'].to(device, non_blocking=True)
                
                inputs = torch.clamp(inputs, 0, 9)
                outputs = torch.clamp(outputs, 0, 9)
                
                input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                target_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                
                with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    model_outputs = model(input_grids, target_grids, mode='inference')
                    losses = loss_fn(model_outputs, outputs, input_grids)
                
                loss = losses['total'] / PROMETHEUS_CONFIG['gradient_accumulation']
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % PROMETHEUS_CONFIG['gradient_accumulation'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), PROMETHEUS_CONFIG['gradient_clip'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    scheduler.step()
                
                train_metrics['loss'] += losses['total'].item()
                train_metrics['exact'] += losses['exact_count'].item()
                train_metrics['samples'] += inputs.size(0)
                
                if batch_idx % 10 == 0:
                    current_exact = train_metrics['exact'] / max(1, train_metrics['samples']) * 100
                    current_loss = train_metrics['loss'] / max(1, batch_idx + 1)
                    pbar.set_postfix({'loss': f"{current_loss:.3f}", 'exact': f"{current_exact:.1f}%"})
            
            # Validation
            if epoch % 5 == 0:
                model.eval()
                val_metrics = defaultdict(float)
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        inputs = batch['inputs'].to(device)
                        outputs = batch['outputs'].to(device)
                        
                        inputs = torch.clamp(inputs, 0, 9)
                        outputs = torch.clamp(outputs, 0, 9)
                        
                        input_grids = F.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float()
                        target_grids = F.one_hot(outputs, num_classes=10).permute(0, 3, 1, 2).float()
                        
                        model_outputs = model(input_grids, target_grids, mode='inference')
                        pred_output = model_outputs['predicted_output']
                        
                        val_loss = F.cross_entropy(pred_output, outputs)
                        pred_classes = pred_output.argmax(dim=1)
                        exact_matches = (pred_classes == outputs).all(dim=[1,2]).sum()
                        pixel_matches = (pred_classes == outputs).float().mean()
                        
                        val_metrics['loss'] += val_loss.item()
                        val_metrics['exact'] += exact_matches.item()
                        val_metrics['pixel'] += pixel_matches.item()
                        val_metrics['samples'] += inputs.size(0)
                
                train_exact = train_metrics['exact'] / train_metrics['samples'] * 100
                train_loss = train_metrics['loss'] / len(train_loader)
                val_exact = val_metrics['exact'] / val_metrics['samples'] * 100
                val_loss = val_metrics['loss'] / len(val_loader)
                val_pixel = val_metrics['pixel'] / len(val_loader) * 100
                
                print(f"\n🎨 PROMETHEUS Stage {stage}, Epoch {epoch+1}:")
                print(f"   🎯 Train: {train_exact:.2f}% exact, Loss: {train_loss:.3f}")
                print(f"   🎯 Val: {val_exact:.2f}% exact, Loss: {val_loss:.3f}, Pixel: {val_pixel:.1f}%")
                
                # Save best model
                if val_exact > best_exact:
                    best_exact = val_exact
                    
                    os.makedirs(models_dir, exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_exact': best_exact,
                        'stage': stage,
                        'epoch': epoch,
                        'config': PROMETHEUS_CONFIG
                    }, best_model_path)
                    print(f"💾 New best model saved: {best_exact:.2f}% exact match")
    
    print("\n🎉 PROMETHEUS Specialized Training Complete!")
    print(f"🏆 Best Performance: {best_exact:.2f}% exact match")
    return model


if __name__ == "__main__":
    print("🚀 Starting PROMETHEUS Specialized Training...")
    model = train_prometheus_specialized()
    print("✅ Training completed successfully!")