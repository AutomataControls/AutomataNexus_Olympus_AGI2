"""
OLYMPUS Ensemble Training V3.2 - ADVANCED CHECKPOINTING & OPTIMIZATIONS
Ultimate ensemble training with robust, versioned checkpointing and explicit user control.
Builds upon V3.1 with a professional-grade training pipeline.
Target: 95%+ performance with ultimate ensemble mastery
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
import numpy as np
import json
import os
import sys
import gc
import time
import re
import glob
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import random
from collections import defaultdict
import argparse
import warnings

# --- OPTIMIZATION: Enable CuDNN benchmarking for static input sizes ---
torch.backends.cudnn.benchmark = True

# Suppress the specific scheduler warning for OneCycleLR
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/scripts/training')

# Import OLYMPUS ensemble
from src.models.olympus_ensemble import OlympusEnsemble, EnsembleDecision

# OLYMPUS V3 Configuration (remains the same)
OLYMPUS_V3_CONFIG = {
    'batch_size': 512, 'learning_rate': 0.0002, 'num_epochs': 50, 'gradient_accumulation': 1,
    'epochs_per_stage': 3, 'curriculum_stages': 15, 'ensemble_loss_weight': 2.5, 'specialist_sync_weight': 0.1,
    'consensus_weight': 0.0, 'fusion_regularization': 0.1, 'transform_penalty': 0.01, 'exact_match_bonus': 2.0,
    'gradient_clip': 0.5, 'weight_decay': 2e-6, 'ultra_teal_iou_weight': 0.85, 'strict_match_weight': 0.15,
    'freeze_specialists': False, 'fusion_training_only': False, 'specialist_learning_rate': 5e-05,
    'consensus_threshold': 0.6, 'specialist_dropout': 0.2, 'ensemble_coordination': True, 'adaptive_weights': True,
    'meta_ensemble_learning': True, 'label_smoothing': 0.01, 'ensemble_diversity_bonus': True,
    'specialist_agreement_bonus': True, 'consensus_building_bonus': True, 'fusion_optimization': True,
    'advanced_meta_learning': True, 'cross_specialist_attention': True, 'dynamic_fusion_weights': True,
    'ultimate_coordination': True, 'ensemble_self_attention': True, 'adaptive_curriculum': True,
    'ultimate_fusion_networks': True, 'warmup_epochs': 7, 'cosine_restarts': True, 'restart_multiplier': 1.0,
    'plateau_patience': 10, 'lr_cycle_mult': 2.0, 'min_lr_ratio': 0.001
}

# STAGE_CONFIG (remains the same)
STAGE_CONFIG = [
    {'stage': 0, 'max_grid_size': 3, 'focus': 'ultimate_tiny_grid_basic_transformations'},
    {'stage': 1, 'max_grid_size': 4, 'focus': 'ultimate_micro_grid_specialist_coordination'},
    {'stage': 2, 'max_grid_size': 5, 'focus': 'ultimate_ensemble_shape_coordination'},
    {'stage': 3, 'max_grid_size': 6, 'focus': 'ultimate_decision_fusion_learning'},
    {'stage': 4, 'max_grid_size': 7, 'focus': 'ultimate_pattern_synchronization_training'},
    {'stage': 5, 'max_grid_size': 8, 'focus': 'ultimate_specialist_consensus'},
    {'stage': 6, 'max_grid_size': 9, 'focus': 'ultimate_intermediate_fusion_protocols'},
    {'stage': 7, 'max_grid_size': 10, 'focus': 'ultimate_composite_ensemble_decisions'},
    {'stage': 8, 'max_grid_size': 11, 'focus': 'ultimate_scaling_coordination_protocols'},
    {'stage': 9, 'max_grid_size': 12, 'focus': 'ultimate_complex_consensus_building'},
    {'stage': 10, 'max_grid_size': 14, 'focus': 'ultimate_pattern_ensemble_coordination'},
    {'stage': 11, 'max_grid_size': 16, 'focus': 'ultimate_ensemble_intelligence_emergence'},
    {'stage': 12, 'max_grid_size': 18, 'focus': 'ultimate_multiscale_ensemble_reasoning'},
    {'stage': 13, 'max_grid_size': 22, 'focus': 'ultimate_coordination_protocols_mastery'},
    {'stage': 14, 'max_grid_size': 27, 'focus': 'ultimate_ensemble_coordination_mastery'},
    {'stage': 15, 'max_grid_size': 30, 'focus': 'ultimate_olympus_god_intelligence_mastery'}
]


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/checkpoints_v3'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


print(f"\033[96m{'=' * 130}\033[0m")
print(f"\033[96m🏛️ OLYMPUS Ensemble Training V3.2 (Advanced Checkpointing & Optimizations)\033[0m")
print(f"\033[96mTarget: 95%+ Performance with Ultimate Ensemble Mastery\033[0m")
print(f"\033[96m{'=' * 130}\033[0m")

# The optimized OlympusV3Loss, OlympusV3UltimateDataset, etc. from the previous version remain unchanged.
# For brevity, I will omit their code here, assuming they are defined as in the v3.1 script.
# ... (Paste OlympusV3Loss and OlympusV3UltimateDataset class definitions here) ...
# (Pasting them back in for completeness)

class OlympusV3Loss(nn.Module):
    """Ultimate loss function for OLYMPUS ensemble V3 training"""
    def __init__(self, config):
        super().__init__()
        self.config = config; self.ensemble_weight = config['ensemble_loss_weight']; self.sync_weight = config['specialist_sync_weight']
        self.consensus_weight = config['consensus_weight']; self.fusion_reg_weight = config['fusion_regularization']
        self.transform_penalty = config['transform_penalty']; self.exact_match_bonus = config['exact_match_bonus']
        self.label_smoothing = config['label_smoothing']; self.focal_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.meta_learning_weight = 0.35; self.cross_attention_weight = 0.3; self.adaptive_weight_bonus = 0.25
        self.self_attention_weight = 0.2; self.ultimate_coordination_weight = 0.15
    def forward(self, ensemble_decision: EnsembleDecision, targets: torch.Tensor, inputs: torch.Tensor, stage_idx: int) -> Dict:
        pred_output = ensemble_decision.prediction; B = pred_output.shape[0]
        target_indices = targets.argmax(dim=1) if targets.dim() > 3 else targets.view(B, -1)
        pred_flat = pred_output.view(B, -1); target_flat = target_indices.view(B, -1) if target_indices.dim() > 1 else target_indices
        if pred_flat.shape[1] > 10: pred_flat = pred_flat[:, :10]
        if target_flat.shape[1] > 1: target_flat = target_flat[:, 0]
        ensemble_loss = self.focal_loss(pred_flat, target_flat.long())
        pred_classes = pred_flat.argmax(dim=1); exact_matches = (pred_classes == target_flat).float()
        exact_count = exact_matches.sum(); exact_bonus = -exact_matches.mean() * self.exact_match_bonus
        specialist_predictions = ensemble_decision.specialist_predictions
        sync_loss = torch.tensor(0.0, device=pred_output.device); cross_attention_loss = torch.tensor(0.0, device=pred_output.device); self_attention_loss = torch.tensor(0.0, device=pred_output.device)
        if len(specialist_predictions) > 1:
            try:
                preds_reshaped = [p.view(B, -1)[:, :10] for p in specialist_predictions.values()]; pred_stack = torch.stack(preds_reshaped, dim=1)
                p1 = pred_stack.unsqueeze(2); p2 = pred_stack.unsqueeze(1); mse_pairwise = F.mse_loss(p1, p2, reduction='none').mean(dim=-1)
                sync_loss = torch.triu(mse_pairwise.mean(dim=0), diagonal=1).sum()
                if stage_idx > 3:
                    attention_scores = F.softmax(torch.matmul(pred_stack, pred_stack.transpose(-1, -2)), dim=-1)
                    cross_attention_loss = -torch.log(torch.diagonal(attention_scores, dim1=-2, dim2=-1) + 1e-8).mean()
                    attended_predictions = torch.matmul(attention_scores, pred_stack); self_attention_loss = F.mse_loss(attended_predictions, pred_stack)
            except (RuntimeError, IndexError): pass
        consensus_score = ensemble_decision.consensus_score; consensus_bonus = -torch.tensor(consensus_score, device=pred_output.device) * self.consensus_weight
        fusion_weights = list(ensemble_decision.fusion_weights.values()); fusion_reg = torch.tensor(0.0, device=pred_output.device); adaptive_weight_loss = torch.tensor(0.0, device=pred_output.device); ultimate_coordination_loss = torch.tensor(0.0, device=pred_output.device)
        if len(fusion_weights) > 1:
            fusion_tensor = torch.tensor(fusion_weights, device=pred_output.device); fusion_entropy = -(fusion_tensor * torch.log(fusion_tensor + 1e-8)).sum(); fusion_reg = -fusion_entropy * self.fusion_reg_weight
            if stage_idx > 5:
                target_distribution = torch.ones_like(fusion_tensor) / len(fusion_weights); weight_kl_div = F.kl_div(F.log_softmax(fusion_tensor, dim=0), target_distribution, reduction='sum'); ultimate_coordination_loss = weight_kl_div * self.ultimate_coordination_weight
                weight_variance = fusion_tensor.var(); adaptive_weight_loss = weight_variance * self.adaptive_weight_bonus
        meta_learning_bonus = torch.tensor(0.0, device=pred_output.device)
        if hasattr(ensemble_decision, 'meta_features') and ensemble_decision.meta_features is not None and stage_idx > 1:
            meta_entropy = -(F.softmax(ensemble_decision.meta_features, dim=-1) * F.log_softmax(ensemble_decision.meta_features, dim=-1)).sum(dim=-1).mean(); meta_learning_bonus = -meta_entropy * self.meta_learning_weight
        if inputs.dim() > 1:
            input_flat = inputs.view(B, -1)[:, :10] if inputs.numel() > B*10 else inputs.view(B, -1); copy_penalty = F.mse_loss(pred_flat, input_flat) * self.transform_penalty
        else: copy_penalty = torch.tensor(0.0, device=pred_output.device)
        total_loss = (ensemble_loss + exact_bonus + sync_loss * self.sync_weight + consensus_bonus + fusion_reg + copy_penalty + cross_attention_loss * self.cross_attention_weight + adaptive_weight_loss + meta_learning_bonus + self_attention_loss * self.self_attention_weight + ultimate_coordination_loss)
        return {'total': total_loss, 'ensemble': ensemble_loss, 'sync': sync_loss * self.sync_weight, 'consensus_bonus': consensus_bonus, 'fusion_reg': fusion_reg, 'exact_bonus': exact_bonus, 'copy_penalty': copy_penalty, 'cross_attention': cross_attention_loss * self.cross_attention_weight, 'adaptive_weights': adaptive_weight_loss, 'meta_learning': meta_learning_bonus, 'self_attention': self_attention_loss * self.self_attention_weight, 'ultimate_coordination': ultimate_coordination_loss, 'exact_count': exact_count, 'consensus_score': consensus_score}

from train_olympus_ensemble_v2 import OlympusV2AugmentedDataset, olympus_v2_augmented_collate_fn as foundation_collate_fn
class OlympusV3UltimateDataset(OlympusV2AugmentedDataset):
    def __init__(self, data_dir: str, max_grid_size: int, stage_config: Dict, augmentation_factor: int = 6):
        super().__init__(data_dir, max_grid_size, stage_config, augmentation_factor)
        if max_grid_size <= 5: self._add_synthetic_tiny_grid_samples()
    def _add_synthetic_tiny_grid_samples(self):
        original_count = len(self.samples); target_samples = 5000
        patterns_map = { 3: [([[0,0,0],[0,1,0],[0,0,0]], [[1,1,1],[1,0,1],[1,1,1]])], 4: [([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]], [[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]])], 5: [([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]], [[1,1,0,1,1],[1,1,0,1,1],[0,0,0,0,0],[1,1,0,1,1],[1,1,0,1,1]])] }
        patterns = patterns_map.get(self.max_grid_size, [])
        if not patterns: return
        while len(self.samples) < target_samples:
            inp, out = random.choice(patterns); variation_type = random.randint(0, 4); inp_arr, out_arr = np.array(inp), np.array(out)
            if variation_type == 1: k = np.random.randint(1, 4); inp_arr, out_arr = np.rot90(inp_arr, k), np.rot90(out_arr, k)
            elif variation_type == 2: axis = np.random.randint(0, 2); inp_arr, out_arr = np.flip(inp_arr, axis), np.flip(out_arr, axis)
            self.samples.append({'input': inp_arr, 'output': out_arr, 'is_arc': True, 'complexity': self.stage_config.get('complexity', 'ensemble')})


def find_latest_checkpoint():
    """Finds the checkpoint from the highest stage in the checkpoint directory."""
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "olympus_v3_stage_*.pt"))
    if not checkpoints:
        return None, -1, 0.0

    latest_stage = -1
    best_perf = 0.0
    latest_checkpoint = None

    for ckpt in checkpoints:
        match = re.search(r"stage_(\d+)_perf_([\d.]+)\.pt", os.path.basename(ckpt))
        if match:
            stage = int(match.group(1))
            perf = float(match.group(2))
            if stage > latest_stage:
                latest_stage = stage
                best_perf = perf
                latest_checkpoint = ckpt
            elif stage == latest_stage and perf > best_perf:
                best_perf = perf
                latest_checkpoint = ckpt
                
    return latest_checkpoint, latest_stage, best_perf

def train_olympus_ensemble_v3(args):
    """Main training function for OLYMPUS Ensemble V3"""
    stage_start, stage_end = args.start_stage, args.end_stage
    print(f"\033[96m🏛️ Initializing OLYMPUS Ensemble V3 (Advanced Checkpointing) Training...\033[0m")
    
    torch.cuda.empty_cache(); gc.collect()
    
    olympus = OlympusEnsemble(max_grid_size=30, d_model=256, device=device).to(device)
    if torch.__version__.startswith('2'):
        print("\033[92m🔥 PyTorch 2.x detected. Activating torch.compile() for MAXIMUM SPEED!\033[0m")
        olympus = torch.compile(olympus)

    # --- ENHANCEMENT: Consolidated Optimizer Creation ---
    # This must be done before loading a checkpoint to load optimizer state
    param_groups = [{'params': olympus.fusion_engine.parameters(), 'name': 'fusion'}]
    for idx, (name, spec) in enumerate(olympus.specialists.items()):
        param_groups.append({'params': spec.parameters(), 'name': f'{name}'})
    optimizer = optim.AdamW(param_groups, lr=OLYMPUS_V3_CONFIG['learning_rate'], weight_decay=OLYMPUS_V3_CONFIG['weight_decay'])
    scaler = GradScaler()
    
    best_performance = 0.0
    
    # --- ENHANCEMENT: Advanced Checkpoint Loading Logic ---
    if not args.no_resume:
        checkpoint_to_load = None
        if args.resume_from:
            if os.path.exists(args.resume_from):
                checkpoint_to_load = args.resume_from
                print(f"\033[92m🏛️ User specified checkpoint found. Loading from: {checkpoint_to_load}\033[0m")
            else:
                print(f"\033[93m⚠️ User specified checkpoint not found at: {args.resume_from}. Searching for latest.\033[0m")
        
        if not checkpoint_to_load:
            checkpoint_to_load, last_stage, last_perf = find_latest_checkpoint()
            if checkpoint_to_load:
                print(f"\033[92m🏛️ Auto-resuming from latest checkpoint: {os.path.basename(checkpoint_to_load)} (Stage {last_stage}, Perf {last_perf:.2%})\033[0m")
                # Adjust start stage to continue from where we left off
                if stage_start <= last_stage:
                    stage_start = last_stage + 1
                    print(f"\033[93m   Adjusting start stage to {stage_start} to avoid re-training.\033[0m")

        if checkpoint_to_load:
            try:
                checkpoint = torch.load(checkpoint_to_load, map_location=device)
                # Use a compatible model state dict loading
                model_state_dict = olympus.state_dict()
                compatible_state_dict = {k: v for k, v in checkpoint['ensemble_state_dict'].items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                model_state_dict.update(compatible_state_dict)
                olympus.load_state_dict(model_state_dict)
                
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                best_performance = checkpoint.get('best_performance', 0.0)
                print(f"\033[92m   Successfully loaded model and optimizer state. Best known performance: {best_performance:.2%}\033[0m")
            except Exception as e:
                print(f"\033[91m⚠️ Failed to load checkpoint: {e}. Starting with fresh model weights.\033[0m")
        else:
            print(f"\033[96m🏛️ No V3 checkpoint found. Starting with a fresh model or V2 weights.\033[0m")
            # Fallback to V2/V1 can be added here if needed
    else:
        print(f"\033[91m🔥 --no-resume flag set. Starting training from scratch.\033[0m")


    global_epoch_counter = 0
    print(f"\033[96m🏛️ Starting Training from Stage {stage_start} to {stage_end}\033[0m")
    
    for stage_idx in range(stage_start, stage_end + 1):
        # ... (The per-stage setup logic from v3.1 is largely the same) ...
        stage_config = STAGE_CONFIG[stage_idx]
        print(f"\n\033[96m{'=' * 135}\033[0m"); print(f"\033[38;2;255;204;153m🏛️ V3 Opt Stage {stage_idx}: Grid Size {stage_config['max_grid_size']} | Focus: {stage_config['focus']}\033[0m"); print(f"\033[96m{'=' * 135}\033[0m")
        if stage_config['max_grid_size'] <= 5: aug_factor, epoch_mult, bs = 10, 4.0, 1024
        elif stage_config['max_grid_size'] <= 8: aug_factor, epoch_mult, bs = 8, 3.0, 512
        elif stage_config['max_grid_size'] <= 16: aug_factor, epoch_mult, bs = 6, 2.0, 256
        else: aug_factor, epoch_mult, bs = 4, 1.0, 64
        stage_epochs = int(OLYMPUS_V3_CONFIG['epochs_per_stage'] * epoch_mult)
        lr_mult = 2.0 if stage_config['max_grid_size'] <= 6 else 1.0
        for pg in optimizer.param_groups:
            if 'fusion' in pg['name']: pg['lr'] = OLYMPUS_V3_CONFIG['learning_rate'] * lr_mult
            else: pg['lr'] = OLYMPUS_V3_CONFIG['specialist_learning_rate'] * lr_mult
        print(f"\033[96m🏛️ Stage {stage_idx}: Batch={bs}, Epochs={stage_epochs}, LR_mult={lr_mult}x, Aug={aug_factor}x\033[0m")
        dataset = OlympusV3UltimateDataset(data_dir='/content/AutomataNexus_Olympus_AGI2/data', max_grid_size=stage_config['max_grid_size'], stage_config=stage_config, augmentation_factor=aug_factor)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=foundation_collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
        acc_steps = 4 if stage_config['max_grid_size'] <= 8 else (8 if stage_config['max_grid_size'] >= 22 else 6)
        max_lrs = [pg['lr'] * 3 for pg in optimizer.param_groups]; total_steps = ((len(dataloader) + acc_steps - 1) // acc_steps) * stage_epochs
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, total_steps=total_steps, pct_start=0.3)
        criterion = OlympusV3Loss(OLYMPUS_V3_CONFIG)

        stage_performance = train_ultimate_mastery_stage(
            olympus, dataloader, criterion, optimizer, scheduler, scaler, 
            stage_idx, stage_config, stage_epochs, acc_steps, global_epoch_counter
        )
        global_epoch_counter += stage_epochs

        # --- ENHANCEMENT: Versioned Checkpoint Saving ---
        if stage_performance > best_performance:
            best_performance = stage_performance
            
            # 1. Save the versioned checkpoint
            versioned_filename = f"olympus_v3_stage_{stage_idx}_perf_{best_performance:.4f}.pt"
            versioned_save_path = os.path.join(CHECKPOINT_DIR, versioned_filename)
            
            # 2. Also save a "best" file for convenience
            best_save_path = os.path.join(CHECKPOINT_DIR, "olympus_v3_best.pt")

            ensemble_state = {
                'ensemble_state_dict': olympus.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_performance': best_performance,
                'stage': stage_idx,
            }
            torch.save(ensemble_state, versioned_save_path)
            torch.save(ensemble_state, best_save_path) # Overwrite the general 'best' file
            print(f"\033[96m🏛️ New best performance: {best_performance:.2%}! Saved checkpoint to:\n   {versioned_filename}\033[0m")

        torch.cuda.empty_cache(); gc.collect()
    
    print(f"\n\033[96m{'=' * 140}\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Training Complete! Overall Best Performance: {best_performance:.2%}\033[0m")
    print(f"\033[96m{'=' * 140}\033[0m")

# The train_ultimate_mastery_stage function from v3.1 remains the same
# For brevity, I will omit its code here, assuming it is defined as in the v3.1 script.
# (Pasting it back in for completeness)
def train_ultimate_mastery_stage(olympus, dataloader, criterion, optimizer, scheduler, scaler, stage_idx, stage_config, stage_epochs, accumulation_steps=1, global_epoch_counter=0):
    olympus.train(); best_stage_performance = 0.0
    for epoch in range(stage_epochs):
        epoch_losses = defaultdict(float); total_exact_matches = 0; total_samples = 0
        stage_focus = stage_config['focus'].replace('_', ' ').title()
        pbar = tqdm(dataloader, desc=f"\033[38;2;255;204;153m🏛️ {stage_focus} Stage {stage_idx} Epoch {epoch}\033[0m")
        for batch_idx, (inputs, targets, metadata) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                ensemble_decision = olympus(inputs, targets, mode='train'); loss_dict = criterion(ensemble_decision, targets, inputs, stage_idx)
                loss = loss_dict['total'] / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(olympus.parameters(), OLYMPUS_V3_CONFIG['gradient_clip'])
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            for key, value in loss_dict.items():
                if torch.is_tensor(value): epoch_losses[key] += value.item()
                else: epoch_losses[key] += value
            total_exact_matches += loss_dict['exact_count'].item(); total_samples += inputs.size(0)
            current_performance = total_exact_matches / max(total_samples, 1) * 100
            pbar.set_postfix({'Loss': f"{loss_dict['total'].item():.4f}", 'Perf': f"{current_performance:.1f}%", 'LR': f"{optimizer.param_groups[0]['lr']:.2e}"})
        epoch_performance = total_exact_matches / total_samples
        best_stage_performance = max(best_stage_performance, epoch_performance)
        if epoch % 2 == 0 or epoch == stage_epochs - 1:
            avg_loss = epoch_losses['total'] / len(dataloader)
            print(f"\033[38;2;255;204;153m⏰ OLYMPUS V3 Stage {stage_idx}, Epoch {epoch} (Global: {global_epoch_counter + epoch + 1}): Perf: {epoch_performance:.2%}, Loss: {avg_loss:.4f}\033[0m")
    print(f"\033[96m✅ Ultimate Stage {stage_idx} complete! Best exact: {best_stage_performance:.2%}\033[0m")
    return best_stage_performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train OLYMPUS V3 Ensemble (v3.2) with Advanced Checkpointing')
    
    # --- ENHANCEMENT: Add new checkpoint control arguments ---
    parser.add_argument('--no-resume', action='store_true', help='Start training from scratch, ignoring any existing checkpoints.')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to a specific checkpoint file to resume training from.')

    parser.add_argument('--lower-stages-only', action='store_true', help='Convenience flag to train stages 0-5')
    parser.add_argument('--tiny-grids-only', action='store_true', help='Convenience flag to train stages 0-2')
    parser.add_argument('--upper-stages-only', action='store_true', help='Convenience flag to train stages 6-15')
    parser.add_argument('--start-stage', type=int, default=0, help='Start from specific stage (0-15)')
    parser.add_argument('--end-stage', type=int, default=15, help='End at specific stage (0-15)')
    args = parser.parse_args()
    
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    
    # Let convenience flags override start/end stages
    if args.tiny_grids_only: args.start_stage, args.end_stage = 0, 2
    elif args.lower_stages_only: args.start_stage, args.end_stage = 0, 5
    elif args.upper_stages_only: args.start_stage, args.end_stage = 6, 15
        
    train_olympus_ensemble_v3(args)