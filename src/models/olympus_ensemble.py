"""
OLYMPUS Ensemble - Ultimate AGI2 System for ARC Challenge
All 5 specialists process every problem → Advanced fusion → Best solution
MINERVA + ATLAS + IRIS + CHRONOS + PROMETHEUS = OLYMPUS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import defaultdict
import os

# Import actual enhanced specialist models that match your checkpoints
from .minerva_v6_enhanced import MinervaV6Enhanced      # MINERVA V6 Enhanced
from .atlas_v5_enhanced import AtlasV5Enhanced          # ATLAS V5 Enhanced  
from .iris_v6_enhanced import IrisV6Enhanced            # IRIS V6 Enhanced
from .chronos_v4_enhanced import ChronosV4Enhanced      # CHRONOS V4 Enhanced (stays V4)
from .prometheus_v6_enhanced import PrometheusV6Enhanced # PROMETHEUS V6 Enhanced


class EnsembleDecision:
    """Container for OLYMPUS ensemble decision with full metadata"""
    def __init__(self, prediction: torch.Tensor, confidence: float, 
                 specialist_predictions: Dict[str, torch.Tensor],
                 specialist_confidences: Dict[str, float],
                 fusion_weights: Dict[str, float],
                 consensus_score: float,
                 **kwargs): # Allow for additional metadata like meta_features
        self.prediction = prediction
        self.confidence = confidence
        self.specialist_predictions = specialist_predictions
        self.specialist_confidences = specialist_confidences
        self.fusion_weights = fusion_weights
        self.consensus_score = consensus_score
        self.metadata = {}
        # Store any additional attributes passed during training
        for key, value in kwargs.items():
            setattr(self, key, value)


class DecisionFusionEngine(nn.Module):
    """Advanced decision fusion combining all 5 specialist outputs"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.num_specialists = 5
        
        # Enhanced confidence analysis network
        self.confidence_analyzer = nn.Sequential(
            nn.Linear(self.num_specialists, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.num_specialists), nn.Softmax(dim=-1)
        )
        
        # Grid-size adaptive weighting
        self.grid_size_adapter = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, self.num_specialists), nn.Softmax(dim=-1)
        )
        
        # Specialist expertise router based on problem complexity
        self.expertise_router = nn.Sequential(
            nn.Linear(self.num_specialists * 2, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.num_specialists), nn.Softmax(dim=-1)
        )
        
        # Fixed-size adaptive networks for proper state saving/loading
        max_feature_size = 30 * 30 * 10 * self.num_specialists
        
        # Prediction similarity analyzer (fixed size)
        self.similarity_network = nn.Sequential(
            nn.Linear(max_feature_size + self.num_specialists, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        
        # Meta-fusion network for final decision (fixed size)
        self.meta_fusion = nn.Sequential(
            nn.Linear(max_feature_size + self.num_specialists + 1, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10), nn.Softmax(dim=-1)
        )
        
        self.max_feature_size = max_feature_size
        
    def calculate_iou_scores(self, predictions: List[torch.Tensor], target: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = predictions[0].shape[0]
        iou_scores = torch.zeros(len(predictions), batch_size, device=predictions[0].device)
        
        for i, pred in enumerate(predictions):
            pred_indices = pred.argmax(dim=1)
            
            if target is not None:
                target_indices = target.argmax(dim=1) if target.dim() > 3 else target
                intersection = (pred_indices == target_indices).float().sum(dim=[1,2])
                union = float(pred_indices.shape[1] * pred_indices.shape[2])
                iou_scores[i] = intersection / union
            else:
                for j, other_pred in enumerate(predictions):
                    if i != j:
                        other_indices = other_pred.argmax(dim=1)
                        intersection = (pred_indices == other_indices).float().sum(dim=[1,2])
                        union = float(pred_indices.shape[1] * pred_indices.shape[2])
                        iou_scores[i] += intersection / union
                iou_scores[i] /= (len(predictions) - 1)
        
        return iou_scores.transpose(0, 1)
    
    def _pad_features_to_max_size(self, features: torch.Tensor) -> torch.Tensor:
        current_size = features.shape[1]
        if current_size < self.max_feature_size:
            padding = torch.zeros(features.shape[0], self.max_feature_size - current_size, 
                                device=features.device, dtype=features.dtype)
            return torch.cat([features, padding], dim=1)
        elif current_size > self.max_feature_size:
            return features[:, :self.max_feature_size]
        return features
    
    # --- MODIFIED: ACCEPTS A DICTIONARY OF TENSORS FOR CONFIDENCES ---
    def forward(self, specialist_predictions: Dict[str, torch.Tensor],
                specialist_confidences_tensors: Dict[str, torch.Tensor],
                target: Optional[torch.Tensor] = None) -> EnsembleDecision:
        """Fuse all specialist predictions into final OLYMPUS decision"""
        
        specialist_names = ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']
        predictions = []
        confidences_tensors_list = []
        
        ref_device = list(specialist_predictions.values())[0].device
        
        for name in specialist_names:
            pred = specialist_predictions.get(name, torch.zeros_like(list(specialist_predictions.values())[0]))
            conf = specialist_confidences_tensors.get(name, torch.tensor(0.5, device=ref_device))
            predictions.append(pred)
            confidences_tensors_list.append(conf)
        
        stacked_predictions = torch.stack(predictions, dim=0)
        # --- FIX: Efficiently stack the list of tensors directly ---
        confidence_tensor = torch.stack(confidences_tensors_list).unsqueeze(0)  # Shape: [1, 5]
        batch_size = predictions[0].shape[0]
        
        grid_size = float(max(predictions[0].shape[-2:]))
        grid_size_tensor = torch.tensor([grid_size], dtype=torch.float32, device=ref_device).unsqueeze(0)
        
        flat_predictions = stacked_predictions.transpose(0, 1).reshape(batch_size, -1)
        padded_predictions = self._pad_features_to_max_size(flat_predictions)
        
        iou_scores = self.calculate_iou_scores(predictions, target)
        
        base_fusion_weights = self.confidence_analyzer(confidence_tensor).expand(batch_size, -1)
        grid_weights = self.grid_size_adapter(grid_size_tensor).expand(batch_size, -1)
        
        router_input = torch.cat([confidence_tensor.expand(batch_size, -1), grid_weights], dim=1)
        expertise_weights = self.expertise_router(router_input)
        
        fusion_weights = base_fusion_weights * grid_weights * expertise_weights
        combined_weights = F.softmax(fusion_weights * iou_scores, dim=-1)
        
        consensus_input = torch.cat([padded_predictions, confidence_tensor.expand(batch_size, -1)], dim=1)
        consensus_score_tensor = self.similarity_network(consensus_input) # Shape: [batch, 1]
        
        meta_input = torch.cat([padded_predictions, combined_weights, consensus_score_tensor], dim=1)
        final_prediction = self.meta_fusion(meta_input)
        
        weighted_confidences = torch.sum(combined_weights * confidence_tensor.expand(batch_size, -1), dim=1, keepdim=True)
        final_confidence_tensor = (weighted_confidences * consensus_score_tensor).mean()
        
        # --- FINAL CONVERSION TO FLOATS (SAFE TO DO HERE) ---
        final_confidence = final_confidence_tensor.item()
        consensus_score = consensus_score_tensor.mean().item()
        
        specialist_conf_dict = {name: conf.item() for name, conf in specialist_confidences_tensors.items()}
        fusion_weights_dict = {name: combined_weights[:, i].mean().item() for i, name in enumerate(specialist_names)}
        
        return EnsembleDecision(
            prediction=final_prediction,
            confidence=final_confidence,
            specialist_predictions=specialist_predictions,
            specialist_confidences=specialist_conf_dict,
            fusion_weights=fusion_weights_dict,
            consensus_score=consensus_score,
            meta_features=meta_input # Pass meta features for loss calculation in training
        )


class OlympusEnsemble(nn.Module):
    """OLYMPUS - All specialists process every problem for ultimate performance"""
    
    def __init__(self, max_grid_size: int = 30, d_model: int = 256, device: str = 'cuda'):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.device_name = device
        
        print(f"\033[96m🏛️ Initializing OLYMPUS Ensemble - Ultimate AGI2 System\033[0m")
        
        self.specialists = nn.ModuleDict({
            'minerva': MinervaV6Enhanced(max_grid_size, d_model, preserve_weights=True),
            'atlas': AtlasV5Enhanced(max_grid_size, d_model, 2, preserve_weights=True),
            'iris': IrisV6Enhanced(max_grid_size, d_model, 3, preserve_weights=True),
            'chronos': ChronosV4Enhanced(max_grid_size, d_model, 8, preserve_weights=True),
            'prometheus': PrometheusV6Enhanced(max_grid_size, d_model, 8, preserve_weights=True)
        })
        
        self.fusion_engine = DecisionFusionEngine(d_model)
        self.ensemble_performance = []
        self.specialist_performance = defaultdict(list)
        self.decision_history = []
        
        self.logger = logging.getLogger('OLYMPUS')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('🏛️ OLYMPUS: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\033[96m🏛️ OLYMPUS initialized with {total_params:,} total parameters across all specialists\033[0m")
        
    def load_all_specialists(self, weight_dir: str) -> Dict[str, bool]:
        print(f"\033[96m🏛️ Loading all specialist weights...\033[0m")
        weight_patterns = {
            'minerva': ['minerva_v1_best.pt', 'minerva_v2_best.pt', 'minerva_v3_best.pt', 'minerva_best.pt'], 
            'atlas': ['atlas_v1_best.pt', 'atlas_v2_best.pt', 'atlas_v3_best.pt', 'atlas_best.pt'], 
            'iris': ['iris_v1_best.pt', 'iris_v2_best.pt', 'iris_v3_best.pt', 'iris_best.pt'], 
            'chronos': ['chronos_v1_best.pt', 'chronos_v2_best.pt', 'chronos_v3_best.pt', 'chronos_best.pt'], 
            'prometheus': ['prometheus_v1_best.pt', 'prometheus_v2_best.pt', 'prometheus_v3_best.pt', 'prometheus_best.pt']
        }
        load_results = {}
        for specialist_name, patterns in weight_patterns.items():
            loaded = False
            for pattern in patterns:
                weight_path = os.path.join(weight_dir, pattern)
                if os.path.exists(weight_path):
                    try:
                        success = self.specialists[specialist_name].load_compatible_weights(weight_path)
                        if success:
                            print(f"\033[96m✅ {specialist_name.upper()}: Loaded weights from {pattern}\033[0m")
                            loaded = True; break
                    except Exception as e:
                        print(f"\033[93m⚠️  {specialist_name.upper()}: Loading error: {e}\033[0m")
            load_results[specialist_name] = loaded
            if not loaded: print(f"\033[93m⚠️  {specialist_name.upper()}: No compatible weights found\033[0m")
        print(f"\033[96m🏛️ Successfully loaded {sum(load_results.values())}/5 specialists\033[0m")
        return load_results
    
    def forward(self, input_grid: torch.Tensor, 
                target_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference') -> EnsembleDecision:
        """OLYMPUS forward pass - all specialists process every problem"""
        
        specialist_predictions = {}
        # --- MODIFIED: Store confidences as tensors to avoid breaking the computation graph ---
        specialist_confidences_tensors = {} 
        specialist_features = {}
        
        specialist_names = ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']
        
        for name in specialist_names:
            try:
                specialist = self.specialists[name]
                
                if name == 'chronos':
                    inputs = [input_grid] if not isinstance(input_grid, list) else input_grid
                    output = specialist(inputs, output_grid=target_grid, mode=mode)
                else:
                    output = specialist(input_grid, output_grid=target_grid, mode=mode)
                
                specialist_predictions[name] = output['predicted_output']
                
                # --- FIX: Extract confidence but KEEP IT AS A TENSOR ---
                confidence = output.get('confidence', torch.tensor(0.5, device=input_grid.device))
                if not torch.is_tensor(confidence):
                    confidence = torch.tensor(float(confidence), device=input_grid.device, dtype=torch.float32)

                if confidence.numel() > 1:
                    specialist_confidences_tensors[name] = confidence.mean()
                else:
                    specialist_confidences_tensors[name] = confidence.squeeze()
                    
                specialist_features[name] = output.get('features', None)
                
                if mode == 'inference':
                    print(f"\033[96m   {name.upper()}: Confidence = {specialist_confidences_tensors[name].item():.3f}\033[0m")
                
            except Exception as e:
                print(f"\033[91m❌ {name.upper()} failed: {e}\033[0m")
                specialist_predictions[name] = torch.zeros_like(list(specialist_predictions.values())[0] if specialist_predictions else input_grid)
                specialist_confidences_tensors[name] = torch.tensor(0.0, device=input_grid.device)
                specialist_features[name] = None
        
        # --- MODIFIED: Pass the dictionary of tensors to the fusion engine ---
        ensemble_decision = self.fusion_engine(
            specialist_predictions, 
            specialist_confidences_tensors,
            target_grid
        )
        
        ensemble_decision.metadata.update({
            'active_specialists': list(specialist_predictions.keys()),
            'olympus_version': 'AGI2_V1.0'
        })
        
        if mode == 'inference':
            primary = max(ensemble_decision.fusion_weights.items(), key=lambda x: x[1])
            print(f"\033[96m🏛️ OLYMPUS Decision: Primary={primary[0].upper()} ({primary[1]:.2f}), "
                  f"Consensus={ensemble_decision.consensus_score:.3f}, Final_Confidence={ensemble_decision.confidence:.3f}\033[0m")
        
        return ensemble_decision
    
    def evaluate_performance(self, test_dataset, max_samples: int = 100) -> Dict[str, float]:
        print(f"\033[96m🏛️ Evaluating OLYMPUS performance on {max_samples} samples...\033[0m")
        self.eval()
        correct_predictions, total_samples = 0, 0
        specialist_correct = defaultdict(int)
        
        with torch.no_grad():
            for i, (input_grid, target_grid, metadata) in enumerate(test_dataset):
                if i >= max_samples: break
                decision = self.forward(input_grid, target_grid, mode='inference')
                pred_indices = decision.prediction.argmax(dim=1)
                target_indices = target_grid.argmax(dim=1) if target_grid.dim() > 3 else target_grid
                if torch.equal(pred_indices, target_indices): correct_predictions += 1
                for name, pred in decision.specialist_predictions.items():
                    if torch.equal(pred.argmax(dim=1), target_indices): specialist_correct[name] += 1
                total_samples += 1
        
        ensemble_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        specialist_accuracies = {name: sc / total_samples for name, sc in specialist_correct.items()}
        
        print(f"\033[96m🏛️ OLYMPUS Evaluation Complete: Accuracy: {ensemble_accuracy:.1%}\033[0m")
        return {'ensemble_accuracy': ensemble_accuracy, **{f'{name}_accuracy': acc for name, acc in specialist_accuracies.items()}}

    def get_ensemble_state(self) -> Dict[str, Any]:
        return {'ensemble_name': 'OLYMPUS_AGI2', 'total_specialists': len(self.specialists), 'total_parameters': sum(p.numel() for p in self.parameters())}
    
    def save_ensemble(self, save_path: str):
        torch.save({'ensemble_state_dict': self.state_dict(), 'ensemble_config': {'max_grid_size': self.max_grid_size, 'd_model': self.d_model}}, save_path)
        print(f"\033[96m🏛️ OLYMPUS ensemble saved to {save_path}\033[0m")
    
    def load_ensemble(self, load_path: str):
        try:
            state = torch.load(load_path, map_location=self.device_name)
            self.load_state_dict(state['ensemble_state_dict'], strict=False)
            print(f"\033[96m🏛️ OLYMPUS ensemble loaded from {load_path}\033[0m")
            return True
        except Exception as e:
            print(f"\033[96m🏛️ Error loading OLYMPUS ensemble: {e}. Starting fresh.\033[0m")
            return False