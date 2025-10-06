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
                 consensus_score: float):
        self.prediction = prediction
        self.confidence = confidence
        self.specialist_predictions = specialist_predictions
        self.specialist_confidences = specialist_confidences
        self.fusion_weights = fusion_weights
        self.consensus_score = consensus_score
        self.metadata = {}


class DecisionFusionEngine(nn.Module):
    """Advanced decision fusion combining all 5 specialist outputs"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.num_specialists = 5
        
        # Confidence analysis network
        self.confidence_analyzer = nn.Sequential(
            nn.Linear(self.num_specialists, 64),  # 5 confidence scores
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_specialists),  # Output fusion weights
            nn.Softmax(dim=-1)
        )
        
        # Adaptive networks - will be initialized on first forward pass
        self.similarity_network = None
        self.meta_fusion = None
        
        # IoU-based selection weights
        self.iou_weight = 0.85  # ULTRA TEAL formula
        self.exact_weight = 0.15
        
        # Track expected feature sizes
        self.expected_pred_features = 10 * self.num_specialists
        self.networks_initialized = False
        
    def calculate_iou_scores(self, predictions: List[torch.Tensor], target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate IoU scores between specialist predictions"""
        batch_size = predictions[0].shape[0]
        iou_scores = torch.zeros(len(predictions), batch_size, device=predictions[0].device)
        
        for i, pred in enumerate(predictions):
            pred_indices = pred.argmax(dim=1)  # [batch, H, W]
            
            if target is not None:
                target_indices = target.argmax(dim=1) if target.dim() > 3 else target
                # Calculate IoU with target
                intersection = (pred_indices == target_indices).float().sum(dim=[1,2])
                union = (pred_indices.shape[1] * pred_indices.shape[2])
                iou_scores[i] = intersection / union
            else:
                # Calculate average IoU with other predictions
                for j, other_pred in enumerate(predictions):
                    if i != j:
                        other_indices = other_pred.argmax(dim=1)
                        intersection = (pred_indices == other_indices).float().sum(dim=[1,2])
                        union = (pred_indices.shape[1] * pred_indices.shape[2])
                        iou_scores[i] += intersection / union
                iou_scores[i] /= (len(predictions) - 1)
        
        return iou_scores.transpose(0, 1)  # [batch, num_specialists]
    
    def _initialize_networks(self, feature_size: int, device: torch.device):
        """Initialize adaptive networks based on actual feature size"""
        if not self.networks_initialized:
            # Prediction similarity analyzer
            self.similarity_network = nn.Sequential(
                nn.Linear(feature_size + self.num_specialists, 256),  # predictions + confidences
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),  # Consensus score
                nn.Sigmoid()
            ).to(device)
            
            # Meta-fusion network for final decision
            self.meta_fusion = nn.Sequential(
                nn.Linear(feature_size + self.num_specialists + 1, 128),  # Predictions + confidences + consensus
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),  # Final prediction logits
                nn.Softmax(dim=-1)
            ).to(device)
            
            self.networks_initialized = True
    
    def forward(self, specialist_predictions: Dict[str, torch.Tensor],
                specialist_confidences: Dict[str, float],
                target: Optional[torch.Tensor] = None) -> EnsembleDecision:
        """Fuse all specialist predictions into final OLYMPUS decision"""
        
        # Organize specialist data
        specialist_names = ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']
        predictions = []
        confidences = []
        
        for name in specialist_names:
            pred = specialist_predictions.get(name, torch.zeros_like(list(specialist_predictions.values())[0]))
            conf = specialist_confidences.get(name, 0.5)
            predictions.append(pred)
            confidences.append(conf)
        
        # Stack predictions and confidences
        stacked_predictions = torch.stack(predictions, dim=0)  # [5, batch, C, H, W]
        confidence_tensor = torch.tensor(confidences, device=predictions[0].device).unsqueeze(0)  # [1, 5]
        batch_size = predictions[0].shape[0]
        
        # Calculate consensus score - make robust to different prediction shapes
        flat_predictions = stacked_predictions.transpose(0, 1).reshape(batch_size, -1)  # [batch, 5*C*H*W]
        
        # Initialize networks adaptively based on actual feature size
        self._initialize_networks(flat_predictions.shape[1], predictions[0].device)
        
        # Calculate IoU-based quality scores
        iou_scores = self.calculate_iou_scores(predictions, target)  # [batch, 5]
        
        # Generate confidence-based fusion weights
        fusion_weights = self.confidence_analyzer(confidence_tensor)  # [1, 5]
        fusion_weights = fusion_weights.expand(batch_size, -1)  # [batch, 5]
        
        # Combine with IoU scores for final weights
        combined_weights = fusion_weights * iou_scores
        combined_weights = F.softmax(combined_weights, dim=-1)
        
        consensus_input = torch.cat([flat_predictions, confidence_tensor.expand(batch_size, -1)], dim=1)
        
        # Re-initialize if input size changed or networks not initialized
        if (self.similarity_network is None or 
            self.similarity_network[0].in_features != consensus_input.shape[1]):
            self.networks_initialized = False
            self._initialize_networks(flat_predictions.shape[1], predictions[0].device)
            
        consensus_score = self.similarity_network(consensus_input).squeeze()  # [batch]
        
        # Generate final prediction using meta-fusion
        meta_input = torch.cat([
            flat_predictions,  # All prediction features
            combined_weights,  # Final fusion weights
            consensus_score.unsqueeze(-1) if consensus_score.dim() == 1 else consensus_score  # Consensus score
        ], dim=1)
        
        # Check if meta_fusion network size matches
        if (self.meta_fusion is None or 
            self.meta_fusion[0].in_features != meta_input.shape[1]):
            self.networks_initialized = False
            self._initialize_networks(flat_predictions.shape[1], predictions[0].device)
        
        final_prediction = self.meta_fusion(meta_input)  # [batch, 10]
        
        # Calculate final confidence
        weighted_confidences = torch.sum(combined_weights * confidence_tensor.expand(batch_size, -1), dim=1)
        final_confidence = (weighted_confidences * consensus_score).mean().item()
        
        # Create specialist confidence dict
        specialist_conf_dict = {name: conf for name, conf in zip(specialist_names, confidences)}
        
        # Create fusion weights dict
        fusion_weights_dict = {name: combined_weights[:, i].mean().item() 
                              for i, name in enumerate(specialist_names)}
        
        return EnsembleDecision(
            prediction=final_prediction,
            confidence=final_confidence,
            specialist_predictions=specialist_predictions,
            specialist_confidences=specialist_conf_dict,
            fusion_weights=fusion_weights_dict,
            consensus_score=consensus_score.mean().item()
        )


class OlympusEnsemble(nn.Module):
    """OLYMPUS - All specialists process every problem for ultimate performance"""
    
    def __init__(self, max_grid_size: int = 30, d_model: int = 256, device: str = 'cuda'):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.d_model = d_model
        self.device_name = device
        
        print(f"\033[96m🏛️ Initializing OLYMPUS Ensemble - Ultimate AGI2 System\033[0m")
        
        # Initialize all 5 specialist models (latest enhanced versions that match your checkpoints)
        self.specialists = nn.ModuleDict({
            'minerva': MinervaV6Enhanced(max_grid_size, d_model, preserve_weights=True),
            'atlas': AtlasV5Enhanced(max_grid_size, d_model, 2, preserve_weights=True),
            'iris': IrisV6Enhanced(max_grid_size, d_model, 3, preserve_weights=True),
            'chronos': ChronosV4Enhanced(max_grid_size, d_model, 8, preserve_weights=True),
            'prometheus': PrometheusV6Enhanced(max_grid_size, d_model, 8, preserve_weights=True)
        })
        
        # Decision fusion engine
        self.fusion_engine = DecisionFusionEngine(d_model)
        
        # Performance tracking
        self.ensemble_performance = []
        self.specialist_performance = defaultdict(list)
        self.decision_history = []
        
        # Logging
        self.logger = logging.getLogger('OLYMPUS')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('🏛️ OLYMPUS: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\033[96m🏛️ OLYMPUS initialized with {total_params:,} total parameters across all specialists\033[0m")
        
    def load_all_specialists(self, weight_dir: str = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels') -> Dict[str, bool]:
        """Load pre-trained weights for all specialists"""
        print(f"\033[96m🏛️ Loading all specialist weights...\033[0m")
        
        # Define weight file patterns for each specialist (from InputBestModels directory)
        weight_patterns = {
            'minerva': ['minerva_best.pt'],      # MINERVA V6 Enhanced
            'atlas': ['atlas_best.pt'],          # ATLAS V5 Enhanced  
            'iris': ['iris_best.pt'],            # IRIS V6 Enhanced
            'chronos': ['chronos_best.pt'],      # CHRONOS V4/V5 Enhanced
            'prometheus': ['prometheus_best.pt'] # PROMETHEUS V6 Enhanced
        }
        
        load_results = {}
        
        for specialist_name, patterns in weight_patterns.items():
            loaded = False
            for pattern in patterns:
                weight_path = os.path.join(weight_dir, pattern)
                print(f"\033[96m   Checking: {weight_path}\033[0m")
                if os.path.exists(weight_path):
                    try:
                        if hasattr(self.specialists[specialist_name], 'load_compatible_weights'):
                            success = self.specialists[specialist_name].load_compatible_weights(weight_path)
                        else:
                            # Fallback manual loading
                            checkpoint = torch.load(weight_path, map_location=self.device_name)
                            if 'model_state_dict' in checkpoint:
                                state_dict = checkpoint['model_state_dict']
                            else:
                                state_dict = checkpoint
                            
                            model_dict = self.specialists[specialist_name].state_dict()
                            compatible_params = {}
                            for name, param in state_dict.items():
                                if name in model_dict and model_dict[name].shape == param.shape:
                                    compatible_params[name] = param
                            
                            model_dict.update(compatible_params)
                            self.specialists[specialist_name].load_state_dict(model_dict)
                            success = len(compatible_params) > 0
                        
                        if success:
                            print(f"\033[96m✅ {specialist_name.upper()}: Loaded weights from {pattern}\033[0m")
                            loaded = True
                            break
                        
                    except Exception as e:
                        print(f"\033[93m⚠️  {specialist_name.upper()}: Loading error: {e}\033[0m")
                        continue
            
            load_results[specialist_name] = loaded
            if not loaded:
                print(f"\033[93m⚠️  {specialist_name.upper()}: No compatible weights found, using random initialization\033[0m")
        
        successful_loads = sum(load_results.values())
        print(f"\033[96m🏛️ Successfully loaded {successful_loads}/5 specialists\033[0m")
        
        return load_results
    
    def forward(self, input_grid: torch.Tensor, 
                target_grid: Optional[torch.Tensor] = None,
                mode: str = 'inference') -> EnsembleDecision:
        """OLYMPUS forward pass - all specialists process every problem"""
        
        if mode == 'train':
            print(f"\033[96m🏛️ OLYMPUS Training Mode: All specialists processing problem...\033[0m")
        
        # Prepare inputs for all specialists
        specialist_predictions = {}
        specialist_confidences = {}
        specialist_features = {}
        
        # Process with all 5 specialists in parallel
        specialist_names = ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']
        
        for name in specialist_names:
            try:
                specialist = self.specialists[name]
                
                # Handle different input requirements
                if name == 'chronos':
                    # CHRONOS expects sequence input
                    if isinstance(input_grid, list):
                        inputs = input_grid
                    else:
                        inputs = [input_grid]  # Convert single grid to sequence
                    output = specialist(inputs, target_grid, mode=mode)
                else:
                    # Other specialists expect single grid input
                    output = specialist(input_grid, target_grid, mode=mode)
                
                # Extract outputs
                specialist_predictions[name] = output['predicted_output']
                
                # Extract confidence robustly
                confidence = output.get('confidence', 0.5)
                if torch.is_tensor(confidence):
                    if confidence.numel() == 1:
                        specialist_confidences[name] = confidence.item()
                    else:
                        # Multiple confidence values - take mean
                        specialist_confidences[name] = confidence.mean().item()
                else:
                    specialist_confidences[name] = float(confidence)
                    
                specialist_features[name] = output.get('features', None)
                
                if mode == 'inference':
                    print(f"\033[96m   {name.upper()}: Confidence = {specialist_confidences[name]:.3f}\033[0m")
                
            except Exception as e:
                # Handle specialist errors gracefully
                print(f"\033[91m❌ {name.upper()} failed: {e}\033[0m")
                specialist_predictions[name] = torch.zeros_like(list(specialist_predictions.values())[0] if specialist_predictions else input_grid)
                specialist_confidences[name] = 0.0
                specialist_features[name] = None
        
        # Fuse all specialist predictions
        ensemble_decision = self.fusion_engine(
            specialist_predictions, 
            specialist_confidences,
            target_grid
        )
        
        # Add OLYMPUS metadata
        ensemble_decision.metadata.update({
            'active_specialists': list(specialist_predictions.keys()),
            'total_specialists': len(self.specialists),
            'processing_mode': 'parallel_all_specialists',
            'olympus_version': 'AGI2_V1.0'
        })
        
        # Log decision summary
        if mode == 'inference':
            primary_specialist = max(ensemble_decision.fusion_weights.items(), key=lambda x: x[1])
            print(f"\033[96m🏛️ OLYMPUS Decision: Primary={primary_specialist[0].upper()} ({primary_specialist[1]:.2f}), "
                  f"Consensus={ensemble_decision.consensus_score:.3f}, Final_Confidence={ensemble_decision.confidence:.3f}\033[0m")
        
        return ensemble_decision
    
    def evaluate_performance(self, test_dataset, max_samples: int = 100) -> Dict[str, float]:
        """Evaluate OLYMPUS ensemble performance on test dataset"""
        print(f"\033[96m🏛️ Evaluating OLYMPUS performance on {max_samples} samples...\033[0m")
        
        self.eval()
        correct_predictions = 0
        total_samples = 0
        specialist_correct = defaultdict(int)
        
        with torch.no_grad():
            for i, (input_grid, target_grid, metadata) in enumerate(test_dataset):
                if i >= max_samples:
                    break
                
                # Get OLYMPUS decision
                decision = self.forward(input_grid, target_grid, mode='inference')
                
                # Check if ensemble prediction is correct
                pred_indices = decision.prediction.argmax(dim=1)
                target_indices = target_grid.argmax(dim=1) if target_grid.dim() > 3 else target_grid
                
                is_correct = torch.equal(pred_indices, target_indices)
                if is_correct:
                    correct_predictions += 1
                
                # Check individual specialist correctness
                for name, pred in decision.specialist_predictions.items():
                    spec_pred_indices = pred.argmax(dim=1)
                    if torch.equal(spec_pred_indices, target_indices):
                        specialist_correct[name] += 1
                
                total_samples += 1
                
                if (i + 1) % 10 == 0:
                    current_accuracy = correct_predictions / total_samples
                    print(f"\033[96m   Progress: {i+1}/{max_samples}, Current Accuracy: {current_accuracy:.1%}\033[0m")
        
        # Calculate final performance metrics
        ensemble_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        specialist_accuracies = {name: specialist_correct[name] / total_samples 
                               for name in specialist_correct.keys()}
        
        # Update performance history
        self.ensemble_performance.append(ensemble_accuracy)
        for name, acc in specialist_accuracies.items():
            self.specialist_performance[name].append(acc)
        
        results = {
            'ensemble_accuracy': ensemble_accuracy,
            'total_samples': total_samples,
            **{f'{name}_accuracy': acc for name, acc in specialist_accuracies.items()}
        }
        
        print(f"\033[96m🏛️ OLYMPUS Evaluation Complete:\033[0m")
        print(f"\033[96m   Ensemble Accuracy: {ensemble_accuracy:.1%}\033[0m")
        for name, acc in specialist_accuracies.items():
            print(f"\033[96m   {name.upper()} Accuracy: {acc:.1%}\033[0m")
        
        return results
    
    def get_ensemble_state(self) -> Dict[str, Any]:
        """Get comprehensive OLYMPUS ensemble state"""
        return {
            'ensemble_name': 'OLYMPUS_AGI2',
            'total_specialists': len(self.specialists),
            'specialist_names': list(self.specialists.keys()),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'ensemble_performance_history': self.ensemble_performance,
            'specialist_performance_history': dict(self.specialist_performance),
            'decision_count': len(self.decision_history),
            'device': self.device_name,
            'architecture': 'All_Specialists_Every_Problem',
            'target_accuracy': 0.95,
            'fusion_strategy': 'confidence_weighted_consensus'
        }
    
    def save_ensemble(self, save_path: str):
        """Save complete OLYMPUS ensemble state"""
        ensemble_state = {
            'ensemble_state_dict': self.state_dict(),
            'ensemble_config': {
                'max_grid_size': self.max_grid_size,
                'd_model': self.d_model,
                'device': self.device_name
            },
            'performance_metrics': self.get_ensemble_state()
        }
        
        torch.save(ensemble_state, save_path)
        print(f"\033[96m🏛️ OLYMPUS ensemble saved to {save_path}\033[0m")
    
    def load_ensemble(self, load_path: str):
        """Load complete OLYMPUS ensemble state"""
        ensemble_state = torch.load(load_path, map_location=self.device_name)
        
        self.load_state_dict(ensemble_state['ensemble_state_dict'])
        self.ensemble_performance = ensemble_state['performance_metrics']['ensemble_performance_history']
        self.specialist_performance = defaultdict(list, ensemble_state['performance_metrics']['specialist_performance_history'])
        
        print(f"\033[96m🏛️ OLYMPUS ensemble loaded from {load_path}\033[0m")