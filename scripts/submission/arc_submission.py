#!/usr/bin/env python3
"""
ARC Prize 2025 Submission Script
OLYMPUS Ensemble V3 - Ultimate Multi-Specialist System
"""

import torch
import torch.nn as nn
import json
import numpy as np
import sys
import os
from pathlib import Path

# Add project paths
sys.path.append('/kaggle/working')
sys.path.append('/kaggle/working/src')
sys.path.append('/kaggle/working/scripts/training')

# Import OLYMPUS ensemble
from src.models.olympus_ensemble import OlympusEnsemble

def load_olympus_model(model_path):
    """Load the trained OLYMPUS V3 model"""
    print(f"Loading OLYMPUS V3 model from: {model_path}")
    
    # Initialize OLYMPUS ensemble with same config as training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    olympus = OlympusEnsemble(
        max_grid_size=30,
        d_model=512,
        device=device
    )
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    olympus.load_state_dict(checkpoint['ensemble_state_dict'])
    olympus.eval()
    
    print(f"✅ OLYMPUS V3 loaded successfully on {device}")
    print(f"📊 Best performance: {checkpoint.get('best_performance', 'Unknown'):.2%}")
    
    return olympus

def preprocess_grid(grid):
    """Convert grid to tensor format"""
    if isinstance(grid, list):
        grid = np.array(grid, dtype=np.int64)
    
    # Ensure grid is at least 3x3 (minimum training size)
    h, w = grid.shape
    if h < 3 or w < 3:
        # Pad to 3x3 minimum
        new_h, new_w = max(h, 3), max(w, 3)
        padded = np.zeros((new_h, new_w), dtype=np.int64)
        padded[:h, :w] = grid
        grid = padded
    
    # Convert to tensor and add batch dimension
    tensor = torch.tensor(grid, dtype=torch.long).unsqueeze(0)
    return tensor

def postprocess_prediction(prediction, target_shape=None):
    """Convert model prediction back to list format"""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Remove batch dimension if present
    if prediction.ndim == 3:
        prediction = prediction[0]
    
    # Ensure prediction is 2D
    if prediction.ndim == 1:
        # Try to reshape to square
        size = int(np.sqrt(len(prediction)))
        if size * size == len(prediction):
            prediction = prediction.reshape(size, size)
        else:
            # Default to 3x3 if can't determine shape
            prediction = prediction[:9].reshape(3, 3)
    
    # Clip values to valid range (0-9)
    prediction = np.clip(prediction, 0, 9).astype(int)
    
    # Resize to target shape if specified
    if target_shape is not None:
        target_h, target_w = target_shape
        current_h, current_w = prediction.shape
        
        if (current_h, current_w) != (target_h, target_w):
            # Simple resize by cropping or padding
            resized = np.zeros((target_h, target_w), dtype=int)
            copy_h = min(current_h, target_h)
            copy_w = min(current_w, target_w)
            resized[:copy_h, :copy_w] = prediction[:copy_h, :copy_w]
            prediction = resized
    
    return prediction.tolist()

def predict_task(olympus, task_data):
    """Generate predictions for a single task"""
    predictions = []
    
    # Extract training examples for context
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    
    for test_idx, test_example in enumerate(test_examples):
        test_input = test_example['input']
        
        # Preprocess input
        input_tensor = preprocess_grid(test_input)
        
        # Move to device
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        # Generate predictions
        with torch.no_grad():
            try:
                # Get ensemble prediction
                result = olympus.forward_with_consensus(input_tensor)
                
                # Extract the main prediction
                if isinstance(result, dict):
                    prediction1 = result.get('final_output', result.get('ensemble_output', input_tensor))
                else:
                    prediction1 = result
                
                # Generate second attempt by using different routing
                olympus.eval()  # Ensure eval mode
                result2 = olympus(input_tensor)
                if isinstance(result2, dict):
                    prediction2 = result2.get('ensemble_output', result2.get('final_output', input_tensor))
                else:
                    prediction2 = result2
                
            except Exception as e:
                print(f"⚠️ Prediction error for test {test_idx}: {e}")
                # Fallback: return input or simple transformation
                prediction1 = input_tensor
                prediction2 = input_tensor
        
        # Postprocess predictions
        target_shape = None
        if len(train_examples) > 0:
            # Try to infer output shape from training examples
            output_shapes = [np.array(ex['output']).shape for ex in train_examples]
            if output_shapes:
                target_shape = output_shapes[0]  # Use first example's output shape
        
        pred1_list = postprocess_prediction(prediction1, target_shape)
        pred2_list = postprocess_prediction(prediction2, target_shape)
        
        predictions.append({
            "attempt_1": pred1_list,
            "attempt_2": pred2_list
        })
    
    return predictions

def main():
    """Main submission function"""
    print("🏛️ OLYMPUS V3 ARC Prize 2025 Submission")
    print("=" * 50)
    
    # Model path - adjust this based on where your trained model is saved
    MODEL_PATHS = [
        '/kaggle/working/olympus_v3_best.pt',
        '/kaggle/working/bestmodels/olympus_v3_best.pt',
        '/kaggle/input/olympus-models/olympus_v3_best.pt',
        '/kaggle/input/trained-models/olympus_v3_best.pt'
    ]
    
    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError("❌ OLYMPUS V3 model not found in any expected location")
    
    # Load test challenges
    test_file = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    if not os.path.exists(test_file):
        # Fallback locations
        test_alternatives = [
            '/kaggle/working/data/arc-agi_test_challenges.json',
            'data/arc-agi_test_challenges.json'
        ]
        for alt in test_alternatives:
            if os.path.exists(alt):
                test_file = alt
                break
    
    print(f"📁 Loading test challenges from: {test_file}")
    
    with open(test_file, 'r') as f:
        test_challenges = json.load(f)
    
    print(f"📊 Found {len(test_challenges)} test tasks")
    
    # Load OLYMPUS model
    olympus = load_olympus_model(model_path)
    
    # Generate predictions
    submission = {}
    
    for task_id, task_data in test_challenges.items():
        print(f"🔮 Predicting task: {task_id}")
        
        try:
            predictions = predict_task(olympus, task_data)
            submission[task_id] = predictions
            
        except Exception as e:
            print(f"❌ Error processing task {task_id}: {e}")
            # Create fallback predictions
            num_tests = len(task_data.get('test', []))
            fallback_predictions = []
            for _ in range(num_tests):
                fallback_predictions.append({
                    "attempt_1": [[0, 0], [0, 0]],
                    "attempt_2": [[0, 0], [0, 0]]
                })
            submission[task_id] = fallback_predictions
    
    # Save submission
    output_file = 'submission.json'
    with open(output_file, 'w') as f:
        json.dump(submission, f)
    
    print(f"✅ Submission saved to: {output_file}")
    print(f"📊 Total tasks processed: {len(submission)}")
    print("🏆 OLYMPUS V3 submission ready for ARC Prize 2025!")

if __name__ == "__main__":
    main()