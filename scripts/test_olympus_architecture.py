#!/usr/bin/env python3
"""
OLYMPUS Architecture Test - Verify enhanced ensemble system
Tests that all 5 specialists can be loaded and run inference
"""

import sys
import os
import torch
import numpy as np

# Add project paths
sys.path.append('/content/AutomataNexus_Olympus_AGI2')
sys.path.append('/content/AutomataNexus_Olympus_AGI2/src')

from src.models.olympus_ensemble import OlympusEnsemble

def test_olympus_architecture():
    """Test OLYMPUS ensemble architecture"""
    print(f"\033[96m{'=' * 80}\033[0m")
    print(f"\033[96m🏛️ OLYMPUS Architecture Test - Enhanced Ensemble System\033[0m")
    print(f"\033[96m{'=' * 80}\033[0m")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\033[96m📱 Device: {device}\033[0m")
    
    # Initialize OLYMPUS ensemble
    try:
        olympus = OlympusEnsemble(max_grid_size=30, d_model=128, device=str(device))
        print(f"\033[96m✅ OLYMPUS ensemble initialized successfully\033[0m")
    except Exception as e:
        print(f"\033[91m❌ Failed to initialize OLYMPUS: {e}\033[0m")
        return False
    
    # Test specialist loading
    print(f"\033[96m\n🔄 Testing specialist weight loading...\033[0m")
    load_results = olympus.load_all_specialists()
    
    for specialist, loaded in load_results.items():
        status = "✅ Loaded" if loaded else "⚠️ No weights"
        print(f"\033[96m   {specialist.upper()}: {status}\033[0m")
    
    # Create test input (simple 5x5 grid)
    test_input = torch.randint(0, 10, (1, 1, 5, 5)).to(device)
    test_target = torch.randint(0, 10, (1, 1, 5, 5)).to(device)
    
    print(f"\033[96m\n🧪 Testing ensemble inference...\033[0m")
    print(f"\033[96m   Input shape: {test_input.shape}\033[0m")
    
    try:
        # Test inference
        olympus.eval()
        with torch.no_grad():
            decision = olympus.forward(test_input, test_target, mode='inference')
        
        print(f"\033[96m✅ Ensemble inference successful\033[0m")
        print(f"\033[96m   Final confidence: {decision.confidence:.3f}\033[0m")
        print(f"\033[96m   Consensus score: {decision.consensus_score:.3f}\033[0m")
        
        # Show specialist contributions
        print(f"\033[96m\n📊 Specialist Fusion Weights:\033[0m")
        for spec, weight in decision.fusion_weights.items():
            print(f"\033[96m   {spec.upper()}: {weight:.3f}\033[0m")
        
        return True
        
    except Exception as e:
        print(f"\033[91m❌ Ensemble inference failed: {e}\033[0m")
        return False

def test_individual_specialists():
    """Test individual specialist models"""
    print(f"\033[96m\n🔧 Testing individual specialists...\033[0m")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = torch.randint(0, 10, (1, 1, 5, 5)).to(device)
    
    # Test each specialist individually
    specialists = {
        'MINERVA V6': ('src.models.minerva_v6_enhanced', 'MinervaV6Enhanced'),
        'ATLAS V5': ('src.models.atlas_v5_enhanced', 'AtlasV5Enhanced'), 
        'IRIS V6': ('src.models.iris_v6_enhanced', 'IrisV6Enhanced'),
        'CHRONOS V4': ('src.models.chronos_v4_enhanced', 'ChronosV4Enhanced'),
        'PROMETHEUS V6': ('src.models.prometheus_v6_enhanced', 'PrometheusV6Enhanced')
    }
    
    for name, (module_path, class_name) in specialists.items():
        try:
            # Dynamic import
            exec(f"from {module_path} import {class_name}")
            model_class = eval(class_name)
            
            # Initialize model
            model = model_class(max_grid_size=30).to(device)
            
            # Test forward pass
            with torch.no_grad():
                output = model(test_input, mode='inference')
            
            print(f"\033[96m   ✅ {name}: Working\033[0m")
            
        except Exception as e:
            print(f"\033[91m   ❌ {name}: {str(e)[:60]}...\033[0m")

if __name__ == "__main__":
    print("Starting OLYMPUS Architecture Test...")
    
    # Test individual specialists first
    test_individual_specialists()
    
    # Test full ensemble
    success = test_olympus_architecture()
    
    if success:
        print(f"\033[96m\n🎉 OLYMPUS Architecture Test PASSED\033[0m")
        print(f"\033[96m🏛️ Enhanced ensemble system ready for training!\033[0m")
    else:
        print(f"\033[91m\n❌ OLYMPUS Architecture Test FAILED\033[0m")
        print(f"\033[91m🏛️ Check individual specialist models\033[0m")