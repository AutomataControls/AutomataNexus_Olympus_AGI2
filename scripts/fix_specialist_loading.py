"""
Fix specialist model loading for OLYMPUS ensemble
Converts individual specialist models to OLYMPUS-compatible format
"""

import torch
import os

# Paths
input_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports'
output_dir = '/content/AutomataNexus_Olympus_AGI2/src/models/reports/Olympus/InputBestModels'
os.makedirs(output_dir, exist_ok=True)

# Map individual model files to OLYMPUS names
specialist_models = {
    'minerva': [
        f'{input_dir}/Minerva/Stage5_Weights/minerva_v6_enhanced_stage5_best_state_dict_only.pt',
        f'{input_dir}/Minerva/minerva_v6_enhanced_best.pt',
        f'{input_dir}/Minerva/minerva_v6_enhanced_best_state_dict_only.pt'
    ],
    'atlas': [
        f'{input_dir}/Atlas/Stage5_Weights/atlas_v5_enhanced_stage5_best_state_dict_only.pt',
        f'{input_dir}/Atlas/atlas_v5_enhanced_best.pt',
        f'{input_dir}/Atlas/atlas_v5_enhanced_best_state_dict_only.pt'
    ],
    'iris': [
        f'{input_dir}/Iris/Stage5_Weights/iris_v6_enhanced_stage5_best_state_dict_only.pt',
        f'{input_dir}/Iris/iris_v6_enhanced_best.pt',
        f'{input_dir}/Iris/iris_v6_enhanced_best_state_dict_only.pt'
    ],
    'chronos': [
        f'{input_dir}/Chronos/Stage5_Weights/chronos_v4_enhanced_stage5_best_state_dict_only.pt',
        f'{input_dir}/Chronos/chronos_v4_enhanced_best.pt',
        f'{input_dir}/Chronos/chronos_v4_enhanced_best_state_dict_only.pt'
    ],
    'prometheus': [
        f'{input_dir}/Prometheus/Stage5_Weights/prometheus_v6_enhanced_stage5_best_state_dict_only.pt',
        f'{input_dir}/Prometheus/prometheus_v6_enhanced_best.pt',
        f'{input_dir}/Prometheus/prometheus_v6_enhanced_best_state_dict_only.pt'
    ]
}

print("🔧 Converting individual specialist models to OLYMPUS format...")
print("=" * 60)

for specialist_name, possible_paths in specialist_models.items():
    loaded = False
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Load the checkpoint
                checkpoint = torch.load(path, map_location='cpu')
                
                # Extract state dict
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Save in OLYMPUS-compatible format
                output_path = os.path.join(output_dir, f'{specialist_name}_best.pt')
                torch.save({
                    'model_state_dict': state_dict,
                    'model_name': specialist_name.upper(),
                    'source': 'individual_training',
                    'converted': True
                }, output_path)
                
                print(f"✅ {specialist_name.upper()}: Converted from {os.path.basename(path)}")
                loaded = True
                break
                
            except Exception as e:
                print(f"⚠️  {specialist_name.upper()}: Failed to load {os.path.basename(path)} - {str(e)}")
    
    if not loaded:
        print(f"❌ {specialist_name.upper()}: No valid checkpoint found")

print("\n" + "=" * 60)
print("🏛️ Conversion complete! Your specialist models are now OLYMPUS-ready.")
print("🚀 You can now run OLYMPUS training with your high-performance specialists!")