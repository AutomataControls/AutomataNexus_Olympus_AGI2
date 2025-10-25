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

# Check what files are actually in the directory
print("📁 Files found in InputBestModels:")
if os.path.exists(output_dir):
    for f in os.listdir(output_dir):
        if f.endswith('.pt'):
            print(f"  - {f}")
print()

# Map individual model files to OLYMPUS names
specialist_models = {
    'minerva': [f'{output_dir}/minerva_best.pt'],
    'atlas': [f'{output_dir}/atlas_best.pt'],
    'iris': [f'{output_dir}/iris_best.pt'],
    'chronos': [f'{output_dir}/chronos_best.pt'],
    'prometheus': [f'{output_dir}/prometheus_best.pt']
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
                
                # Check if this is already the output file
                if path == os.path.join(output_dir, f'{specialist_name}_best.pt'):
                    print(f"✓ {specialist_name.upper()}: Already exists, checking format...")
                    
                    # Just verify it's in the right format
                    if 'model_state_dict' in checkpoint:
                        print(f"✅ {specialist_name.upper()}: Format is correct")
                    else:
                        # Reformat it
                        torch.save({
                            'model_state_dict': state_dict,
                            'model_name': specialist_name.upper()
                        }, path)
                        print(f"✅ {specialist_name.upper()}: Reformatted successfully")
                else:
                    # Save in OLYMPUS-compatible format
                    output_path = os.path.join(output_dir, f'{specialist_name}_best.pt')
                    torch.save({
                        'model_state_dict': state_dict,
                        'model_name': specialist_name.upper()
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