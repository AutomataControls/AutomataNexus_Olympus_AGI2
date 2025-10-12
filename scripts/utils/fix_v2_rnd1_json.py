"""
Convert V2_Rnd_1.json to match the format of other training reports
"""

import json
import os

# Read the original file
filepath = '/mnt/d/opt/AutomataNexus_Olympus_AGI2/src/models/reports/TrainingReport/V2_Rnd_1.json'
with open(filepath, 'r') as f:
    old_data = json.load(f)

# Convert to new format
new_data = {
    "title": old_data["title"],
    "summary": {
        "description": old_data["overview"]["description"],
        "approach": old_data["overview"]["approach"],
        "target": old_data["overview"]["target"]
    },
    "initialization": {
        "parameters": old_data["initialization"]["parameters"],
        "fusionEngine": {
            "loaded": old_data["initialization"]["fusion_engine"]["loaded"],
            "missing": old_data["initialization"]["fusion_engine"]["missing"], 
            "status": old_data["initialization"]["fusion_engine"]["status"]
        },
        "weights": {
            "v1": "Loaded V1 ensemble weights for V2 training",
            "v2": old_data["initialization"]["model_loading"]["v2_state"]
        }
    },
    "stages": []
}

# Convert stages
for stage in old_data.get("training_stages", []):
    new_stage = {
        "stageId": stage["stage"],
        "gridSize": stage["grid_size"],
        "complexity": stage["complexity"],
        "focus": stage["focus"],
        "augmentation": {
            "original": stage["samples"]["original"],
            "augmented": stage["samples"]["augmented"],
            "factor": stage["samples"]["augmentation_factor"]
        },
        "epochs": []
    }
    
    # Add single epoch entry with final metrics
    if "final_metrics" in stage:
        new_stage["epochs"].append({
            "epoch": stage.get("epochs", 15) - 1,  # Last epoch
            "global": stage["stage"] * 15 + stage.get("epochs", 15),
            "performance": stage["final_metrics"]["exact"],
            "loss": stage["final_metrics"]["loss"],
            "fusionLr": stage["final_metrics"]["fusion_lr"],
            "grid": f"{stage['grid_size']}x{stage['grid_size']}",
            "consensus": stage["final_metrics"]["consensus"],
            "final": True
        })
    
    new_data["stages"].append(new_stage)

# Set completion status based on stages
new_data["completed"] = len(new_data["stages"]) == 15

# Find final performance
if new_data["stages"]:
    final_stage = new_data["stages"][-1]
    if final_stage["epochs"]:
        new_data["finalPerformance"] = final_stage["epochs"][-1]["performance"]
    else:
        new_data["finalPerformance"] = 0.0
else:
    new_data["finalPerformance"] = 0.0

# Backup original
os.rename(filepath, filepath.replace('.json', '_original.json'))

# Save new format
with open(filepath, 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print("✅ Successfully converted V2_Rnd_1.json to new format")
print(f"   Original backed up as V2_Rnd_1_original.json")
print(f"   Found {len(new_data['stages'])} stages")
print(f"   Final performance: {new_data['finalPerformance']:.1f}%")