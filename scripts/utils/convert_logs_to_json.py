"""
Convert raw training logs to properly formatted JSON
Following the V2_Rnd_2.json format as template
"""

import json
import os
import re
from typing import Dict, List, Any
from datetime import datetime

def parse_log_to_json(filepath: str) -> Dict[str, Any]:
    """Convert a raw training log to structured JSON"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Determine version from filename or content
    version = "V3" if "V3" in os.path.basename(filepath) else "V2"
    
    # Initialize JSON structure
    result = {
        "title": f"OLYMPUS {version} {'Ultimate' if version == 'V3' else 'Advanced'} Ensemble Training",
        "summary": {},
        "initialization": {},
        "stages": []
    }
    
    # Parse header information
    for i, line in enumerate(lines[:50]):
        if "Full Specialist Coordination" in line or "Partial Specialist Fine-Tuning" in line:
            result["summary"]["approach"] = line.strip()
        elif "Target:" in line:
            result["summary"]["target"] = line.split("Target:")[-1].strip()
        elif "total parameters" in line:
            params_match = re.search(r'(\d+,?\d+,?\d+)\s+total parameters', line)
            if params_match:
                result["initialization"]["parameters"] = int(params_match.group(1).replace(',', ''))
        elif "FUSION ENGINE:" in line:
            fusion_match = re.search(r'(\d+) loaded, (\d+) missing', line)
            if fusion_match:
                result["initialization"]["fusionEngine"] = {
                    "loaded": int(fusion_match.group(1)),
                    "missing": int(fusion_match.group(2))
                }
        elif "SUCCESS: All" in line and "fusion engine" in line:
            result["initialization"]["fusionEngine"]["status"] = line.strip()
    
    # Parse stages
    current_stage = None
    current_epochs = []
    stage_data = {}
    
    for line in lines:
        # New stage detection
        stage_match = re.search(r'Stage (\d+): Grid Size (\d+) \| Complexity: ([\w_]+) \| Focus: ([\w_]+)', line)
        if stage_match:
            # Save previous stage if exists
            if current_stage is not None and stage_data:
                stage_data["epochs"] = current_epochs
                result["stages"].append(stage_data)
            
            # Start new stage
            current_stage = int(stage_match.group(1))
            current_epochs = []
            stage_data = {
                "stageId": current_stage,
                "gridSize": int(stage_match.group(2)),
                "complexity": stage_match.group(3),
                "focus": stage_match.group(4)
            }
        
        # Augmentation info
        aug_match = re.search(r'Augmented from (\d+) to (\d+) samples \((\d+)x\)', line)
        if aug_match and stage_data:
            stage_data["augmentation"] = {
                "original": int(aug_match.group(1)),
                "augmented": int(aug_match.group(2)),
                "factor": int(aug_match.group(3))
            }
        
        # Epoch performance lines (with emoji indicators)
        if "⏰ OLYMPUS" in line and "Epoch" in line:
            epoch_match = re.search(r'Stage (\d+), Epoch (\d+) \(Global: (\d+)\)', line)
            if epoch_match:
                epoch_num = int(epoch_match.group(2))
                global_epoch = int(epoch_match.group(3))
                
                # Get performance from next line
                perf_data = {
                    "epoch": epoch_num,
                    "global": global_epoch
                }
                
                # Look for performance line
                for j, next_line in enumerate(lines[lines.index(line)+1:lines.index(line)+5]):
                    if "Ultimate Ensemble:" in next_line or "Advanced Ensemble:" in next_line:
                        perf_match = re.search(r'(\d+\.\d+)% exact, Loss: ([-]?\d+\.?\d*)', next_line)
                        if perf_match:
                            perf_data["performance"] = float(perf_match.group(1))
                            perf_data["loss"] = float(perf_match.group(2))
                    elif "Fusion:" in next_line:
                        lr_match = re.search(r'Fusion: ([\d.]+) \| .*Grid: (\d+)x\d+ \| Consensus: ([\d.]+)', next_line)
                        if lr_match:
                            perf_data["fusionLr"] = float(lr_match.group(1))
                            perf_data["grid"] = f"{lr_match.group(2)}x{lr_match.group(2)}"
                            perf_data["consensus"] = float(lr_match.group(3))
                
                current_epochs.append(perf_data)
        
        # Stage completion
        if "✅" in line and "Stage" in line and "complete!" in line:
            final_perf_match = re.search(r'Final exact: (\d+\.\d+)%', line)
            if final_perf_match and current_epochs:
                current_epochs[-1]["final"] = True
    
    # Save last stage
    if current_stage is not None and stage_data:
        stage_data["epochs"] = current_epochs
        result["stages"].append(stage_data)
    
    # Parse final results
    for line in lines[-20:]:
        if "Training Complete!" in line:
            result["completed"] = True
        if "Best" in line and "Performance:" in line:
            best_match = re.search(r'Performance: (\d+\.\d+)%', line)
            if best_match:
                result["finalPerformance"] = float(best_match.group(1))
    
    return result

def main():
    """Convert all log files to JSON"""
    reports_dir = '/mnt/d/opt/AutomataNexus_Olympus_AGI2/src/models/reports/TrainingReport'
    
    # Files to convert (skip the already properly formatted ones)
    files_to_convert = [
        'V2_Rnd_1.json',
        'V2_Rnd_3.json', 
        'V2_Rnd_4.json',
        'V3_Lower_Rnd_2.json',
        'V3_Rnd_2.json',
        'V3_Rnd_3.json',
        'v3_Lower_Rnd_1.json',
        'v3_rND_1.json'
    ]
    
    for filename in files_to_convert:
        filepath = os.path.join(reports_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Skipping {filename} - file not found")
            continue
            
        print(f"Converting {filename}...")
        
        try:
            # Parse the log file
            json_data = parse_log_to_json(filepath)
            
            # Create backup of original
            backup_path = filepath.replace('.json', '_raw.txt')
            os.rename(filepath, backup_path)
            
            # Save as proper JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            
            print(f"  ✓ Converted successfully")
            print(f"  ✓ Original backed up to {os.path.basename(backup_path)}")
            print(f"  ✓ Found {len(json_data.get('stages', []))} stages")
            
        except Exception as e:
            print(f"  ✗ Error converting {filename}: {e}")
    
    print("\n✅ Conversion complete!")

if __name__ == "__main__":
    main()