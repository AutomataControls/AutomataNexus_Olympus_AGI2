"""
Organize and restructure the TrainingReport directory
"""

import os
import shutil
import json
from datetime import datetime

def organize_training_reports():
    base_dir = '/mnt/d/opt/AutomataNexus_Olympus_AGI2/src/models/reports/TrainingReport'
    
    # Create new directory structure
    directories = [
        'v2_runs',
        'v3_runs',
        'v3_lower_runs',
        'raw_logs',
        'analysis',
        'other_reports'
    ]
    
    for dir_name in directories:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
    
    # File organization mapping
    file_moves = {
        # V2 JSON runs
        'V2_Rnd_1.json': 'v2_runs/',
        'V2_Rnd_2.json': 'v2_runs/',
        'V2_Rnd_3.json': 'v2_runs/',
        'V2_Rnd_4.json': 'v2_runs/',
        
        # V3 full runs
        'V3_Rnd_2.json': 'v3_runs/',
        'V3_Rnd_3.json': 'v3_runs/',
        'v3_rND_1.json': 'v3_runs/',
        
        # V3 lower stages only
        'v3_Lower_Rnd_1.json': 'v3_lower_runs/',
        'V3_Lower_Rnd_2.json': 'v3_lower_runs/',
        
        # Raw/backup logs
        'V2_Rnd_1_original.json': 'raw_logs/',
        'V2_Rnd_3_raw.txt': 'raw_logs/',
        'V2_Rnd_4_raw.txt': 'raw_logs/',
        'V3_Lower_Rnd_2_raw.txt': 'raw_logs/',
        'V3_Rnd_2_raw.txt': 'raw_logs/',
        'V3_Rnd_3_raw.txt': 'raw_logs/',
        'v3_Lower_Rnd_1_raw.txt': 'raw_logs/',
        'v3_rND_1_raw.txt': 'raw_logs/',
        
        # Analysis outputs
        'training_summary.csv': 'analysis/',
        'training_analysis_summary.csv': 'analysis/',
        'training_performance_plot.png': 'analysis/',
        
        # Other reports
        'WarrenChillerReport_A_Basement.html': 'other_reports/',
        'WarrenFahl_Electrical_Usage.html': 'other_reports/',
        'WarrenFahl_Raw_data.csv': 'other_reports/',
        'nvidia_cuda_wsl.md': 'other_reports/'
    }
    
    # Additional metric CSVs to move
    metric_files = [f for f in os.listdir(base_dir) if f.endswith('_metrics.csv')]
    for metric_file in metric_files:
        file_moves[metric_file] = 'analysis/'
    
    # Move files
    moved_count = 0
    for filename, destination in file_moves.items():
        src_path = os.path.join(base_dir, filename)
        dst_path = os.path.join(base_dir, destination, filename)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"  ✓ Moved {filename} → {destination}")
            moved_count += 1
        else:
            print(f"  ⚠ Skipped {filename} (not found)")
    
    # Create README for each directory
    readmes = {
        'README.md': """# OLYMPUS Training Reports

This directory contains all training reports and analysis for the OLYMPUS ensemble models.

## Directory Structure

- **v2_runs/** - V2 training runs (full 15 stages)
- **v3_runs/** - V3 training runs (full 15 stages)
- **v3_lower_runs/** - V3 runs focusing only on lower stages (0-5)
- **raw_logs/** - Original raw training logs and backups
- **analysis/** - Generated analysis, summaries, and visualizations
- **other_reports/** - Miscellaneous reports and documentation

## Quick Stats

Best performances achieved:
- Overall: 64.5% (v3_Lower_Rnd_1)
- 4x4 grids: 59.8%
- 5x5 grids: 63.9%
- 6x6 grids: 63.8%

Target: 85%+ on lower stages
""",
        'v2_runs/README.md': """# V2 Training Runs

OLYMPUS V2 Advanced Ensemble Training runs with:
- Partial Specialist Fine-Tuning
- Advanced Fusion
- Meta-Learning Coordination

Files:
- V2_Rnd_1.json - First run, achieved 56.2% best (44.3% final)
- V2_Rnd_2.json - Incomplete run, achieved 58.9% best
- V2_Rnd_3.json - Empty/failed run
- V2_Rnd_4.json - Empty/failed run
""",
        'v3_runs/README.md': """# V3 Training Runs

OLYMPUS V3 Ultimate Ensemble Training runs with:
- Full Specialist Coordination
- Ultimate Meta-Learning
- Ensemble Self-Attention

Files:
- V3_Rnd_2.json - Complete run, 59.9% final (59.8% best on 4x4)
- V3_Rnd_3.json - Complete run, 63.1% final (62.0% best on 6x6)
- v3_rND_1.json - Empty/failed run
""",
        'v3_lower_runs/README.md': """# V3 Lower Stages Training

Focused training on lower stages (0-5, grids 4x4-9x9) for 85%+ target.

Files:
- v3_Lower_Rnd_1.json - Best performer! 64.5% final (63.9% on 5x5)
- V3_Lower_Rnd_2.json - Empty/failed run

These runs use aggressive hyperparameters:
- Higher learning rates
- More epochs
- Increased augmentation
- Smaller batch sizes for more gradient updates
"""
    }
    
    for readme_file, content in readmes.items():
        readme_path = os.path.join(base_dir, readme_file)
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"  ✓ Created {readme_file}")
    
    # Create a summary JSON with best performances
    best_performances = {
        "last_updated": datetime.now().isoformat(),
        "overall_best": {
            "performance": 64.5,
            "file": "v3_Lower_Rnd_1.json",
            "type": "V3 Lower Stages"
        },
        "grid_bests": {
            "4x4": {"performance": 59.8, "file": "V3_Rnd_2.json"},
            "5x5": {"performance": 63.9, "file": "v3_Lower_Rnd_1.json"},
            "6x6": {"performance": 63.8, "file": "v3_Lower_Rnd_1.json"},
            "8x8": {"performance": 59.5, "file": "V3_Rnd_3.json"},
            "10x10": {"performance": 59.6, "file": "V3_Rnd_3.json"},
            "16x16": {"performance": 57.0, "file": "V3_Rnd_2.json"},
            "30x30": {"performance": 50.4, "file": "V3_Rnd_2.json"}
        },
        "target": "85%+ on lower stages (4x4-9x9)"
    }
    
    with open(os.path.join(base_dir, 'analysis', 'best_performances.json'), 'w') as f:
        json.dump(best_performances, f, indent=4)
    
    print(f"\n✅ Organization complete! Moved {moved_count} files.")
    print(f"✅ Created {len(readmes)} README files.")
    print(f"✅ Generated best_performances.json summary.")
    
    # List final structure
    print("\n📁 New directory structure:")
    for dir_name in directories:
        dir_path = os.path.join(base_dir, dir_name)
        file_count = len(os.listdir(dir_path))
        print(f"  {dir_name}/ ({file_count} files)")

if __name__ == "__main__":
    organize_training_reports()