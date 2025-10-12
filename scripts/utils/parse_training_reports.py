"""
Parse and refactor OLYMPUS training reports from JSON format
Extracts key metrics and creates clean summaries
"""

import json
import os
import pandas as pd
from typing import Dict, List, Tuple
import re
from datetime import datetime
import numpy as np

def parse_training_line(line: str) -> Dict:
    """Parse a single line from training output"""
    metrics = {}
    
    # Extract stage info
    stage_match = re.search(r'Stage (\d+):', line)
    if stage_match:
        metrics['stage'] = int(stage_match.group(1))
    
    # Extract epoch info
    epoch_match = re.search(r'Epoch (\d+)', line)
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
    
    # Extract performance
    perf_match = re.search(r'Performance=(\d+\.?\d*)%', line)
    if perf_match:
        metrics['performance'] = float(perf_match.group(1))
    
    # Extract loss
    loss_match = re.search(r'Loss=([-]?\d+\.?\d*)', line)
    if loss_match:
        metrics['loss'] = float(loss_match.group(1))
    
    # Extract consensus
    consensus_match = re.search(r'Consensus=(\d+\.?\d*)', line)
    if consensus_match:
        metrics['consensus'] = float(consensus_match.group(1))
    
    # Extract grid size
    grid_match = re.search(r'Grid Size (\d+)', line)
    if grid_match:
        metrics['grid_size'] = int(grid_match.group(1))
        
    # Extract exact match percentage
    exact_match = re.search(r'(\d+\.?\d*)% exact', line)
    if exact_match:
        metrics['exact_match'] = float(exact_match.group(1))
        
    # Check if it's a warmup epoch
    if 'Warmup' in line:
        metrics['is_warmup'] = True
    
    return metrics

def parse_json_report(filepath: str) -> Tuple[List[Dict], Dict]:
    """Parse a JSON training report and extract metrics"""
    
    with open(filepath, 'r') as f:
        # Read as text first since these are actually training logs
        content = f.read()
    
    all_metrics = []
    summary_info = {
        'file': os.path.basename(filepath),
        'version': 'V2' if 'V2' in filepath else 'V3',
        'best_performance': 0,
        'final_performance': 0,
        'total_epochs': 0,
        'stages_completed': 0,
        'stage_performances': {}
    }
    
    lines = content.split('\n')
    current_stage = -1
    
    for line in lines:
        if not line.strip():
            continue
            
        # Parse metrics from line
        metrics = parse_training_line(line)
        
        if metrics:
            if 'stage' in metrics:
                current_stage = metrics['stage']
            elif current_stage >= 0:
                metrics['stage'] = current_stage
            
            if 'performance' in metrics or 'exact_match' in metrics:
                all_metrics.append(metrics)
                
                # Update summary
                perf = metrics.get('exact_match', metrics.get('performance', 0))
                if perf > summary_info['best_performance']:
                    summary_info['best_performance'] = perf
                    
                # Track stage performances
                if 'stage' in metrics:
                    stage = metrics['stage']
                    if stage not in summary_info['stage_performances']:
                        summary_info['stage_performances'][stage] = {
                            'best': perf,
                            'start': perf,
                            'end': perf,
                            'grid_size': metrics.get('grid_size', 0)
                        }
                    else:
                        if perf > summary_info['stage_performances'][stage]['best']:
                            summary_info['stage_performances'][stage]['best'] = perf
                        summary_info['stage_performances'][stage]['end'] = perf
        
        # Check for completion messages
        if 'Training Complete!' in line:
            complete_match = re.search(r'Best.*Performance: (\d+\.?\d*)%', line)
            if complete_match:
                summary_info['final_performance'] = float(complete_match.group(1))
    
    summary_info['stages_completed'] = len(summary_info['stage_performances'])
    summary_info['total_epochs'] = len(all_metrics)
    
    return all_metrics, summary_info

def create_summary_dataframe(summaries: List[Dict]) -> pd.DataFrame:
    """Create a summary DataFrame from multiple training runs"""
    
    rows = []
    for summary in summaries:
        row = {
            'file': summary['file'],
            'version': summary['version'],
            'best_performance': summary['best_performance'],
            'final_performance': summary['final_performance'],
            'stages_completed': summary['stages_completed'],
            'total_epochs': summary['total_epochs']
        }
        
        # Add stage-specific columns
        for stage, perf in summary['stage_performances'].items():
            row[f'stage_{stage}_best'] = perf['best']
            row[f'stage_{stage}_improvement'] = perf['end'] - perf['start']
            row[f'stage_{stage}_grid'] = perf['grid_size']
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    """Process all training reports"""
    reports_dir = '/mnt/d/opt/AutomataNexus_Olympus_AGI2/src/models/reports/TrainingReport'
    
    # Get all JSON files
    json_files = [f for f in os.listdir(reports_dir) if f.endswith('.json')]
    
    print(f"Found {len(json_files)} training reports to process\n")
    
    all_summaries = []
    all_metrics_by_file = {}
    
    for json_file in sorted(json_files):
        filepath = os.path.join(reports_dir, json_file)
        print(f"Processing {json_file}...")
        
        try:
            metrics, summary = parse_json_report(filepath)
            all_summaries.append(summary)
            all_metrics_by_file[json_file] = metrics
            
            # Print summary for this file
            print(f"  Version: {summary['version']}")
            print(f"  Best Performance: {summary['best_performance']:.2f}%")
            print(f"  Stages Completed: {summary['stages_completed']}")
            print(f"  Stage Performances:")
            for stage, perf in sorted(summary['stage_performances'].items()):
                print(f"    Stage {stage} ({perf['grid_size']}x{perf['grid_size']}): "
                      f"{perf['start']:.1f}% → {perf['end']:.1f}% (best: {perf['best']:.1f}%)")
            print()
        except Exception as e:
            print(f"  Error processing file: {e}")
            print()
    
    # Create summary DataFrame
    df_summary = create_summary_dataframe(all_summaries)
    
    # Save summary to CSV
    summary_path = os.path.join(reports_dir, 'training_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    # Print overall statistics
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total runs analyzed: {len(all_summaries)}")
    
    v2_runs = [s for s in all_summaries if s['version'] == 'V2']
    v3_runs = [s for s in all_summaries if s['version'] == 'V3']
    
    if v2_runs:
        v2_best = max(s['best_performance'] for s in v2_runs)
        print(f"\nV2 Statistics:")
        print(f"  Total runs: {len(v2_runs)}")
        print(f"  Best performance: {v2_best:.2f}%")
        
    if v3_runs:
        v3_best = max(s['best_performance'] for s in v3_runs)
        print(f"\nV3 Statistics:")
        print(f"  Total runs: {len(v3_runs)}")
        print(f"  Best performance: {v3_best:.2f}%")
    
    # Save detailed metrics for each file
    for filename, metrics in all_metrics_by_file.items():
        if metrics:
            df_metrics = pd.DataFrame(metrics)
            metrics_path = os.path.join(reports_dir, f"{filename.replace('.json', '')}_metrics.csv")
            df_metrics.to_csv(metrics_path, index=False)
    
    print("\n✅ Training report parsing complete!")
    
    return df_summary, all_metrics_by_file

if __name__ == "__main__":
    main()