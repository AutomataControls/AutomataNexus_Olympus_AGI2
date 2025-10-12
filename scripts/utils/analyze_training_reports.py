"""
Analyze all JSON training reports and create comprehensive summary
"""

import json
import os
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

def load_json_report(filepath: str) -> Dict:
    """Load and parse a JSON training report"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_performance_data(report: Dict) -> Dict[str, Any]:
    """Extract key performance metrics from a report"""
    result = {
        'title': report.get('title', 'Unknown'),
        'final_performance': report.get('finalPerformance', 0),
        'completed': report.get('completed', False),
        'stages': []
    }
    
    # Handle different JSON formats
    stages_key = 'stages' if 'stages' in report else 'training_stages'
    stages = report.get(stages_key, [])
    
    for stage in stages:
        stage_data = {
            'stage_id': stage.get('stageId', stage.get('stage_id', -1)),
            'grid_size': stage.get('gridSize', stage.get('grid_size', 0)),
            'epochs': []
        }
        
        # Get epochs data
        epochs = stage.get('epochs', stage.get('performance_history', []))
        for epoch in epochs:
            if isinstance(epoch, dict):
                perf = epoch.get('performance', epoch.get('exact_match', 0))
                if perf > 0:
                    stage_data['epochs'].append({
                        'epoch': epoch.get('epoch', 0),
                        'performance': perf,
                        'loss': epoch.get('loss', 0),
                        'is_final': epoch.get('final', False)
                    })
        
        if stage_data['epochs']:
            stage_data['start_perf'] = stage_data['epochs'][0]['performance']
            stage_data['end_perf'] = stage_data['epochs'][-1]['performance']
            stage_data['best_perf'] = max(e['performance'] for e in stage_data['epochs'])
            stage_data['improvement'] = stage_data['end_perf'] - stage_data['start_perf']
        else:
            stage_data['start_perf'] = 0
            stage_data['end_perf'] = 0
            stage_data['best_perf'] = 0
            stage_data['improvement'] = 0
            
        result['stages'].append(stage_data)
    
    # Calculate overall best
    if result['stages']:
        result['best_performance'] = max(s['best_perf'] for s in result['stages'])
    else:
        result['best_performance'] = result['final_performance']
    
    return result

def create_summary_report():
    """Create comprehensive summary of all training reports"""
    reports_dir = '/mnt/d/opt/AutomataNexus_Olympus_AGI2/src/models/reports/TrainingReport'
    
    # Get all JSON files
    json_files = [f for f in os.listdir(reports_dir) if f.endswith('.json') and not f.endswith('_raw.txt')]
    
    all_reports = []
    
    for json_file in sorted(json_files):
        filepath = os.path.join(reports_dir, json_file)
        
        try:
            report = load_json_report(filepath)
            perf_data = extract_performance_data(report)
            perf_data['filename'] = json_file
            all_reports.append(perf_data)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Create summary table
    summary_rows = []
    for report in all_reports:
        row = {
            'File': report['filename'],
            'Version': 'V3' if 'V3' in report['filename'] else 'V2',
            'Best Performance': f"{report['best_performance']:.1f}%",
            'Final Performance': f"{report['final_performance']:.1f}%",
            'Stages Completed': len(report['stages']),
            'Status': '✅' if report['completed'] else '❌'
        }
        
        # Add key stage performances
        for stage in report['stages']:
            if stage['grid_size'] in [4, 8, 16, 30]:
                col_name = f"{stage['grid_size']}x{stage['grid_size']} Best"
                row[col_name] = f"{stage['best_perf']:.1f}%"
        
        summary_rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_rows)
    
    # Print summary
    print("\n" + "="*120)
    print("OLYMPUS TRAINING REPORTS SUMMARY")
    print("="*120)
    
    # Group by version
    for version in ['V2', 'V3']:
        version_df = df[df['Version'] == version]
        if not version_df.empty:
            print(f"\n{version} Training Runs:")
            print("-"*60)
            for _, row in version_df.iterrows():
                print(f"  {row['File']:<30} Best: {row['Best Performance']:<8} Final: {row['Final Performance']:<8} Status: {row['Status']}")
    
    # Find best performers
    print("\n" + "-"*60)
    print("TOP PERFORMERS:")
    print("-"*60)
    
    # Get best overall
    best_overall = max(all_reports, key=lambda x: x['best_performance'])
    print(f"  Best Overall: {best_overall['filename']} - {best_overall['best_performance']:.1f}%")
    
    # Get best by stage
    stage_bests = {}
    for report in all_reports:
        for stage in report['stages']:
            grid = stage['grid_size']
            if grid not in stage_bests or stage['best_perf'] > stage_bests[grid]['perf']:
                stage_bests[grid] = {
                    'perf': stage['best_perf'],
                    'file': report['filename']
                }
    
    print("\n  Best by Grid Size:")
    for grid in sorted(stage_bests.keys()):
        if grid > 0:
            print(f"    {grid}x{grid}: {stage_bests[grid]['perf']:.1f}% ({stage_bests[grid]['file']})")
    
    # Save detailed CSV
    csv_path = os.path.join(reports_dir, 'training_analysis_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved detailed summary to {csv_path}")
    
    # Create performance trends plot
    create_performance_plot(all_reports, reports_dir)
    
    return df, all_reports

def create_performance_plot(all_reports: List[Dict], output_dir: str):
    """Create performance visualization"""
    
    # Group reports by version
    v2_reports = [r for r in all_reports if 'V2' in r['filename']]
    v3_reports = [r for r in all_reports if 'V3' in r['filename']]
    
    # Get grid sizes
    all_grids = set()
    for report in all_reports:
        for stage in report['stages']:
            all_grids.add(stage['grid_size'])
    grid_sizes = sorted(list(all_grids))
    
    # Plot data
    plt.figure(figsize=(12, 8))
    
    # Plot V2 runs
    for report in v2_reports:
        perfs = []
        grids = []
        for grid in grid_sizes:
            stage = next((s for s in report['stages'] if s['grid_size'] == grid), None)
            if stage:
                perfs.append(stage['best_perf'])
                grids.append(grid)
        if perfs:
            plt.plot(grids, perfs, 'o-', alpha=0.6, label=f"V2: {report['filename'][:15]}")
    
    # Plot V3 runs
    for report in v3_reports:
        perfs = []
        grids = []
        for grid in grid_sizes:
            stage = next((s for s in report['stages'] if s['grid_size'] == grid), None)
            if stage:
                perfs.append(stage['best_perf'])
                grids.append(grid)
        if perfs:
            plt.plot(grids, perfs, 's-', alpha=0.8, label=f"V3: {report['filename'][:15]}", linewidth=2)
    
    plt.xlabel('Grid Size')
    plt.ylabel('Best Performance (%)')
    plt.title('OLYMPUS Training Performance by Grid Size')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'training_performance_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Saved performance plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    df, reports = create_summary_report()