#!/usr/bin/env python3
"""Analyze grid sizes in ARC-AGI training data."""

import json
from collections import Counter, defaultdict
import sys

def analyze_grid_sizes(json_path):
    """Analyze the distribution of grid sizes in ARC-AGI data."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Track grid sizes
    all_grid_sizes = []
    small_grid_counts = defaultdict(int)
    min_size = float('inf')
    min_example = None
    
    # Analyze each task
    for task_id, task_data in data.items():
        if 'train' in task_data:
            for example in task_data['train']:
                input_grid = example['input']
                output_grid = example['output']
                
                input_h, input_w = len(input_grid), len(input_grid[0]) if input_grid else 0
                output_h, output_w = len(output_grid), len(output_grid[0]) if output_grid else 0
                
                # Record all sizes
                all_grid_sizes.extend([(input_h, input_w), (output_h, output_w)])
                
                # Track small grids specifically
                for h, w in [(input_h, input_w), (output_h, output_w)]:
                    if h <= 5 and w <= 5:
                        small_grid_counts[f"{h}x{w}"] += 1
                    
                    # Track minimum
                    size = h * w
                    if size < min_size and size > 0:
                        min_size = size
                        min_example = (task_id, h, w)
        
        # Also check test examples
        if 'test' in task_data:
            for example in task_data['test']:
                input_grid = example['input']
                input_h, input_w = len(input_grid), len(input_grid[0]) if input_grid else 0
                
                all_grid_sizes.append((input_h, input_w))
                
                if input_h <= 5 and input_w <= 5:
                    small_grid_counts[f"{input_h}x{input_w}"] += 1
                
                size = input_h * input_w
                if size < min_size and size > 0:
                    min_size = size
                    min_example = (task_id, input_h, input_w)
    
    # Calculate size distribution
    size_counter = Counter(all_grid_sizes)
    
    # Print results
    print("=== ARC-AGI Grid Size Analysis ===\n")
    
    print(f"1. SMALLEST GRID SIZE:")
    if min_example:
        print(f"   - Size: {min_example[1]}x{min_example[2]} (area={min_example[1]*min_example[2]})")
        print(f"   - Found in task: {min_example[0]}\n")
    
    print(f"2. SMALL GRID COUNTS (≤5x5):")
    # Check for specific sizes
    for size in ["2x2", "3x3", "4x4", "5x5"]:
        count = small_grid_counts.get(size, 0)
        print(f"   - {size}: {count} grids")
    
    # Also check non-square small grids
    print(f"\n   Other small grids:")
    for size, count in sorted(small_grid_counts.items()):
        if size not in ["2x2", "3x3", "4x4", "5x5"]:
            print(f"   - {size}: {count} grids")
    
    print(f"\n3. OVERALL DISTRIBUTION:")
    print(f"   Total grids analyzed: {len(all_grid_sizes)}")
    print(f"   Unique grid sizes: {len(size_counter)}")
    
    # Most common sizes
    print(f"\n   Top 10 most common sizes:")
    for (h, w), count in size_counter.most_common(10):
        percentage = (count / len(all_grid_sizes)) * 100
        print(f"   - {h}x{w}: {count} grids ({percentage:.1f}%)")
    
    # Size ranges
    print(f"\n   Size distribution by area:")
    area_ranges = {
        "Tiny (1-9)": 0,
        "Small (10-25)": 0,
        "Medium (26-100)": 0,
        "Large (101-400)": 0,
        "Huge (400+)": 0
    }
    
    for (h, w), count in size_counter.items():
        area = h * w
        if area <= 9:
            area_ranges["Tiny (1-9)"] += count
        elif area <= 25:
            area_ranges["Small (10-25)"] += count
        elif area <= 100:
            area_ranges["Medium (26-100)"] += count
        elif area <= 400:
            area_ranges["Large (101-400)"] += count
        else:
            area_ranges["Huge (400+)"] += count
    
    for range_name, count in area_ranges.items():
        percentage = (count / len(all_grid_sizes)) * 100
        print(f"   - {range_name}: {count} grids ({percentage:.1f}%)")
    
    # Check for 2x2 grids specifically
    print(f"\n4. 2x2 GRID ANALYSIS:")
    has_2x2 = any(size == (2, 2) for size, _ in size_counter.items())
    print(f"   - Contains 2x2 grids: {'YES' if has_2x2 else 'NO'}")
    if has_2x2:
        count_2x2 = size_counter[(2, 2)]
        print(f"   - Number of 2x2 grids: {count_2x2}")
        print(f"   - Percentage of total: {(count_2x2/len(all_grid_sizes))*100:.2f}%")

if __name__ == "__main__":
    json_path = "/mnt/d/opt/AutomataNexus_Olympus_AGI2/data/arc-agi_training_challenges.json"
    analyze_grid_sizes(json_path)