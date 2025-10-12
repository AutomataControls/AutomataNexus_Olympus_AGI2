# OLYMPUS Training Reports

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



  Fast Models (5-15s):
  ollama pull gemma2:2b
  ollama pull llama3.2:latest
  ollama pull stable-code:3b
  ollama pull gemma3:4b

  Medium Models (20-30s):
  ollama pull qwen2.5-coder:7b
  ollama pull qwen2-math:7b
  ollama pull command-r:7b
  ollama pull llama3-chatqa:8b

    For Complex Reasoning & Architecture:
  # Best balance for your setup (12B - recommended!)
  ollama pull mistral-nemo:12b

  # Alternative if you want even better reasoning (14B)
  ollama pull qwen2.5:14b

  # For complex code architecture (16B - will be slower)
  ollama pull deepseek-coder-v2:16b

