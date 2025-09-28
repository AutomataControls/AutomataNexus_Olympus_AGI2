# AutomataNexus OLYMPUS AGI2

![AutomataNexus](https://img.shields.io/badge/AutomataNexus-AI-06b6d4?labelColor=64748b)
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

OLYMPUS AGI2 is an advanced ensemble system for solving Abstract Reasoning Corpus (ARC) tasks. The system integrates 5 specialized neural networks with intelligent task routing, shape prediction, and post-processing heuristics to achieve high accuracy on complex reasoning problems.

## Architecture

The OLYMPUS system consists of:

- **5 Specialized Neural Networks**: Each model focuses on different aspects of reasoning
- **Ensemble Voting System**: Weighted voting based on task characteristics
- **Grid Size Predictor**: Advanced shape prediction with 50+ rule patterns
- **Task Router**: Analyzes tasks to assign optimal model weights
- **Heuristic Pipeline**: Post-processing to fix common prediction errors

## Key Features

- **Smart Ensemble Strategy**: Dynamic model weighting based on task analysis
- **Comprehensive Shape Prediction**: Handles scaling, cropping, object-based transformations
- **Multi-Stage Training**: Curriculum learning with progressive difficulty
- **Enhanced Loss Functions**: Exact match bonuses and transformation penalties
- **Robust Error Handling**: Heuristic solvers for near-miss corrections

## Project Structure

```
AutomataNexus_Olympus_AGI2/
├── src/
│   ├── core/                 # Main OLYMPUS components
│   │   ├── olympus_ensemble_runner.py    # Complete system integration
│   │   ├── ensemble_test_bench.py        # Base ensemble with 5 models
│   │   ├── ensemble_with_size_prediction.py  # Enhanced with shape prediction
│   │   ├── task_router.py                # Intelligent task analysis
│   │   └── heuristic_solvers.py          # Post-processing fixes
│   ├── models/               # Neural network architectures
│   │   └── arc_models_enhanced.py        # All 5 OLYMPUS models
│   ├── utils/                # Utilities
│   │   └── grid_size_predictor_v2.py     # Advanced shape prediction
│   └── evaluation/           # Evaluation scripts
├── scripts/
│   └── training/             # Training scripts
│       └── colab_training_v4_megascale_curriculum.py
├── tests/                    # Unit tests
├── data/                     # ARC dataset
└── archive/                  # Old/reference files
```

## Neural Network Models

Our solution features 5 custom PyTorch models, each specialized for different reasoning tasks:

### 1. **MINERVA** - Strategic Pattern Analysis
- Vision Transformer architecture with pattern memory bank
- 8.7M parameters
- Handles strategic reasoning and decision making

### 2. **ATLAS** - Spatial Transformations  
- Spatial Transformer Network (STN)
- 3.5M parameters
- Specializes in geometric transformations and structural analysis

### 3. **IRIS** - Color Pattern Recognition
- Attention-based color relationship analyzer
- 4.2M parameters
- Masters color mappings and harmony detection

### 4. **CHRONOS** - Temporal Sequences
- Bidirectional LSTM with evolution prediction
- 6.1M parameters
- Tracks pattern evolution and sequences

### 5. **PROMETHEUS** - Creative Pattern Generation
- Variational Autoencoder (VAE) architecture
- 9.3M parameters
- Generates novel pattern solutions

## System Components

### Task Router
Analyzes task characteristics to determine optimal model weights:
- Grid size and aspect ratio analysis
- Color complexity detection
- Transformation pattern recognition
- Object counting and density analysis

### Grid Size Predictor V2
Comprehensive shape prediction with 50+ rule patterns:
- Exact match and consistent size detection
- Fractional scaling (1/2, 1/3, 1/4, 2x, 3x, etc.)
- Object-based sizing (bounding box, object count)
- Pattern-based transformations
- Extreme reduction rules for edge cases

### Heuristic Solvers
Post-processing pipeline to fix common errors:
- Symmetry completion
- Boundary fixes
- Color consistency
- Pattern filling
- Edge cleanup

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- SciPy
- CUDA GPU (for training)
- 16GB+ RAM
- 40GB+ GPU memory (for mega-scale training)

## Performance

- **Inference Speed**: ~100ms per task
- **Training Time**: 300 epochs with curriculum learning
- **Batch Size**: 512 (effective 2048 with gradient accumulation)
- **Current Accuracy**: Improving from 0.1% baseline

## Installation

```bash
# Clone repository
git clone https://github.com/AutomataControls/AutomataNexus_Olympus_AGI2.git
cd AutomataNexus_Olympus_AGI2

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```python
# Run mega-scale training with curriculum learning
python scripts/training/colab_training_v4_megascale_curriculum.py
```

### Evaluation

```python
from src.core.olympus_ensemble_runner import OLYMPUSRunner

# Initialize system
olympus = OLYMPUSRunner(model_dir='/path/to/models')

# Evaluate on training set
results = olympus.evaluate_on_training_set(
    'data/arc-agi_training_challenges.json',
    n_tasks=50
)

# Make predictions
result = olympus.predict(input_grid, train_examples)
```

### Testing Individual Components

```python
# Test ensemble
from src.core.ensemble_test_bench import OLYMPUSEnsemble
ensemble = OLYMPUSEnsemble()

# Test shape prediction
from src.utils.grid_size_predictor_v2 import GridSizePredictorV2
predictor = GridSizePredictorV2()
shape = predictor.predict_output_shape(input_grid, train_examples)
```

## Training Strategy

### Curriculum Learning
- **Stage 0**: Easy tasks (small grids, few colors)
- **Stage 1**: Medium complexity
- **Stage 2**: Full dataset

### Loss Function Enhancements
- Exact match bonus (5.0x reward)
- Edge-aware weighting
- Transformation penalty (prevents copying)
- Active region focus

### Data Augmentation
- 10x augmentation with rotations and flips
- Maintains transformation consistency

## Technical Details

### Ensemble Voting
- Weighted voting based on task characteristics
- Dynamic weight adjustment per model
- Confidence scoring system

### Model Specializations
- **MINERVA**: General pattern recognition, strategic analysis
- **ATLAS**: Spatial transformations, geometric reasoning
- **IRIS**: Color patterns, palette transformations
- **CHRONOS**: Sequential patterns, temporal evolution
- **PROMETHEUS**: Creative solutions, novel patterns

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Andrew Jewell Sr.**  
AutomataNexus, LLC