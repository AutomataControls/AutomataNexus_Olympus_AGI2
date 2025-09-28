"""
ARC Domain-Specific Language (DSL) Module
For deterministic program synthesis and exact transformations
"""

from .arc_dsl import (
    Operation,
    DSLProgram,
    DSLExecutor,
    DSLProgramGenerator,
    DSLProgramPredictor
)

from .dsl_training import (
    DSLAugmentedDataset,
    DSLTrainingIntegration
)

__all__ = [
    'Operation',
    'DSLProgram',
    'DSLExecutor',
    'DSLProgramGenerator',
    'DSLProgramPredictor',
    'DSLAugmentedDataset',
    'DSLTrainingIntegration'
]