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

# Model-specific DSL imports
try:
    from .minerva_dsl import (
        MINERVADSLGenerator,
        MINERVADSLTraining
    )
    MINERVA_DSL_AVAILABLE = True
except ImportError:
    MINERVA_DSL_AVAILABLE = False

__all__ = [
    'Operation',
    'DSLProgram',
    'DSLExecutor',
    'DSLProgramGenerator',
    'DSLProgramPredictor',
    'DSLAugmentedDataset',
    'DSLTrainingIntegration'
]

# Add MINERVA DSL exports if available
if MINERVA_DSL_AVAILABLE:
    __all__.extend(['MINERVADSLGenerator', 'MINERVADSLTraining'])