"""
Model-specific training systems for AutomataNexus OLYMPUS AGI2
Each model has its own specialized LEAP, MEPT, and PRISM implementations
"""

# MINERVA training systems
try:
    from .minerva_leap import create_minerva_leap_system, MinervaLEAPTrainer, MinervaPatternGenerator
    from .minerva_mept import create_minerva_mept_system, MinervaMEPTLoss, MinervaExperienceReplayBuffer, MinervaPatternBank
    from .minerva_prism import create_minerva_prism_system, MinervaProgramSynthesizer, MinervaProgramLibrary
    MINERVA_SYSTEMS_AVAILABLE = True
except ImportError as e:
    MINERVA_SYSTEMS_AVAILABLE = False
    print(f"Warning: MINERVA training systems not available: {e}")

# Export what's available
__all__ = []

if MINERVA_SYSTEMS_AVAILABLE:
    __all__.extend([
        'create_minerva_leap_system', 'MinervaLEAPTrainer', 'MinervaPatternGenerator',
        'create_minerva_mept_system', 'MinervaMEPTLoss', 'MinervaExperienceReplayBuffer', 'MinervaPatternBank',
        'create_minerva_prism_system', 'MinervaProgramSynthesizer', 'MinervaProgramLibrary'
    ])

# IRIS training systems
try:
    from .iris_leap import create_iris_leap_system, IrisLEAPTrainer, IrisPatternGenerator
    from .iris_mept import create_iris_mept_system, IrisMEPTLoss, IrisExperienceReplayBuffer, IrisPatternBank
    from .iris_prism import create_iris_prism_system, IrisProgramSynthesizer, IrisProgramLibrary
    IRIS_SYSTEMS_AVAILABLE = True
except ImportError as e:
    IRIS_SYSTEMS_AVAILABLE = False
    print(f"Warning: IRIS training systems not available: {e}")

if IRIS_SYSTEMS_AVAILABLE:
    __all__.extend([
        'create_iris_leap_system', 'IrisLEAPTrainer', 'IrisPatternGenerator',
        'create_iris_mept_system', 'IrisMEPTLoss', 'IrisExperienceReplayBuffer', 'IrisPatternBank',
        'create_iris_prism_system', 'IrisProgramSynthesizer', 'IrisProgramLibrary'
    ])

# Future model-specific systems will be added here:
# - ATLAS systems (spatial transformation focus)  
# - CHRONOS systems (temporal sequence focus)
# - PROMETHEUS systems (meta-learning focus)