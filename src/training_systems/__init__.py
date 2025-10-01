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

# Future model-specific systems will be added here:
# - IRIS systems (color pattern focus)
# - ATLAS systems (spatial transformation focus)  
# - CHRONOS systems (temporal sequence focus)
# - PROMETHEUS systems (meta-learning focus)