"""NaviDiv - Molecular Diversity Analysis Package.

This package provides tools for molecular diversity analysis including:
- Fragment analysis
- Scaffold analysis  
- Similarity scoring
- Clustering
- t-SNE visualization
"""

# Configure RDKit logging to suppress warnings throughout the package
try:
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    # RDKit not available - continue without logging configuration
    pass

__version__ = "0.1.0"
__author__ = "NaviDiv Team"
