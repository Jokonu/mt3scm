"""
MT3SCM
======
This module provides 'Multivariate Time Series Sub-Sequence CLustering Metric' computation function
"""
__version__ = "0.1.0"

from .mt3scm_computation import mt3scm_score, MT3SCM

__all__ = ["mt3scm_score", "MT3SCM"]
