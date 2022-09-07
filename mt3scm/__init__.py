"""
MT3SCM
======
This module provides 'Multivariate Time Series Sub-Sequence CLustering Metric' computation function
"""
__version__ = "0.4.7"

from .mt3scm_computation import MT3SCM, mt3scm_score

__all__ = ["mt3scm_score", "MT3SCM"]
