""" Smoothing Spline
"""
# TODO: friendlier wrappers
from ._sbart import _sbart as sbart, _bvalues as bvalues
from .core import SmoothSpline

__all__ = [
    'SmoothSpline',
    'bvalues',
    'sbart'
]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
