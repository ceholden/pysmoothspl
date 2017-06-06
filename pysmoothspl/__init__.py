""" Smoothing Spline
"""
from ._sbart import _sbart as sbart  # TODO: friendlier wrapper

__version__ = '0.0.1'


__all__ = [
    'sbart'
]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
