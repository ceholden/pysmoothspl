""" Declarations for _array_wrap.pyx
"""
import numpy as np
cimport numpy as np

np.import_array()


cdef np.ndarray to_ndarray(int size, int dtype, void *array_ptr)
