""" Array wrapper to help with memory deallocation

Modified from code originally published by Gael Varoquaux under BSD license:
https://gist.github.com/GaelVaroquaux/1249305

"""
from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport free
import logging

import numpy as np
cimport numpy as np

np.import_array()

logger = logging.getLogger(__name__)


cdef class ArrayWrapper:
    #: pointer to data
    cdef void *data_ptr
    #: number of elements in array
    cdef int size
    #: data type
    cdef int dtype

    cdef set_data(self, int size, int dtype, void* data_ptr):
        logger.debug('Setting array wrapper data')
        self.size = size
        self.dtype = dtype
        self.data_ptr = data_ptr

    def __array__(self):
        logger.debug('Converting to np.ndarray')
        # This gets called when NumPy tries to get an array
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size

        ndarray = np.PyArray_SimpleNewFromData(
            1, shape, self.dtype, self.data_ptr
        )
        return ndarray

    def __dealloc__(self):
        logger.debug('Deallocating memory...')
        free(<void*> self.data_ptr)


cdef np.ndarray to_ndarray(int size, int dtype, void *array_ptr):
    """ Wrap an array pointer with a deallocator and return as np.ndarray
    """
    cdef np.ndarray ndarray

    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(size, dtype, array_ptr)

    ndarray = np.array(array_wrapper, copy=False)
    ndarray.base = <PyObject*> array_wrapper

    Py_INCREF(array_wrapper)

    return ndarray
