""" Wrapper around C versions of `sbart.c` for R's `smooth.spline`
"""
import logging

from libc.stdlib cimport malloc, free

cimport cython
import numpy as np
cimport numpy as np

import _array_wrap
cimport _array_wrap

logger = logging.getLogger(__name__)

# Initialize NumPy
np.import_array()


cdef extern from "sbart.h":
    int sbart(
        double *xs,
        double *ys,
        double *ws,
        int n,
        double *knot,
        int nk,
        double *coef,
        double *sz,
        double spar,
        int ispar,
        double *lspar,
        int *isetup,
        double *xwy,
        double *hs0,
        double *hs1,
        double *hs2,
        double *hs3,
        double *sg0,
        double *sg1,
        double *sg2,
        double *sg3,
        double *abd,
        int ld4
    )
    double c_bvalue(
        double *t,
        double *bcoef,
        int n,
        int k,
        double x,
        int jderiv
    )


cdef _nknots(int n):
    """ Calculate a "good" amount of knots
    """
    if n < 500:
        return n
    cdef double a, a1, a2, a3, a4
    a1 = np.log2(50)
    a2 = np.log2(100)
    a3 = np.log2(140)
    a4 = np.log2(200)

    if n < 200:
        a = 2 ** (a1 + (a2 -  a1) * (n - 50.0) / 150.0)
    elif n < 800:
        a = 2 ** (a2 + (a3 - a2) * (n - 200.0) / 600.0)
    elif n < 3200:
        a = 2 ** (a3 + (a4 - a3) * (n - 800.0) / 2400.0)
    else:
        a = 200 + (n - 3200.0) ** 0.2
    return int(a)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _sbart(np.ndarray[np.double_t, ndim=1] xs,
             np.ndarray[np.double_t, ndim=1] ys,
             np.ndarray[np.double_t, ndim=1] ws,
             double spar):
    """ Compute a smoothing spline using `sbart` (R's `smooth.spline`)
    """
    assert ys.shape[0] == ws.shape[0] == xs.shape[0]

    # Defaults in `contr.sp` or from `smooth.spline`
    cdef double lspar = -1.5
    cdef double uspar = 1.5
    cdef int ld4 = 4
    cdef int ldnk = 1
    cdef int isetup = 0
    cdef int ier, n, nk
    cdef double xs_range, xs_min

    # TODO: cross-validation type
    # TODO: ispar (if we want it estimated)
    cdef int ispar = 1

    n = xs.shape[0]
    # Prepare inputs
    # 1. Normalize weights
    ws = (ws * (ws > 0).sum()) / ws.sum()

    # 2. Scale xs to [0, 1]
    xs_min = xs[0]
    xs_range = (xs[n - 1] - xs[0])
    xs = (xs - xs[0]) / xs_range

    # 3. Calculate knots
    nk = _nknots(n)
    cdef np.ndarray[np.double_t, ndim=1] knots = np.concatenate((
        np.repeat(xs[0], 3),
        xs[np.linspace(0, n - 1, nk, dtype=np.int)],
        np.repeat(xs[n - 1], 3)
    ))
    nk += 2

    logger.debug('Calculated {nk} knots for {n} observations'.
                 format(nk=nk, n=n))

    # 3. Allocate
    logger.debug('Allocating work and output arrays')
    cdef double *sz = <double*> malloc(sizeof(double) * n)
    cdef double *coef = <double*> malloc(sizeof(double) * nk)
    cdef double *xwy = <double*> malloc(sizeof(double) * nk)
    cdef double *hs0 = <double*> malloc(sizeof(double) * nk)
    cdef double *hs1 = <double*> malloc(sizeof(double) * nk)
    cdef double *hs2 = <double*> malloc(sizeof(double) * nk)
    cdef double *hs3 = <double*> malloc(sizeof(double) * nk)
    cdef double *sg0 = <double*> malloc(sizeof(double) * nk)
    cdef double *sg1 = <double*> malloc(sizeof(double) * nk)
    cdef double *sg2 = <double*> malloc(sizeof(double) * nk)
    cdef double *sg3 = <double*> malloc(sizeof(double) * nk)
    cdef double *abd = <double*> malloc(sizeof(double) * ld4 * nk)
    cdef double *p1ip = <double*> malloc(sizeof(double) * ld4 * nk)
    cdef double *p2ip = <double*> malloc(sizeof(double) * ldnk * nk)
    logger.debug('Allocated memory...')

    try:
        ier = sbart(<double*> xs.data,
                    <double*> ys.data,
                    <double*> ws.data,
                    n,
                    <double*> knots.data, nk,
                    coef, sz,
                    spar, ispar, &lspar, &isetup,
                    xwy,
                    hs0, hs1, hs2, hs3,
                    sg0, sg1, sg2, sg3,
                    abd, ld4)
        if ier != 0:
            raise RuntimeError('An error occurred within `sbart`. '
                               'Return code {0}'.format(ier))
    finally:
        # Make sure we don't leak even if there's an exception
        logger.debug('Deallocating workspace memory')
        free(xwy)
        free(hs0)
        free(hs1)
        free(hs2)
        free(hs3)
        free(sg0)
        free(sg1)
        free(sg2)
        free(sg3)
        free(abd)
        free(p1ip)
        free(p2ip)

    logger.debug('Converting membuffer into np.ndarray')
    coefarr = _array_wrap.to_ndarray(nk, np.NPY_DOUBLE, coef)
    szarr = _array_wrap.to_ndarray(n, np.NPY_DOUBLE, sz)

    return knots, coefarr, szarr, xs_min, xs_range


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray _bvalues(np.ndarray[np.double_t, ndim=1] knots,
                          np.ndarray[np.double_t, ndim=1] coef,
                          np.ndarray[np.double_t, ndim=1] xs,
                          double fit_xs_min,
                          double fit_xs_range,
                          int derivative):
    """ Calculate spline fit for some x
    """
    cdef int n, nk
    xs = (xs - fit_xs_min) / fit_xs_range

    n = xs.shape[0]
    nk = knots.shape[0]

    logger.debug('Allocating memory...')
    cdef double *yhat = <double*> malloc(sizeof(double) * n)

    # We have to do it in loop over native code since `c_bvalue` evaluates for
    # one `x`
    for i in range(n):
        yhat[i] = c_bvalue(
            <double*> knots.data,
            <double*> coef.data,
            nk - 4, 4, xs[i], derivative
        )

    yhatarr = _array_wrap.to_ndarray(n, np.NPY_DOUBLE, yhat)

    return yhatarr
