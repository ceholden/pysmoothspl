""" Wrapper around C versions of `sbart.c` for R's `smooth.spline`
"""
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

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


def _nknots(n):
    if n < 500:
        return n
    a1 = np.log2(50)
    a2 = np.log2(100)
    a3 = np.log2(140)
    a4 = np.log2(200)

    if n < 200:
        a = 2 ** (a1 + (a2 -  a1) * (n - 50) / 150)
    elif n < 800:
        a = 2 ** (a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200:
        a = 2 ** (a3 + (a4 - a3) * (n - 800) / 2400)
    else:
        a = 200 + (n - 3200) ** 0.2
    return int(a)


cdef _sbart(np.ndarray[np.double_t, ndim=1] xs,
           np.ndarray[np.double_t, ndim=1] ys,
           np.ndarray[np.double_t, ndim=1] ws,
           float spar):
    """ Compute a smoothing spline using `sbart` (R's `smooth.spline`)
    """
    # Defaults in `contr.sp` or from `smooth.spline`
    cdef double lspar = -1.5
    cdef double uspar = 1.5
    cdef int ld4 = 4
    cdef int ldnk = 1
    cdef int isetup = 0

#    if cv is True:
#        int icrit = 2
#    else:
#        int icrit = 0
    # TODO: ispar (if we want it estimated)
    cdef int ispar = 1

    cdef int n = xs.shape[0]
    assert ys.shape[0] == ws.shape[0] == n

    cdef int nk = _nknots(n)
    cdef np.ndarray[np.double_t, ndim=1] knots = np.concatenate((
        np.repeat(xs[0], 3),
        xs[np.linspace(0, n - 1, nk, dtype=np.int)],
        np.repeat(xs[-1], 3)
    ))
    nk += 2

    # Prepare inputs
    # 1. normalize weights
    ws = (ws * (ws > 0).sum()) / ws.sum()
    # 2. scale xs to [0, 1]
    xs = (xs - xs[0]) / (xs[-1] - xs[0])
    # 3. allocate
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
        raise RuntimeError('An error occurred within `sbart`. Return code {0}'
                           .format(ier))

    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> n

    cdef np.ndarray[np.double_t, ndim=1] szarr = np.PyArray_SimpleNewFromData(
        1, shape, np.NPY_DOUBLE, sz)

    free(sz)
    free(coef)
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

    return ier, szarr
