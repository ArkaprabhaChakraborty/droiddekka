import numpy as np


def reshape_x(x, dim_x, ndim):
    """ ensure x is a (dim_x, 1) shaped vector"""
    x = np.atleast_2d(x)
    if x.shape[1] == dim_x:
        x = x.T
    if x.shape != (dim_x, 1):
        raise ValueError('x (shape {}) must be convertible to shape ({}, 1)'.format(x.shape, dim_x))

    if ndim == 1:
        x = x[:, 0]

    if ndim == 0:
        x = x[0, 0]

    return x