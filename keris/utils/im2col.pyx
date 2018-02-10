"""
This module is an optimized version of im2col implementation from
[Stanford CS231n assignment](http://cs231n.stanford.edu/assignments.html)
"""
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float32
ctypedef np.float32_t[:, :] ndarray2D_t
ctypedef np.float32_t[:, :, :, :] ndarray4D_t


@cython.wraparound(False)
cpdef ndarray2D_t im2col(ndarray4D_t x, int field_height, int field_width,
                                int padding, int stride) except *:
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]

    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1

    cdef int p = padding
    cdef ndarray4D_t x_padded = np.pad(
        x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    cdef ndarray2D_t cols = np.zeros(
        (C * field_height * field_width, N * HH * WW),
        dtype=DTYPE)

    # Moving the inner loop to a C function with no bounds checking works, but
    # does not seem to help performance in any measurable way.

    im2col_inner(cols, x_padded, N, C, H, W, HH, WW,
                        field_height, field_width, padding, stride)
    return cols


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_inner(ndarray2D_t cols, ndarray4D_t x_padded, int N,
                             int C, int H, int W, int HH, int WW,
                             int field_height, int field_width,
                             int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for yy in range(HH):
            for xx in range(WW):
                for ii in range(field_height):
                    for jj in range(field_width):
                        row = c * field_width * field_height + ii * field_height + jj
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            cols[row, col] = x_padded[i, c,
                                                      stride * yy + ii,
                                                      stride * xx + jj]


cpdef ndarray4D_t col2im(ndarray2D_t cols, int N, int C, int H, int W,
                                int field_height, int field_width,
                                int padding, int stride) except *:
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=DTYPE)
    cdef int HH = (H + 2 * padding - field_height) / stride + 1
    cdef int WW = (W + 2 * padding - field_width) / stride + 1
    cdef ndarray4D_t x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding),
                                         dtype=DTYPE)

    # Moving the inner loop to a C-function with no bounds checking improves
    # performance quite a bit for col2im.
    col2im_inner(cols, x_padded, N, C, H, W, HH, WW,
                        field_height, field_width, padding, stride)
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_inner(ndarray2D_t cols, ndarray4D_t x_padded, int N,
                             int C, int H, int W, int HH, int WW,
                             int field_height, int field_width,
                             int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, i, col

    for c in range(C):
        for ii in range(field_height):
            for jj in range(field_width):
                row = c * field_width * field_height + ii * field_height + jj
                for yy in range(HH):
                    for xx in range(WW):
                        for i in range(N):
                            col = yy * WW * N + xx * N + i
                            x_padded[i, c, stride * yy + ii,
                                     stride * xx + jj] += cols[row, col]
